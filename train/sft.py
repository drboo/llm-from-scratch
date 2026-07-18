"""
Day 23: SFT (Supervised Fine-Tuning) training loop.

Loads a base pretrained checkpoint and fine-tunes on instruction data
produced by sft/data.py.  Key differences from pretrain.py:

  - Per-example data loading from sft_{split}.bin + sft_{split}_labels.bin +
    sft_{split}_offsets.npy (written by sft/data.py)
  - Loss computed with ignore_index=-100 so only assistant tokens contribute
  - Much lower LR (1e-5 default)
  - Epochs-based: loops over the dataset 2–3 times; stops early if val loss
    rises for 3 consecutive eval periods (overfitting guard)
  - Samples stop at <|endofturn|> / <|eos|> — the key quality check

Usage:
    # Fine-tune from a pretrained checkpoint:
    python train/sft.py --base-ckpt checkpoints/ckpt_050000.pt --data-dir data/sft

    # Override defaults:
    python train/sft.py --base-ckpt checkpoints/ckpt_050000.pt \\
        --data-dir data/sft --epochs 3 --lr 1e-5 --batch-size 16 \\
        --out-dir checkpoints/sft --wandb-project my-llm-sft
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tokeniser"))

from model.gpt import GPT, ModelConfig
from train.checkpoint import save as save_ckpt, load as load_ckpt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SFTConfig:
    # ── data ────────────────────────────────────────────────────────────────
    data_dir:   str = "data/sft"
    base_ckpt:  str = ""          # pretrained checkpoint to start from
    out_dir:    str = "checkpoints/sft"

    # ── model (inferred from base_ckpt if provided, else these are used) ───
    vocab_size: int = 32_000
    d_model:    int = 384
    n_head:     int = 6
    n_layer:    int = 6
    ctx:        int = 256

    # ── training ────────────────────────────────────────────────────────────
    epochs:     int   = 2
    batch_size: int   = 8
    accum_steps: int  = 1

    # ── optimizer ───────────────────────────────────────────────────────────
    lr:           float = 1e-5
    min_lr:       float = 1e-6
    weight_decay: float = 0.1
    beta1:        float = 0.9
    beta2:        float = 0.95
    grad_clip:    float = 1.0
    warmup_frac:  float = 0.03    # fraction of total steps for warmup

    # ── eval / logging ──────────────────────────────────────────────────────
    eval_every:   int = 50
    eval_batches: int = 20
    ckpt_every:   int = 200
    log_every:    int = 10
    patience:     int = 3         # early-stop after N eval periods of rising val loss

    # ── optional integrations ────────────────────────────────────────────────
    wandb_project: str = ""
    resume:        str = ""       # resume a previous SFT run

    # ── sample prompt ───────────────────────────────────────────────────────
    sample_prompt: str = "Write a short email declining a Friday meeting."
    sample_n:      int = 200
    sample_top_k:  int = 50
    sample_temp:   float = 0.8

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_head=self.n_head,
            n_layer=self.n_layer,
            ctx=self.ctx,
        )


# ---------------------------------------------------------------------------
# SFT-specific data loader
# ---------------------------------------------------------------------------


# Module-level cache so we don't re-open memmaps on every batch call
_DATA_CACHE: dict[str, object] = {}


def _load_split(data_dir: Path, split: str):
    key = f"{data_dir}/{split}"
    if key not in _DATA_CACHE:
        offsets = np.load(str(data_dir / f"sft_{split}_offsets.npy"))
        tokens  = np.memmap(str(data_dir / f"sft_{split}.bin"),
                            dtype=np.uint16, mode="r")
        labels  = np.memmap(str(data_dir / f"sft_{split}_labels.bin"),
                            dtype=np.int32, mode="r")
        _DATA_CACHE[key] = (offsets, tokens, labels)
    return _DATA_CACHE[key]


def sft_get_batch(
    split: str,
    data_dir: str | Path,
    ctx: int,
    batch_size: int,
    device: torch.device,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample batch_size examples from the SFT dataset.

    Returns:
        x       (B, T)  token ids, dtype=long
        labels  (B, T)  -100 on prompt/pad tokens, token_id on response tokens
    """
    offsets, tokens, labels = _load_split(Path(data_dir), split)
    n_examples = len(offsets) - 1
    idxs = np.random.randint(0, n_examples, size=(batch_size,))

    batch_x: list[list[int]] = []
    batch_y: list[list[int]] = []
    for i in idxs:
        s, e = int(offsets[i]), int(offsets[i + 1])
        t = list(map(int, tokens[s:e]))[:ctx]
        l = list(map(int, labels[s:e]))[:ctx]
        batch_x.append(t)
        batch_y.append(l)

    # Pad to the longest example in this batch
    max_len = max(len(t) for t in batch_x)
    for i in range(batch_size):
        pad = max_len - len(batch_x[i])
        batch_x[i].extend([pad_id] * pad)
        batch_y[i].extend([-100] * pad)

    x = torch.tensor(batch_x, dtype=torch.long,  device=device)
    y = torch.tensor(batch_y, dtype=torch.long,  device=device)
    return x, y


def count_examples(data_dir: str | Path, split: str) -> int:
    offsets = np.load(str(Path(data_dir) / f"sft_{split}_offsets.npy"))
    return len(offsets) - 1


# ---------------------------------------------------------------------------
# LR schedule (same shape as pretrain but shorter)
# ---------------------------------------------------------------------------


def get_lr(step: int, total_steps: int, cfg: SFTConfig) -> float:
    warmup = max(1, int(total_steps * cfg.warmup_frac))
    if step < warmup:
        return cfg.lr * step / warmup
    if step >= total_steps:
        return cfg.min_lr
    progress = (step - warmup) / max(total_steps - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ---------------------------------------------------------------------------
# Masked loss
# ---------------------------------------------------------------------------


def sft_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Cross-entropy loss ignoring -100 labels (prompt + padding tokens)."""
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def make_optimizer(model: GPT, cfg: SFTConfig) -> torch.optim.AdamW:
    decay    = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_loss(model: GPT, cfg: SFTConfig, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = sft_get_batch("val", cfg.data_dir, cfg.ctx, cfg.batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits, _ = model(x, targets=None)
        losses.append(sft_loss(logits, y, cfg.vocab_size).item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def _sample_response(model: GPT, cfg: SFTConfig, device: torch.device) -> str:
    """Encode the sample prompt in chat format and generate a response."""
    try:
        from tokenizer import Codec  # type: ignore
        tok_path = ROOT / "tokeniser" / "tokenizer.json"
        codec = Codec(str(tok_path))
        # Build the prompt up to the <|assistant|> tag (no response yet)
        head = [codec.bos, codec.user,
                *codec.encode(cfg.sample_prompt),
                codec.endofturn, codec.assistant]
        prompt_ids = torch.tensor([head], dtype=torch.long, device=device)
        stop_ids = {codec.endofturn, codec.eos}
    except Exception:
        return "[tokenizer unavailable]"

    from inference.sample import sample as gen
    out = gen(
        model, prompt_ids,
        n_new=cfg.sample_n,
        temperature=cfg.sample_temp,
        top_k=cfg.sample_top_k,
        eos_id=codec.eos,
    )
    new_ids = out[0, len(head):].tolist()
    # Strip at first stop token
    for i, t in enumerate(new_ids):
        if t in stop_ids:
            new_ids = new_ids[:i]
            break
    return codec.decode(new_ids)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _infer_model_config(ckpt_path: str) -> dict:
    """Read model config from a pretrain checkpoint's state dict shapes."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state)
    # d_model from embed weight, n_layer from block count, n_head from attn
    d_model   = sd["embed.weight"].shape[1]
    vocab_size = sd["embed.weight"].shape[0]
    n_layer   = max(
        int(k.split(".")[1]) + 1
        for k in sd if k.startswith("blocks.")
    )
    # n_head from qkv projection: shape (3*d_model, d_model), inferred elsewhere
    # Try to get from attn weight
    n_head = 6  # fallback; real value in model config
    for k, v in sd.items():
        if "attn.qkv" in k or "attn.c_attn" in k:
            # q weight shape: (n_head * head_dim, d_model)
            break
    ctx = 256  # fallback; stored in model config if available
    return dict(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer,
                n_head=n_head, ctx=ctx)


def train(cfg: SFTConfig) -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    data_dir = Path(cfg.data_dir)
    out_dir  = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ───────────────────────────────────────────────────────────────
    # Load architecture from base checkpoint if provided
    if cfg.base_ckpt and Path(cfg.base_ckpt).exists():
        inferred = _infer_model_config(cfg.base_ckpt)
        cfg.vocab_size = inferred["vocab_size"]
        cfg.d_model    = inferred["d_model"]
        cfg.n_layer    = inferred["n_layer"]

    model = GPT(cfg.model_config()).to(device)

    if cfg.base_ckpt and Path(cfg.base_ckpt).exists():
        state = torch.load(cfg.base_ckpt, map_location=device, weights_only=False)
        sd    = state.get("model_state_dict", state)
        model.load_state_dict(sd, strict=True)
        print(f"Loaded base weights from {cfg.base_ckpt}")
    else:
        print("No base checkpoint — training from random init")

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = make_optimizer(model, cfg)
    start_step = 0

    if cfg.resume and Path(cfg.resume).exists():
        start_step, last_loss = load_ckpt(cfg.resume, model, optimizer)
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Resumed SFT run from {cfg.resume}  (step {start_step})")

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Steps ───────────────────────────────────────────────────────────────
    n_train   = count_examples(data_dir, "train")
    steps_per_epoch = max(1, n_train // (cfg.batch_size * cfg.accum_steps))
    total_steps = steps_per_epoch * cfg.epochs
    print(f"params        : {model.num_params()/1e6:.2f}M")
    print(f"train examples: {n_train:,}")
    print(f"steps/epoch   : {steps_per_epoch:,}")
    print(f"total steps   : {total_steps:,}  ({cfg.epochs} epochs)")
    print(f"device        : {device}")

    # ── W&B ─────────────────────────────────────────────────────────────────
    wb = None
    if cfg.wandb_project:
        try:
            import wandb
            wb = wandb.init(project=cfg.wandb_project, config=cfg.__dict__)
        except Exception as e:
            print(f"[wandb] {e}")

    # ── Loop ────────────────────────────────────────────────────────────────
    val_loss = float("nan")
    best_val = float("inf")
    patience_count = 0
    t_start = time.time()

    for step in range(start_step + 1, total_steps + 1):
        lr = get_lr(step, total_steps, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        train_loss = 0.0
        for _ in range(cfg.accum_steps):
            x, y = sft_get_batch("train", data_dir, cfg.ctx, cfg.batch_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                logits, _ = model(x, targets=None)
                loss = sft_loss(logits, y, cfg.vocab_size) / cfg.accum_steps
            scaler.scale(loss).backward()
            train_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            epoch   = step / steps_per_epoch
            print(
                f"step {step:>6}/{total_steps}  "
                f"epoch={epoch:.2f}  "
                f"loss={train_loss:.4f}  "
                f"val={val_loss:.4f}  "
                f"lr={lr:.2e}  "
                f"gnorm={grad_norm:.2f}  "
                f"t={elapsed:.0f}s"
            )
            if wb:
                wb.log({"train/loss": train_loss, "train/lr": lr,
                        "train/grad_norm": grad_norm, "step": step})

        if step % cfg.eval_every == 0:
            val_loss = eval_loss(model, cfg, device)
            print(f"  ↳ val loss: {val_loss:.4f}")
            if wb:
                wb.log({"val/loss": val_loss, "step": step})
            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                patience_count = 0
                best_path = out_dir / "ckpt_best.pt"
                save_ckpt(best_path, model, optimizer, step, val_loss)
                print(f"  ↳ new best val — saved {best_path}")
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    print(f"  ↳ early stop (val loss rising for {cfg.patience} evals)")
                    break

        if step % cfg.ckpt_every == 0:
            ckpt_path = out_dir / f"ckpt_{step:06d}.pt"
            save_ckpt(ckpt_path, model, optimizer, step, train_loss)
            print(f"  ↳ saved {ckpt_path}")

            if cfg.sample_prompt:
                model.eval()
                resp = _sample_response(model, cfg, device)
                model.train()
                samples_dir = out_dir / "samples"
                samples_dir.mkdir(exist_ok=True)
                p = samples_dir / f"step_{step:06d}.txt"
                p.write_text(
                    f"=== step {step} ===\n"
                    f"Prompt: {cfg.sample_prompt}\n\n"
                    f"Response:\n{resp}\n"
                )
                print(f"  ↳ sample → {p}")

    if wb:
        wb.finish()
    print(f"\nSFT training complete. Best val loss: {best_val:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> SFTConfig:
    p = argparse.ArgumentParser(description="SFT training (Day 23)")
    p.add_argument("--data-dir",      default="data/sft")
    p.add_argument("--base-ckpt",     default="",
                   help="pretrained checkpoint to fine-tune from")
    p.add_argument("--out-dir",       default="checkpoints/sft")
    p.add_argument("--epochs",        type=int,   default=2)
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--accum-steps",   type=int,   default=1)
    p.add_argument("--lr",            type=float, default=1e-5)
    p.add_argument("--min-lr",        type=float, default=1e-6)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--warmup-frac",   type=float, default=0.03)
    p.add_argument("--eval-every",    type=int,   default=50)
    p.add_argument("--eval-batches",  type=int,   default=20)
    p.add_argument("--ckpt-every",    type=int,   default=200)
    p.add_argument("--patience",      type=int,   default=3)
    p.add_argument("--resume",        default="")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--sample-prompt", default="Write a short email declining a Friday meeting.")
    p.add_argument("--sample-n",      type=int,   default=200)
    p.add_argument("--vocab-size",    type=int,   default=32_000)
    p.add_argument("--ctx",           type=int,   default=256)
    args = p.parse_args()

    cfg = SFTConfig()
    for k, v in vars(args).items():
        setattr(cfg, k.replace("-", "_"), v)
    return cfg


if __name__ == "__main__":
    train(_parse())
