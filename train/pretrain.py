"""
Day 10 / 13: Production-quality pretraining loop.

Features:
  - Cosine LR schedule with linear warmup
  - Gradient accumulation  (tokens/step = batch_size × ctx × accum_steps)
  - bf16 autocast on CUDA
  - Gradient clipping (max_norm=1.0)
  - Periodic val loss evaluation
  - Checkpoint save / resume
  - Optional W&B logging
  - Sample progression: save generated text at every checkpoint (Day 13)

Usage:
    python train/pretrain.py                          # nano defaults
    python train/pretrain.py --resume checkpoints/ckpt_0500.pt
    python train/pretrain.py --wandb-project my-llm
    python train/pretrain.py --sample-prompt "Once upon a time" --sample-steps 500
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPT, ModelConfig
from data.dataloader import get_batch
from train.checkpoint import save as save_ckpt, load as load_ckpt
from inference.sample import sample as generate_sample


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    # ── data ────────────────────────────────────────────────────────────────
    data_dir:    str  = "data/toy"
    out_dir:     str  = "checkpoints"

    # ── model ───────────────────────────────────────────────────────────────
    vocab_size:  int  = 32_000
    d_model:     int  = 384
    n_head:      int  = 6
    n_layer:     int  = 6
    ctx:         int  = 256

    # ── training ────────────────────────────────────────────────────────────
    max_steps:   int   = 5_000
    warmup_steps: int  = 200
    batch_size:  int   = 8       # micro-batch (sequences per GPU step)
    accum_steps: int   = 4       # gradient accumulation steps
    # effective tokens/step = batch_size × ctx × accum_steps

    # ── optimizer ───────────────────────────────────────────────────────────
    lr:           float = 3e-4
    min_lr:       float = 3e-5
    weight_decay: float = 0.1
    beta1:        float = 0.9
    beta2:        float = 0.95
    grad_clip:    float = 1.0

    # ── eval / logging ──────────────────────────────────────────────────────
    eval_every:   int  = 250
    eval_batches: int  = 10
    ckpt_every:   int  = 500
    log_every:    int  = 10

    # ── optional integrations ────────────────────────────────────────────────
    wandb_project: str = ""   # empty = disabled
    resume:        str = ""   # path to checkpoint, empty = fresh start

    # ── sample progression (Day 13) ──────────────────────────────────────────
    sample_prompt: str = ""   # if set, generate a sample at each checkpoint
    sample_steps:  int = 0    # generate every N steps (0 = same as ckpt_every)
    sample_n:      int = 200  # tokens to generate per sample
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

    def tokens_per_step(self) -> int:
        return self.batch_size * self.ctx * self.accum_steps


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ---------------------------------------------------------------------------
# Optimizer — separate weight-decay groups
# ---------------------------------------------------------------------------


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """Weight decay on 2D+ params (weight matrices); no decay on 1D (norms)."""
    decay     = [p for p in model.parameters() if p.dim() >= 2]
    no_decay  = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
    )


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_loss(model: GPT, cfg: TrainConfig, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch("val", cfg.data_dir, cfg.ctx, cfg.batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _load_codec(cfg: TrainConfig):
    """Return a Codec for BPE models, or None for byte-level models."""
    if cfg.vocab_size <= 256:
        return None
    tok_path = ROOT / "tokeniser" / "tokenizer.json"
    if not tok_path.exists():
        return None
    try:
        sys.path.insert(0, str(ROOT / "tokeniser"))
        from tokenizer import Codec  # type: ignore
        return Codec(str(tok_path))
    except Exception:
        return None


def _save_sample(model: GPT, cfg: TrainConfig, step: int, device: torch.device,
                 codec=None) -> None:
    """Generate from cfg.sample_prompt and write to out_dir/samples/step_XXXXXX.txt."""
    if codec is not None:
        ids = codec.encode(cfg.sample_prompt)
    else:
        ids = list(cfg.sample_prompt.encode("utf-8"))

    prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)

    out = generate_sample(
        model, prompt_ids,
        n_new=cfg.sample_n,
        temperature=cfg.sample_temp,
        top_k=cfg.sample_top_k,
    )
    new_ids = out[0, prompt_ids.shape[1]:].tolist()
    if codec is not None:
        text = codec.decode(new_ids)
    else:
        try:
            text = bytes([i for i in new_ids if i < 256]).decode("utf-8", errors="replace")
        except Exception:
            text = str(new_ids)

    samples_dir = Path(cfg.out_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / f"step_{step:06d}.txt"
    path.write_text(f"=== step {step} | prompt: {cfg.sample_prompt!r} ===\n{text}\n")
    print(f"  ↳ sample  → {path}")


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Model + optimiser ───────────────────────────────────────────────────
    model     = GPT(cfg.model_config()).to(device)
    optimizer = make_optimizer(model, cfg)
    start_step = 0

    if cfg.resume:
        start_step, last_loss = load_ckpt(cfg.resume, model, optimizer)
        model.to(device)
        # move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Resumed from {cfg.resume}  (step {start_step}, loss {last_loss:.4f})")

    n_params = model.num_params()
    tps      = cfg.tokens_per_step()
    print(f"params     : {n_params/1e6:.2f}M")
    print(f"tokens/step: {tps:,}  "
          f"(batch={cfg.batch_size} × ctx={cfg.ctx} × accum={cfg.accum_steps})")
    print(f"device     : {device}")

    # ── Codec (for BPE sample decoding) ─────────────────────────────────────
    codec = _load_codec(cfg) if cfg.sample_prompt else None
    if codec:
        print(f"[sample] BPE codec loaded (vocab={codec.vocab_size})")

    # ── W&B ─────────────────────────────────────────────────────────────────
    wb = None
    if cfg.wandb_project:
        try:
            import wandb
            wb = wandb.init(project=cfg.wandb_project, config=cfg.__dict__)
        except Exception as e:
            print(f"[wandb] init failed: {e}  — continuing without logging")

    # ── Loop ────────────────────────────────────────────────────────────────
    val_loss  = float("nan")
    t_start   = time.time()

    for step in range(start_step + 1, cfg.max_steps + 1):
        # ── LR update ───────────────────────────────────────────────────────
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ────────────────────────────────────────────
        optimizer.zero_grad()
        train_loss = 0.0
        for micro in range(cfg.accum_steps):
            x, y = get_batch("train", cfg.data_dir, cfg.ctx, cfg.batch_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                _, loss = model(x, y)
            loss = loss / cfg.accum_steps
            scaler.scale(loss).backward()
            train_loss += loss.item()

        # ── Grad clip → step ─────────────────────────────────────────────────
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ── Logging ──────────────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            tok_s   = tps * step / elapsed
            print(
                f"step {step:>6}/{cfg.max_steps}  "
                f"loss={train_loss:.4f}  "
                f"val={val_loss:.4f}  "
                f"lr={lr:.2e}  "
                f"gnorm={grad_norm:.2f}  "
                f"tok/s={tok_s:,.0f}"
            )
            if wb:
                wb.log({"train/loss": train_loss, "train/lr": lr,
                        "train/grad_norm": grad_norm, "step": step})

        # ── Validation ───────────────────────────────────────────────────────
        if step % cfg.eval_every == 0 and Path(cfg.data_dir, "val.bin").exists():
            val_loss = eval_loss(model, cfg, device)
            print(f"  ↳ val loss: {val_loss:.4f}")
            if wb:
                wb.log({"val/loss": val_loss, "step": step})

        # ── Checkpoint ───────────────────────────────────────────────────────
        if step % cfg.ckpt_every == 0:
            ckpt_path = Path(cfg.out_dir) / f"ckpt_{step:06d}.pt"
            save_ckpt(ckpt_path, model, optimizer, step, train_loss)
            print(f"  ↳ saved {ckpt_path}")

        # ── Sample progression ────────────────────────────────────────────────
        sample_every = cfg.sample_steps if cfg.sample_steps > 0 else cfg.ckpt_every
        if cfg.sample_prompt and step % sample_every == 0:
            _save_sample(model, cfg, step, device, codec=codec)

    if wb:
        wb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> TrainConfig:
    p = argparse.ArgumentParser(description="Pretrain GPT (Day 10)")
    p.add_argument("--data-dir",     default="data/toy")
    p.add_argument("--out-dir",      default="checkpoints")
    p.add_argument("--max-steps",    type=int,   default=5_000)
    p.add_argument("--warmup-steps", type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--accum-steps",  type=int,   default=4)
    p.add_argument("--ctx",          type=int,   default=256)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--min-lr",       type=float, default=3e-5)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--eval-every",   type=int,   default=250)
    p.add_argument("--ckpt-every",   type=int,   default=500)
    p.add_argument("--log-every",    type=int,   default=10)
    p.add_argument("--resume",        default="",  help="path to checkpoint")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--sample-prompt", default="",  help="generate sample text at each checkpoint")
    p.add_argument("--sample-steps",  type=int, default=0,
                   help="sample every N steps (0 = same as --ckpt-every)")
    p.add_argument("--sample-n",      type=int,   default=200)
    p.add_argument("--sample-top-k",  type=int,   default=50)
    p.add_argument("--sample-temp",   type=float, default=0.8)
    args = p.parse_args()

    cfg = TrainConfig()
    for k, v in vars(args).items():
        setattr(cfg, k.replace("-", "_"), v)
    return cfg


if __name__ == "__main__":
    train(_parse())
