"""
Day 30: Direct Preference Optimization (DPO) training loop.

DPO (Rafailov et al. 2023) fine-tunes a language model to prefer
chosen responses over rejected ones — without a reward model.

The loss:
    L_DPO = -E [ log σ( β · (log π/π_ref|chosen − log π/π_ref|rejected) ) ]

where π is the policy being trained and π_ref is the frozen reference.
The β hyper-parameter controls how far the policy is allowed to deviate
from the reference.

Usage:
    python train/dpo.py --ref-ckpt checkpoints/sft/ckpt_best.pt \\
                        --out-dir  checkpoints/dpo

Or without a checkpoint (random model, for smoke-testing):
    python train/dpo.py
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPT, ModelConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DPOConfig:
    ref_ckpt:   str | None = None     # frozen reference model (SFT checkpoint)
    out_dir:    str        = "checkpoints/dpo"
    ctx:        int        = 256
    beta:       float      = 0.1      # KL-divergence penalty
    lr:         float      = 5e-7     # very small — DPO is sensitive to LR
    n_steps:    int        = 200      # steps over preference pairs
    batch_size: int        = 4        # pairs per step
    max_tokens: int        = 512      # max tokens per (prompt+response)
    eval_every: int        = 50
    log_every:  int        = 10
    # GQA: 0 = infer from checkpoint (or default to n_head for a new model)
    n_kv_head:  int        = 0


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def sequence_logprobs(
    model:     GPT,
    input_ids: torch.Tensor,   # (1, T)
    labels:    torch.Tensor,   # (1, T)  -100 on prompt
) -> torch.Tensor:
    """
    Sum of log-probabilities over response tokens.

        log π(y|x) = Σ_{t ∈ response} log π(y_t | x, y_<t)

    Args:
        model:     GPT in train or eval mode
        input_ids: (1, T)
        labels:    (1, T) with -100 on prompt positions

    Returns:
        scalar tensor (sum of token log-probs over non-masked positions)
    """
    logits, _ = model(input_ids)           # (1, T, V)
    # Align: logits[t] predicts token at position t+1
    shift_logits = logits[:, :-1, :]       # (1, T-1, V)
    shift_labels = labels[:, 1:]           # (1, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather the log prob of each target token; clamp labels to avoid
    # invalid indices on masked positions (the mask zeroes them out anyway)
    token_logps = log_probs.gather(
        2, shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)                          # (1, T-1)

    mask = (shift_labels != -100).float()
    return (token_logps * mask).sum()      # scalar


def dpo_loss(
    policy_chosen_logps:   torch.Tensor,  # (B,)
    policy_rejected_logps: torch.Tensor,  # (B,)
    ref_chosen_logps:      torch.Tensor,  # (B,)
    ref_rejected_logps:    torch.Tensor,  # (B,)
    beta:                  float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DPO loss and implicit reward margin.

    Implicit reward:  r(x,y) = β · (log π(y|x) − log π_ref(y|x))
    Loss:             −E[ log σ(r_chosen − r_rejected) ]

    Returns:
        loss:   scalar
        margin: mean(r_chosen − r_rejected)  — positive means policy prefers chosen
    """
    chosen_ratio   = policy_chosen_logps   - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    margin = beta * (chosen_ratio - rejected_ratio)     # (B,)
    loss   = -F.logsigmoid(margin).mean()
    return loss, margin.mean().detach()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _infer_n_head(sd: dict, d_model: int) -> int:
    """
    Infer n_head from the RoPE cosine buffer shape.

    rope_cos has shape (max_seq_len, d_head // 2), so
    d_head = 2 * rope_cos.shape[1]  →  n_head = d_model // d_head.
    """
    d_head = 2 * sd["blocks.0.attn.rope_cos"].shape[1]
    return d_model // d_head


def _infer_n_kv_head(sd: dict, n_head: int, d_model: int) -> int:
    """
    Infer n_kv_head from the qkv_proj weight shape in a state dict.

    qkv_proj.weight has shape (q_dim + 2*kv_dim, d_model) where
    q_dim = n_head * d_head and kv_dim = n_kv_head * d_head.

    Solving: n_kv_head = (out_features / d_head - n_head) / 2
    """
    d_head     = d_model // n_head
    out_feats  = sd["blocks.0.attn.qkv_proj.weight"].shape[0]
    n_kv_head  = (out_feats // d_head - n_head) // 2
    return n_kv_head


def _build_nano(n_kv_head: int = 0) -> GPT:
    cfg = ModelConfig(vocab_size=256, d_model=128, n_head=2, n_layer=2,
                      ctx=64, n_kv_head=n_kv_head)
    return GPT(cfg)


def _load_model(ckpt_path: str | None, ctx: int,
                device: torch.device, n_kv_head: int = 0) -> GPT:
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd    = state.get("model_state_dict", state.get("model", state))
        vocab_size = sd["embed.weight"].shape[0]
        d_model    = sd["embed.weight"].shape[1]
        n_layer    = max(int(k.split(".")[1])
                         for k in sd if k.startswith("blocks.")) + 1
        n_head     = _infer_n_head(sd, d_model)
        # Infer GQA config from the qkv_proj shape so GQA checkpoints load correctly
        inferred_kv = _infer_n_kv_head(sd, n_head, d_model)
        resolved_kv = n_kv_head if n_kv_head > 0 else inferred_kv
        cfg   = ModelConfig(vocab_size=vocab_size, d_model=d_model,
                            n_head=n_head, n_kv_head=resolved_kv,
                            n_layer=n_layer, ctx=ctx)
        model = GPT(cfg)
        model.load_state_dict(sd, strict=True)
        gqa_str = f"GQA n_kv={resolved_kv}" if resolved_kv != n_head else "MHA"
        print(f"[dpo] Loaded {Path(ckpt_path).name} — "
              f"{model.num_params()/1e6:.1f}M params  {gqa_str}")
    else:
        model = _build_nano(n_kv_head)
        print("[dpo] No checkpoint — random nano model")
    return model.to(device)


def _batch_sequence_logprobs(
    model:      GPT,
    ids_list:   list[list[int]],
    labels_list: list[list[int]],
    device:     torch.device,
) -> torch.Tensor:
    """Compute sequence_logprobs for a batch of variable-length sequences in one forward pass.

    Returns a (B,) tensor of summed log-probs, one per sequence.
    """
    max_len = max(len(ids) for ids in ids_list)
    B = len(ids_list)
    batch_ids    = torch.zeros(B, max_len, dtype=torch.long, device=device)
    batch_labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
    for i, (ids, labs) in enumerate(zip(ids_list, labels_list)):
        batch_ids[i, :len(ids)]    = torch.tensor(ids,  dtype=torch.long)
        batch_labels[i, :len(labs)] = torch.tensor(labs, dtype=torch.long)

    logits, _ = model(batch_ids)                  # (B, T, V)
    shift_logits = logits[:, :-1, :]              # (B, T-1, V)
    shift_labels = batch_labels[:, 1:]            # (B, T-1)

    log_probs  = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(
        2, shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)                                 # (B, T-1)

    mask = (shift_labels != -100).float()
    return (token_logps * mask).sum(dim=-1)       # (B,)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: DPOConfig) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load reference model (frozen) and policy (trained) ──────────────────
    ref_model = _load_model(cfg.ref_ckpt, cfg.ctx, device, cfg.n_kv_head)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    policy = copy.deepcopy(ref_model)
    policy.train()
    for p in policy.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)

    # ── Preference data ──────────────────────────────────────────────────────
    from dpo.data import TRAIN_PAIRS, VAL_PAIRS, load_split_inmemory

    codec = None
    tok_path = ROOT / "tokeniser" / "tokenizer.json"
    if tok_path.exists():
        try:
            sys.path.insert(0, str(ROOT / "tokeniser"))
            from tokenizer import Codec  # type: ignore
            codec = Codec(str(tok_path))
        except Exception:
            pass

    train_examples = load_split_inmemory(TRAIN_PAIRS, codec, cfg.max_tokens)
    val_examples   = load_split_inmemory(VAL_PAIRS,   codec, cfg.max_tokens)

    if not train_examples:
        print("[dpo] No training examples — exiting.")
        return {}

    print(f"[dpo] {len(train_examples)} train pairs, "
          f"{len(val_examples)} val pairs, device={device}")

    best_val_loss = float("inf")
    history: list[dict] = []

    # ── Steps loop ───────────────────────────────────────────────────────────
    rng   = torch.Generator()
    rng.manual_seed(42)
    step  = 0

    while step < cfg.n_steps:
        # Sample a batch of pairs (with replacement)
        indices = torch.randint(len(train_examples),
                                (cfg.batch_size,), generator=rng)

        ctx = policy.cfg.ctx
        c_ids_batch, c_lab_batch = [], []
        r_ids_batch, r_lab_batch = [], []
        for idx in indices:
            (c_ids, c_lab), (r_ids, r_lab) = train_examples[idx.item()]
            c_ids_batch.append(c_ids[:ctx]); c_lab_batch.append(c_lab[:ctx])
            r_ids_batch.append(r_ids[:ctx]); r_lab_batch.append(r_lab[:ctx])

        # Two forwards (ref + policy) over 2*B sequences instead of 4*B forwards.
        all_ids  = c_ids_batch + r_ids_batch
        all_labs = c_lab_batch + r_lab_batch

        with torch.no_grad():
            ref_logps = _batch_sequence_logprobs(ref_model, all_ids, all_labs, device)
        pol_logps = _batch_sequence_logprobs(policy, all_ids, all_labs, device)

        B = cfg.batch_size
        loss, margin = dpo_loss(
            pol_logps[:B], pol_logps[B:],
            ref_logps[:B], ref_logps[B:],
            cfg.beta,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        step += 1

        batch_loss   = loss.detach()
        batch_margin = margin.detach()

        if step % cfg.log_every == 0:
            loss_val   = batch_loss.item()
            margin_val = batch_margin.item()
            print(f"  step {step:4d}/{cfg.n_steps}  "
                  f"loss={loss_val:.4f}  "
                  f"margin={margin_val:.4f}")
            history.append({
                "step":   step,
                "loss":   loss_val,
                "margin": margin_val,
            })

        # ── Validation ───────────────────────────────────────────────────────
        if step % cfg.eval_every == 0 and val_examples:
            policy.eval()
            val_loss = _eval(policy, ref_model, val_examples, cfg, device)
            policy.train()
            print(f"  [val] step {step}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "step":              step,
                    "model_state_dict":  policy.state_dict(),
                    "val_loss":          val_loss,
                    "cfg":               cfg,
                }
                torch.save(ckpt, out_dir / "ckpt_best.pt")
                print(f"  [val] saved best checkpoint (val_loss={val_loss:.4f})")

    # Save final checkpoint regardless
    torch.save({
        "step":             step,
        "model_state_dict": policy.state_dict(),
        "cfg":              cfg,
    }, out_dir / "ckpt_final.pt")
    print(f"[dpo] Done. Best val loss: {best_val_loss:.4f}")

    return {
        "best_val_loss": best_val_loss,
        "steps":         step,
        "history":       history,
    }


def _eval(policy: GPT, ref: GPT, examples: list, cfg: DPOConfig,
          device: torch.device) -> float:
    ctx   = policy.cfg.ctx
    total = 0.0
    count = 0
    batch_size = cfg.batch_size

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            chunk = examples[i : i + batch_size]
            c_ids_b = [c[:ctx] for (c, _), _ in chunk]
            c_lab_b = [l[:ctx] for (_, l), _ in chunk]
            r_ids_b = [c[:ctx] for _, (c, _) in chunk]
            r_lab_b = [l[:ctx] for _, (_, l) in chunk]

            all_ids  = c_ids_b + r_ids_b
            all_labs = c_lab_b + r_lab_b

            ref_logps = _batch_sequence_logprobs(ref,    all_ids, all_labs, device)
            pol_logps = _batch_sequence_logprobs(policy, all_ids, all_labs, device)

            B = len(chunk)
            loss, _ = dpo_loss(
                pol_logps[:B], pol_logps[B:],
                ref_logps[:B], ref_logps[B:],
                cfg.beta,
            )
            total += loss.item() * B
            count += B

    return total / count if count > 0 else float("inf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO training (Day 30)")
    p.add_argument("--ref-ckpt",   default="")
    p.add_argument("--out-dir",    default="checkpoints/dpo")
    p.add_argument("--ctx",        type=int,   default=256)
    p.add_argument("--beta",       type=float, default=0.1)
    p.add_argument("--lr",         type=float, default=5e-7)
    p.add_argument("--n-steps",    type=int,   default=200)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--eval-every", type=int,   default=50)
    p.add_argument("--n-kv-head", type=int,   default=0,
                   help="GQA KV heads (0=infer from ckpt or use n_head)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    cfg  = DPOConfig(
        ref_ckpt   = args.ref_ckpt or None,
        out_dir    = args.out_dir,
        ctx        = args.ctx,
        beta       = args.beta,
        lr         = args.lr,
        n_steps    = args.n_steps,
        batch_size = args.batch_size,
        eval_every = args.eval_every,
        n_kv_head  = args.n_kv_head,
    )
    train(cfg)
