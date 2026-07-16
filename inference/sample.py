"""
Day 11: Autoregressive sampling / generation.

No KV cache — naive forward pass each step.  Supports:
  - temperature scaling
  - top-k filtering
  - top-p (nucleus) filtering
  - greedy decode (temperature=0)

Usage:
    # random weights (no checkpoint):
    python inference/sample.py --prompt "Once upon a time"

    # from a checkpoint:
    python inference/sample.py --ckpt checkpoints/ckpt_000500.pt \\
        --prompt "Once upon a time" --temperature 0.8 --top-k 50

    # greedy:
    python inference/sample.py --prompt "hello" --temperature 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPT, ModelConfig


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits outside the top-k."""
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, min(k, logits.size(-1)))
    threshold  = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out tokens whose cumulative softmax probability exceeds p (nucleus)."""
    if p >= 1.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Mark tokens to remove in sorted order (shift by one to keep at least one)
    sorted_remove = (cumulative - sorted_probs) > p
    # Scatter the boolean mask back to the original token ordering
    remove_mask = torch.zeros_like(sorted_remove).scatter_(-1, sorted_idx, sorted_remove)
    return logits.masked_fill(remove_mask, float("-inf"))


@torch.no_grad()
def sample(
    model: GPT,
    prompt_ids: torch.Tensor,
    n_new: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_id: int | None = None,
) -> torch.Tensor:
    """
    Autoregressively generate `n_new` tokens from `prompt_ids`.

    Args:
        model:       GPT instance in eval mode.
        prompt_ids:  LongTensor of shape (1, T_prompt).
        n_new:       number of new tokens to generate.
        temperature: >1 = more random, <1 = more peaked, 0 = greedy.
        top_k:       keep only top-k logits (0 = disabled).
        top_p:       nucleus threshold (1.0 = disabled).
        eos_id:      stop early if this token is sampled.

    Returns:
        LongTensor of shape (1, T_prompt + n_generated).
    """
    model.eval()
    idx = prompt_ids.clone()
    ctx = model.cfg.ctx

    for _ in range(n_new):
        # Crop to context window
        idx_cond   = idx[:, -ctx:]
        logits, _  = model(idx_cond)
        logits     = logits[:, -1, :]          # (1, vocab)

        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = top_k_filter(logits, top_k)
            logits = top_p_filter(logits, top_p)
            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break

    return idx


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _encode_prompt(prompt: str, vocab_size: int, device: torch.device) -> torch.Tensor:
    """
    Encode the prompt.  If a tokenizer.json is present next to the tokeniser
    directory, use the Codec; otherwise fall back to raw bytes (vocab ≤ 256).
    """
    tok_path = ROOT / "tokeniser" / "tokenizer.json"
    if tok_path.exists() and vocab_size > 256:
        try:
            sys.path.insert(0, str(ROOT / "tokeniser"))
            from tokenizer import Codec  # type: ignore
            codec = Codec(str(tok_path))
            ids = codec.encode(prompt)
            return torch.tensor([ids], dtype=torch.long, device=device)
        except Exception as e:
            print(f"[warn] Codec unavailable ({e}), falling back to byte encoding")

    # Byte fallback — works for vocab_size=256 models
    ids = [b for b in prompt.encode("utf-8")]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _decode_ids(ids: list[int], vocab_size: int) -> str:
    tok_path = ROOT / "tokeniser" / "tokenizer.json"
    if tok_path.exists() and vocab_size > 256:
        try:
            sys.path.insert(0, str(ROOT / "tokeniser"))
            from tokenizer import Codec  # type: ignore
            codec = Codec(str(tok_path))
            return codec.decode(ids)
        except Exception:
            pass
    # Byte fallback
    return bytes([i for i in ids if i < 256]).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Sample from a GPT model (Day 11)")
    p.add_argument("--ckpt",        default="",    help="path to .pt checkpoint")
    p.add_argument("--prompt",      default="Once upon a time",
                   help="text prompt (or comma-separated token ids with --raw-ids)")
    p.add_argument("--raw-ids",     action="store_true",
                   help="treat --prompt as comma-separated integer token ids")
    p.add_argument("--n-new",       type=int,   default=200,  help="tokens to generate")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k",       type=int,   default=0)
    p.add_argument("--top-p",       type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=42)
    # model overrides (used when running without a checkpoint)
    p.add_argument("--vocab-size",  type=int,   default=32_000)
    p.add_argument("--d-model",     type=int,   default=384)
    p.add_argument("--n-head",      type=int,   default=6)
    p.add_argument("--n-layer",     type=int,   default=6)
    p.add_argument("--ctx",         type=int,   default=256)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build or load model ──────────────────────────────────────────────────
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        # Infer config from state dict embedding shape
        embed_w = ckpt["model"]["embed.weight"]
        vocab_size, d_model = embed_w.shape
        # Infer n_layer from block keys
        n_layer = max(
            int(k.split(".")[1]) for k in ckpt["model"] if k.startswith("blocks.")
        ) + 1
        n_head  = args.n_head    # not stored in plain state dict; pass via flag
        ctx     = args.ctx
        cfg = ModelConfig(vocab_size=vocab_size, d_model=d_model,
                          n_head=n_head, n_layer=n_layer, ctx=ctx)
        model = GPT(cfg)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        cfg = ModelConfig(vocab_size=args.vocab_size, d_model=args.d_model,
                          n_head=args.n_head, n_layer=args.n_layer, ctx=args.ctx)
        model = GPT(cfg)
        print("No checkpoint — using random weights")

    model = model.to(device)
    model.eval()
    print(f"Model: {model.num_params()/1e6:.2f}M params  device={device}")

    # ── Encode prompt ────────────────────────────────────────────────────────
    if args.raw_ids:
        ids = [int(t) for t in args.prompt.split(",")]
        prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)
    else:
        prompt_ids = _encode_prompt(args.prompt, cfg.vocab_size, device)

    print(f"Prompt  : {args.prompt!r}  ({prompt_ids.shape[1]} tokens)")
    print(f"Settings: temp={args.temperature}  top_k={args.top_k}  top_p={args.top_p}")
    print("-" * 60)

    # ── Sample ───────────────────────────────────────────────────────────────
    out_ids = sample(
        model, prompt_ids,
        n_new=args.n_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    new_ids  = out_ids[0, prompt_ids.shape[1]:].tolist()
    full_ids = out_ids[0].tolist()
    text     = _decode_ids(full_ids, cfg.vocab_size)
    print(text)
    print("-" * 60)
    print(f"Generated {len(new_ids)} new tokens.")


if __name__ == "__main__":
    main()
