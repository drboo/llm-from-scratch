"""
Day 26: KV-cache benchmark — tokens/sec with vs. without cache.

Verifies numerical identity under greedy decoding, then measures throughput
at various context lengths.  Run without a checkpoint to use a random-init
nano model (same architecture, real speed numbers).

Usage:
    # Random-init model (no checkpoint needed):
    python inference/benchmark.py

    # With a real checkpoint:
    python inference/benchmark.py --ckpt checkpoints/ckpt_050000.pt

    # Longer generation for more accurate timing:
    python inference/benchmark.py --n-new 400 --prompt-len 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPT, ModelConfig
from inference.sample import sample as naive_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_or_build(ckpt: str | None, ctx: int, device: torch.device) -> GPT:
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        sd    = state.get("model_state_dict", state)
        vocab_size = sd["embed.weight"].shape[0]
        d_model    = sd["embed.weight"].shape[1]
        n_layer    = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
        cfg   = ModelConfig(vocab_size=vocab_size, d_model=d_model,
                            n_head=6, n_layer=n_layer, ctx=ctx)
        model = GPT(cfg)
        model.load_state_dict(sd, strict=True)
        print(f"Loaded checkpoint: {ckpt}")
    else:
        cfg   = ModelConfig()   # nano defaults
        model = GPT(cfg)
        print("Using random-init nano model (no checkpoint)")

    model.to(device)
    model.eval()
    return model


def _make_prompt(prompt_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randint(0, vocab_size, (1, prompt_len), device=device)


def _timed_naive(model: GPT, prompt: torch.Tensor, n_new: int,
                 warmup: int = 1) -> tuple[torch.Tensor, float]:
    """Generate n_new tokens without cache, return (output, tok/s)."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            naive_sample(model, prompt, n_new=min(n_new, 10), temperature=0.0)

    if prompt.device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = naive_sample(model, prompt, n_new=n_new, temperature=0.0)
    if prompt.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return out, n_new / elapsed


def _timed_cached(model: GPT, prompt: torch.Tensor, n_new: int,
                  warmup: int = 1) -> tuple[torch.Tensor, float]:
    """Generate n_new tokens with KV cache, return (output, tok/s)."""
    for _ in range(warmup):
        model.generate_cached(prompt, n_new=min(n_new, 10), temperature=0.0)

    if prompt.device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = model.generate_cached(prompt, n_new=n_new, temperature=0.0)
    if prompt.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return out, n_new / elapsed


# ---------------------------------------------------------------------------
# Identity check
# ---------------------------------------------------------------------------


def verify_identity(model: GPT, prompt_len: int = 8, n_new: int = 20) -> bool:
    """
    Verify cached and non-cached greedy generation produce identical tokens.

    Returns True if they match, False otherwise.
    """
    device  = next(model.parameters()).device
    prompt  = _make_prompt(prompt_len, model.cfg.vocab_size, device)

    with torch.no_grad():
        out_naive  = naive_sample(model, prompt, n_new=n_new, temperature=0.0)
        out_cached = model.generate_cached(prompt, n_new=n_new, temperature=0.0)

    new_naive  = out_naive[ 0, prompt_len:].tolist()
    new_cached = out_cached[0, prompt_len:].tolist()

    match = new_naive == new_cached
    if match:
        print(f"  ✓ Identity check passed ({n_new} tokens, prompt_len={prompt_len})")
    else:
        print(f"  ✗ Identity check FAILED")
        print(f"    naive:  {new_naive[:10]}")
        print(f"    cached: {new_cached[:10]}")
    return match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def benchmark(
    ckpt:       str | None = None,
    ctx:        int = 256,
    prompt_len: int = 32,
    n_new:      int = 200,
    warmup:     int = 1,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    model = _load_or_build(ckpt, ctx, device)
    print(f"Params: {model.num_params()/1e6:.1f}M  |  ctx={ctx}")

    prompt = _make_prompt(prompt_len, model.cfg.vocab_size, device)

    # Identity check first
    print(f"\n{'─'*50}")
    print("Identity check (greedy, must match exactly):")
    ok = verify_identity(model, prompt_len=prompt_len, n_new=min(n_new, 50))

    # Benchmark
    print(f"\n{'─'*50}")
    print(f"Benchmark: prompt_len={prompt_len}, n_new={n_new}")
    print()

    _, naive_tps  = _timed_naive( model, prompt, n_new, warmup)
    _, cached_tps = _timed_cached(model, prompt, n_new, warmup)

    speedup = cached_tps / naive_tps

    print(f"  Without cache : {naive_tps:>8,.1f} tok/s")
    print(f"  With KV cache : {cached_tps:>8,.1f} tok/s")
    print(f"  Speedup       : {speedup:>8.1f}×")
    print(f"\n{'─'*50}")

    if speedup < 1.5:
        print("  NOTE: small speedup expected on CPU at short ctx.")
        print("  On GPU with ctx ≥ 512 the speedup is typically 5–20×.")

    return {
        "identity_ok":  ok,
        "naive_tps":    naive_tps,
        "cached_tps":   cached_tps,
        "speedup":      speedup,
        "device":       str(device),
        "prompt_len":   prompt_len,
        "n_new":        n_new,
    }


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KV cache benchmark (Day 26)")
    p.add_argument("--ckpt",       default="",  help="checkpoint path")
    p.add_argument("--ctx",        type=int,   default=256)
    p.add_argument("--prompt-len", type=int,   default=32)
    p.add_argument("--n-new",      type=int,   default=200)
    p.add_argument("--warmup",     type=int,   default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    benchmark(
        ckpt       = args.ckpt or None,
        ctx        = args.ctx,
        prompt_len = args.prompt_len,
        n_new      = args.n_new,
        warmup     = args.warmup,
    )
