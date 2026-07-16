"""
Day 12: Perplexity evaluator.

Perplexity = exp(mean cross-entropy loss over the validation set).
At random init on a 32k-vocab model, PPL ≈ 32000.
A well-trained nano should reach PPL < 100 on the toy corpus.

Usage:
    # from a checkpoint:
    python eval/perplexity.py --ckpt checkpoints/ckpt_005000.pt --data data/toy

    # against random weights (sanity check — should be near vocab_size):
    python eval/perplexity.py --data data/toy

    # evaluate multiple checkpoints:
    python eval/perplexity.py --ckpt-dir checkpoints --data data/toy
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPT, ModelConfig
from data.dataloader import get_batch


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_perplexity(
    model: GPT,
    data_dir: str | Path,
    split: str = "val",
    ctx: int = 256,
    batch_size: int = 8,
    n_batches: int = 50,
    device: torch.device | None = None,
) -> float:
    """
    Compute perplexity of `model` on `split`.bin in `data_dir`.

    Returns:
        Perplexity (exp of mean cross-entropy loss).
    """
    if device is None:
        device = next(model.parameters()).device

    bin_path = Path(data_dir) / f"{split}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"Data file not found: {bin_path}")

    model.eval()
    total_loss = 0.0
    use_amp = device.type == "cuda"

    for _ in range(n_batches):
        x, y = get_batch(split, str(data_dir), ctx, batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_amp):
            _, loss = model(x, y)
        total_loss += loss.item()

    mean_loss = total_loss / n_batches
    return math.exp(mean_loss)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _load_model(ckpt_path: str, n_head: int, ctx: int, device: torch.device) -> tuple[GPT, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    embed_w = ckpt["model"]["embed.weight"]
    vocab_size, d_model = embed_w.shape
    n_layer = max(
        int(k.split(".")[1]) for k in ckpt["model"] if k.startswith("blocks.")
    ) + 1
    cfg = ModelConfig(vocab_size=vocab_size, d_model=d_model,
                      n_head=n_head, n_layer=n_layer, ctx=ctx)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    step = ckpt.get("step", 0)
    return model, step


def _default_model(args, device: torch.device) -> GPT:
    cfg = ModelConfig(
        vocab_size=args.vocab_size, d_model=args.d_model,
        n_head=args.n_head, n_layer=args.n_layer, ctx=args.ctx,
    )
    return GPT(cfg).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Compute val perplexity (Day 12)")
    p.add_argument("--ckpt",       default="",         help="single checkpoint path")
    p.add_argument("--ckpt-dir",   default="",         help="directory of checkpoints (eval all)")
    p.add_argument("--data",       default="data/toy", help="data directory with val.bin")
    p.add_argument("--split",      default="val",      choices=["train", "val"])
    p.add_argument("--n-batches",  type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--ctx",        type=int, default=256)
    p.add_argument("--n-head",     type=int, default=6,
                   help="n_head (not stored in state dict; must match checkpoint)")
    # fallback model flags (used when no --ckpt)
    p.add_argument("--vocab-size", type=int, default=32_000)
    p.add_argument("--d-model",    type=int, default=384)
    p.add_argument("--n-layer",    type=int, default=6)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
        ckpts = sorted(ckpt_dir.glob("*.pt"))
        if not ckpts:
            print(f"No .pt files found in {ckpt_dir}")
            return
        print(f"{'step':>8}  {'ppl':>10}  file")
        print("-" * 50)
        for path in ckpts:
            model, step = _load_model(str(path), args.n_head, args.ctx, device)
            ppl = compute_perplexity(model, args.data, args.split,
                                     args.ctx, args.batch_size, args.n_batches, device)
            print(f"{step:>8}  {ppl:>10.2f}  {path.name}")
    elif args.ckpt:
        model, step = _load_model(args.ckpt, args.n_head, args.ctx, device)
        print(f"Checkpoint: {args.ckpt}  (step {step})")
        print(f"Params    : {model.num_params()/1e6:.2f}M")
        ppl = compute_perplexity(model, args.data, args.split,
                                 args.ctx, args.batch_size, args.n_batches, device)
        print(f"Perplexity ({args.split}): {ppl:.2f}")
    else:
        model = _default_model(args, device)
        print("No checkpoint — using random weights")
        print(f"Params    : {model.num_params()/1e6:.2f}M")
        ppl = compute_perplexity(model, args.data, args.split,
                                 args.ctx, args.batch_size, args.n_batches, device)
        print(f"Perplexity ({args.split}): {ppl:.2f}")
        print(f"Expected at random init ≈ {args.vocab_size:,}")
