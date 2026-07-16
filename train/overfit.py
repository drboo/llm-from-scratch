"""
Day 9: The overfit test — prove the model can learn.

A single fixed batch is memorised: loss must fall from ~10.4 to < 0.1,
then greedy sampling must regurgitate the training tokens verbatim.

If this test fails, debug these common causes (in order):
  1. off-by-one targets       — y should be x shifted left by 1
  2. causal mask wrong        — check attention.py
  3. RoPE applied to V        — must be Q and K only
  4. zero_grad missing        — optimizer.zero_grad() before loss.backward()
  5. loss wiring              — logits.reshape(-1, vocab), targets.reshape(-1)

Usage:
    python train/overfit.py                            # nano, random ids
    python train/overfit.py --data data/toy/train.bin  # nano, real data
    python train/overfit.py --tiny                     # tiny model, ~30s on CPU
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPT, ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_model(tiny: bool, ctx: int, device: torch.device) -> GPT:
    cfg = (
        ModelConfig(vocab_size=256,    d_model=64,  n_head=4, n_layer=2, ctx=ctx)
        if tiny else
        ModelConfig(vocab_size=32_000, d_model=384, n_head=6, n_layer=6, ctx=ctx)
    )
    return GPT(cfg).to(device)


def fixed_batch(
    data_path: str | None,
    ctx: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed (x, y) pair — same batch every step."""
    if data_path and Path(data_path).exists():
        data = np.memmap(str(data_path), dtype=np.uint16, mode="r")
        g    = torch.Generator().manual_seed(42)
        ix   = torch.randint(len(data) - ctx, (batch_size,), generator=g)
        x    = torch.stack([torch.from_numpy(data[i    : i + ctx    ].astype(np.int64)) for i in ix])
        y    = torch.stack([torch.from_numpy(data[i + 1: i + ctx + 1].astype(np.int64)) for i in ix])
    else:
        g   = torch.Generator().manual_seed(42)
        seq = torch.randint(0, vocab_size, (batch_size, ctx + 1), generator=g)
        x, y = seq[:, :-1], seq[:, 1:]
    return x.to(device), y.to(device)


@torch.no_grad()
def generate_greedy(model: GPT, prompt: torch.Tensor, n_new: int) -> torch.Tensor:
    """Greedy autoregressive generation from a (1, T_prompt) prompt tensor."""
    model.eval()
    idx = prompt.clone()
    for _ in range(n_new):
        logits, _ = model(idx[:, -model.cfg.ctx:])
        next_tok  = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx       = torch.cat([idx, next_tok], dim=1)
    model.train()
    return idx


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")

    torch.manual_seed(0)
    model = make_model(args.tiny, args.ctx, device)
    print(f"params : {model.num_params()/1e6:.2f}M")

    x, y = fixed_batch(args.data, args.ctx, args.batch_size, model.cfg.vocab_size, device)
    print(f"batch  : {tuple(x.shape)}  ({x.numel():,} tokens)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\n{'step':>6}  {'loss':>10}")
    print("-" * 20)

    t0 = time.time()
    loss_val = float("inf")
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()

        if step % args.log_every == 0 or step == 1:
            print(f"{step:>6}  {loss_val:>10.4f}")

        if loss_val < 0.01:  # early exit once well overfit
            print(f"{step:>6}  {loss_val:>10.4f}  (early stop)")
            break

    elapsed = time.time() - t0
    print(f"\nfinal loss : {loss_val:.4f}  ({elapsed:.1f}s)")

    # ── Checkpoint ──────────────────────────────────────────────────────────
    if loss_val < 0.1:
        print("PASS  loss < 0.1")
    else:
        print(f"FAIL  loss {loss_val:.4f} >= 0.1  — see docstring for debug tips")

    # ── Regurgitation check ──────────────────────────────────────────────────
    n_prompt = min(8, args.ctx // 4)
    prompt   = x[0:1, :n_prompt]
    n_gen    = args.ctx - n_prompt
    generated = generate_greedy(model, prompt, n_gen)

    gen_tokens    = generated[0, n_prompt:].tolist()
    target_tokens = x[0, n_prompt:].tolist()

    print(f"\nprompt    : {prompt[0].tolist()}")
    print(f"generated : {gen_tokens[:32]}")
    print(f"target    : {target_tokens[:32]}")

    if gen_tokens == target_tokens:
        print("\nPASS  verbatim regurgitation ✓")
    else:
        mismatches = sum(g != t for g, t in zip(gen_tokens, target_tokens))
        print(f"\nPARTIAL  {len(target_tokens) - mismatches}/{len(target_tokens)} tokens match")

    return loss_val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 9 overfit test")
    p.add_argument("--steps",      type=int,   default=1_000)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--batch-size", type=int,   default=8)
    p.add_argument("--ctx",        type=int,   default=256)
    p.add_argument("--log-every",  type=int,   default=100)
    p.add_argument("--data",       type=str,   default=None,
                   help="path to .bin file; uses random ids if omitted")
    p.add_argument("--tiny",       action="store_true",
                   help="use tiny model (vocab=256, d=64, L=2) for quick CPU test")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
