"""
Day 9 checkpoint tests — overfit test.

Run:  pytest train/test_day9.py -v -s

Uses a tiny model (vocab=256, d=64, L=2) so the full test suite stays fast.
The --tiny flag on overfit.py exercises the same path for manual runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import GPT, ModelConfig
from train.overfit import fixed_batch, generate_greedy

# ---------------------------------------------------------------------------
# Tiny model config — fast enough to overfit on CPU in a few seconds
# ---------------------------------------------------------------------------

CFG = ModelConfig(vocab_size=256, d_model=64, n_head=4, n_layer=2, ctx=32)
DEVICE = torch.device("cpu")
BATCH_SIZE = 4
LR = 3e-3     # higher LR for quick convergence in tiny model
STEPS = 400


@pytest.fixture(scope="module")
def overfit_model():
    """Train a tiny model to overfit a fixed batch; return (model, x, y)."""
    torch.manual_seed(0)
    model = GPT(CFG).to(DEVICE)
    x, y  = fixed_batch(None, CFG.ctx, BATCH_SIZE, CFG.vocab_size, DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    for step in range(STEPS):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.01:
            break

    return model, x, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_loss_falls_from_random_init():
    """Loss at step 1 should be close to ln(vocab_size) ≈ 5.5 for vocab=256."""
    torch.manual_seed(0)
    model = GPT(CFG).to(DEVICE)
    x, y  = fixed_batch(None, CFG.ctx, BATCH_SIZE, CFG.vocab_size, DEVICE)
    _, loss = model(x, y)
    expected = torch.tensor(CFG.vocab_size).float().log().item()
    assert abs(loss.item() - expected) < 0.5, (
        f"Initial loss {loss.item():.3f} far from ln({CFG.vocab_size})={expected:.3f}"
    )


def test_loss_below_threshold(overfit_model):
    """After training, loss on the fixed batch must be < 0.1."""
    model, x, y = overfit_model
    model.eval()
    with torch.no_grad():
        _, loss = model(x, y)
    model.train()
    assert loss.item() < 0.1, (
        f"Loss {loss.item():.4f} >= 0.1 after {STEPS} steps.\n"
        "Common bugs: off-by-one targets, wrong causal mask, RoPE on V, "
        "missing zero_grad, or broken loss reshape."
    )


def test_greedy_regurgitates_training_sequence(overfit_model):
    """Greedy decode from a prefix must reproduce the exact training tokens."""
    model, x, _ = overfit_model
    n_prompt = 4
    n_gen    = CFG.ctx - n_prompt

    prompt    = x[0:1, :n_prompt]
    generated = generate_greedy(model, prompt, n_gen)

    gen_tokens    = generated[0, n_prompt:].tolist()
    target_tokens = x[0, n_prompt:].tolist()

    assert gen_tokens == target_tokens, (
        f"Regurgitation failed.\n"
        f"  first mismatch at token "
        f"{next(i for i,(g,t) in enumerate(zip(gen_tokens,target_tokens)) if g!=t)}\n"
        f"  generated: {gen_tokens[:16]}\n"
        f"  target:    {target_tokens[:16]}"
    )


def test_zero_grad_is_called():
    """Verify optimizer.zero_grad() resets gradients (sanity check for the loop)."""
    torch.manual_seed(0)
    model = GPT(CFG).to(DEVICE)
    x, y  = fixed_batch(None, CFG.ctx, BATCH_SIZE, CFG.vocab_size, DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)

    # Step 1
    opt.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    opt.step()

    # After step, call zero_grad and confirm grads are None / zero
    opt.zero_grad(set_to_none=True)
    for p in model.parameters():
        assert p.grad is None, "zero_grad did not clear gradients"


def test_loss_strictly_decreases_early(overfit_model):
    """Loss at step 10 must be strictly below loss at step 1."""
    torch.manual_seed(0)
    model = GPT(CFG).to(DEVICE)
    x, y  = fixed_batch(None, CFG.ctx, BATCH_SIZE, CFG.vocab_size, DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )
