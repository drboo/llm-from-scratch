"""
Day 7 checkpoint tests — full GPT model.

Run:  pytest model/test_day7.py -v

Key checkpoint: initial loss ≈ ln(vocab_size) ≈ 10.4.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import GPT, ModelConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Tiny config for fast unit tests
TINY = ModelConfig(vocab_size=256, d_model=64, n_head=4, n_layer=4, ctx=32)

# Nano config (matches configs/nano.yaml)
NANO = ModelConfig(vocab_size=32_000, d_model=384, n_head=6, n_layer=6, ctx=256)

B, T = 2, 16


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return GPT(TINY)


@pytest.fixture(scope="module")
def nano_model():
    torch.manual_seed(0)
    return GPT(NANO)


# ---------------------------------------------------------------------------
# Shape and forward pass
# ---------------------------------------------------------------------------


def test_logits_shape(tiny_model):
    idx = torch.randint(0, TINY.vocab_size, (B, T))
    logits, loss = tiny_model(idx)
    assert logits.shape == (B, T, TINY.vocab_size)
    assert loss is None


def test_loss_returned_with_targets(tiny_model):
    idx     = torch.randint(0, TINY.vocab_size, (B, T))
    targets = torch.randint(0, TINY.vocab_size, (B, T))
    _, loss = tiny_model(idx, targets)
    assert loss is not None
    assert loss.shape == ()          # scalar
    assert torch.isfinite(loss)


def test_standard_next_token_target(tiny_model):
    """targets = idx shifted left by one (the standard LM setup)."""
    seq = torch.randint(0, TINY.vocab_size, (B, T + 1))
    idx, targets = seq[:, :-1], seq[:, 1:]
    _, loss = tiny_model(idx, targets)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Initial loss ≈ ln(vocab_size)  — THE checkpoint
# ---------------------------------------------------------------------------


def test_initial_loss_nano():
    """Freshly initialised nano model: loss should be within 0.5 of ln(32000)."""
    torch.manual_seed(42)
    model = GPT(NANO)
    seq   = torch.randint(0, NANO.vocab_size, (4, NANO.ctx + 1))
    idx, targets = seq[:, :-1], seq[:, 1:]
    with torch.no_grad():
        _, loss = model(idx, targets)
    expected = math.log(NANO.vocab_size)   # ln(32000) ≈ 10.37
    assert abs(loss.item() - expected) < 0.5, (
        f"Initial loss {loss.item():.4f} is far from ln({NANO.vocab_size}) = {expected:.4f}.\n"
        "Check: head/loss wiring, weight init, no softmax applied twice."
    )


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------


def test_weight_tying(tiny_model):
    """head.weight and embed.weight must be the same tensor object."""
    assert tiny_model.head.weight is tiny_model.embed.weight


def test_tied_weights_counted_once(tiny_model):
    """num_params() must not double-count the shared embedding/head tensor."""
    n = tiny_model.num_params()
    embed_params = TINY.vocab_size * TINY.d_model
    # If counted twice, num_params would be too large
    n_if_double_counted = n + embed_params
    # Raw sum without dedup
    raw = sum(p.numel() for name, p in tiny_model.named_parameters())
    assert n < raw or n == raw  # n <= raw always; meaningful check is n < raw when tied
    # More direct: head adds 0 params on top of embedding
    assert n == raw - embed_params or n == raw  # either dedup works or both count it once


# ---------------------------------------------------------------------------
# Scaled residual init
# ---------------------------------------------------------------------------


def test_residual_init_smaller_than_other_projections(tiny_model):
    """out_proj and w_down std should be smaller than qkv_proj std."""
    block = tiny_model.blocks[0]
    qkv_std  = block.attn.qkv_proj.weight.std().item()
    out_std  = block.attn.out_proj.weight.std().item()
    down_std = block.ffn.w_down.weight.std().item()
    assert out_std  < qkv_std, f"out_proj std {out_std:.4f} >= qkv std {qkv_std:.4f}"
    assert down_std < qkv_std, f"w_down std {down_std:.4f} >= qkv std {qkv_std:.4f}"


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


def test_nano_param_count(nano_model):
    """Nano model should be in the tens-of-millions range."""
    n = nano_model.num_params()
    print(f"\n  nano param count: {n/1e6:.2f}M")
    assert 5e6 < n < 100e6, f"Unexpected param count: {n/1e6:.2f}M"


def test_config_from_yaml():
    cfg = ModelConfig.from_yaml("configs/nano.yaml")
    assert cfg.n_layer == 6
    assert cfg.d_model == 384
