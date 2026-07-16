"""
Day 6 checkpoint tests — SwiGLU FFN and transformer block.

Run:  pytest model/test_day6.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.ffn import SwiGLUFFN, _swiglu_hidden
from model.block import TransformerBlock

# ---------------------------------------------------------------------------
# Shared config (nano)
# ---------------------------------------------------------------------------

D_MODEL = 384
N_HEAD  = 6
N_LAYER = 6
B, T    = 2, 32


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------


def test_swiglu_hidden_dim_matches_gelu_params():
    """3 · d · h ≈ 2 · d · 4d  →  h ≈ 8d/3, param counts within 5%."""
    for d in [128, 256, 384, 512]:
        h = _swiglu_hidden(d)
        swiglu_params = 3 * d * h
        gelu_params   = 2 * d * (4 * d)
        ratio = swiglu_params / gelu_params
        assert 0.95 <= ratio <= 1.15, (
            f"d_model={d}: SwiGLU params {swiglu_params} vs GELU {gelu_params} "
            f"(ratio {ratio:.3f})"
        )


def test_swiglu_output_shape():
    ffn = SwiGLUFFN(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    assert ffn(x).shape == (B, T, D_MODEL)


def test_swiglu_no_bias():
    ffn = SwiGLUFFN(D_MODEL)
    for name, param in ffn.named_parameters():
        assert "bias" not in name, f"unexpected bias: {name}"


def test_swiglu_no_nan():
    ffn = SwiGLUFFN(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    out = ffn(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


def test_block_output_shape():
    block = TransformerBlock(D_MODEL, N_HEAD)
    x = torch.randn(B, T, D_MODEL)
    assert block(x).shape == (B, T, D_MODEL)


def test_block_no_nan():
    block = TransformerBlock(D_MODEL, N_HEAD)
    x = torch.randn(B, T, D_MODEL)
    assert torch.isfinite(block(x)).all()


def test_block_residual_changes_input():
    """Output must differ from input — the block is doing something."""
    block = TransformerBlock(D_MODEL, N_HEAD)
    x = torch.randn(B, T, D_MODEL)
    assert not torch.allclose(block(x), x)


# ---------------------------------------------------------------------------
# Stack 6 blocks: no NaNs, activations don't blow up layer to layer
# ---------------------------------------------------------------------------


def test_stacked_blocks_stable():
    """Forward a random batch through 6 blocks.

    Prints per-layer RMS so anomalies are visible.  The hard assertion is that
    every layer's RMS stays in [0.01, 1000] — a very loose bound that would
    only fail on genuine explosion or collapse.
    """
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList(
        [TransformerBlock(D_MODEL, N_HEAD) for _ in range(N_LAYER)]
    )

    x = torch.randn(B, T, D_MODEL)
    print()  # newline so per-layer output starts on its own line
    for i, block in enumerate(blocks):
        x = block(x)
        rms = x.pow(2).mean().sqrt().item()
        print(f"  layer {i+1:2d}  RMS = {rms:.4f}")
        assert torch.isfinite(x).all(), f"NaN/Inf at layer {i+1}"
        assert 0.01 < rms < 1000, f"activations exploded/collapsed at layer {i+1}: RMS={rms:.4f}"
