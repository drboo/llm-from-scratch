
"""
Day 4 checkpoint tests.

Run:  pytest model/test_day4.py -v
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.norm import RMSNorm
from model.model import TokenEmbedding, causal_mask


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def test_rmsnorm_output_shape():
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    assert norm(x).shape == x.shape


def test_rmsnorm_unit_rms():
    """With weight=ones the output RMS over the last dim should be ~1."""
    norm = RMSNorm(128)
    x = torch.randn(4, 16, 128) * 5.0  # deliberately large scale
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rmsnorm_learnable_weight():
    norm = RMSNorm(32)
    assert norm.weight.requires_grad
    assert norm.weight.shape == (32,)


# ---------------------------------------------------------------------------
# TokenEmbedding
# ---------------------------------------------------------------------------


def test_embedding_output_shape():
    emb = TokenEmbedding(vocab_size=1000, d_model=64)
    x = torch.randint(0, 1000, (2, 16))
    assert emb(x).shape == (2, 16, 64)


def test_embedding_init_std():
    """Weights should be initialised with std ≈ 0.02."""
    emb = TokenEmbedding(vocab_size=32_000, d_model=384)
    std = emb.embedding.weight.std().item()
    assert abs(std - 0.02) < 0.003, f"std={std:.4f}, expected ~0.02"


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------


def test_causal_mask_shape():
    mask = causal_mask(8)
    assert mask.shape == (8, 8)


def test_causal_mask_blocks_future():
    """Position i must not attend to any j > i."""
    T = 12
    mask = causal_mask(T)
    for i in range(T):
        for j in range(i + 1, T):
            assert mask[i, j] == float("-inf"), f"mask[{i},{j}] should be -inf"


def test_causal_mask_allows_past_and_present():
    """Position i must be able to attend to all j <= i."""
    T = 12
    mask = causal_mask(T)
    for i in range(T):
        for j in range(i + 1):
            assert mask[i, j] == 0.0, f"mask[{i},{j}] should be 0"


def test_causal_mask_softmax_zeroes_future():
    """Softmax of (logits + causal_mask) should assign zero weight to future."""
    T = 6
    mask = causal_mask(T)
    logits = torch.randn(T, T)
    weights = torch.softmax(logits + mask, dim=-1)
    for i in range(T):
        for j in range(i + 1, T):
            assert weights[i, j].item() == pytest.approx(0.0, abs=1e-6), (
                f"position {i} attends to future position {j} with weight {weights[i,j]:.2e}"
            )


def test_causal_mask_respects_device():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    mask = causal_mask(8, device=torch.device("cuda"))
    assert mask.device.type == "cuda"
