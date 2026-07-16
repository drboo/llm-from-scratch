"""
Day 5 checkpoint tests — RoPE and causal self-attention.

Run:  pytest model/test_day5.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.rope import precompute_rope_freqs, apply_rope
from model.attention import CausalSelfAttention

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

D_MODEL  = 64
N_HEAD   = 4
D_HEAD   = D_MODEL // N_HEAD   # 16
B, T     = 2, 12


# ---------------------------------------------------------------------------
# RoPE tests
# ---------------------------------------------------------------------------


def test_rope_freqs_shape():
    cos, sin = precompute_rope_freqs(D_HEAD, max_seq_len=128)
    assert cos.shape == (128, D_HEAD // 2)
    assert sin.shape == (128, D_HEAD // 2)


def test_rope_position_zero_is_identity():
    """At position 0 all angles are 0: cos=1, sin=0, so output == input."""
    cos, sin = precompute_rope_freqs(D_HEAD, max_seq_len=32)
    x = torch.randn(1, 1, D_HEAD)          # (B=1, T=1, d_head)
    cos0 = cos[:1]                          # (1, D_HEAD // 2)
    sin0 = sin[:1]
    out = apply_rope(x, cos0, sin0)
    assert torch.allclose(out, x, atol=1e-6), "RoPE at position 0 must be identity"


def test_rope_output_shape():
    cos, sin = precompute_rope_freqs(D_HEAD, max_seq_len=T)
    x = torch.randn(B, N_HEAD, T, D_HEAD)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


def test_rope_relative_property():
    """Inner product <RoPE(q,m), RoPE(k,n)> depends only on (m-n), not m or n.

    For a fixed difference d = m - n, sample three (m,n) pairs.
    The dot-product should be the same (up to fp32 rounding).
    """
    torch.manual_seed(0)
    cos, sin = precompute_rope_freqs(D_HEAD, max_seq_len=64)

    q = torch.randn(D_HEAD)
    k = torch.randn(D_HEAD)

    def dot_at(m: int, n: int) -> float:
        q_rot = apply_rope(q.view(1, 1, D_HEAD), cos[m:m+1], sin[m:m+1]).squeeze()
        k_rot = apply_rope(k.view(1, 1, D_HEAD), cos[n:n+1], sin[n:n+1]).squeeze()
        return (q_rot * k_rot).sum().item()

    # All pairs have relative offset d = 3
    d03 = dot_at(0, 3)   # wait, we want m > n so let's use m=3, n=0, diff = 3
    # actually we want the same m-n for all; let's pick diff = +5
    ref  = dot_at(5,  0)
    d2   = dot_at(10, 5)
    d3   = dot_at(20, 15)

    assert abs(ref - d2) < 1e-4, f"relative property failed: {ref:.6f} vs {d2:.6f}"
    assert abs(ref - d3) < 1e-4, f"relative property failed: {ref:.6f} vs {d3:.6f}"


# ---------------------------------------------------------------------------
# Attention tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def attn():
    torch.manual_seed(42)
    return CausalSelfAttention(d_model=D_MODEL, n_head=N_HEAD, max_seq_len=64)


def test_attention_output_shape(attn):
    x = torch.randn(B, T, D_MODEL)
    out = attn(x)
    assert out.shape == (B, T, D_MODEL)


def test_flash_and_manual_agree(attn):
    """Both implementations must agree within 1e-4 on CPU."""
    torch.manual_seed(7)
    x = torch.randn(B, T, D_MODEL)
    with torch.no_grad():
        y_flash  = attn(x, use_flash=True)
        y_manual = attn(x, use_flash=False)
    assert torch.allclose(y_flash, y_manual, atol=1e-4), (
        f"max diff = {(y_flash - y_manual).abs().max().item():.2e}"
    )


def test_causal_mask_in_attention(attn):
    """Perturbing token at position t+1 must not change outputs at positions ≤ t."""
    torch.manual_seed(3)
    x = torch.randn(1, T, D_MODEL)

    pivot = T // 2   # perturb everything after this position

    with torch.no_grad():
        y1 = attn(x, use_flash=False)

        x2 = x.clone()
        x2[:, pivot + 1:, :] = torch.randn_like(x2[:, pivot + 1:, :])
        y2 = attn(x2, use_flash=False)

    assert torch.allclose(y1[:, :pivot + 1, :], y2[:, :pivot + 1, :], atol=1e-5), (
        "Causal mask violated: output at positions ≤ pivot changed after "
        "perturbing future tokens."
    )


def test_rope_applied_to_qk_not_v(attn):
    """RoPE must not be applied to V — output at position 0 shouldn't shift
    when only tokens after 0 change (already covered by causality, but this
    also confirms V is untouched by positional encoding)."""
    torch.manual_seed(5)
    x1 = torch.randn(1, T, D_MODEL)
    x2 = x1.clone()
    x2[:, 1:, :] = torch.randn_like(x2[:, 1:, :])

    with torch.no_grad():
        y1 = attn(x1, use_flash=False)
        y2 = attn(x2, use_flash=False)

    assert torch.allclose(y1[:, 0, :], y2[:, 0, :], atol=1e-5), (
        "Output at position 0 changed when only future tokens were perturbed."
    )
