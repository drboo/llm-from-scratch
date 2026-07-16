from __future__ import annotations

import torch


def precompute_rope_freqs(
    d_head: int,
    max_seq_len: int,
    theta: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for RoPE.

    For each dimension pair i = 0…d_head/2-1:
        freq_i = theta^(-2i / d_head)
    For each position m = 0…max_seq_len-1:
        angle[m, i] = m * freq_i

    Returns:
        cos, sin — both shape (max_seq_len, d_head // 2)
    """
    i = torch.arange(0, d_head, 2, dtype=torch.float32)
    freqs = 1.0 / (theta ** (i / d_head))               # (d_head // 2,)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)               # (max_seq_len, d_head // 2)
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to x.

    Each consecutive pair of dimensions (2i, 2i+1) is rotated by the angle
    for position m and frequency i:
        [x0, x1] → [x0·cos - x1·sin,  x0·sin + x1·cos]

    Args:
        x:   (..., T, d_head)  — Q or K tensor (not V)
        cos: (T, d_head // 2)  — already sliced to current sequence length
        sin: (T, d_head // 2)

    Returns:
        Rotated tensor, same shape as x.
    """
    x_even = x[..., 0::2]   # (..., T, d_head // 2)
    x_odd  = x[..., 1::2]

    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)

    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos

    # Re-interleave: stack on last dim then flatten
    # [..., d_head//2, 2] → [..., d_head]
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2)
