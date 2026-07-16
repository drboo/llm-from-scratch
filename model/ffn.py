from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _swiglu_hidden(d_model: int, multiple_of: int = 64) -> int:
    """Hidden dim that matches standard GELU FFN parameter count.

    A GELU FFN has 2 matrices of shape (d_model, 4·d_model).
    SwiGLU has 3 matrices (gate, up, down) so to keep the same total params:
        3 · d_model · h = 2 · d_model · 4·d_model  →  h = 8/3 · d_model
    Rounded up to the nearest `multiple_of` for memory-alignment efficiency.
    """
    raw = int(8 * d_model / 3)
    return multiple_of * ((raw + multiple_of - 1) // multiple_of)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (Noam Shazeer, 2020).

    forward(x) = W_down( silu(W_gate(x)) ⊙ W_up(x) )

    The gating mechanism lets the network selectively suppress dimensions,
    which empirically outperforms vanilla GELU FFNs at the same parameter count.
    """

    def __init__(self, d_model: int, multiple_of: int = 64):
        super().__init__()
        h = _swiglu_hidden(d_model, multiple_of)
        self.w_gate = nn.Linear(d_model, h, bias=False)
        self.w_up   = nn.Linear(d_model, h, bias=False)
        self.w_down = nn.Linear(h, d_model, bias=False)

    @property
    def hidden_dim(self) -> int:
        return self.w_gate.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
