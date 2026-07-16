from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import precompute_rope_freqs, apply_rope
from model.model import causal_mask


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    Supports two forward paths that must agree numerically:
      - flash  (use_flash=True):  F.scaled_dot_product_attention with is_causal=True
      - manual (use_flash=False): explicit scores → mask → softmax → values
    """

    def __init__(self, d_model: int, n_head: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_model = d_model

        self.qkv_proj  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj   = nn.Linear(d_model, d_model, bias=False)

        cos, sin = precompute_rope_freqs(self.d_head, max_seq_len)
        self.register_buffer("rope_cos", cos)  # (max_seq_len, d_head // 2)
        self.register_buffer("rope_sin", sin)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """(B, T, d_model) → (B, n_head, T, d_head)"""
        return x.view(B, T, self.n_head, self.d_head).transpose(1, 2)

    def _manual_attn(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """Explicit scaled dot-product attention (for learning / verification).

        Q, K, V: (B, n_head, T, d_head)
        """
        T = Q.size(2)
        scale = math.sqrt(self.d_head)
        scores = (Q @ K.transpose(-2, -1)) / scale           # (B, H, T, T)
        scores = scores + causal_mask(T, device=Q.device)    # add -inf to future
        weights = F.softmax(scores.float(), dim=-1).to(Q.dtype)
        return weights @ V                                    # (B, H, T, d_head)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, use_flash: bool = True) -> torch.Tensor:
        """
        Args:
            x:         (B, T, d_model)
            use_flash: True → F.scaled_dot_product_attention (fast)
                       False → manual implementation (identical numerics on CPU)
        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)                        # (B, T, 3·d_model)
        Q, K, V = qkv.split(self.d_model, dim=-1)

        Q = self._split_heads(Q, B, T)                # (B, H, T, d_head)
        K = self._split_heads(K, B, T)
        V = self._split_heads(V, B, T)

        # Apply RoPE to Q and K only (not V)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # Attend
        if use_flash:
            out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        else:
            out = self._manual_attn(Q, K, V)

        # Merge heads: (B, H, T, d_head) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)
