"""
Causal self-attention with optional Grouped Query Attention (GQA).

Standard MHA:  n_kv_head == n_head   (every Q head has its own K/V pair)
GQA:           n_kv_head < n_head    (each KV head is shared by n_groups Q heads)
MQA:           n_kv_head == 1        (all Q heads share one KV pair)

KV cache savings over MHA: n_head / n_kv_head × (can fit larger batch or longer ctx).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import precompute_rope_freqs, apply_rope
from model.model import causal_mask


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional GQA.

    For GQA, Q keeps n_head heads while K and V use n_kv_head heads
    (n_kv_head must evenly divide n_head).  Before SDPA the K/V heads are
    expanded (repeat_interleave) so Q×K^T has the right shape.
    The KV cache stores only n_kv_head slices, so GQA directly reduces
    cache memory and bandwidth.
    """

    def __init__(self, d_model: int, n_head: int, n_kv_head: int = 0,
                 max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        n_kv_head = n_kv_head if n_kv_head > 0 else n_head
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head    = n_head
        self.n_kv_head = n_kv_head
        self.n_groups  = n_head // n_kv_head   # Q heads per KV head
        self.d_head    = d_model // n_head
        self.d_model   = d_model

        # Fused projection: Q gets n_head channels, K and V get n_kv_head each.
        # When n_kv_head == n_head this reduces to the original 3×d_model shape.
        q_dim  = n_head    * self.d_head
        kv_dim = n_kv_head * self.d_head
        self.qkv_proj = nn.Linear(d_model, q_dim + 2 * kv_dim, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

        cos, sin = precompute_rope_freqs(self.d_head, max_seq_len)
        self.register_buffer("rope_cos", cos)  # (max_seq_len, d_head // 2)
        self.register_buffer("rope_sin", sin)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor, B: int, T: int,
                     n: int) -> torch.Tensor:
        """(B, T, n*d_head) → (B, n, T, d_head)"""
        return x.view(B, T, n, self.d_head).transpose(1, 2)

    def _expand_kv(self, K: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand n_kv_head → n_head via repeat_interleave (no-op when n_groups==1)."""
        if self.n_groups == 1:
            return K, V
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)
        return K, V

    def _manual_attn(self, Q: torch.Tensor, K: torch.Tensor,
                     V: torch.Tensor) -> torch.Tensor:
        """Explicit scaled dot-product attention (for verification).

        Q, K, V: (B, n_head, T, d_head)
        """
        T     = Q.size(2)
        scale = math.sqrt(self.d_head)
        scores  = (Q @ K.transpose(-2, -1)) / scale
        scores  = scores + causal_mask(T, device=Q.device)
        weights = F.softmax(scores.float(), dim=-1).to(Q.dtype)
        return weights @ V

    def _project(self, x: torch.Tensor, B: int, T: int):
        """Project x → (Q, K, V) split-heads tensors."""
        qkv   = self.qkv_proj(x)
        q_dim  = self.n_head    * self.d_head
        kv_dim = self.n_kv_head * self.d_head
        Q, K, V = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        Q = self._split_heads(Q, B, T, self.n_head)
        K = self._split_heads(K, B, T, self.n_kv_head)
        V = self._split_heads(V, B, T, self.n_kv_head)
        return Q, K, V

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, use_flash: bool = True) -> torch.Tensor:
        """
        Args:
            x:         (B, T, d_model)
            use_flash: True → F.scaled_dot_product_attention
                       False → manual (identical numerics on CPU)
        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        Q, K, V = self._project(x, B, T)

        # RoPE: positions 0..T-1 (standard causal forward)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # Expand KV heads for GQA
        K, V = self._expand_kv(K, V)

        if use_flash:
            out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        else:
            out = self._manual_attn(Q, K, V)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Cached forward (Day 26 + GQA)
    # ------------------------------------------------------------------

    def forward_cached(
        self,
        x:       torch.Tensor,
        start_pos: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cached forward for autoregressive generation.

        Cache tensors have shape (1, n_kv_head, max_seq_len, d_head).
        GQA heads are expanded after reading from the cache.

        Args:
            x:         (1, T, d_model)  T=prompt_len on prefill, T=1 on decode
            start_pos: absolute position of the first token in x
            k_cache:   (1, n_kv_head, max_seq_len, d_head)  mutated in-place
            v_cache:   (1, n_kv_head, max_seq_len, d_head)  mutated in-place

        Returns:
            (1, T, d_model)
        """
        B, T, _ = x.shape

        Q, K, V = self._project(x, B, T)

        # Chunked prefill (start_pos > 0 and T > 1) would need a bottom-right-
        # aligned causal mask; only full prefill (start_pos=0) and single-token
        # decode (T=1) are supported today.
        assert start_pos == 0 or T == 1, (
            "Cached forward only supports full prefill (start_pos=0) or "
            "single-token decode (T=1). Chunked prefill is not implemented."
        )

        # RoPE with ABSOLUTE positions — assert bounds to catch rope-buffer overflow early.
        rope_len = self.rope_cos.shape[0]
        assert start_pos + T <= rope_len, (
            f"Sequence position {start_pos + T} exceeds RoPE buffer size {rope_len}. "
            "Extend rope buffers via _extend_rope_buffers() before long generations."
        )
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # Write new KV into cache (n_kv_head slots)
        k_cache[:, :, start_pos : start_pos + T, :] = K
        v_cache[:, :, start_pos : start_pos + T, :] = V

        # Read full context from cache, then expand for GQA.
        # Note: repeat_interleave materialises n_head-sized copies on every
        # decode step, partially offsetting the GQA cache-memory savings.
        K_ctx = k_cache[:, :, : start_pos + T, :]
        V_ctx = v_cache[:, :, : start_pos + T, :]
        K_ctx, V_ctx = self._expand_kv(K_ctx, V_ctx)

        out = F.scaled_dot_product_attention(
            Q, K_ctx, V_ctx, is_causal=(T > 1)
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)
