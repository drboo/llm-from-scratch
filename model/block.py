from __future__ import annotations

import torch
import torch.nn as nn

from model.norm import RMSNorm
from model.attention import CausalSelfAttention
from model.ffn import SwiGLUFFN


class TransformerBlock(nn.Module):
    """One transformer block: pre-norm attention + pre-norm FFN, both residual.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Pre-norm keeps the residual stream clean — gradients flow back through
    the residual path unscaled, so early layers receive meaningful signal
    even in deep stacks.
    """

    def __init__(self, d_model: int, n_head: int, n_kv_head: int = 0,
                 max_seq_len: int = 2048):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_head, n_kv_head, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLUFFN(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_cached(
        self,
        x:        torch.Tensor,
        start_pos: int,
        k_cache:  torch.Tensor,
        v_cache:  torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn.forward_cached(self.norm1(x), start_pos, k_cache, v_cache)
        x = x + self.ffn(self.norm2(x))
        return x
