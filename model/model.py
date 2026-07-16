from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learnable token embedding table: (vocab_size, d_model).

    Initialised with std=0.02 (GPT-2 convention) so logits start near zero
    and the initial loss is close to ln(vocab_size).
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) int64 token ids  →  (B, T, d_model)
        return self.embedding(x)


def causal_mask(T: int, device: torch.device | None = None) -> torch.Tensor:
    """Additive causal attention mask of shape (T, T).

    mask[i, j] = 0      when j <= i  (position i can attend to j)
    mask[i, j] = -inf   when j > i   (position i cannot see the future)

    Add this to raw attention logits before softmax.
    """
    mask = torch.full((T, T), float("-inf"), device=device)
    # triu(diagonal=1) zeroes the diagonal and below, leaving -inf only above
    return torch.triu(mask, diagonal=1)
