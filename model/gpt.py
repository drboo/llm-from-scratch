"""
Day 7: Full GPT model — embed → N blocks → RMSNorm → tied head → CE loss.

Usage:
    from model.gpt import GPT, ModelConfig
    cfg = ModelConfig()          # nano defaults
    model = GPT(cfg)
    logits, loss = model(idx, targets)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.block import TransformerBlock
from model.norm import RMSNorm


@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    d_model:    int = 384
    n_head:     int = 6
    n_layer:    int = 6
    ctx:        int = 256

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class GPT(nn.Module):
    """Decoder-only transformer (GPT-style).

    Architecture:
        token embed  →  N × TransformerBlock  →  RMSNorm  →  linear head
    Weight tying: head.weight == embed.weight (saves vocab_size × d_model params).
    Scaled residual init: out_proj and ffn.w_down scaled by 1/sqrt(2·n_layer)
    so residual-stream variance stays bounded at init regardless of depth.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_head, max_seq_len=cfg.ctx)
            for _ in range(cfg.n_layer)
        ])
        self.norm   = RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying — head and embedding share the same tensor.
        # Parameters() deduplicates by id so this is counted only once.
        self.head.weight = self.embed.weight

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        # Embedding / tied head: std=0.02 (GPT-2 convention)
        nn.init.normal_(self.embed.weight, std=0.02)

        # All linear projections inside blocks: std=0.02
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)

        # Residual projections scaled down by 1/sqrt(2·n_layer).
        # These write back into the residual stream; scaling keeps the
        # total residual variance ≈ 1 at init regardless of depth.
        scale = (2 * self.cfg.n_layer) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=0.02 * scale)
            nn.init.normal_(block.ffn.w_down.weight,    std=0.02 * scale)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_params(self) -> int:
        """Total parameters, counting tied weights only once."""
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:     (B, T) int64 token ids
            targets: (B, T) int64 — next-token labels (idx shifted left by 1).
                     Pass None to get logits only (e.g. at inference).
        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss, or None if targets is None
        """
        x = self.embed(idx)                    # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)                  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss
