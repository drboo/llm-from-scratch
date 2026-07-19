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
    # GQA: number of KV heads. 0 means n_head (standard MHA).
    # Must divide n_head evenly. 1 = MQA, n_head = MHA.
    n_kv_head:  int = 0

    @property
    def kv_heads(self) -> int:
        """Resolved number of KV heads (never 0)."""
        return self.n_kv_head if self.n_kv_head > 0 else self.n_head

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
            TransformerBlock(cfg.d_model, cfg.n_head,
                             n_kv_head=cfg.kv_heads, max_seq_len=cfg.ctx)
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

    @classmethod
    def from_checkpoint(cls, path: str | Path,
                        device: str | torch.device = "cpu") -> "GPT":
        """Load a GPT from a checkpoint, inferring architecture automatically.

        Prefers the 'model_cfg' key written by checkpoint.save(); falls back
        to inferring the config from state-dict shapes for older checkpoints.
        """
        state = torch.load(path, map_location=device, weights_only=False)
        sd    = state.get("model_state_dict", state.get("model", state))

        if "model_cfg" in state and state["model_cfg"] is not None:
            cfg = ModelConfig(**{k: v for k, v in state["model_cfg"].items()
                                 if k in ModelConfig.__dataclass_fields__})
        else:
            vocab_size = sd["embed.weight"].shape[0]
            d_model    = sd["embed.weight"].shape[1]
            n_layer    = max(int(k.split(".")[1]) for k in sd
                             if k.startswith("blocks.")) + 1
            d_head    = 2 * sd["blocks.0.attn.rope_cos"].shape[1]
            n_head    = d_model // d_head
            out_feats = sd["blocks.0.attn.qkv_proj.weight"].shape[0]
            n_kv_head = (out_feats // d_head - n_head) // 2
            ctx       = sd["blocks.0.attn.rope_cos"].shape[0]
            cfg = ModelConfig(vocab_size=vocab_size, d_model=d_model,
                              n_head=n_head, n_kv_head=n_kv_head,
                              n_layer=n_layer, ctx=ctx)

        model = cls(cfg)
        model.load_state_dict(sd, strict=True)
        return model.to(device)

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

    # ------------------------------------------------------------------
    # Cached generation (Day 26)
    # ------------------------------------------------------------------

    def forward_cached(
        self,
        idx:       torch.Tensor,
        start_pos: int,
        cache,
    ) -> torch.Tensor:
        """
        Single forward pass using the KV cache.

        Args:
            idx:       (1, T) token ids — T=prompt_len on prefill, T=1 on decode
            start_pos: absolute sequence position of the first token in idx
            cache:     KVCache instance (mutated in-place)

        Returns:
            logits (1, T, vocab_size)
        """
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block.forward_cached(x, start_pos, cache.k[i], cache.v[i])
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate_cached(
        self,
        prompt_ids:  torch.Tensor,
        n_new:       int,
        temperature: float = 1.0,
        top_k:       int | None = None,
        top_p:       float | None = None,
        eos_id:      int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV cache.

        Prefills the prompt in a single pass, then decodes one token at a
        time.  Each decode step runs only one token through the network;
        previous K/V are read from the cache.

        Args:
            prompt_ids: (1, T) int64
            n_new:      maximum new tokens to generate
            temperature: > 0; 0 = greedy
            top_k:      top-k filtering (None = disabled)
            top_p:      nucleus filtering (None = disabled)
            eos_id:     stop early if this token is sampled

        Returns:
            (1, T + n_generated) token ids
        """
        from model.kv_cache import KVCache
        from inference.sample import top_k_filter, top_p_filter

        device    = prompt_ids.device
        prompt_len = prompt_ids.shape[1]
        max_len   = prompt_len + n_new

        cache = KVCache.for_model(self, max_len, device,
                                   dtype=next(self.parameters()).dtype)

        # ── Prefill ─────────────────────────────────────────────────────
        logits = self.forward_cached(prompt_ids, start_pos=0, cache=cache)
        # logits: (1, prompt_len, vocab_size)

        generated: list[int] = []
        next_logits = logits[:, -1, :]   # (1, vocab_size)

        # ── Decode loop ─────────────────────────────────────────────────
        for step in range(n_new):
            # Sample from next_logits
            if temperature == 0.0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            else:
                scaled = next_logits / temperature
                if top_k is not None:
                    scaled = top_k_filter(scaled, top_k)
                if top_p is not None:
                    scaled = top_p_filter(scaled, top_p)
                probs   = F.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)

            tok = next_id.item()
            generated.append(tok)

            if eos_id is not None and tok == eos_id:
                break

            if prompt_len + step + 1 >= max_len:
                break

            # Feed the new token back
            pos     = prompt_len + step
            logits  = self.forward_cached(next_id, start_pos=pos, cache=cache)
            next_logits = logits[:, -1, :]

        gen_tensor = torch.tensor([generated], dtype=torch.long, device=device)
        return torch.cat([prompt_ids, gen_tensor], dim=1)
