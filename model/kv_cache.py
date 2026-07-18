"""KV cache for autoregressive generation."""

from __future__ import annotations

import torch


class KVCache:
    """
    Pre-allocated key/value cache for all transformer layers.

    Each layer gets a pair of tensors of shape
    (1, n_head, max_seq_len, d_head).  Positions are filled in-place as
    generation proceeds, so there are no repeated allocations.

    Usage:
        cache = KVCache.for_model(model, max_seq_len, device)
        # then pass cache.k[i], cache.v[i] to layer i's forward_cached()
    """

    def __init__(
        self,
        n_layer:     int,
        n_head:      int,
        d_head:      int,
        max_seq_len: int,
        device:      torch.device,
        dtype:       torch.dtype = torch.float32,
    ):
        shape = (1, n_head, max_seq_len, d_head)
        self.k = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(n_layer)]
        self.n_layer     = n_layer
        self.max_seq_len = max_seq_len

    @classmethod
    def for_model(
        cls,
        model,
        max_seq_len: int,
        device:      torch.device,
        dtype:       torch.dtype = torch.float32,
    ) -> "KVCache":
        """Convenience constructor that reads shape from a GPT model.

        With GQA, the cache uses n_kv_head rather than n_head so it is
        proportionally smaller (e.g. 3× smaller when n_head=6, n_kv_head=2).
        """
        cfg    = model.cfg
        d_head = cfg.d_model // cfg.n_head
        return cls(cfg.n_layer, cfg.kv_heads, d_head, max_seq_len, device, dtype)

    def reset(self) -> None:
        """Zero out the cache (call between independent generation requests)."""
        for i in range(self.n_layer):
            self.k[i].zero_()
            self.v[i].zero_()
