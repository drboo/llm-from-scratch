"""
Day 26 tests — KV cache and cached generation.

Run:  pytest model/test_day26.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import GPT, ModelConfig
from model.kv_cache import KVCache
from inference.sample import sample as naive_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nano(ctx: int = 64) -> GPT:
    """Tiny deterministic model for tests."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, d_model=64, n_head=2, n_layer=2, ctx=ctx)
    return GPT(cfg).eval()


def _prompt(length: int = 8, vocab: int = 256) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, vocab, (1, length))


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------


class TestKVCache:
    def test_shape(self):
        model = _nano()
        cache = KVCache.for_model(model, max_seq_len=64, device=torch.device("cpu"))
        cfg = model.cfg
        d_head = cfg.d_model // cfg.n_head
        for layer_k, layer_v in zip(cache.k, cache.v):
            assert layer_k.shape == (1, cfg.n_head, 64, d_head)
            assert layer_v.shape == (1, cfg.n_head, 64, d_head)

    def test_n_layers(self):
        model = _nano()
        cache = KVCache.for_model(model, max_seq_len=32, device=torch.device("cpu"))
        assert len(cache.k) == model.cfg.n_layer
        assert len(cache.v) == model.cfg.n_layer

    def test_reset_zeros(self):
        model = _nano()
        cache = KVCache.for_model(model, max_seq_len=32, device=torch.device("cpu"))
        cache.k[0].fill_(1.0)
        cache.reset()
        assert cache.k[0].sum() == 0.0

    def test_initially_zero(self):
        model = _nano()
        cache = KVCache.for_model(model, max_seq_len=32, device=torch.device("cpu"))
        assert all(c.sum() == 0.0 for c in cache.k)
        assert all(c.sum() == 0.0 for c in cache.v)


# ---------------------------------------------------------------------------
# forward_cached — attention layer
# ---------------------------------------------------------------------------


class TestAttentionForwardCached:
    def test_output_shape_single_token(self):
        model = _nano()
        attn  = model.blocks[0].attn
        cache = KVCache.for_model(model, 64, torch.device("cpu"))
        x     = torch.randn(1, 1, model.cfg.d_model)
        out   = attn.forward_cached(x, start_pos=0, k_cache=cache.k[0], v_cache=cache.v[0])
        assert out.shape == (1, 1, model.cfg.d_model)

    def test_output_shape_prefill(self):
        model = _nano()
        attn  = model.blocks[0].attn
        cache = KVCache.for_model(model, 64, torch.device("cpu"))
        T     = 8
        x     = torch.randn(1, T, model.cfg.d_model)
        out   = attn.forward_cached(x, start_pos=0, k_cache=cache.k[0], v_cache=cache.v[0])
        assert out.shape == (1, T, model.cfg.d_model)

    def test_cache_is_written(self):
        model = _nano()
        attn  = model.blocks[0].attn
        cache = KVCache.for_model(model, 64, torch.device("cpu"))
        x     = torch.randn(1, 4, model.cfg.d_model)
        attn.forward_cached(x, start_pos=0, k_cache=cache.k[0], v_cache=cache.v[0])
        # First 4 positions of cache must be non-zero
        assert cache.k[0][:, :, :4, :].abs().sum() > 0
        # Positions 4+ must still be zero
        assert cache.k[0][:, :, 4:, :].abs().sum() == 0.0

    def test_rope_uses_absolute_position(self):
        """
        Generating token at pos 10 must differ from pos 0 due to RoPE.
        We verify by comparing the cached K values.
        """
        model = _nano()
        attn  = model.blocks[0].attn
        cache_a = KVCache.for_model(model, 64, torch.device("cpu"))
        cache_b = KVCache.for_model(model, 64, torch.device("cpu"))
        x = torch.randn(1, 1, model.cfg.d_model)

        attn.forward_cached(x, start_pos=0,  k_cache=cache_a.k[0], v_cache=cache_a.v[0])
        attn.forward_cached(x, start_pos=10, k_cache=cache_b.k[0], v_cache=cache_b.v[0])

        k0  = cache_a.k[0][:, :, 0,  :]
        k10 = cache_b.k[0][:, :, 10, :]
        assert not torch.allclose(k0, k10), "RoPE must differ at different positions"


# ---------------------------------------------------------------------------
# generate_cached — identity with naive sample
# ---------------------------------------------------------------------------


class TestGenerateCachedIdentity:
    """The core correctness invariant: cached == non-cached under greedy."""

    def _check(self, prompt_len: int, n_new: int, ctx: int = 64):
        model  = _nano(ctx=ctx)
        prompt = _prompt(prompt_len)

        with torch.no_grad():
            out_naive  = naive_sample(model, prompt, n_new=n_new, temperature=0.0)
            out_cached = model.generate_cached(prompt, n_new=n_new, temperature=0.0)

        new_naive  = out_naive[ 0, prompt_len:].tolist()
        new_cached = out_cached[0, prompt_len:].tolist()
        assert new_naive == new_cached, (
            f"Mismatch at prompt_len={prompt_len} n_new={n_new}\n"
            f"  naive:  {new_naive}\n"
            f"  cached: {new_cached}"
        )

    def test_short_prompt_short_gen(self):
        self._check(prompt_len=4, n_new=8)

    def test_medium_prompt(self):
        self._check(prompt_len=16, n_new=20)

    def test_single_token_prompt(self):
        self._check(prompt_len=1, n_new=10)

    def test_long_generation(self):
        self._check(prompt_len=8, n_new=40, ctx=64)

    def test_prompt_preserved_in_output(self):
        model  = _nano()
        prompt = _prompt(6)
        out    = model.generate_cached(prompt, n_new=10, temperature=0.0)
        assert out.shape[1] >= 6
        assert out[0, :6].tolist() == prompt[0].tolist()

    def test_output_longer_than_prompt(self):
        model  = _nano()
        prompt = _prompt(4)
        out    = model.generate_cached(prompt, n_new=10, temperature=0.0)
        assert out.shape[1] > prompt.shape[1]


# ---------------------------------------------------------------------------
# EOS stopping
# ---------------------------------------------------------------------------


class TestEOSStopping:
    def test_stops_at_eos(self):
        """If the model emits eos_id it should stop early."""
        model  = _nano()
        prompt = _prompt(4)
        # Run without stopping to find what the first generated token is
        with torch.no_grad():
            full = model.generate_cached(prompt, n_new=20, temperature=0.0)
        first_new = full[0, 4].item()
        # Now use that token as eos — should get exactly 1 new token
        out = model.generate_cached(prompt, n_new=20, temperature=0.0,
                                     eos_id=first_new)
        assert out.shape[1] == prompt.shape[1] + 1

    def test_no_eos_generates_full_length(self):
        model  = _nano()
        prompt = _prompt(4)
        n_new  = 15
        out = model.generate_cached(prompt, n_new=n_new, temperature=0.0,
                                     eos_id=None)
        assert out.shape[1] == prompt.shape[1] + n_new


# ---------------------------------------------------------------------------
# Benchmark — cached must be faster than naive on repeated calls
# ---------------------------------------------------------------------------


class TestCachedFaster:
    def test_cached_not_slower(self):
        """
        On CPU with tiny model, the speedup may be modest but cached
        must not be significantly slower than naive.
        """
        import time
        model  = _nano(ctx=64)
        prompt = _prompt(8)
        n_new  = 30

        # Warmup
        with torch.no_grad():
            naive_sample(model, prompt, n_new=5, temperature=0.0)
            model.generate_cached(prompt, n_new=5, temperature=0.0)

        t0 = time.perf_counter()
        with torch.no_grad():
            naive_sample(model, prompt, n_new=n_new, temperature=0.0)
        naive_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.generate_cached(prompt, n_new=n_new, temperature=0.0)
        cached_time = time.perf_counter() - t0

        # Cached must not be more than 50% slower (allowing for overhead on tiny CPU model)
        assert cached_time < naive_time * 1.5, (
            f"Cached ({cached_time:.3f}s) much slower than naive ({naive_time:.3f}s)"
        )
