"""
Day 29 tests — Grouped Query Attention (GQA).

Tests cover:
  - Config & parameter count
  - MHA backwards-compatibility (n_kv_head == n_head)
  - GQA (n_kv_head < n_head): shapes, output, KV cache
  - MQA (n_kv_head == 1)
  - Identity with cached generation under GQA
  - KV cache memory savings

Run:  pytest model/test_day29.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.attention import CausalSelfAttention
from model.gpt import GPT, ModelConfig
from model.kv_cache import KVCache
from inference.sample import sample as naive_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attn(n_head: int, n_kv_head: int, d_model: int = 64,
          max_seq_len: int = 64) -> CausalSelfAttention:
    torch.manual_seed(0)
    return CausalSelfAttention(d_model, n_head, n_kv_head, max_seq_len).eval()


def _gpt(n_head: int = 4, n_kv_head: int = 2, ctx: int = 64) -> GPT:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, d_model=64, n_head=n_head,
                      n_kv_head=n_kv_head, n_layer=2, ctx=ctx)
    return GPT(cfg).eval()


def _prompt(length: int = 8, vocab: int = 256) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, vocab, (1, length))


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_kv_heads_default_equals_n_head(self):
        cfg = ModelConfig(n_head=6)
        assert cfg.kv_heads == 6

    def test_kv_heads_gqa(self):
        cfg = ModelConfig(n_head=6, n_kv_head=2)
        assert cfg.kv_heads == 2

    def test_kv_heads_mqa(self):
        cfg = ModelConfig(n_head=6, n_kv_head=1)
        assert cfg.kv_heads == 1

    def test_invalid_n_kv_head_raises(self):
        with pytest.raises(AssertionError):
            CausalSelfAttention(d_model=64, n_head=6, n_kv_head=4)  # 6 % 4 != 0


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


class TestParamCount:
    def test_gqa_fewer_params_than_mha(self):
        """GQA has a smaller qkv_proj than MHA."""
        mha = _attn(n_head=4, n_kv_head=4)
        gqa = _attn(n_head=4, n_kv_head=2)
        mha_p = sum(p.numel() for p in mha.parameters())
        gqa_p = sum(p.numel() for p in gqa.parameters())
        assert gqa_p < mha_p

    def test_mqa_fewest_params(self):
        mha = _attn(n_head=4, n_kv_head=4)
        mqa = _attn(n_head=4, n_kv_head=1)
        mha_p = sum(p.numel() for p in mha.parameters())
        mqa_p = sum(p.numel() for p in mqa.parameters())
        assert mqa_p < mha_p

    def test_qkv_proj_size_formula(self):
        """qkv_proj output = (n_head + 2*n_kv_head) * d_head."""
        n_head, n_kv_head, d_model = 4, 2, 64
        attn   = _attn(n_head, n_kv_head, d_model)
        d_head = d_model // n_head
        expected_out = (n_head + 2 * n_kv_head) * d_head
        assert attn.qkv_proj.out_features == expected_out

    def test_mha_qkv_proj_size_unchanged(self):
        """MHA (n_kv_head == n_head) keeps the original 3*d_model qkv size."""
        n_head, d_model = 4, 64
        attn = _attn(n_head, n_kv_head=n_head, d_model=d_model)
        assert attn.qkv_proj.out_features == 3 * d_model


# ---------------------------------------------------------------------------
# Forward — output shapes
# ---------------------------------------------------------------------------


class TestForwardShape:
    @pytest.mark.parametrize("n_kv_head", [1, 2, 4])
    def test_output_shape_matches_input(self, n_kv_head):
        attn = _attn(n_head=4, n_kv_head=n_kv_head)
        x    = torch.randn(2, 16, 64)
        out  = attn(x)
        assert out.shape == x.shape

    def test_mha_default_shape(self):
        attn = _attn(n_head=4, n_kv_head=0)   # 0 → n_head
        x    = torch.randn(1, 8, 64)
        assert attn(x).shape == x.shape


# ---------------------------------------------------------------------------
# GQA matches MHA when n_kv_head == n_head
# ---------------------------------------------------------------------------


class TestMHACompatibility:
    def test_gqa_equals_mha_when_groups_equal_1(self):
        """With n_groups=1 (n_kv_head==n_head), GQA must match MHA output."""
        torch.manual_seed(7)
        x   = torch.randn(1, 8, 64)
        # Same weights → same output
        attn_mha = _attn(n_head=4, n_kv_head=4)
        attn_gqa = _attn(n_head=4, n_kv_head=4)
        # Copy weights so they're identical
        attn_gqa.load_state_dict(attn_mha.state_dict())
        with torch.no_grad():
            out_mha = attn_mha(x)
            out_gqa = attn_gqa(x)
        assert torch.allclose(out_mha, out_gqa, atol=1e-6)

    def test_flash_manual_agree_under_gqa(self):
        """Flash and manual attention must agree numerically for GQA."""
        torch.manual_seed(3)
        attn = _attn(n_head=4, n_kv_head=2)
        x    = torch.randn(1, 8, 64)
        with torch.no_grad():
            out_flash  = attn(x, use_flash=True)
            out_manual = attn(x, use_flash=False)
        assert torch.allclose(out_flash, out_manual, atol=1e-5)


# ---------------------------------------------------------------------------
# KV cache with GQA
# ---------------------------------------------------------------------------


class TestKVCacheGQA:
    def test_cache_shape_uses_n_kv_head(self):
        model = _gpt(n_head=4, n_kv_head=2)
        cache = KVCache.for_model(model, max_seq_len=64, device=torch.device("cpu"))
        d_head = model.cfg.d_model // model.cfg.n_head
        assert cache.k[0].shape == (1, 2, 64, d_head)

    def test_mha_cache_uses_n_head(self):
        model = _gpt(n_head=4, n_kv_head=4)
        cache = KVCache.for_model(model, max_seq_len=64, device=torch.device("cpu"))
        d_head = model.cfg.d_model // model.cfg.n_head
        assert cache.k[0].shape == (1, 4, 64, d_head)

    def test_gqa_cache_smaller_than_mha(self):
        """GQA cache uses n_kv_head/n_head fraction of MHA cache memory."""
        model_mha = _gpt(n_head=4, n_kv_head=4)
        model_gqa = _gpt(n_head=4, n_kv_head=2)
        cache_mha = KVCache.for_model(model_mha, 64, torch.device("cpu"))
        cache_gqa = KVCache.for_model(model_gqa, 64, torch.device("cpu"))
        mha_bytes = cache_mha.k[0].numel() * cache_mha.k[0].element_size()
        gqa_bytes = cache_gqa.k[0].numel() * cache_gqa.k[0].element_size()
        assert gqa_bytes == mha_bytes // 2   # 2 KV heads vs 4

    def test_mqa_cache_smallest(self):
        model = _gpt(n_head=4, n_kv_head=1)
        cache = KVCache.for_model(model, 64, torch.device("cpu"))
        d_head = model.cfg.d_model // model.cfg.n_head
        assert cache.k[0].shape == (1, 1, 64, d_head)


# ---------------------------------------------------------------------------
# forward_cached — GQA identity with cached generation
# ---------------------------------------------------------------------------


class TestGenerateCachedGQA:
    def _check(self, n_head: int, n_kv_head: int, prompt_len: int,
               n_new: int, ctx: int = 64):
        model  = _gpt(n_head=n_head, n_kv_head=n_kv_head, ctx=ctx)
        prompt = _prompt(prompt_len)
        with torch.no_grad():
            out_naive  = naive_sample(model, prompt, n_new=n_new, temperature=0.0)
            out_cached = model.generate_cached(prompt, n_new=n_new, temperature=0.0)
        new_naive  = out_naive[ 0, prompt_len:].tolist()
        new_cached = out_cached[0, prompt_len:].tolist()
        assert new_naive == new_cached, (
            f"GQA identity failed n_head={n_head} n_kv={n_kv_head}\n"
            f"  naive:  {new_naive}\n"
            f"  cached: {new_cached}"
        )

    def test_gqa_2_kv_heads(self):
        self._check(n_head=4, n_kv_head=2, prompt_len=8, n_new=16)

    def test_mqa_1_kv_head(self):
        self._check(n_head=4, n_kv_head=1, prompt_len=8, n_new=16)

    def test_mha_still_works(self):
        self._check(n_head=4, n_kv_head=4, prompt_len=8, n_new=16)

    def test_single_token_prompt(self):
        self._check(n_head=4, n_kv_head=2, prompt_len=1, n_new=12)

    def test_medium_prompt(self):
        self._check(n_head=4, n_kv_head=2, prompt_len=20, n_new=24, ctx=64)


# ---------------------------------------------------------------------------
# forward_cached — cache is written to n_kv_head slots
# ---------------------------------------------------------------------------


class TestCacheWriteGQA:
    def test_prefill_writes_n_kv_head_slots(self):
        model = _gpt(n_head=4, n_kv_head=2)
        cache = KVCache.for_model(model, 64, torch.device("cpu"))
        prompt = _prompt(8)
        with torch.no_grad():
            model.forward_cached(prompt, start_pos=0, cache=cache)
        # First 8 positions of layer-0 K cache must be filled
        assert cache.k[0][:, :, :8, :].abs().sum() > 0
        # Positions 8+ still zero
        assert cache.k[0][:, :, 8:, :].abs().sum() == 0.0
        # Cache has 2 KV heads, not 4
        assert cache.k[0].shape[1] == 2

    def test_decode_extends_cache(self):
        model = _gpt(n_head=4, n_kv_head=2)
        cache = KVCache.for_model(model, 64, torch.device("cpu"))
        prompt = _prompt(4)
        with torch.no_grad():
            model.forward_cached(prompt, start_pos=0, cache=cache)
            next_tok = torch.tensor([[42]], dtype=torch.long)
            model.forward_cached(next_tok, start_pos=4, cache=cache)
        # Position 4 must now be filled
        assert cache.k[0][:, :, 4, :].abs().sum() > 0
