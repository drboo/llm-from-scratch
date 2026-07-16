"""
Day 11 tests — sampling / generation.

Run:  pytest inference/test_day11.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import GPT, ModelConfig
from inference.sample import sample, top_k_filter, top_p_filter

TINY = ModelConfig(vocab_size=256, d_model=64, n_head=4, n_layer=2, ctx=32)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    m = GPT(TINY)
    m.eval()
    return m


def _prompt(n: int = 4, vocab: int = 256) -> torch.Tensor:
    return torch.randint(0, vocab, (1, n))


# ---------------------------------------------------------------------------
# top_k_filter
# ---------------------------------------------------------------------------


def test_top_k_filter_keeps_k_tokens():
    logits = torch.randn(1, 256)
    out = top_k_filter(logits.clone(), k=10)
    finite = (out != float("-inf")).sum().item()
    assert finite == 10


def test_top_k_filter_zero_disabled():
    logits = torch.randn(1, 256)
    out = top_k_filter(logits.clone(), k=0)
    assert torch.equal(out, logits)


# ---------------------------------------------------------------------------
# top_p_filter
# ---------------------------------------------------------------------------


def test_top_p_filter_one_disabled():
    logits = torch.randn(1, 256)
    out = top_p_filter(logits.clone(), p=1.0)
    assert torch.equal(out, logits)


def test_top_p_filter_very_small_p_keeps_at_least_one():
    logits = torch.randn(1, 256)
    out = top_p_filter(logits, p=0.0)
    finite = (out != float("-inf")).sum().item()
    assert finite >= 1


def test_top_p_filter_cumulative_prob():
    """After filtering, surviving token probs should sum to ≥ p (for p=0.5)."""
    torch.manual_seed(1)
    logits = torch.randn(1, 256)
    p = 0.5
    out = top_p_filter(logits.clone(), p=p)
    surviving_prob = F.softmax(out, dim=-1).sum().item()
    assert surviving_prob >= p - 1e-4


# ---------------------------------------------------------------------------
# sample() output shape and type
# ---------------------------------------------------------------------------


def test_sample_output_shape(tiny_model):
    prompt = _prompt(4)
    out = sample(tiny_model, prompt, n_new=10)
    assert out.shape == (1, 14)


def test_sample_preserves_prompt(tiny_model):
    prompt = _prompt(4)
    out = sample(tiny_model, prompt, n_new=10)
    assert torch.equal(out[:, :4], prompt)


def test_sample_greedy_deterministic(tiny_model):
    prompt = _prompt(4)
    out1 = sample(tiny_model, prompt, n_new=20, temperature=0.0)
    out2 = sample(tiny_model, prompt, n_new=20, temperature=0.0)
    assert torch.equal(out1, out2)


def test_sample_stochastic_varies(tiny_model):
    """With temperature=1.0 and different seeds, outputs should differ."""
    prompt = _prompt(4)
    torch.manual_seed(0)
    out1 = sample(tiny_model, prompt, n_new=20, temperature=1.0)
    torch.manual_seed(99)
    out2 = sample(tiny_model, prompt, n_new=20, temperature=1.0)
    assert not torch.equal(out1, out2)


def test_sample_ids_in_vocab_range(tiny_model):
    prompt = _prompt(4)
    out = sample(tiny_model, prompt, n_new=30, temperature=1.0)
    assert out.min().item() >= 0
    assert out.max().item() < TINY.vocab_size


# ---------------------------------------------------------------------------
# Temperature effect on diversity
# ---------------------------------------------------------------------------


def test_high_temp_higher_entropy(tiny_model):
    """Entropy of the distribution at high temp should exceed low temp."""
    prompt = _prompt(4)
    torch.manual_seed(0)
    idx_cond = prompt[:, -TINY.ctx:]
    logits, _ = tiny_model(idx_cond)
    logits = logits[:, -1, :]

    def entropy(t):
        p = F.softmax(logits / t, dim=-1)
        return -(p * p.log()).sum().item()

    assert entropy(2.0) > entropy(0.1)


# ---------------------------------------------------------------------------
# EOS early stopping
# ---------------------------------------------------------------------------


def test_eos_stops_early(tiny_model):
    """If the model ever samples eos_id, generation stops."""
    # Force eos to be the top token by patching logits temporarily
    eos_id = 0
    prompt = torch.tensor([[1, 2, 3, 4]])
    # Use greedy at temperature=0 with eos_id=top token of the model
    # Get what greedy would pick first and set that as eos
    with torch.no_grad():
        logits, _ = tiny_model(prompt[:, -TINY.ctx:])
        first_greedy = logits[0, -1, :].argmax().item()

    out = sample(tiny_model, prompt, n_new=50, temperature=0.0, eos_id=first_greedy)
    # Should have stopped after 1 new token (the eos itself)
    assert out.shape[1] == prompt.shape[1] + 1
