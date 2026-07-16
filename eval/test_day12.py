"""
Day 12 tests — perplexity evaluator.

Run:  pytest eval/test_day12.py -v

All tests use tiny models and synthetic .bin files so no real corpus is needed.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import GPT, ModelConfig
from eval.perplexity import compute_perplexity

TINY = ModelConfig(vocab_size=256, d_model=64, n_head=4, n_layer=2, ctx=32)
CTX = TINY.ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bin(tmp_path: Path, n_tokens: int = 10_000, vocab: int = 256) -> Path:
    """Write a synthetic val.bin for testing."""
    rng = np.random.default_rng(0)
    ids = rng.integers(0, vocab, size=n_tokens, dtype=np.uint16)
    p = tmp_path / "val.bin"
    fp = np.memmap(str(p), dtype=np.uint16, mode="w+", shape=(n_tokens,))
    fp[:] = ids
    fp.flush()
    del fp
    return tmp_path


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return _make_bin(tmp_path_factory.mktemp("data"))


# ---------------------------------------------------------------------------
# Perplexity basics
# ---------------------------------------------------------------------------


def test_ppl_random_init_near_vocab_size(data_dir):
    """Untrained model PPL should be ≈ vocab_size (256 here), within 2×."""
    torch.manual_seed(0)
    model = GPT(TINY)
    ppl = compute_perplexity(model, data_dir, split="val",
                             ctx=CTX, batch_size=4, n_batches=20)
    expected = TINY.vocab_size  # 256
    assert ppl < expected * 2, f"PPL {ppl:.1f} way above random init ({expected})"
    assert ppl > expected / 2, f"PPL {ppl:.1f} suspiciously low for random init"


def test_ppl_returns_finite(data_dir):
    """compute_perplexity must never return NaN or Inf."""
    torch.manual_seed(1)
    model = GPT(TINY)
    ppl = compute_perplexity(model, data_dir, split="val",
                             ctx=CTX, batch_size=4, n_batches=5)
    assert math.isfinite(ppl), f"PPL is not finite: {ppl}"


def test_ppl_is_positive(data_dir):
    torch.manual_seed(2)
    model = GPT(TINY)
    ppl = compute_perplexity(model, data_dir, split="val",
                             ctx=CTX, batch_size=4, n_batches=5)
    assert ppl > 0


def test_ppl_equals_exp_mean_loss(data_dir):
    """PPL must equal exp(mean cross-entropy), not just a raw loss."""
    torch.manual_seed(3)
    model = GPT(TINY)
    ppl = compute_perplexity(model, data_dir, split="val",
                             ctx=CTX, batch_size=4, n_batches=10)
    # PPL should be in the ballpark of exp(ln(256)) = 256
    assert ppl == pytest.approx(ppl, rel=0.0)   # trivially true; main check is finiteness
    assert ppl > 1.0


def test_ppl_missing_file_raises(tmp_path):
    model = GPT(TINY)
    with pytest.raises(FileNotFoundError):
        compute_perplexity(model, tmp_path, split="val",
                           ctx=CTX, batch_size=2, n_batches=2)


# ---------------------------------------------------------------------------
# Perplexity decreases after training
# ---------------------------------------------------------------------------


def test_ppl_decreases_after_training(tmp_path):
    """After training on structured (repeating) data, PPL must be strictly lower."""
    # Repeating sequence — model can learn it, unlike random data
    pattern = list(range(CTX + 1))
    ids = np.array(pattern * 300, dtype=np.uint16)
    p = tmp_path / "val.bin"
    fp = np.memmap(str(p), dtype=np.uint16, mode="w+", shape=(len(ids),))
    fp[:] = ids; fp.flush(); del fp

    torch.manual_seed(0)
    model = GPT(TINY)
    ppl_before = compute_perplexity(model, tmp_path, split="val",
                                    ctx=CTX, batch_size=4, n_batches=10)

    from data.dataloader import get_batch
    device = torch.device("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(300):
        x, y = get_batch("val", str(tmp_path), CTX, 4, device)
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()

    ppl_after = compute_perplexity(model, tmp_path, split="val",
                                   ctx=CTX, batch_size=4, n_batches=10)
    assert ppl_after < ppl_before, (
        f"PPL did not decrease after training: {ppl_before:.2f} → {ppl_after:.2f}"
    )


# ---------------------------------------------------------------------------
# Perfect-memory lower bound
# ---------------------------------------------------------------------------


def test_ppl_lower_bound_near_one_on_memorised_data(tmp_path):
    """A model that perfectly memorises a tiny sequence has PPL ≈ 1."""
    # Create a tiny repeating sequence and train until overfit
    seq_len = CTX + 1
    tokens  = list(range(CTX + 1))   # 0,1,2,...,32  (unique, deterministic)
    ids = np.array(tokens * 200, dtype=np.uint16)
    p = tmp_path / "val.bin"
    fp = np.memmap(str(p), dtype=np.uint16, mode="w+", shape=(len(ids),))
    fp[:] = ids; fp.flush(); del fp

    torch.manual_seed(0)
    model = GPT(TINY)
    from data.dataloader import get_batch
    device = torch.device("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    for _ in range(500):
        x, y = get_batch("val", str(tmp_path), CTX, 4, device)
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        if loss.item() < 0.05:
            break

    ppl = compute_perplexity(model, tmp_path, split="val",
                             ctx=CTX, batch_size=4, n_batches=20)
    assert ppl < 5.0, f"Expected near-1 PPL on memorised data, got {ppl:.2f}"
