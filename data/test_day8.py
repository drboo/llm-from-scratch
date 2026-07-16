"""
Day 8 checkpoint tests — memmap dataloader.

Run:  pytest data/test_day8.py -v

The core tests use synthetic .bin files and run instantly.
The integration tests (decode real text) need train.bin to exist —
run `python data/prepare_toy.py` first; they skip otherwise.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataloader import get_batch

DATA_DIR = Path(__file__).parent / "toy"   # written by prepare_toy.py

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_bin(directory: Path, split: str, n_tokens: int = 10_000) -> np.ndarray:
    """Write an arange uint16 .bin file; returns the underlying array."""
    arr = np.arange(n_tokens, dtype=np.uint16)
    path = directory / f"{split}.bin"
    arr.tofile(str(path))
    return arr


# ---------------------------------------------------------------------------
# Core tests (synthetic data — always run)
# ---------------------------------------------------------------------------


def test_get_batch_shapes(tmp_path):
    _write_fake_bin(tmp_path, "train")
    x, y = get_batch("train", tmp_path, ctx=32, batch_size=4)
    assert x.shape == (4, 32)
    assert y.shape == (4, 32)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_y_is_x_shifted_by_one(tmp_path):
    """With arange data, y[i,j] == x[i,j] + 1 for all i,j."""
    _write_fake_bin(tmp_path, "train", n_tokens=50_000)
    g = torch.Generator().manual_seed(42)
    x, y = get_batch("train", tmp_path, ctx=128, batch_size=8, generator=g)
    assert torch.all(y == x + 1), "y must equal x shifted left by exactly one token"


def test_batch_within_data_bounds(tmp_path):
    n_tokens = 5_000
    _write_fake_bin(tmp_path, "train", n_tokens=n_tokens)
    x, y = get_batch("train", tmp_path, ctx=64, batch_size=16)
    assert x.max().item() <  n_tokens
    assert y.max().item() <  n_tokens
    assert x.min().item() >= 0


def test_val_split_loadable(tmp_path):
    _write_fake_bin(tmp_path, "train")
    _write_fake_bin(tmp_path, "val", n_tokens=2_000)
    x, y = get_batch("val", tmp_path, ctx=32, batch_size=2)
    assert x.shape == (2, 32)


def test_reproducible_with_generator(tmp_path):
    _write_fake_bin(tmp_path, "train")
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    x1, y1 = get_batch("train", tmp_path, ctx=32, batch_size=4, generator=g1)
    x2, y2 = get_batch("train", tmp_path, ctx=32, batch_size=4, generator=g2)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_different_seeds_give_different_batches(tmp_path):
    _write_fake_bin(tmp_path, "train", n_tokens=50_000)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    x1, _ = get_batch("train", tmp_path, ctx=64, batch_size=4, generator=g1)
    x2, _ = get_batch("train", tmp_path, ctx=64, batch_size=4, generator=g2)
    assert not torch.equal(x1, x2)


# ---------------------------------------------------------------------------
# Integration tests — require prepare_toy.py to have been run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_train_bin():
    path = DATA_DIR / "train.bin"
    if not path.exists():
        pytest.skip(
            "data/toy/train.bin not found — run `python data/prepare_toy.py` first"
        )
    return path


def test_real_batch_shapes(real_train_bin):
    x, y = get_batch("train", DATA_DIR, ctx=256, batch_size=4)
    assert x.shape == (4, 256)
    assert y.shape == (4, 256)


def test_real_y_is_x_shifted(real_train_bin):
    g = torch.Generator().manual_seed(0)
    x, y = get_batch("train", DATA_DIR, ctx=256, batch_size=4, generator=g)
    # Load raw data to verify the shift directly
    data = np.memmap(str(real_train_bin), dtype=np.uint16, mode="r")
    # Re-sample with same seed
    g2 = torch.Generator().manual_seed(0)
    ix = torch.randint(len(data) - 256, (4,), generator=g2)
    for b, i in enumerate(ix):
        expected_y = torch.from_numpy(data[i + 1: i + 257].astype(np.int64))
        assert torch.equal(y[b], expected_y), f"y mismatch at batch index {b}"


def test_decode_batch_reads_as_real_text(real_train_bin):
    """Decoded output should be non-empty natural text, not garbage."""
    from tokeniser.tokenizer import Codec
    tok_path = Path(__file__).parent.parent / "tokeniser" / "tokenizer.json"
    if not tok_path.exists():
        pytest.skip("tokenizer.json not found")

    codec = Codec(tok_path)
    g = torch.Generator().manual_seed(0)
    x, _ = get_batch("train", DATA_DIR, ctx=256, batch_size=1, generator=g)

    text = codec.decode(x[0].tolist())
    assert len(text) > 50, "Decoded text is suspiciously short"
    # Should contain at least some ASCII letters/spaces
    printable = sum(c.isalpha() or c.isspace() for c in text)
    assert printable / len(text) > 0.4, "Decoded text looks like binary garbage"
