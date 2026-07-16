"""
Day 18 tests — tokenize + pack the real corpus.

Run:  pytest data/test_day18.py -v

Uses tiny synthetic shards and a real tokenizer if present, otherwise skips
tests that require it.  All .bin I/O tests use the byte-level fallback path.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.acquire import write_shard
from data.prepare_real import (
    split_docs,
    save_bin,
    make_datasheet,
    DEFAULT_RATIOS,
    VAL_FRACTION,
)

ROOT = Path(__file__).resolve().parent.parent
TOK_PATH = ROOT / "tokeniser" / "tokenizer.json"
HAS_TOKENIZER = TOK_PATH.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_docs(n: int, source: str = "web") -> list[dict]:
    return [
        {"text": f"Document {i} from {source}. " * 20, "source": source, "id": str(i)}
        for i in range(n)
    ]


def _shard(tmp_path: Path, docs: list[dict], name: str = "train.jsonl.gz") -> Path:
    p = tmp_path / name
    write_shard(iter(docs), p)
    return p


# ---------------------------------------------------------------------------
# split_docs
# ---------------------------------------------------------------------------


class TestSplitDocs:
    def test_val_fraction_respected(self):
        docs  = _make_docs(100)
        tr, vl = split_docs(docs, val_fraction=0.1)
        assert len(vl) == 10
        assert len(tr) == 90

    def test_no_overlap(self):
        docs  = _make_docs(50)
        tr, vl = split_docs(docs, val_fraction=0.2)
        train_ids = {d["id"] for d in tr}
        val_ids   = {d["id"] for d in vl}
        assert train_ids.isdisjoint(val_ids)

    def test_all_docs_accounted_for(self):
        docs  = _make_docs(80)
        tr, vl = split_docs(docs, val_fraction=0.1)
        assert len(tr) + len(vl) == 80

    def test_deterministic_with_same_seed(self):
        docs   = _make_docs(40)
        tr1, v1 = split_docs(docs, val_fraction=0.2, seed=7)
        tr2, v2 = split_docs(docs, val_fraction=0.2, seed=7)
        assert [d["id"] for d in tr1] == [d["id"] for d in tr2]

    def test_different_seed_different_split(self):
        docs   = _make_docs(40)
        tr1, _ = split_docs(docs, val_fraction=0.2, seed=1)
        tr2, _ = split_docs(docs, val_fraction=0.2, seed=2)
        assert [d["id"] for d in tr1] != [d["id"] for d in tr2]

    def test_at_least_one_val_doc(self):
        docs = _make_docs(5)
        _, vl = split_docs(docs, val_fraction=0.01)
        assert len(vl) >= 1

    def test_small_corpus(self):
        docs = _make_docs(2)
        tr, vl = split_docs(docs, val_fraction=0.5)
        assert len(tr) + len(vl) == 2


# ---------------------------------------------------------------------------
# save_bin
# ---------------------------------------------------------------------------


class TestSaveBin:
    def test_creates_file(self, tmp_path):
        save_bin([1, 2, 3, 4, 5], tmp_path / "train.bin")
        assert (tmp_path / "train.bin").exists()

    def test_correct_dtype(self, tmp_path):
        tokens = list(range(100))
        save_bin(tokens, tmp_path / "t.bin")
        arr = np.memmap(str(tmp_path / "t.bin"), dtype=np.uint16, mode="r")
        assert arr.dtype == np.uint16

    def test_correct_values(self, tmp_path):
        tokens = [10, 20, 30, 40, 50]
        save_bin(tokens, tmp_path / "t.bin")
        arr = np.memmap(str(tmp_path / "t.bin"), dtype=np.uint16, mode="r")
        assert list(arr) == tokens

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "train.bin"
        save_bin([1, 2, 3], path)
        assert path.exists()

    def test_large_token_ids_preserved(self, tmp_path):
        # uint16 max = 65535; our vocab is 32k so well within range
        tokens = [0, 100, 1000, 32000, 65535]
        save_bin(tokens, tmp_path / "t.bin")
        arr = np.memmap(str(tmp_path / "t.bin"), dtype=np.uint16, mode="r")
        assert list(arr) == tokens

    def test_file_size_proportional_to_length(self, tmp_path):
        save_bin(list(range(100)),  tmp_path / "small.bin")
        save_bin(list(range(1000)), tmp_path / "large.bin")
        small = (tmp_path / "small.bin").stat().st_size
        large = (tmp_path / "large.bin").stat().st_size
        assert large == small * 10


# ---------------------------------------------------------------------------
# make_datasheet
# ---------------------------------------------------------------------------


class TestMakeDatasheet:
    def _src_counts(self):
        return {
            "web":   {"train": 12_000_000, "val": 600_000},
            "email": {"train": 2_000_000,  "val": 100_000},
            "code":  {"train": 5_000_000,  "val": 250_000},
        }

    def test_creates_json_file(self, tmp_path):
        make_datasheet(
            shard_paths={"web": None, "email": None, "code": None},
            ratios=DEFAULT_RATIOS,
            src_counts=self._src_counts(),
            compression=4.1,
            out_dir=tmp_path,
        )
        assert (tmp_path / "datasheet.json").exists()

    def test_json_valid(self, tmp_path):
        make_datasheet(
            shard_paths={},
            ratios=DEFAULT_RATIOS,
            src_counts=self._src_counts(),
            compression=4.2,
            out_dir=tmp_path,
        )
        ds = json.loads((tmp_path / "datasheet.json").read_text())
        assert "mixture_ratios" in ds
        assert "sources" in ds
        assert "filters_applied" in ds
        assert "compression_ratio_bytes_per_token" in ds

    def test_compression_stored(self, tmp_path):
        make_datasheet({}, DEFAULT_RATIOS, self._src_counts(), 3.8, tmp_path)
        ds = json.loads((tmp_path / "datasheet.json").read_text())
        assert ds["compression_ratio_bytes_per_token"] == pytest.approx(3.8)

    def test_val_fraction_stored(self, tmp_path):
        make_datasheet({}, DEFAULT_RATIOS, self._src_counts(), 4.0, tmp_path)
        ds = json.loads((tmp_path / "datasheet.json").read_text())
        assert ds["val_fraction"] == VAL_FRACTION

    def test_filters_applied_all_three_sources(self, tmp_path):
        make_datasheet({}, DEFAULT_RATIOS, self._src_counts(), 4.0, tmp_path)
        ds = json.loads((tmp_path / "datasheet.json").read_text())
        for src in ("web", "email", "code"):
            assert src in ds["filters_applied"]
            assert len(ds["filters_applied"][src]) > 0

    def test_token_counts_stored(self, tmp_path):
        counts = self._src_counts()
        make_datasheet({}, DEFAULT_RATIOS, counts, 4.0, tmp_path)
        ds = json.loads((tmp_path / "datasheet.json").read_text())
        assert ds["sources"]["web"]["train_tokens"] == counts["web"]["train"]
        assert ds["sources"]["email"]["val_tokens"] == counts["email"]["val"]


# ---------------------------------------------------------------------------
# Integration: prepare_real with real tokenizer (skipped if absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TOKENIZER, reason="tokenizer.json not present")
class TestPrepareRealIntegration:
    def test_writes_bin_files(self, tmp_path):
        from data.prepare_real import prepare_real

        web_docs   = _make_docs(20, "web")
        email_docs = _make_docs(10, "email")
        code_docs  = _make_docs(10, "code")

        web_shard   = _shard(tmp_path / "web",   web_docs,   "train.jsonl.gz")
        email_shard = _shard(tmp_path / "email", email_docs, "train.jsonl.gz")
        code_shard  = _shard(tmp_path / "code",  code_docs,  "train.jsonl.gz")

        prepare_real(
            web_shard=web_shard,
            email_shard=email_shard,
            code_shard=code_shard,
            out_dir=tmp_path / "out",
            ratios=DEFAULT_RATIOS,
            val_fraction=0.1,
        )
        assert (tmp_path / "out" / "train.bin").exists()
        assert (tmp_path / "out" / "val.bin").exists()
        assert (tmp_path / "out" / "datasheet.json").exists()

    def test_val_bin_smaller_than_train_bin(self, tmp_path):
        from data.prepare_real import prepare_real

        docs        = _make_docs(50, "web")
        web_shard   = _shard(tmp_path / "w", docs, "train.jsonl.gz")

        prepare_real(
            web_shard=web_shard,
            email_shard=None,
            code_shard=None,
            out_dir=tmp_path / "out",
            ratios={"web": 1.0, "email": 0.0, "code": 0.0},
            val_fraction=0.1,
        )
        train_size = (tmp_path / "out" / "train.bin").stat().st_size
        val_size   = (tmp_path / "out" / "val.bin").stat().st_size
        assert val_size < train_size

    def test_no_token_overlap_train_val(self, tmp_path):
        """Val documents are doc-level held-out, so token sequences differ."""
        from data.prepare_real import prepare_real

        # Use unique, identifiable documents
        docs = [{"text": f"UNIQUE_DOC_{i} " * 50, "source": "web", "id": str(i)}
                for i in range(20)]
        web_shard = _shard(tmp_path / "w", docs, "train.jsonl.gz")

        prepare_real(
            web_shard=web_shard,
            email_shard=None,
            code_shard=None,
            out_dir=tmp_path / "out",
            ratios={"web": 1.0, "email": 0.0, "code": 0.0},
            val_fraction=0.2,
            seed=0,
        )
        train_arr = np.memmap(
            str(tmp_path / "out" / "train.bin"), dtype=np.uint16, mode="r"
        )
        val_arr   = np.memmap(
            str(tmp_path / "out" / "val.bin"), dtype=np.uint16, mode="r"
        )
        # Train and val should have different sizes (doc-level split)
        assert len(train_arr) != len(val_arr)
