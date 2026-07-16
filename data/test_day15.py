"""
Day 15 tests — corpus acquisition helpers.

Run:  pytest data/test_day15.py -v

Tests cover the shard writer and reader; they use synthetic in-memory data
so no internet connection or HuggingFace token is needed.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.acquire import write_shard, iter_shard, list_shards, report_mixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_docs(n: int, source: str = "web") -> list[dict]:
    return [
        {"text": f"Hello world document number {i}. " * 5, "source": source, "id": str(i)}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# write_shard
# ---------------------------------------------------------------------------


def test_write_shard_creates_file(tmp_path):
    out = tmp_path / "web" / "train.jsonl.gz"
    write_shard(iter(_make_docs(10)), out)
    assert out.exists()


def test_write_shard_gzipped_jsonl(tmp_path):
    docs = _make_docs(5, "web")
    out  = tmp_path / "train.jsonl.gz"
    write_shard(iter(docs), out)

    with gzip.open(out, "rt") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 5
    parsed = json.loads(lines[0])
    assert "text" in parsed and "source" in parsed


def test_write_shard_returns_stats(tmp_path):
    docs = _make_docs(20, "email")
    out  = tmp_path / "train.jsonl.gz"
    stats = write_shard(iter(docs), out)
    assert stats["n_docs"] == 20
    assert stats["n_bytes"] > 0
    assert stats["est_tokens"] == stats["n_bytes"] // 4


def test_write_shard_empty(tmp_path):
    out = tmp_path / "empty.jsonl.gz"
    stats = write_shard(iter([]), out)
    assert stats["n_docs"] == 0
    assert stats["n_bytes"] == 0
    assert out.exists()


def test_write_shard_creates_parent_dirs(tmp_path):
    out = tmp_path / "deep" / "nested" / "dir" / "train.jsonl.gz"
    write_shard(iter(_make_docs(3)), out)
    assert out.exists()


# ---------------------------------------------------------------------------
# iter_shard
# ---------------------------------------------------------------------------


def test_iter_shard_roundtrip(tmp_path):
    docs = _make_docs(15, "code")
    out  = tmp_path / "train.jsonl.gz"
    write_shard(iter(docs), out)

    recovered = list(iter_shard(out))
    assert len(recovered) == 15
    assert recovered[0]["source"] == "code"
    assert recovered[7]["text"] == docs[7]["text"]


def test_iter_shard_preserves_unicode(tmp_path):
    docs = [{"text": "Héllo wörld — 日本語テスト 🐍", "source": "web", "id": "0"}]
    out  = tmp_path / "uni.jsonl.gz"
    write_shard(iter(docs), out)
    recovered = list(iter_shard(out))
    assert recovered[0]["text"] == docs[0]["text"]


# ---------------------------------------------------------------------------
# list_shards
# ---------------------------------------------------------------------------


def test_list_shards_finds_all(tmp_path):
    for name in ["web", "email", "code"]:
        (tmp_path / name).mkdir()
        write_shard(iter(_make_docs(2, name)), tmp_path / name / "train.jsonl.gz")

    shards = list_shards(tmp_path)
    assert len(shards) == 3
    assert all(p.suffix == ".gz" for p in shards)


def test_list_shards_empty_dir(tmp_path):
    assert list_shards(tmp_path) == []


def test_list_shards_sorted(tmp_path):
    names = ["code", "email", "web"]
    for name in names:
        (tmp_path / name).mkdir()
        write_shard(iter(_make_docs(1, name)), tmp_path / name / "train.jsonl.gz")

    shards = list_shards(tmp_path)
    paths  = [p.parent.name for p in shards]
    assert paths == sorted(paths)


# ---------------------------------------------------------------------------
# report_mixture (smoke test — just checks it doesn't crash)
# ---------------------------------------------------------------------------


def test_report_mixture_no_crash(capsys):
    stats = {
        "web":   {"n_docs": 10_000, "n_bytes": 50_000_000, "est_tokens": 12_500_000},
        "email": {"n_docs": 2_000,  "n_bytes": 5_000_000,  "est_tokens": 1_250_000},
        "code":  {"n_docs": 5_000,  "n_bytes": 20_000_000, "est_tokens": 5_000_000},
    }
    report_mixture(stats)
    out = capsys.readouterr().out
    assert "web" in out
    assert "email" in out
    assert "code" in out
    assert "TOTAL" in out
    assert "Chinchilla" in out


def test_report_mixture_zero_total(capsys):
    stats = {s: {"n_docs": 0, "n_bytes": 0, "est_tokens": 0}
             for s in ("web", "email", "code")}
    report_mixture(stats)
    out = capsys.readouterr().out
    assert "No data" in out


def test_write_shard_est_tokens_proportional_to_bytes(tmp_path):
    """Longer docs → more bytes → more estimated tokens."""
    short_docs = [{"text": "hi", "source": "web", "id": str(i)} for i in range(10)]
    long_docs  = [{"text": "x " * 500, "source": "web", "id": str(i)} for i in range(10)]

    s1 = write_shard(iter(short_docs), tmp_path / "short.jsonl.gz")
    s2 = write_shard(iter(long_docs),  tmp_path / "long.jsonl.gz")

    assert s2["est_tokens"] > s1["est_tokens"]
