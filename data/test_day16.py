"""
Day 16 tests — cleaning and deduplication.

Run:  pytest data/test_day16.py -v

All tests use synthetic in-memory data; no internet or fasttext model needed.
The language-ID filter is disabled (--no-langid path) in all tests.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.acquire import write_shard, iter_shard
from data.clean import (
    enforce_utf8,
    strip_html,
    looks_like_html,
    clean_doc,
    clean_shard,
)
from data.dedup import (
    normalise,
    doc_hash,
    exact_dedup,
    near_dedup,
    _shingles,
    dedup_shard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shard(tmp_path: Path, docs: list[dict], name: str = "train.jsonl.gz") -> Path:
    p = tmp_path / name
    write_shard(iter(docs), p)
    return p


def _web_doc(text: str, i: int = 0) -> dict:
    return {"text": text, "source": "web", "id": str(i)}


def _email_doc(text: str, i: int = 0) -> dict:
    return {"text": text, "source": "email", "id": str(i)}


def _code_doc(text: str, i: int = 0) -> dict:
    return {"text": text, "source": "code", "id": str(i)}


LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 5  # > 100 chars


# ===========================================================================
# clean.py tests
# ===========================================================================


class TestEnforceUtf8:
    def test_clean_string_unchanged(self):
        t = "Hello world!"
        assert enforce_utf8(t) == t

    def test_replacement_char_survives(self):
        # bytes with invalid UTF-8 sequences become replacement chars
        bad = "hello \x80 world"
        result = enforce_utf8(bad)
        assert isinstance(result, str)
        assert "hello" in result


class TestLooksLikeHtml:
    def test_detects_html_tags(self):
        assert looks_like_html("<p>Hello</p>")
        assert looks_like_html("<div class='x'>text</div>")

    def test_plain_text_false(self):
        assert not looks_like_html("Just plain text here.")

    def test_angle_bracket_in_code_detected(self):
        # conservative: anything with a tag-like pattern is flagged
        assert looks_like_html("<br>")


class TestStripHtml:
    def test_removes_tags(self):
        result = strip_html("<p>Hello <b>world</b></p>")
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_plain_text_unchanged(self):
        t = "Just plain text."
        assert strip_html(t) == t

    def test_entities_decoded(self):
        result = strip_html("&lt;h1&gt;Title&lt;/h1&gt;")
        assert "Title" in result


class TestCleanDoc:
    def test_short_doc_dropped(self):
        doc = _web_doc("hi", 0)
        assert clean_doc(doc, min_chars=100, use_langid=False) is None

    def test_long_doc_kept(self):
        doc = _web_doc(LONG_TEXT, 0)
        result = clean_doc(doc, min_chars=100, use_langid=False)
        assert result is not None

    def test_html_stripped_from_web(self):
        html = "<p>" + "word " * 30 + "</p>"
        doc  = _web_doc(html, 0)
        result = clean_doc(doc, min_chars=50, use_langid=False)
        assert result is not None
        assert "<p>" not in result["text"]

    def test_html_stripped_from_email(self):
        html = "<html><body>" + "word " * 30 + "</body></html>"
        doc  = _email_doc(html, 0)
        result = clean_doc(doc, min_chars=50, use_langid=False)
        assert result is not None
        assert "<html>" not in result["text"]

    def test_code_html_not_stripped(self):
        """Code docs must NOT have HTML stripped — they may contain angle brackets."""
        code = "def foo():\n    return '<bar>'\n" * 10
        doc  = _code_doc(code, 0)
        result = clean_doc(doc, min_chars=50, use_langid=False)
        assert result is not None
        assert "<bar>" in result["text"]

    def test_excess_newlines_collapsed(self):
        text = "Hello\n\n\n\n\nworld. " * 5
        doc  = _web_doc(text, 0)
        result = clean_doc(doc, min_chars=10, use_langid=False)
        assert "\n\n\n" not in result["text"]

    def test_langid_disabled_keeps_all(self):
        # non-English-looking text should survive when langid is off
        doc = _web_doc("Bonjour le monde! " * 10, 0)
        assert clean_doc(doc, min_chars=10, use_langid=False) is not None

    def test_source_preserved(self):
        doc    = _web_doc(LONG_TEXT, 7)
        result = clean_doc(doc, use_langid=False)
        assert result["source"] == "web"
        assert result["id"] == "7"


class TestCleanShard:
    def test_drops_short_docs(self, tmp_path):
        docs = [
            _web_doc("short", 0),
            _web_doc(LONG_TEXT, 1),
            _web_doc("x", 2),
            _web_doc(LONG_TEXT, 3),
        ]
        in_p  = _shard(tmp_path, docs, "web/in.jsonl.gz")
        out_p = tmp_path / "web" / "out.jsonl.gz"
        stats = clean_shard(in_p, out_p, min_chars=100, use_langid=False)
        assert stats["in"] == 4
        assert stats["out"] == 2
        assert stats["dropped"] == 2

    def test_all_pass(self, tmp_path):
        docs = [_web_doc(LONG_TEXT, i) for i in range(5)]
        in_p  = _shard(tmp_path, docs, "w/in.jsonl.gz")
        out_p = tmp_path / "w" / "out.jsonl.gz"
        stats = clean_shard(in_p, out_p, min_chars=10, use_langid=False)
        assert stats["out"] == 5
        assert stats["dropped"] == 0


# ===========================================================================
# dedup.py tests
# ===========================================================================


class TestNormalise:
    def test_lowercases(self):
        assert normalise("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert normalise("a   b\t\nc") == "a b c"

    def test_strips(self):
        assert normalise("  hi  ") == "hi"


class TestDocHash:
    def test_same_text_same_hash(self):
        assert doc_hash("Hello World") == doc_hash("Hello World")

    def test_normalised_equals_original(self):
        assert doc_hash("HELLO") == doc_hash("hello")
        assert doc_hash("a  b") == doc_hash("a b")

    def test_different_text_different_hash(self):
        assert doc_hash("apple") != doc_hash("orange")


class TestShingles:
    def test_non_empty(self):
        assert len(_shingles("hello world")) > 0

    def test_short_text(self):
        shingles = _shingles("hi", k=5)
        assert len(shingles) >= 1

    def test_shingle_length(self):
        for s in _shingles("hello world", k=5):
            assert len(s) <= 5


class TestExactDedup:
    def test_removes_exact_duplicates(self):
        docs = [
            _web_doc("The cat sat on the mat. " * 5, i)
            for i in range(3)
        ]
        kept, stats = exact_dedup(docs)
        assert len(kept) == 1
        assert stats["dropped"] == 2

    def test_normalised_duplicates_removed(self):
        docs = [
            _web_doc("Hello World", 0),
            _web_doc("hello world", 1),   # same after normalise
            _web_doc("HELLO WORLD", 2),   # same after normalise
        ]
        kept, stats = exact_dedup(docs)
        assert len(kept) == 1

    def test_unique_docs_kept(self):
        docs = [_web_doc(f"Unique content number {i}. " * 5, i) for i in range(5)]
        kept, stats = exact_dedup(docs)
        assert len(kept) == 5
        assert stats["dropped"] == 0

    def test_empty_input(self):
        kept, stats = exact_dedup([])
        assert kept == []
        assert stats["in"] == 0

    def test_stats_drop_pct(self):
        docs = [_web_doc("same text " * 10, i) for i in range(4)]
        _, stats = exact_dedup(docs)
        assert stats["drop_pct"] == pytest.approx(75.0)


class TestNearDedup:
    def test_removes_near_duplicates(self):
        base = "The quick brown fox jumps over the lazy dog. " * 10
        docs = [
            _web_doc(base, 0),
            _web_doc(base + " Extra sentence.", 1),   # very similar
        ]
        kept, stats = near_dedup(docs, threshold=0.8)
        assert len(kept) == 1

    def test_keeps_dissimilar_docs(self):
        docs = [
            _web_doc("The quick brown fox. " * 10, 0),
            _web_doc("Machine learning transforms data into models. " * 10, 1),
        ]
        kept, stats = near_dedup(docs, threshold=0.8)
        assert len(kept) == 2

    def test_empty_input(self):
        kept, stats = near_dedup([], threshold=0.8)
        assert kept == []

    def test_single_doc_kept(self):
        kept, _ = near_dedup([_web_doc("hello " * 20, 0)], threshold=0.8)
        assert len(kept) == 1


class TestDedupShard:
    def test_exact_dups_removed(self, tmp_path):
        text = LONG_TEXT
        docs = [_web_doc(text, i) for i in range(4)]   # all identical
        in_p  = _shard(tmp_path, docs, "web/train.jsonl.gz")
        out_p = tmp_path / "web" / "out.jsonl.gz"
        stats = dedup_shard(in_p, out_p, use_minhash=False)
        assert stats["out"] == 1
        assert stats["dropped"] == 3

    def test_unique_docs_all_kept(self, tmp_path):
        docs = [_web_doc(f"Unique doc {i}. " * 20, i) for i in range(5)]
        in_p  = _shard(tmp_path, docs, "web/train.jsonl.gz")
        out_p = tmp_path / "web" / "out.jsonl.gz"
        stats = dedup_shard(in_p, out_p, use_minhash=False)
        assert stats["out"] == 5

    def test_output_file_created(self, tmp_path):
        docs = [_web_doc(LONG_TEXT, 0)]
        in_p  = _shard(tmp_path, docs, "code/train.jsonl.gz")
        out_p = tmp_path / "code" / "out.jsonl.gz"
        dedup_shard(in_p, out_p)
        assert out_p.exists()
