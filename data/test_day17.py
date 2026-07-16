"""
Day 17 tests — PII scrubbing and code filtering.

Run:  pytest data/test_day17.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.scrub import (
    scrub_regex,
    strip_signature,
    strip_quoted_replies,
    scrub_email,
    audit_pii,
    scrub_shard,
)
from data.filter_code import (
    shannon_entropy,
    data_literal_fraction,
    max_line_length,
    is_generated,
    filter_code_doc,
    filter_shard,
)
from data.acquire import write_shard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shard(tmp_path: Path, docs: list[dict], name: str = "train.jsonl.gz") -> Path:
    p = tmp_path / name
    write_shard(iter(docs), p)
    return p

LONG_CODE = "def foo(x):\n    return x + 1\n" * 20   # ~580 chars, good code


# ===========================================================================
# scrub.py
# ===========================================================================


class TestScrubRegex:
    def test_email_replaced(self):
        result = scrub_regex("Contact me at alice@example.com for details.")
        assert "@" not in result
        assert "<|email_addr|>" in result

    def test_phone_replaced(self):
        result = scrub_regex("Call us: 555-867-5309 or (800) 555-1234.")
        assert "<|phone|>" in result

    def test_ssn_replaced(self):
        result = scrub_regex("SSN: 123-45-6789")
        assert "123-45-6789" not in result
        assert "<|ssn|>" in result

    def test_cc_replaced(self):
        result = scrub_regex("Card: 4111 1111 1111 1111")
        assert "4111" not in result
        assert "<|cc|>" in result

    def test_no_false_positive_on_clean_text(self):
        text   = "The weather today is sunny and warm."
        result = scrub_regex(text)
        assert result == text

    def test_multiple_emails_all_replaced(self):
        text   = "From: alice@a.com To: bob@b.org CC: carol@c.net"
        result = scrub_regex(text)
        assert "@" not in result
        assert result.count("<|email_addr|>") == 3

    def test_plain_number_not_flagged_as_phone(self):
        text   = "The year 2024 had 365 days."
        result = scrub_regex(text)
        assert "<|phone|>" not in result


class TestStripSignature:
    def test_strips_after_double_dash(self):
        text   = "Hello there.\n\n--\nAlice Smith\nalice@example.com"
        result = strip_signature(text)
        assert "Alice Smith" not in result
        assert "Hello there" in result

    def test_strips_after_regards(self):
        text   = "Please find attached.\n\nBest regards,\nBob"
        result = strip_signature(text)
        assert "Bob" not in result
        assert "Please find attached" in result

    def test_no_signature_unchanged(self):
        text   = "Just a plain email with no signature block here."
        result = strip_signature(text)
        assert result == text

    def test_strips_thanks(self):
        text   = "See you there.\n\nThanks\nCarol"
        result = strip_signature(text)
        assert "Carol" not in result


class TestStripQuotedReplies:
    def test_removes_quoted_lines(self):
        text   = "My reply here.\n> Original message\n> More original"
        result = strip_quoted_replies(text)
        assert ">" not in result
        assert "My reply here" in result

    def test_deeply_quoted_removed(self):
        text   = "Top.\n>> deeply nested\n>>> triply nested"
        result = strip_quoted_replies(text)
        assert ">" not in result

    def test_no_quotes_unchanged(self):
        text   = "Clean email with no quotes."
        result = strip_quoted_replies(text)
        assert "Clean email" in result


class TestScrubEmail:
    def test_full_pipeline(self):
        email = (
            "Hi Bob,\n"
            "Please call me at 555-123-4567 or email alice@corp.com.\n"
            "> Original message here\n"
            "\nBest regards,\nAlice"
        )
        result = scrub_email(email, use_scrubadub=False)
        assert "@" not in result
        assert "555-123-4567" not in result
        # quoted-reply lines starting with ">" are removed; placeholder tokens
        # like <|phone|> legitimately contain ">" so check line-start quoting
        import re
        assert not re.search(r"^>", result, re.MULTILINE), \
            "Quoted reply lines (^>) should be stripped"
        assert "Best regards" not in result

    def test_empty_string(self):
        assert scrub_email("", use_scrubadub=False) == ""


class TestAuditPii:
    def test_detects_email(self):
        hits = audit_pii(["Contact us at admin@example.com"])
        assert hits["email addresses"] >= 1

    def test_clean_text_zero_hits(self):
        hits = audit_pii(["The weather is nice today. Call us at extension 42."])
        assert hits["email addresses"] == 0

    def test_multiple_docs(self):
        texts = ["a@b.com text", "c@d.com text", "clean text"]
        hits  = audit_pii(texts)
        assert hits["email addresses"] == 2


class TestScrubShard:
    def test_creates_output(self, tmp_path):
        email = (
            "Hello.\nPlease email me at test@example.com.\n\nThanks,\nAlice"
        )
        docs  = [{"text": email, "source": "email", "id": str(i)} for i in range(5)]
        in_p  = _shard(tmp_path, docs, "email/in.jsonl.gz")
        out_p = tmp_path / "email" / "out.jsonl.gz"
        stats = scrub_shard(in_p, out_p, use_scrubadub=False)
        assert out_p.exists()
        assert stats["in"] == 5

    def test_drops_empty_after_scrub(self, tmp_path):
        # An email that becomes empty after stripping quotes and signature
        docs = [
            {"text": "> quoted only\n\nRegards,\nBob", "source": "email", "id": "0"},
            {"text": "Good email content here. " * 5, "source": "email", "id": "1"},
        ]
        in_p  = _shard(tmp_path, docs, "e/in.jsonl.gz")
        out_p = tmp_path / "e" / "out.jsonl.gz"
        stats = scrub_shard(in_p, out_p, use_scrubadub=False, min_chars=50)
        assert stats["out"] < stats["in"]


# ===========================================================================
# filter_code.py
# ===========================================================================


class TestShannonEntropy:
    def test_uniform_high_entropy(self):
        # Many distinct characters → high entropy
        text = "".join(chr(i) for i in range(32, 128)) * 10
        assert shannon_entropy(text) > 5.0

    def test_repeated_char_low_entropy(self):
        assert shannon_entropy("aaaaaaaaaa") < 0.1

    def test_empty_zero(self):
        assert shannon_entropy("") == 0.0

    def test_code_reasonable_entropy(self):
        assert shannon_entropy(LONG_CODE) > 2.5


class TestDataLiteralFraction:
    def test_pure_data_high_fraction(self):
        data = "1, 2, 3, 4,\n5, 6, 7, 8,\n9, 10, 11,\n"
        assert data_literal_fraction(data) > 0.5

    def test_code_low_fraction(self):
        assert data_literal_fraction(LONG_CODE) < 0.4

    def test_empty_zero(self):
        assert data_literal_fraction("") == 0.0


class TestMaxLineLength:
    def test_short_lines(self):
        assert max_line_length("hello\nworld\n") <= 5

    def test_long_line_detected(self):
        assert max_line_length("a" * 2000) == 2000

    def test_empty(self):
        assert max_line_length("") == 0


class TestIsGenerated:
    def test_detects_auto_generated(self):
        assert is_generated("# Auto-generated by protoc. Do not edit.\ndef foo(): pass")

    def test_detects_generated_by(self):
        assert is_generated("/* Generated by swagger-codegen */\n")

    def test_normal_code_not_generated(self):
        assert not is_generated("def add(a, b):\n    return a + b\n")


class TestFilterCodeDoc:
    def test_good_code_passes(self):
        doc = {"text": LONG_CODE, "source": "code", "id": "0"}
        assert filter_code_doc(doc) is not None

    def test_too_short_dropped(self):
        doc = {"text": "x = 1", "source": "code", "id": "0"}
        assert filter_code_doc(doc) is None

    def test_too_long_dropped(self):
        doc = {"text": "x = 1\n" * 10_000, "source": "code", "id": "0"}
        assert filter_code_doc(doc) is None

    def test_minified_dropped(self):
        long_line = "a" * 2000 + "b" * 2000
        doc = {"text": long_line, "source": "code", "id": "0"}
        assert filter_code_doc(doc) is None

    def test_low_entropy_dropped(self):
        doc = {"text": "x = 1\n" * 20, "source": "code", "id": "0"}
        assert filter_code_doc(doc) is None

    def test_generated_dropped(self):
        header = "# This file is generated by protobuf. Do not edit.\n"
        doc = {"text": header + LONG_CODE, "source": "code", "id": "0"}
        assert filter_code_doc(doc) is None

    def test_data_literal_dropped(self):
        data = "1, 2, 3,\n" * 30
        doc  = {"text": data, "source": "code", "id": "0"}
        assert filter_code_doc(doc) is None


class TestFilterShard:
    def test_good_code_kept(self, tmp_path):
        docs = [{"text": LONG_CODE, "source": "code", "id": str(i)} for i in range(5)]
        in_p  = _shard(tmp_path, docs, "code/in.jsonl.gz")
        out_p = tmp_path / "code" / "out.jsonl.gz"
        stats = filter_shard(in_p, out_p)
        assert stats["out"] == 5

    def test_bad_code_dropped(self, tmp_path):
        docs = [
            {"text": "x = 1", "source": "code", "id": "0"},          # too short
            {"text": LONG_CODE, "source": "code", "id": "1"},         # good
            {"text": "a" * 3000, "source": "code", "id": "2"},        # long line
        ]
        in_p  = _shard(tmp_path, docs, "code/in.jsonl.gz")
        out_p = tmp_path / "code" / "out.jsonl.gz"
        stats = filter_shard(in_p, out_p)
        assert stats["out"] == 1
        assert stats["dropped"] == 2

    def test_reasons_tracked(self, tmp_path):
        docs = [
            {"text": "x", "source": "code", "id": "0"},               # too short
            {"text": LONG_CODE, "source": "code", "id": "1"},
        ]
        in_p  = _shard(tmp_path, docs, "c/in.jsonl.gz")
        out_p = tmp_path / "c" / "out.jsonl.gz"
        stats = filter_shard(in_p, out_p)
        assert stats["reasons"]["length"] >= 1
