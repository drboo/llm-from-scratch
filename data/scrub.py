"""
Day 17 (part 1): PII scrubbing for email shards.

Replaces:
  - Email addresses   → <|email_addr|>
  - Phone numbers     → <|phone|>
  - US SSNs           → <|ssn|>
  - Credit card nums  → <|cc|>
  - Names (via scrubadub, optional) → <|name|>

Also strips:
  - Signature blocks  (heuristic: lines after "-- " or "Regards," etc.)
  - Quoted-reply chains (lines starting with ">")

Usage:
    python data/scrub.py --in data/dedup_shards/email/train.jsonl.gz \\
                         --out data/scrubbed/email/train.jsonl.gz
    python data/scrub.py --in data/dedup_shards/email/train.jsonl.gz \\
                         --out data/scrubbed/email/train.jsonl.gz --no-scrubadub
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.acquire import iter_shard, write_shard

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)
_PHONE_RE = re.compile(
    r"""
    (?:
        \+?1[\s.\-]?       # optional country code
    )?
    (?:\(\d{3}\)|\d{3})   # area code
    [\s.\-]?
    \d{3}
    [\s.\-]?
    \d{4}
    """,
    re.VERBOSE,
)
_SSN_RE = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")
_CC_RE  = re.compile(
    r"\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[- ]?"
    r"\d{4}[- ]?\d{4}[- ]?\d{4}\b"
)

# Signature-block sentinel patterns
_SIG_SENTINELS = re.compile(
    r"^(?:--|best regards?|regards?|sincerely|cheers|thanks?|"
    r"warm regards?|kind regards?|yours truly|yours sincerely)"
    r"[,.]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Quoted-reply lines
_QUOTED_LINE = re.compile(r"^>+.*$", re.MULTILINE)

# ---------------------------------------------------------------------------
# Individual scrubbers
# ---------------------------------------------------------------------------


def scrub_regex(text: str) -> str:
    """Replace PII patterns with placeholder tokens."""
    text = _EMAIL_RE.sub("<|email_addr|>", text)
    text = _PHONE_RE.sub("<|phone|>",      text)
    text = _SSN_RE.sub("<|ssn|>",          text)
    text = _CC_RE.sub("<|cc|>",            text)
    return text


def scrub_names(text: str) -> str:
    """Replace detected names using scrubadub (optional)."""
    try:
        import scrubadub
        scrubber = scrubadub.Scrubber()
        scrubber.remove_detector("email")   # handled by regex already
        return scrubber.clean(text)
    except ImportError:
        return text
    except Exception:
        return text


def strip_signature(text: str) -> str:
    """Remove everything from the first signature-sentinel line onward."""
    m = _SIG_SENTINELS.search(text)
    if m:
        text = text[: m.start()].rstrip()
    return text


def strip_quoted_replies(text: str) -> str:
    """Remove lines that are quoted replies (start with '>')."""
    return _QUOTED_LINE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Full email cleaning pipeline
# ---------------------------------------------------------------------------


def scrub_email(text: str, use_scrubadub: bool = True) -> str:
    """Full PII-scrub + signature/quote strip pipeline for one email body."""
    text = strip_quoted_replies(text)
    text = strip_signature(text)
    text = scrub_regex(text)
    if use_scrubadub:
        text = scrub_names(text)
    # Collapse excess whitespace left by removals
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

_VERIFY_PATTERNS = [
    ("email addresses", _EMAIL_RE),
    ("phone numbers",   re.compile(r"\b\d{3}[.\-\s]\d{3}[.\-\s]\d{4}\b")),
    ("SSNs",            _SSN_RE),
]


def audit_pii(texts: list[str]) -> dict[str, int]:
    """Count surviving PII hits across a list of texts. Expect near-zero."""
    counts: dict[str, int] = {}
    for name, pattern in _VERIFY_PATTERNS:
        counts[name] = sum(len(pattern.findall(t)) for t in texts)
    return counts


# ---------------------------------------------------------------------------
# Shard pipeline
# ---------------------------------------------------------------------------


def scrub_shard(
    in_path: Path,
    out_path: Path,
    use_scrubadub: bool = True,
    min_chars: int = 50,
) -> dict:
    """Scrub all email documents in one shard."""
    n_in = n_out = n_dropped = 0

    def _scrubbed():
        nonlocal n_in, n_out, n_dropped
        for doc in iter_shard(in_path):
            n_in += 1
            text = scrub_email(doc["text"], use_scrubadub=use_scrubadub)
            if len(text) < min_chars:
                n_dropped += 1
                continue
            n_out += 1
            yield {**doc, "text": text}

    write_shard(_scrubbed(), out_path)
    return {
        "in":       n_in,
        "out":      n_out,
        "dropped":  n_dropped,
        "drop_pct": n_dropped / n_in * 100 if n_in else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PII-scrub email shards (Day 17)")
    p.add_argument("--in",  dest="in_path",  required=True, type=Path)
    p.add_argument("--out", dest="out_path", required=True, type=Path)
    p.add_argument("--no-scrubadub", action="store_true",
                   help="skip scrubadub name detection (faster)")
    p.add_argument("--min-chars", type=int, default=50,
                   help="drop scrubbed emails shorter than this")
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse()
    stats = scrub_shard(
        args.in_path, args.out_path,
        use_scrubadub=not args.no_scrubadub,
        min_chars=args.min_chars,
    )
    print(f"Scrubbed: {stats['in']:,} → {stats['out']:,} docs "
          f"(dropped {stats['dropped']:,} = {stats['drop_pct']:.1f}%)")

    # Audit surviving PII
    sample = [doc["text"] for doc in list(iter_shard(args.out_path))[:200]]
    hits   = audit_pii(sample)
    print("\nPII audit (first 200 docs):")
    for name, count in hits.items():
        flag = " ⚠" if count > 0 else " ✓"
        print(f"  {name:<20}: {count:>4} hits{flag}")
