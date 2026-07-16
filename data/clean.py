"""
Day 16: Quality filtering — clean raw corpus shards.

Filters applied per source:

  ALL sources:
    - UTF-8 enforcement (re-encode/replace bad bytes)
    - Drop documents shorter than MIN_CHARS (default 100)

  web:
    - Strip HTML boilerplate via BeautifulSoup
    - Language-ID filter: keep only English (fasttext lid.176.bin)
      Falls back to no language filter if model unavailable.

  email:
    - Strip HTML boilerplate
    - Drop docs shorter than MIN_CHARS after stripping

  code:
    - No HTML stripping (already plain text)
    - No language filter

Usage:
    python data/clean.py --raw data/raw_shards --out data/clean_shards
    python data/clean.py --raw data/raw_shards --out data/clean_shards --min-chars 200
    python data/clean.py --raw data/raw_shards --out data/clean_shards --no-langid
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.acquire import iter_shard, write_shard, list_shards

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CHARS    = 100
LANG_TARGET  = "en"
LANG_THRESH  = 0.7     # minimum fasttext confidence to keep

# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities; fall back to regex if bs4 absent."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ")
    except ImportError:
        return re.sub(r"<[^>]+>", " ", text)


_HTML_PATTERN = re.compile(r"<[a-zA-Z][\s\S]*?>")


def looks_like_html(text: str) -> bool:
    """Quick heuristic: contains an HTML tag."""
    return bool(_HTML_PATTERN.search(text))


# ---------------------------------------------------------------------------
# UTF-8 enforcement
# ---------------------------------------------------------------------------


def enforce_utf8(text: str) -> str:
    """Round-trip through bytes, replacing invalid sequences."""
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_LANG_MODEL = None
_LANG_AVAILABLE = None


def _get_lang_model():
    """Lazy-load fasttext language-ID model (lid.176.bin)."""
    global _LANG_MODEL, _LANG_AVAILABLE
    if _LANG_AVAILABLE is not None:
        return _LANG_MODEL

    # Look for the model in a few standard locations
    candidates = [
        Path.home() / ".cache" / "fasttext" / "lid.176.bin",
        ROOT / "data" / "lid.176.bin",
        Path("/tmp/lid.176.bin"),
    ]
    try:
        import fasttext
        fasttext.FastText.eprint = lambda x: None   # suppress C++ warnings
        for p in candidates:
            if p.exists():
                _LANG_MODEL    = fasttext.load_model(str(p))
                _LANG_AVAILABLE = True
                return _LANG_MODEL
        # Not found — try to download
        import urllib.request
        dest = Path.home() / ".cache" / "fasttext"
        dest.mkdir(parents=True, exist_ok=True)
        url  = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        print(f"  [langid] Downloading fasttext lid.176.bin to {dest} …")
        urllib.request.urlretrieve(url, dest / "lid.176.bin")
        _LANG_MODEL     = fasttext.load_model(str(dest / "lid.176.bin"))
        _LANG_AVAILABLE = True
        return _LANG_MODEL
    except Exception as e:
        print(f"  [langid] Not available ({e}) — language filter disabled")
        _LANG_AVAILABLE = False
        return None


def is_english(text: str, threshold: float = LANG_THRESH) -> bool:
    """Return True if fasttext predicts English above threshold, or if model unavailable."""
    model = _get_lang_model()
    if model is None:
        return True   # fail open: keep the doc
    # fasttext expects single-line input
    sample = text[:500].replace("\n", " ")
    labels, scores = model.predict(sample, k=1)
    label = labels[0].replace("__label__", "")
    return label == LANG_TARGET and scores[0] >= threshold


# ---------------------------------------------------------------------------
# Per-document cleaning
# ---------------------------------------------------------------------------


def clean_doc(doc: dict, min_chars: int = MIN_CHARS,
              use_langid: bool = True) -> dict | None:
    """
    Apply quality filters to one document.

    Returns the cleaned doc dict, or None if it should be dropped.
    """
    text   = doc.get("text", "")
    source = doc.get("source", "web")

    # 1. UTF-8 enforcement
    text = enforce_utf8(text)

    # 2. Strip HTML for web and email
    if source in ("web", "email") and looks_like_html(text):
        text = strip_html(text)

    # 3. Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # 4. Length filter
    if len(text) < min_chars:
        return None

    # 5. Language filter (web only)
    if use_langid and source == "web" and not is_english(text):
        return None

    return {**doc, "text": text}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def clean_shard(
    in_path: Path,
    out_path: Path,
    min_chars: int = MIN_CHARS,
    use_langid: bool = True,
) -> dict:
    """
    Clean one shard file.

    Returns stats: {"in": int, "out": int, "dropped": int, "drop_pct": float}
    """
    def filtered():
        for doc in iter_shard(in_path):
            cleaned = clean_doc(doc, min_chars=min_chars, use_langid=use_langid)
            if cleaned is not None:
                yield cleaned

    # Count input docs
    n_in = sum(1 for _ in iter_shard(in_path))
    write_shard(filtered(), out_path)
    n_out = sum(1 for _ in iter_shard(out_path))

    dropped  = n_in - n_out
    drop_pct = dropped / n_in * 100 if n_in else 0.0
    return {"in": n_in, "out": n_out, "dropped": dropped, "drop_pct": drop_pct}


def clean_all(
    raw_dir: Path,
    out_dir: Path,
    min_chars: int = MIN_CHARS,
    use_langid: bool = True,
) -> dict[str, dict]:
    """Clean all shards under raw_dir, writing to parallel structure under out_dir."""
    shards = list_shards(raw_dir)
    if not shards:
        print(f"No shards found in {raw_dir}")
        return {}

    total_stats: dict[str, dict] = {}

    for in_path in shards:
        rel      = in_path.relative_to(raw_dir)
        out_path = out_dir / rel
        source   = in_path.parent.name
        print(f"\n[clean] {rel}")

        stats = clean_shard(in_path, out_path, min_chars=min_chars, use_langid=use_langid)
        total_stats[str(rel)] = stats
        print(
            f"  {stats['in']:,} → {stats['out']:,} docs  "
            f"(dropped {stats['dropped']:,} = {stats['drop_pct']:.1f}%)"
        )

    return total_stats


def print_summary(stats: dict[str, dict]) -> None:
    total_in  = sum(s["in"]      for s in stats.values())
    total_out = sum(s["out"]     for s in stats.values())
    total_drop = total_in - total_out
    pct        = total_drop / total_in * 100 if total_in else 0
    print(f"\nTotal: {total_in:,} → {total_out:,} docs  "
          f"(dropped {total_drop:,} = {pct:.1f}%)")
    print("Typical range: 10–30% dropped is expected and healthy.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean raw corpus shards (Day 16)")
    p.add_argument("--raw",       type=Path, default=Path("data/raw_shards"))
    p.add_argument("--out",       type=Path, default=Path("data/clean_shards"))
    p.add_argument("--min-chars", type=int,  default=MIN_CHARS)
    p.add_argument("--no-langid", action="store_true",
                   help="skip language-ID filter (faster, keeps non-English)")
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse()
    stats = clean_all(args.raw, args.out,
                      min_chars=args.min_chars,
                      use_langid=not args.no_langid)
    print_summary(stats)
    print(f"\nCleaned shards written to: {args.out}")
    print("Next step: python data/dedup.py --clean data/clean_shards --out data/dedup_shards")
