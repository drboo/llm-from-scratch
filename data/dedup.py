"""
Day 16: Deduplication — exact and near-duplicate removal.

Two passes:
  1. Exact dedup: SHA-256 of normalised text (lower + collapse whitespace).
     Drops identical documents regardless of metadata.

  2. Near-dedup: MinHash + LSH via datasketch.
     Flags pairs with Jaccard similarity > threshold (default 0.8).
     Applied to web + code shards (not email, which is more heterogeneous).

Typical removal rates: 10–30% of the web slice is duplicated.

Usage:
    python data/dedup.py --clean data/clean_shards --out data/dedup_shards
    python data/dedup.py --clean data/clean_shards --out data/dedup_shards \\
        --jaccard 0.7 --no-minhash
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.acquire import iter_shard, write_shard, list_shards

# ---------------------------------------------------------------------------
# Text normalisation (used for exact-dedup key)
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Lower-case + collapse all whitespace — used as the dedup key."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def doc_hash(text: str) -> str:
    """SHA-256 of the normalised text."""
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Exact dedup
# ---------------------------------------------------------------------------


def exact_dedup(docs) -> tuple[list[dict], dict]:
    """
    Remove exact duplicates (by normalised-text hash).

    Args:
        docs: iterable of doc dicts with a "text" key.

    Returns:
        (kept_docs, stats) where stats has "in", "out", "dropped".
    """
    seen: set[str] = set()
    kept: list[dict] = []
    n_in = 0

    for doc in docs:
        n_in += 1
        h = doc_hash(doc["text"])
        if h not in seen:
            seen.add(h)
            kept.append(doc)

    n_out    = len(kept)
    dropped  = n_in - n_out
    return kept, {"in": n_in, "out": n_out, "dropped": dropped,
                  "drop_pct": dropped / n_in * 100 if n_in else 0.0}


# ---------------------------------------------------------------------------
# Near-dedup (MinHash + LSH)
# ---------------------------------------------------------------------------

_MINHASH_PERMS  = 128    # number of hash permutations — higher = more accurate
_SHINGLE_SIZE   = 5      # character n-gram size


def _shingles(text: str, k: int = _SHINGLE_SIZE) -> set[str]:
    """Set of character k-grams from normalised text."""
    t = normalise(text)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def build_minhash(text: str):
    """Return a datasketch MinHash object for one document."""
    from datasketch import MinHash
    m = MinHash(num_perm=_MINHASH_PERMS)
    for shingle in _shingles(text):
        m.update(shingle.encode("utf-8"))
    return m


def near_dedup(docs: list[dict], threshold: float = 0.8) -> tuple[list[dict], dict]:
    """
    Remove near-duplicates using MinHash + LSH.

    Docs that are >= threshold Jaccard similar to an already-seen doc are dropped.

    Args:
        docs:      list of doc dicts (must fit in memory; typically one shard).
        threshold: Jaccard similarity threshold (default 0.8).

    Returns:
        (kept_docs, stats)
    """
    try:
        from datasketch import MinHashLSH
    except ImportError:
        print("  [dedup] datasketch not available — skipping near-dedup")
        n = len(docs)
        return docs, {"in": n, "out": n, "dropped": 0, "drop_pct": 0.0}

    from datasketch import MinHashLSH

    lsh  = MinHashLSH(threshold=threshold, num_perm=_MINHASH_PERMS)
    kept: list[dict] = []
    n_in = len(docs)

    for i, doc in enumerate(docs):
        mh  = build_minhash(doc["text"])
        key = f"doc_{i}"
        # query returns keys of docs already in the index that are near-dups
        if not lsh.query(mh):
            lsh.insert(key, mh)
            kept.append(doc)

    n_out    = len(kept)
    dropped  = n_in - n_out
    return kept, {"in": n_in, "out": n_out, "dropped": dropped,
                  "drop_pct": dropped / n_in * 100 if n_in else 0.0}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_NEAR_DEDUP_SOURCES = {"web", "code"}   # email dedup is exact-only


def dedup_shard(
    in_path: Path,
    out_path: Path,
    jaccard: float = 0.8,
    use_minhash: bool = True,
) -> dict:
    """
    Deduplicate one shard: exact pass then optional MinHash pass.

    Returns combined stats dict.
    """
    source = in_path.parent.name
    docs   = list(iter_shard(in_path))

    # Pass 1: exact dedup
    docs, exact_stats = exact_dedup(docs)
    print(
        f"    exact:  {exact_stats['in']:,} → {exact_stats['out']:,}"
        f"  (dropped {exact_stats['dropped']:,} = {exact_stats['drop_pct']:.1f}%)"
    )

    # Pass 2: near-dedup (web + code only)
    near_stats = {"in": len(docs), "out": len(docs), "dropped": 0, "drop_pct": 0.0}
    if use_minhash and source in _NEAR_DEDUP_SOURCES and docs:
        docs, near_stats = near_dedup(docs, threshold=jaccard)
        print(
            f"    minhash:{near_stats['in']:,} → {near_stats['out']:,}"
            f"  (dropped {near_stats['dropped']:,} = {near_stats['drop_pct']:.1f}%)"
        )

    write_shard(iter(docs), out_path)

    total_in   = exact_stats["in"]
    total_out  = near_stats["out"]
    total_drop = total_in - total_out
    return {
        "in":       total_in,
        "out":      total_out,
        "dropped":  total_drop,
        "drop_pct": total_drop / total_in * 100 if total_in else 0.0,
        "exact":    exact_stats,
        "near":     near_stats,
    }


def dedup_all(
    clean_dir: Path,
    out_dir: Path,
    jaccard: float = 0.8,
    use_minhash: bool = True,
) -> dict[str, dict]:
    """Dedup all shards under clean_dir into out_dir."""
    shards = list_shards(clean_dir)
    if not shards:
        print(f"No shards found in {clean_dir}")
        return {}

    all_stats: dict[str, dict] = {}
    for in_path in shards:
        rel      = in_path.relative_to(clean_dir)
        out_path = out_dir / rel
        print(f"\n[dedup] {rel}")
        all_stats[str(rel)] = dedup_shard(in_path, out_path,
                                          jaccard=jaccard, use_minhash=use_minhash)

    return all_stats


def print_summary(stats: dict[str, dict]) -> None:
    total_in  = sum(s["in"]  for s in stats.values())
    total_out = sum(s["out"] for s in stats.values())
    dropped   = total_in - total_out
    pct       = dropped / total_in * 100 if total_in else 0
    print(f"\nTotal: {total_in:,} → {total_out:,} docs  "
          f"(removed {dropped:,} = {pct:.1f}%)")
    print("Typical range: 10–30% removed is expected and healthy.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deduplicate corpus shards (Day 16)")
    p.add_argument("--clean",      type=Path, default=Path("data/clean_shards"))
    p.add_argument("--out",        type=Path, default=Path("data/dedup_shards"))
    p.add_argument("--jaccard",    type=float, default=0.8,
                   help="MinHash Jaccard threshold (default 0.8)")
    p.add_argument("--no-minhash", action="store_true",
                   help="exact dedup only (faster, less aggressive)")
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse()
    stats = dedup_all(args.clean, args.out,
                      jaccard=args.jaccard,
                      use_minhash=not args.no_minhash)
    print_summary(stats)
    print(f"\nDeduped shards written to: {args.out}")
    print("Next step: python data/prepare_real.py --dedup data/dedup_shards --out data/real")
