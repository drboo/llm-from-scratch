"""
Day 15: Real corpus acquisition.

Downloads raw text from three sources and saves as per-source JSONL shards.
Token counts are estimated (bytes / 4) so you get a quick read before
committing to a full tokenize pass.

Sources and default mixture:
  web   60-70%  FineWeb (HuggingFaceFW/fineweb, sample-10BT)
  email ~10%    Enron email corpus (enron_spam)
  code  20-30%  The Stack Python-only, permissive licenses
                (bigcode/the-stack, lang=Python; falls back to code_search_net)

Deliverable: raw_shards/<source>/<split>.jsonl.gz on disk.

Usage:
    # Quick sample (fits in minutes):
    python data/acquire.py --web 10000 --email 5000 --code 10000 --out data/raw_shards

    # Larger run (hours, ~1-3 GB):
    python data/acquire.py --web 200000 --email 20000 --code 50000 --out data/raw_shards

    # Single source:
    python data/acquire.py --web 50000 --email 0 --code 0 --out data/raw_shards

After this, run data/clean.py (Day 16) to filter and deduplicate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Source iterators
# ---------------------------------------------------------------------------


def _iter_web(n: int):
    """Stream n documents from FineWeb (general web text backbone)."""
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for i, ex in enumerate(ds):
        if i >= n:
            break
        yield {
            "text":   ex["text"],
            "source": "web",
            "id":     ex.get("id", str(i)),
        }


def _iter_email(n: int):
    """Stream n emails from the Enron spam dataset."""
    from datasets import load_dataset
    ds = load_dataset("SetFit/enron_spam", split="train", streaming=True)
    count = 0
    for ex in ds:
        if count >= n:
            break
        subject = ex.get("subject", "")
        body    = ex.get("message", ex.get("text", ""))
        if not body:
            continue
        text = f"Subject: {subject}\n\n{body}" if subject else body
        yield {
            "text":   text,
            "source": "email",
            "id":     str(count),
        }
        count += 1


def _iter_code(n: int):
    """Stream n Python functions from The Stack (permissive) or code_search_net fallback."""
    from datasets import load_dataset

    # Try The Stack (permissive Python only) first
    try:
        ds = load_dataset(
            "bigcode/the-stack",
            data_dir="data/python",
            split="train",
            streaming=True,
        )
        count = 0
        for ex in ds:
            if count >= n:
                break
            text = ex.get("content", "")
            if text:
                yield {
                    "text":   text,
                    "source": "code",
                    "id":     ex.get("hexsha", str(count)),
                }
                count += 1
        return
    except Exception:
        pass  # fall through to public fallback

    # Fallback: code_search_net (Python split)
    ds = load_dataset("code-search-net/code_search_net", "python", split="train", streaming=True)
    count = 0
    for ex in ds:
        if count >= n:
            break
        text = ex.get("whole_func_string", ex.get("func_code_string", ""))
        if text:
            yield {
                "text":   text,
                "source": "code",
                "id":     str(count),
            }
            count += 1


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


def write_shard(docs, out_path: Path, report_every: int = 1_000) -> dict:
    """Write documents to a gzipped JSONL file.

    Returns stats: {"n_docs": int, "n_bytes": int, "est_tokens": int}
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    n_bytes = 0
    t0 = time.time()

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for doc in docs:
            line = json.dumps(doc, ensure_ascii=False)
            f.write(line + "\n")
            n_docs  += 1
            n_bytes += len(doc["text"].encode("utf-8"))
            if n_docs % report_every == 0:
                elapsed = time.time() - t0
                est_tok = n_bytes // 4
                print(
                    f"  {n_docs:>8,} docs  "
                    f"{n_bytes/1e6:>7.1f} MB  "
                    f"~{est_tok/1e6:.1f}M tokens  "
                    f"({elapsed:.0f}s)",
                    end="\r", flush=True,
                )
    print()

    est_tokens = n_bytes // 4   # rough: ~4 bytes per token for BPE
    return {"n_docs": n_docs, "n_bytes": n_bytes, "est_tokens": est_tokens}


# ---------------------------------------------------------------------------
# Acquire
# ---------------------------------------------------------------------------


def acquire(
    n_web:   int,
    n_email: int,
    n_code:  int,
    out_dir: Path,
) -> dict[str, dict]:
    """
    Download raw shards for each source.

    Returns per-source stats suitable for mixture-ratio reporting.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}

    sources = [
        ("web",   n_web,   _iter_web,   "FineWeb (sample-10BT)"),
        ("email", n_email, _iter_email, "Enron spam corpus"),
        ("code",  n_code,  _iter_code,  "The Stack / code_search_net Python"),
    ]

    for name, n, iterator, description in sources:
        if n <= 0:
            print(f"  [{name}] skipped (n=0)")
            stats[name] = {"n_docs": 0, "n_bytes": 0, "est_tokens": 0}
            continue

        print(f"\n[{name}] {description}  (requesting {n:,} docs)")
        out_path = out_dir / name / "train.jsonl.gz"
        stats[name] = write_shard(iterator(n), out_path)
        s = stats[name]
        print(
            f"  → {s['n_docs']:,} docs  "
            f"{s['n_bytes']/1e6:.1f} MB  "
            f"~{s['est_tokens']/1e6:.1f}M tokens"
        )

    return stats


def report_mixture(stats: dict[str, dict]) -> None:
    """Print per-source token estimates and planned mixture ratios."""
    total = sum(s["est_tokens"] for s in stats.values())
    if total == 0:
        print("No data acquired.")
        return

    print("\n" + "=" * 60)
    print(f"{'Source':<10}  {'Docs':>10}  {'~Tokens':>12}  {'Share':>8}")
    print("-" * 60)
    for name, s in stats.items():
        share = s["est_tokens"] / total * 100 if total else 0
        print(
            f"{name:<10}  {s['n_docs']:>10,}  "
            f"{s['est_tokens']/1e6:>10.1f}M  "
            f"{share:>7.1f}%"
        )
    print("-" * 60)
    print(f"{'TOTAL':<10}  {'':>10}  {total/1e6:>10.1f}M  {'100.0%':>8}")
    print("=" * 60)
    print()
    print("Planned mixture ratios (target):")
    print("  web   60-70%  — general text backbone")
    print("  email   ~10%  — Enron email domain")
    print("  code  20-30%  — Python permissive-license code")
    print()

    # Chinchilla check
    # Nano model: ~23M params → need ~460M tokens for compute-optimal training
    for n_params, label in [(23e6, "nano (~23M)"), (124e6, "small (~124M)")]:
        needed = n_params * 20
        print(
            f"  Chinchilla ({label}): need ~{needed/1e6:.0f}M tokens  "
            f"— have {total/1e6:.0f}M ({total/needed*100:.0f}% of target)"
        )


# ---------------------------------------------------------------------------
# Shard reader (used by clean.py, dedup.py, etc.)
# ---------------------------------------------------------------------------


def iter_shard(path: Path):
    """Yield parsed dicts from a .jsonl.gz shard."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def list_shards(raw_dir: Path) -> list[Path]:
    """Return all .jsonl.gz files under raw_dir, sorted."""
    return sorted(raw_dir.rglob("*.jsonl.gz"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Acquire raw corpus shards (Day 15)")
    p.add_argument("--web",   type=int, default=10_000,
                   help="web documents from FineWeb (default 10k)")
    p.add_argument("--email", type=int, default=5_000,
                   help="emails from Enron spam dataset (default 5k)")
    p.add_argument("--code",  type=int, default=10_000,
                   help="Python files from The Stack / code_search_net (default 10k)")
    p.add_argument("--out",   type=Path, default=Path("data/raw_shards"),
                   help="output directory (default data/raw_shards)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    stats = acquire(args.web, args.email, args.code, args.out)
    report_mixture(stats)
    print(f"Raw shards written to: {args.out}")
    print("Next step: python data/clean.py --raw data/raw_shards --out data/clean_shards")
