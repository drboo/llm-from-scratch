"""
Day 18: Tokenize + pack the full corpus into final training binaries.

Steps:
  1. Compression-ratio check — verify bytes/token ≈ 4 on the existing
     tokenizer.  If ratio < 3 or > 6, warn and suggest retraining.
  2. Stream each processed shard (web, email, code), encode with Codec,
     interleave documents according to mixture ratios.
  3. Hold out val by whole document (not token offset) to prevent leakage.
  4. Pack into flat uint16 arrays and write train.bin / val.bin.
  5. Print + save a datasheet: sources, sizes, filters applied, mixture.

Usage:
    # Full run from processed shards:
    python data/prepare_real.py \\
        --web   data/dedup_shards/web/train.jsonl.gz \\
        --email data/scrubbed/email/train.jsonl.gz \\
        --code  data/filtered/code/train.jsonl.gz \\
        --out   data/real

    # Dry-run (estimate token counts, no .bin written):
    python data/prepare_real.py ... --dry-run

    # Override mixture ratios (must sum to 1.0):
    python data/prepare_real.py ... --ratio-web 0.7 --ratio-email 0.1 --ratio-code 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.acquire import iter_shard, list_shards

TOKENIZER_PATH = ROOT / "tokeniser" / "tokenizer.json"
VAL_FRACTION   = 0.05          # 5% of documents held out for val
DOMAIN_MAP     = {"web": "text", "email": "email", "code": "py"}

# Default mixture ratios (web:email:code)
DEFAULT_RATIOS = {"web": 0.65, "email": 0.10, "code": 0.25}


# ---------------------------------------------------------------------------
# Compression-ratio check
# ---------------------------------------------------------------------------


def check_compression_ratio(
    codec,
    shard_paths: list[Path],
    n_sample: int = 500,
) -> float:
    """
    Sample n_sample documents across shards and compute bytes/token.

    Returns mean ratio; warns if outside [3, 6].
    """
    samples: list[dict] = []
    for p in shard_paths:
        for doc in iter_shard(p):
            samples.append(doc)
            if len(samples) >= n_sample:
                break
        if len(samples) >= n_sample:
            break

    if not samples:
        return 0.0

    total_bytes  = 0
    total_tokens = 0
    for doc in samples:
        text          = doc.get("text", "")
        total_bytes  += len(text.encode("utf-8"))
        total_tokens += len(codec.encode(text))

    ratio = total_bytes / total_tokens if total_tokens else 0.0
    status = "OK" if 3.0 <= ratio <= 6.0 else "WARNING"
    print(
        f"  Compression ratio: {ratio:.2f} bytes/token  [{status}]"
        + ("  — ratio looks healthy, reusing existing tokenizer" if status == "OK"
           else "  — consider retraining tokenizer on real mixture (Day 3)")
    )
    return ratio


# ---------------------------------------------------------------------------
# Document-level train / val split
# ---------------------------------------------------------------------------


def split_docs(docs: list[dict], val_fraction: float, seed: int = 42) -> tuple[list, list]:
    """Shuffle and split doc list into (train, val) by whole document."""
    rng = random.Random(seed)
    shuffled = docs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# Interleaved corpus builder
# ---------------------------------------------------------------------------


def build_token_stream(
    source_docs: dict[str, list[dict]],
    ratios: dict[str, float],
    codec,
    split: str = "train",
) -> tuple[list[int], dict[str, int]]:
    """
    Interleave documents from each source according to `ratios` and encode.

    Documents are drawn round-robin weighted by ratio until all sources
    are exhausted; sources that run out early are skipped.

    Returns:
        (flat token list, {source: token_count})
    """
    # Determine how many docs to draw from each source proportionally
    totals   = {src: len(docs) for src, docs in source_docs.items()}
    total_n  = sum(totals.values())
    if total_n == 0:
        return [], {}

    # Normalise ratios to the actual available counts
    ratio_sum = sum(ratios[s] for s in source_docs)
    target    = {
        src: int(total_n * (ratios[src] / ratio_sum))
        for src in source_docs
    }
    # Clamp to available
    target = {src: min(target[src], totals[src]) for src in source_docs}

    print(f"\n[{split}] Drawing:")
    for src, n in target.items():
        print(f"  {src:<8}: {n:,} docs  ({ratios[src]*100:.0f}% ratio)")

    # Build interleaved list
    domain = DOMAIN_MAP
    all_tokens: list[int] = []
    src_counts: dict[str, int] = {src: 0 for src in source_docs}

    # Shuffle each source's doc list with a fixed seed for reproducibility
    rng = random.Random(42)
    pools = {
        src: (lambda d, r=rng: (r.shuffle(d), d)[1])(docs[:target[src]])
        for src, docs in source_docs.items()
    }

    # Interleave: cycle through sources by weight
    weights  = [ratios[src] for src in source_docs]
    sources  = list(source_docs.keys())
    indices  = {src: 0 for src in sources}
    t0 = time.time()
    n_encoded = 0

    while True:
        # Pick a source weighted by ratio, skipping exhausted ones
        available = [s for s in sources if indices[s] < len(pools[s])]
        if not available:
            break
        av_weights = [ratios[s] for s in available]
        total_w    = sum(av_weights)
        r          = rng.random() * total_w
        cumul      = 0.0
        chosen     = available[0]
        for s, w in zip(available, av_weights):
            cumul += w
            if r < cumul:
                chosen = s
                break

        doc  = pools[chosen][indices[chosen]]
        indices[chosen] += 1

        text    = doc.get("text", "")
        dom     = domain.get(chosen, "text")
        ids     = codec.encode_document(text, dom)
        all_tokens.extend(ids)
        src_counts[chosen] += len(ids)
        n_encoded += 1

        if n_encoded % 5_000 == 0:
            elapsed = time.time() - t0
            print(
                f"  {n_encoded:>8,} docs  {len(all_tokens)/1e6:.1f}M tokens  "
                f"({elapsed:.0f}s)",
                end="\r", flush=True,
            )

    print(f"\n  Done: {n_encoded:,} docs → {len(all_tokens)/1e6:.2f}M tokens")
    return all_tokens, src_counts


# ---------------------------------------------------------------------------
# .bin writer
# ---------------------------------------------------------------------------


def save_bin(tokens: list[int], path: Path) -> None:
    """Write flat uint16 array to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(tokens, dtype=np.uint16)
    fp  = np.memmap(str(path), dtype=np.uint16, mode="w+", shape=(len(arr),))
    fp[:] = arr
    fp.flush()
    del fp
    size_mb = path.stat().st_size / 1e6
    print(f"  Wrote {path}  ({len(arr)/1e6:.2f}M tokens, {size_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# Datasheet
# ---------------------------------------------------------------------------


def make_datasheet(
    shard_paths:  dict[str, Path | None],
    ratios:       dict[str, float],
    src_counts:   dict[str, dict[str, int]],
    compression:  float,
    out_dir:      Path,
) -> dict:
    datasheet = {
        "tokenizer":   str(TOKENIZER_PATH),
        "compression_ratio_bytes_per_token": round(compression, 2),
        "val_fraction": VAL_FRACTION,
        "mixture_ratios": ratios,
        "sources": {},
        "filters_applied": {
            "web":   ["clean.py: utf8, html-strip, length≥100, langid=en",
                      "dedup.py: exact SHA-256, MinHash-LSH jaccard≥0.8"],
            "email": ["clean.py: utf8, html-strip, length≥100",
                      "dedup.py: exact SHA-256",
                      "scrub.py: email/phone/SSN/CC regex + scrubadub names, "
                      "sig-strip, quote-strip"],
            "code":  ["clean.py: utf8, length≥100",
                      "dedup.py: exact SHA-256, MinHash-LSH jaccard≥0.8",
                      "filter_code.py: 50–50k chars, line≤1k, entropy≥2.5, "
                      "data-literal<40%, not generated"],
        },
    }
    for src, counts in src_counts.items():
        datasheet["sources"][src] = {
            "shard": str(shard_paths.get(src)),
            "train_tokens": counts.get("train", 0),
            "val_tokens":   counts.get("val",   0),
        }

    path = out_dir / "datasheet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(datasheet, indent=2))
    print(f"\nDatasheet written to {path}")
    return datasheet


def print_datasheet(ds: dict) -> None:
    print("\n" + "=" * 60)
    print("CORPUS DATASHEET")
    print("=" * 60)
    print(f"  Tokenizer     : {ds['tokenizer']}")
    print(f"  Compression   : {ds['compression_ratio_bytes_per_token']} bytes/token")
    print(f"  Val fraction  : {ds['val_fraction']*100:.0f}% (by document)")
    print(f"\n  Mixture:")
    for src, r in ds["mixture_ratios"].items():
        tr = ds["sources"].get(src, {}).get("train_tokens", 0)
        vl = ds["sources"].get(src, {}).get("val_tokens",   0)
        print(f"    {src:<8} {r*100:.0f}%  train={tr/1e6:.1f}M  val={vl/1e6:.1f}M tokens")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def prepare_real(
    web_shard:   Path | None,
    email_shard: Path | None,
    code_shard:  Path | None,
    out_dir:     Path,
    ratios:      dict[str, float],
    dry_run:     bool = False,
    val_fraction: float = VAL_FRACTION,
    seed:        int = 42,
) -> dict:
    """
    Full pipeline: load shards → compress check → split → interleave → pack.

    Returns the datasheet dict.
    """
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. "
            "Run python tokeniser/tokeniser.py first."
        )

    sys.path.insert(0, str(ROOT / "tokeniser"))
    from tokenizer import Codec  # type: ignore
    codec = Codec(str(TOKENIZER_PATH))

    # --- Compression check -------------------------------------------------
    all_shards = [p for p in [web_shard, email_shard, code_shard] if p and p.exists()]
    print("\n[Step 1] Compression-ratio check …")
    compression = check_compression_ratio(codec, all_shards)

    # --- Load docs per source ----------------------------------------------
    print("\n[Step 2] Loading documents …")
    raw: dict[str, list[dict]] = {}
    for src, path in [("web", web_shard), ("email", email_shard), ("code", code_shard)]:
        if path and path.exists():
            docs = list(iter_shard(path))
            print(f"  {src:<8}: {len(docs):,} docs")
            raw[src] = docs
        else:
            print(f"  {src:<8}: skipped (no shard)")
            raw[src] = []

    if not any(raw.values()):
        print("No documents loaded — nothing to do.")
        return {}

    # --- Document-level train / val split ----------------------------------
    print("\n[Step 3] Document-level train/val split …")
    train_docs: dict[str, list] = {}
    val_docs:   dict[str, list] = {}
    for src, docs in raw.items():
        if docs:
            tr, vl = split_docs(docs, val_fraction, seed=seed)
            train_docs[src] = tr
            val_docs[src]   = vl
            print(f"  {src:<8}: {len(tr):,} train + {len(vl):,} val docs")

    if dry_run:
        print("\n[dry-run] Skipping tokenisation and .bin write.")
        return {}

    # --- Tokenise + interleave --------------------------------------------
    print("\n[Step 4] Tokenising train split …")
    train_tokens, train_src_counts = build_token_stream(
        {s: d for s, d in train_docs.items() if d},
        ratios, codec, split="train",
    )
    print("\n[Step 4b] Tokenising val split …")
    val_tokens, val_src_counts = build_token_stream(
        {s: d for s, d in val_docs.items() if d},
        ratios, codec, split="val",
    )

    # --- Write bins --------------------------------------------------------
    print("\n[Step 5] Writing .bin files …")
    save_bin(train_tokens, out_dir / "train.bin")
    save_bin(val_tokens,   out_dir / "val.bin")

    # --- Datasheet ---------------------------------------------------------
    shard_paths = {"web": web_shard, "email": email_shard, "code": code_shard}
    src_counts_combined = {
        src: {
            "train": train_src_counts.get(src, 0),
            "val":   val_src_counts.get(src, 0),
        }
        for src in ("web", "email", "code")
    }
    ds = make_datasheet(shard_paths, ratios, src_counts_combined, compression, out_dir)
    print_datasheet(ds)
    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pack final training corpus (Day 18)")
    p.add_argument("--web",    type=Path, default=None,
                   help="web shard (e.g. data/dedup_shards/web/train.jsonl.gz)")
    p.add_argument("--email",  type=Path, default=None,
                   help="scrubbed email shard")
    p.add_argument("--code",   type=Path, default=None,
                   help="filtered code shard")
    p.add_argument("--out",    type=Path, default=Path("data/real"),
                   help="output directory for train.bin, val.bin, datasheet.json")
    p.add_argument("--ratio-web",   type=float, default=DEFAULT_RATIOS["web"])
    p.add_argument("--ratio-email", type=float, default=DEFAULT_RATIOS["email"])
    p.add_argument("--ratio-code",  type=float, default=DEFAULT_RATIOS["code"])
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="estimate only; do not write .bin files")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse()
    ratios = {
        "web":   args.ratio_web,
        "email": args.ratio_email,
        "code":  args.ratio_code,
    }
    total = sum(ratios.values())
    if abs(total - 1.0) > 0.01:
        print(f"WARNING: ratios sum to {total:.3f}, not 1.0 — normalising")

    prepare_real(
        web_shard    = args.web,
        email_shard  = args.email,
        code_shard   = args.code,
        out_dir      = args.out,
        ratios       = ratios,
        dry_run      = args.dry_run,
        val_fraction = args.val_fraction,
        seed         = args.seed,
    )
