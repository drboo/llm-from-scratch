"""
Day 8: Tokenize a toy corpus and save as uint16 memmap .bin files.

Downloads ~N documents from FineWeb (web text) and Python code from
code_search_net, tokenizes with the trained Codec, packs into a flat
uint16 array, and splits 90/10 into train.bin / val.bin.

Usage:
    python data/prepare_toy.py                    # defaults (~5k docs, quick)
    python data/prepare_toy.py --web 20000 --code 5000 --out data/toy
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokeniser.tokenizer import Codec

TOKENIZER_PATH = ROOT / "tokeniser" / "tokenizer.json"
DEFAULT_OUT     = ROOT / "data" / "toy"
VAL_RATIO       = 0.1


# ---------------------------------------------------------------------------
# Corpus iterators (same sources as tokeniser training)
# ---------------------------------------------------------------------------


def _iter_web(n: int):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= n:
            break
        yield ex["text"], "text"


def _iter_code(n: int):
    from datasets import load_dataset
    ds = load_dataset("code_search_net", split="train", streaming=True)
    count = 0
    for ex in ds:
        if count >= n:
            break
        text = ex.get("func_code_string", "")
        if text:
            count += 1
            yield text, "py"


def _corpus(n_web: int, n_code: int):
    print(f"  sources: {n_web:,} web docs + {n_code:,} Python snippets")
    return itertools.chain(_iter_web(n_web), _iter_code(n_code))


# ---------------------------------------------------------------------------
# Tokenise + save
# ---------------------------------------------------------------------------


def tokenize_corpus(codec: Codec, docs, report_every: int = 1_000) -> list[int]:
    """Encode documents with BOS/domain/EOS wrappers; return flat id list."""
    all_ids: list[int] = []
    for i, (text, domain) in enumerate(docs):
        ids = codec.encode_document(text, domain)
        all_ids.extend(ids)
        if (i + 1) % report_every == 0:
            print(f"  {i+1:,} docs  {len(all_ids)/1e6:.1f}M tokens", end="\r", flush=True)
    print()
    return all_ids


def save_bin(ids: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(ids, dtype=np.uint16)
    fp  = np.memmap(str(path), dtype=np.uint16, mode="w+", shape=(len(arr),))
    fp[:] = arr
    fp.flush()
    del fp
    size_mb = path.stat().st_size / 1e6
    print(f"  saved {path}  ({len(arr)/1e6:.2f}M tokens, {size_mb:.0f} MB)")


def prepare(n_web: int, n_code: int, out_dir: Path) -> dict[str, Path]:
    """Tokenize corpus and write train.bin / val.bin. Returns paths."""
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. "
            "Run `python tokeniser/tokeniser.py` first."
        )

    codec = Codec(TOKENIZER_PATH)
    print("Tokenizing corpus …")
    all_ids = tokenize_corpus(codec, _corpus(n_web, n_code))

    n      = len(all_ids)
    split  = int(n * (1 - VAL_RATIO))
    print(f"Total {n/1e6:.2f}M tokens → {split/1e6:.2f}M train + {(n-split)/1e6:.2f}M val")

    paths = {
        "train": out_dir / "train.bin",
        "val":   out_dir / "val.bin",
    }
    save_bin(all_ids[:split], paths["train"])
    save_bin(all_ids[split:], paths["val"])
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare toy pretraining data (Day 8)")
    p.add_argument("--web",  type=int, default=5_000,  help="web documents")
    p.add_argument("--code", type=int, default=2_000,  help="Python snippets")
    p.add_argument("--out",  type=Path, default=DEFAULT_OUT, help="output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(args.web, args.code, args.out)
    print("Done.")
