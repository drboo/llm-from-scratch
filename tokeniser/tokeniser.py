"""
Day 3: Train a byte-level BPE tokenizer on a small mixed corpus.

Corpus: FineWeb (web text) + The Stack smol Python + Enron emails.
Saves tokeniser/tokenizer.json with vocab_size=32_000 and special tokens.

Usage:
    python tokeniser/tokeniser.py [--max-web N] [--max-code N] [--max-email N]

Defaults stream ~50k web docs + 20k Python files + 10k emails (~few hundred MB).
Expect 30-60 minutes on a modern CPU.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32_000
SAVE_PATH = Path(__file__).parent / "tokenizer.json"

# Defined now so they never need retrofitting into a trained tokenizer.
SPECIALS = ["<|pad|>", "<|user|>", "<|assistant|>", "<|end|>"]

# GPT-4-style split pattern.  The critical arm for Python is \s+(?!\S):
# whitespace runs that are NOT followed by a non-whitespace character stay
# as a single pre-token, letting BPE learn merged indent tokens ("    ")
# instead of four separate single-space tokens.
SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)

# ---------------------------------------------------------------------------
# Corpus iterators
# ---------------------------------------------------------------------------


def iter_web(n: int):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= n:
            break
        yield ex["text"]


def iter_code(n: int):
    from datasets import load_dataset

    # code_search_net is public and Python-only (func_code_string field).
    # Fallback to flytech/python-codes-25k if the primary is unavailable.
    for repo, split, get_text in [
        (
            "code_search_net",
            "train",
            lambda ex: ex.get("func_code_string", ""),
        ),
        (
            "flytech/python-codes-25k",
            "train",
            lambda ex: ex.get("output", ""),
        ),
    ]:
        try:
            ds = load_dataset(repo, split=split, streaming=True)
            count = 0
            for ex in ds:
                if count >= n:
                    break
                text = get_text(ex)
                if text:
                    count += 1
                    yield text
            return
        except Exception as e:
            print(f"  [warning] {repo}: {e}")

    print("  [warning] could not load any Python code dataset — skipping code")


def iter_email(n: int):
    """Try a few common HuggingFace names for the Enron corpus."""
    from datasets import load_dataset

    candidates = [
        ("keirp/enron-emails", "train", "text"),
        ("Andyrasika/Enron_Email_Dataset", "train", "text"),
        ("snilsson/enron-emails", "train", "text"),
    ]
    for repo, split, key in candidates:
        try:
            ds = load_dataset(repo, split=split, streaming=True)
            for i, ex in enumerate(ds):
                if i >= n:
                    break
                text = ex.get(key) or ex.get("body") or ex.get("message") or ""
                if text:
                    yield text
            return
        except Exception:
            continue

    print("  [warning] no Enron email dataset found — continuing without emails")


def build_corpus(max_web: int, max_code: int, max_email: int):
    total = max_web + max_code + max_email
    print(
        f"  sources: {max_web:,} web docs | {max_code:,} Python files | {max_email:,} emails"
        f"  (~{total:,} documents)"
    )
    it = itertools.chain(iter_web(max_web), iter_code(max_code), iter_email(max_email))
    return it, total


# ---------------------------------------------------------------------------
# Build and train
# ---------------------------------------------------------------------------


def build_tokenizer(max_web: int, max_code: int, max_email: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = None  # byte-level: no unicode normalisation

    # 1. Split on our custom pattern to produce pre-tokens.
    # 2. ByteLevel converts each pre-token to its byte representation.
    #    use_regex=False prevents ByteLevel from applying its own GPT-2 regex
    #    on top of the Split we already did.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=SPLIT_PATTERN, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )

    it, total = build_corpus(max_web, max_code, max_email)

    # Write corpus to a temp file then call trainer.train(files) instead of
    # train_from_iterator.  On WSL2 the Rust/Rayon thread-pool inside
    # train_from_iterator can hit a glibc pthread priority assertion; the
    # file-based path avoids that code path entirely.
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    print(f"Writing corpus to {tmp.name} …")
    n_docs = 0
    for text in it:
        tmp.write(text.replace("\x00", "") + "\n")
        n_docs += 1
        if n_docs % 1000 == 0:
            print(f"  {n_docs:,} docs written", end="\r")
    tmp.close()
    print(f"\n  total: {n_docs:,} documents")

    print("Training BPE …")
    tokenizer.train([tmp.name], trainer=trainer)
    os.unlink(tmp.name)
    return tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train byte-level BPE tokenizer (Day 3)")
    p.add_argument("--max-web",   type=int, default=50_000, help="web documents to stream")
    p.add_argument("--max-code",  type=int, default=20_000, help="Python files to stream")
    p.add_argument("--max-email", type=int, default=10_000, help="emails to stream")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tok = build_tokenizer(args.max_web, args.max_code, args.max_email)
    tok.save(str(SAVE_PATH))
    vocab_size = tok.get_vocab_size()
    print(f"\nSaved → {SAVE_PATH}  (vocab size: {vocab_size:,})")
    assert vocab_size == VOCAB_SIZE, f"Expected {VOCAB_SIZE}, got {vocab_size}"
