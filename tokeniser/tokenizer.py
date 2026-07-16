"""
Day 4 — the application layer over the trained tokenizer.

The rest of the project imports THIS, never the raw `tokenizers` object:

    from tokenizer import Codec
    codec = Codec("~/LLM/tokenizer/tokenizer.json")

`Codec` owns three jobs:
  1. encode / decode, with named special-token ids
  2. document formatting (pretrain wrapper, chat wrapper, FIM transform)
  3. streaming a corpus to a uint16 .bin memmap (used on days 5-7)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from tokenizers import Tokenizer

# Domain -> the tag token prepended to every pretraining document.
DOMAIN_TAGS = {
    "text": "<|text|>",
    "email": "<|email|>",
    "py": "<|py|>",
    "rs": "<|rs|>",
    "cpp": "<|cpp|>",
}

# uint16 ceiling. Any id at or above this corrupts the .bin silently.
UINT16_MAX = 65536


class Codec:
    def __init__(self, path: str | Path):
        self.tok = Tokenizer.from_file(str(Path(path).expanduser()))
        self.vocab_size = self.tok.get_vocab_size()

        # Hard guarantee: the whole corpus is about to be written as uint16.
        # If the vocab ever exceeds 65535, catch it here, not after a 40-hour
        # run silently wrapped every high token id around to a low one.
        if self.vocab_size > UINT16_MAX:
            raise ValueError(
                f"vocab {self.vocab_size} > {UINT16_MAX}; uint16 corpus unsafe"
            )

        self._id = self._resolver()
        self.bos = self._id("<|bos|>")
        self.eos = self._id("<|eos|>")
        self.pad = self._id("<|pad|>")
        self.user = self._id("<|user|>")
        self.assistant = self._id("<|assistant|>")
        self.endofturn = self._id("<|endofturn|>")
        self.fim_prefix = self._id("<|fim_prefix|>")
        self.fim_middle = self._id("<|fim_middle|>")
        self.fim_suffix = self._id("<|fim_suffix|>")
        self.tag_id = {d: self._id(t) for d, t in DOMAIN_TAGS.items()}

    def _resolver(self):
        vocab = self.tok.get_vocab()

        def resolve(token: str) -> int:
            if token not in vocab:
                raise KeyError(f"special token {token!r} missing from tokenizer")
            return vocab[token]

        return resolve

    # -- core -------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        # skip_special_tokens=False so round-trip tests can see the wrappers.
        return self.tok.decode(ids, skip_special_tokens=False)

    # -- document formatting ---------------------------------------------
    def encode_document(self, text: str, domain: str) -> list[int]:
        """Pretraining wrapper:  <|bos|> <|domain|> ...text... <|eos|>"""
        if domain not in self.tag_id:
            raise KeyError(f"unknown domain {domain!r}")
        return [self.bos, self.tag_id[domain], *self.encode(text), self.eos]

    def encode_chat(self, instruction: str, response: str):
        """SFT wrapper. Returns (ids, loss_mask). Mask is 1 only on the
        assistant span + its closing turn -- train on the response, never the
        prompt. Defined now because the day-4.4 anneal phase must match it."""
        head = [self.bos, self.user, *self.encode(instruction),
                self.endofturn, self.assistant]
        tail = [*self.encode(response), self.endofturn, self.eos]
        ids = head + tail
        mask = [0] * len(head) + [1] * len(tail)
        return ids, mask

    def encode_fim(self, code: str, rng: random.Random | None = None) -> list[int]:
        """Fill-in-the-middle transform for code. Two random cut points split
        the source into prefix / middle / suffix; the model sees prefix+suffix
        and learns to produce the middle. PSM ordering."""
        rng = rng or random
        n = len(code)
        if n < 3:
            return self.encode(code)
        a, b = sorted(rng.sample(range(n + 1), 2))
        prefix, middle, suffix = code[:a], code[a:b], code[b:]
        return [
            self.bos, self.fim_prefix, *self.encode(prefix),
            self.fim_suffix, *self.encode(suffix),
            self.fim_middle, *self.encode(middle), self.eos,
        ]

    # -- corpus -> disk (days 5-7 use this) -------------------------------
    def write_bin(self, docs: Iterable[tuple[str, str]], out_path: str | Path,
                  report_every: int = 100_000) -> int:
        """Stream (text, domain) pairs to a flat uint16 file. Returns total
        tokens. Appends in chunks so memory stays flat regardless of corpus
        size."""
        out_path = Path(out_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with open(out_path, "wb") as f:
            for i, (text, domain) in enumerate(docs):
                ids = self.encode_document(text, domain)
                arr = np.array(ids, dtype=np.uint16)
                # paranoia: assert nothing overflowed on the way to uint16
                if arr.max(initial=0) >= UINT16_MAX:
                    raise ValueError("token id >= 65536 slipped through")
                f.write(arr.tobytes())
                total += len(ids)
                if report_every and i and i % report_every == 0:
                    print(f"  {i:,} docs, {total/1e6:.1f}M tokens", flush=True)
        return total


def iter_jsonl(path: str | Path) -> Iterator[tuple[str, str]]:
    """Helper: yield (text, domain) from a tagged .jsonl."""
    import json
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield row["text"], row.get("tag", "text")
