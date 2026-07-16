"""
Day 3 checkpoint tests.

Run:  pytest tokeniser/test_tokenizer.py -v

All tests require tokeniser/tokenizer.json to exist.
If it doesn't, run tokeniser/tokeniser.py first.
"""

from pathlib import Path

import pytest
from tokenizers import Tokenizer

TOKENIZER_PATH = Path(__file__).parent / "tokenizer.json"

# Mirrors SPECIALS in tokeniser.py — keep in sync.
SPECIALS = [
    "<|pad|>", "<|bos|>", "<|eos|>",
    "<|user|>", "<|assistant|>", "<|endofturn|>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
    "<|text|>", "<|email|>", "<|py|>", "<|rs|>", "<|cpp|>",
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    if not TOKENIZER_PATH.exists():
        pytest.skip("tokenizer.json not found — run `python tokeniser/tokeniser.py` first")
    return Tokenizer.from_file(str(TOKENIZER_PATH))


# ---------------------------------------------------------------------------
# Round-trip samples
# ---------------------------------------------------------------------------

PROSE = (
    "The quick brown fox jumps over the lazy dog. "
    "Natural language processing enables machines to understand human language."
)

PYTHON_CODE = """\
def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
"""

EMAIL_WITH_URL = """\
From: alice@example.com
To: bob@example.com
Subject: Re: Q3 planning notes

Hi Bob,

Please review the draft at https://docs.example.com/q3-planning?team=ml&draft=true#section-2

Best,
Alice
"""


def round_trip(tok: Tokenizer, text: str) -> str:
    return tok.decode(tok.encode(text).ids)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prose_round_trip(tok):
    assert round_trip(tok, PROSE) == PROSE


def test_python_round_trip(tok):
    result = round_trip(tok, PYTHON_CODE)
    assert result == PYTHON_CODE, "Python code (including indentation) was mangled by the tokenizer"


def test_email_url_round_trip(tok):
    assert round_trip(tok, EMAIL_WITH_URL) == EMAIL_WITH_URL


def test_special_tokens_present(tok):
    vocab = tok.get_vocab()
    for token in SPECIALS:
        assert token in vocab, f"Missing special token: {token}"


def test_vocab_size(tok):
    assert tok.get_vocab_size() == 32_000


def test_python_indentation_not_exploded(tok):
    """One level of Python indentation (4 spaces) should not split into 4 tokens.

    A byte-level tokenizer without the \\s+(?!\\S) pre-tokeniser arm would
    encode each space independently.  With the arm in place, BPE can merge
    the whitespace run into 1–2 tokens.
    """
    four_spaces = "    "
    n_tokens = len(tok.encode(four_spaces).ids)
    assert n_tokens <= 3, (
        f"4 spaces encoded into {n_tokens} tokens — indentation is exploding.\n"
        "Check ByteLevel(use_regex=False) and that the \\s+(?!\\S) arm is in SPLIT_PATTERN."
    )
