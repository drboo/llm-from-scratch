"""
Day 4 unit tests. Run:  pytest tokeniser/test_codec.py -v
(or just `python tokeniser/test_codec.py` to run without pytest)
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

# tokenizer.py lives next to this file; add the directory so the bare
# `import tokenizer` below resolves regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import Codec, DOMAIN_TAGS, iter_jsonl  # noqa: E402

TOK_PATH = Path(__file__).parent / "tokenizer.json"


@pytest.fixture(scope="module")
def c() -> Codec:
    if not TOK_PATH.exists():
        pytest.skip("tokenizer.json not found — run `python tokeniser/tokeniser.py` first")
    try:
        return Codec(str(TOK_PATH))
    except KeyError as e:
        pytest.skip(f"tokenizer missing special token {e} — retrain with updated SPECIALS")


def _strip(codec: Codec, ids: list[int]) -> list[int]:
    """Drop every special-token id, leaving only content tokens."""
    specials = {
        codec.bos, codec.eos, codec.pad,
        codec.user, codec.assistant, codec.endofturn,
        codec.fim_prefix, codec.fim_middle, codec.fim_suffix,
        *codec.tag_id.values(),
    }
    return [i for i in ids if i not in specials]


def test_uint16_safe(c):
    assert c.vocab_size <= 65536


def test_specials_are_single_ids(c):
    for name in ("bos", "eos", "pad", "endofturn"):
        assert isinstance(getattr(c, name), int)
    for d in DOMAIN_TAGS:
        assert d in c.tag_id


def test_document_roundtrip(c):
    for src in ["def f(x):\n    return x\n", "Hi Tom,\n\nSee attached.\n", "just text"]:
        ids = c.encode_document(src, "py")
        assert ids[0] == c.bos and ids[-1] == c.eos
        assert ids[1] == c.tag_id["py"]
        assert c.decode(_strip(c, ids)) == src


def test_tag_selects_domain(c):
    assert c.encode_document("x", "email")[1] == c.tag_id["email"]
    assert c.encode_document("x", "py")[1] == c.tag_id["py"]


def test_chat_mask_covers_response_only(c):
    ids, mask = c.encode_chat("write a function", "def f(): pass")
    assert len(ids) == len(mask)
    assert set(mask) <= {0, 1}
    assert mask[0] == 0               # bos is not a training target
    assert mask[-1] == 1              # eos of the response IS
    first_one = mask.index(1)
    assert ids[first_one] != c.assistant  # mask starts AFTER the assistant tag


def test_fim_reassembles_source(c):
    rng = random.Random(0)
    src = "fn main() {\n    let x = 5;\n    println!(\"{}\", x);\n}\n"
    ids = c.encode_fim(src, rng)
    p = ids.index(c.fim_prefix)
    s = ids.index(c.fim_suffix)
    m = ids.index(c.fim_middle)
    prefix = c.decode(_strip(c, ids[p + 1:s]))
    suffix = c.decode(_strip(c, ids[s + 1:m]))
    middle = c.decode(_strip(c, ids[m + 1:]))
    assert prefix + middle + suffix == src


def test_write_bin_roundtrips_through_memmap(c, tmp_path):
    docs = [("def f(): return 1\n", "py"), ("Hello there.\n", "text")]
    path = tmp_path / "train.bin"
    total = c.write_bin(docs, path, report_every=0)
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    assert len(arr) == total
    assert int(arr.max()) < 65536
    assert arr[0] == c.bos and arr[1] == c.tag_id["py"]
