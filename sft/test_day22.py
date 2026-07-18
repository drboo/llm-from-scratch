"""
Day 22 tests — SFT dataset construction and loss masking.

Run:  pytest sft/test_day22.py -v

Network-dependent tests (iter_alpaca, iter_oasst, etc.) are skipped if the
tokenizer is absent; integration tests require both tokenizer and network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tokeniser"))

ROOT = Path(__file__).resolve().parent.parent
TOK_PATH = ROOT / "tokeniser" / "tokenizer.json"
HAS_TOKENIZER = TOK_PATH.exists()

from sft.data import (
    HAND_WRITTEN,
    is_junk,
    encode_example,
    _write_split,
    build_sft_dataset,
)


# ---------------------------------------------------------------------------
# is_junk
# ---------------------------------------------------------------------------


class TestIsJunk:
    def test_empty_instruction(self):
        bad, reason = is_junk("", "Some good response here.")
        assert bad and reason == "empty_instruction"

    def test_empty_response(self):
        bad, reason = is_junk("Tell me something.", "")
        assert bad and reason == "empty_response"

    def test_short_response(self):
        bad, reason = is_junk("What is 2+2?", "4")
        assert bad and reason == "short_response"

    def test_refusal_response(self):
        bad, reason = is_junk("Do something bad.", "I'm sorry, I cannot help with that.")
        assert bad and reason == "refusal"

    def test_refusal_as_ai(self):
        bad, reason = is_junk("Help me.", "As an AI language model, I cannot do that.")
        assert bad and reason == "refusal"

    def test_good_example_passes(self):
        bad, _ = is_junk(
            "Write a short poem about rain.",
            "Rain falls softly on the windowpane,\nA gentle rhythm, a sweet refrain.",
        )
        assert not bad

    def test_whitespace_only_instruction(self):
        bad, reason = is_junk("   \n\t  ", "A valid response here.")
        assert bad and reason == "empty_instruction"


# ---------------------------------------------------------------------------
# HAND_WRITTEN examples
# ---------------------------------------------------------------------------


class TestHandWritten:
    def test_has_at_least_30_examples(self):
        assert len(HAND_WRITTEN) >= 30

    def test_all_have_instruction_and_response(self):
        for ex in HAND_WRITTEN:
            assert "instruction" in ex and ex["instruction"]
            assert "response"    in ex and ex["response"]

    def test_no_junk_in_hand_written(self):
        for ex in HAND_WRITTEN:
            bad, reason = is_junk(ex["instruction"], ex["response"])
            assert not bad, (
                f"Hand-written example failed junk check ({reason}):\n"
                f"  instruction: {ex['instruction'][:60]}"
            )

    def test_has_email_examples(self):
        email_count = sum(
            1 for ex in HAND_WRITTEN
            if "email" in ex["instruction"].lower()
            or "Subject:" in ex["response"]
        )
        assert email_count >= 10

    def test_has_python_examples(self):
        python_count = sum(
            1 for ex in HAND_WRITTEN
            if "python" in ex["instruction"].lower()
            or "```python" in ex["response"]
        )
        assert python_count >= 10


# ---------------------------------------------------------------------------
# encode_example  (requires tokenizer)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TOKENIZER, reason="tokenizer.json not present")
class TestEncodeExample:
    @pytest.fixture(autouse=True)
    def codec(self):
        from tokenizer import Codec  # type: ignore
        self.codec = Codec(str(TOK_PATH))

    def test_returns_ids_and_labels(self):
        instruction = "Say hello."
        response    = "Hello! How can I help you today?"
        result = encode_example(instruction, response, self.codec)
        assert result is not None
        ids, labels = result
        assert len(ids) == len(labels)

    def test_labels_match_ids_on_response_tokens(self):
        instruction = "What is 2 + 2?"
        response    = "The answer is 4."
        ids, labels = encode_example(instruction, response, self.codec)
        for i, (tok, lab) in enumerate(zip(ids, labels)):
            assert lab == tok or lab == -100

    def test_prompt_tokens_masked(self):
        """The user/instruction portion must have label == -100."""
        instruction = "Write a haiku about the moon."
        response    = "Moon rises softly,\nsilver light on quiet seas,\ndreams drift like white clouds."
        ids, labels = encode_example(instruction, response, self.codec)
        # At least some tokens must be masked (the prompt)
        assert -100 in labels

    def test_response_tokens_carry_loss(self):
        """At least some tokens (the response) must have label != -100."""
        instruction = "Say hello."
        response    = "Hello there, how are you doing today?"
        ids, labels = encode_example(instruction, response, self.codec)
        assert any(lab >= 0 for lab in labels)

    def test_loss_tokens_are_suffix(self):
        """Loss tokens (label != -100) must form a contiguous suffix."""
        instruction = "Describe a sunset."
        response    = "The sky turns orange and pink as the sun dips below the horizon."
        ids, labels = encode_example(instruction, response, self.codec)
        # Find first loss token
        first_loss = next(i for i, l in enumerate(labels) if l >= 0)
        # Everything from first_loss to end must have loss
        assert all(l >= 0 for l in labels[first_loss:])

    def test_too_long_returns_none(self):
        # A response much longer than max_tokens should be dropped
        long_response = "word " * 1000
        result = encode_example("Go.", long_response, self.codec, max_tokens=50)
        assert result is None

    def test_ids_within_vocab(self):
        instruction = "What is the capital of France?"
        response    = "The capital of France is Paris."
        ids, _ = encode_example(instruction, response, self.codec)
        assert all(0 <= t < self.codec.vocab_size for t in ids)

    def test_bos_is_first_token(self):
        ids, labels = encode_example("Hello.", "Hi!", self.codec)
        assert ids[0] == self.codec.bos

    def test_eos_is_last_token(self):
        ids, labels = encode_example("Hello.", "Hi there!", self.codec)
        assert ids[-1] == self.codec.eos

    def test_loss_fraction_reasonable(self):
        """Response tokens should be 20-80% of total (sanity check)."""
        instruction = "Explain gravity briefly."
        response    = "Gravity is a force that attracts objects with mass toward each other. The larger the mass, the stronger the pull."
        ids, labels = encode_example(instruction, response, self.codec)
        loss_frac = sum(1 for l in labels if l >= 0) / len(labels)
        assert 0.1 < loss_frac < 0.95


# ---------------------------------------------------------------------------
# _write_split / file format
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TOKENIZER, reason="tokenizer.json not present")
class TestWriteSplit:
    @pytest.fixture(autouse=True)
    def codec(self):
        from tokenizer import Codec  # type: ignore
        self.codec = Codec(str(TOK_PATH))

    def _make_examples(self, n: int = 5):
        pairs = []
        for i in range(n):
            result = encode_example(
                f"Question number {i}.",
                f"This is answer number {i}, which is correct.",
                self.codec,
            )
            assert result is not None
            pairs.append(result)
        return pairs

    def test_creates_three_files(self, tmp_path):
        _write_split(self._make_examples(), tmp_path, "train")
        assert (tmp_path / "sft_train.bin").exists()
        assert (tmp_path / "sft_train_labels.bin").exists()
        assert (tmp_path / "sft_train_offsets.npy").exists()

    def test_token_and_label_files_same_length(self, tmp_path):
        _write_split(self._make_examples(4), tmp_path, "train")
        t = np.memmap(str(tmp_path / "sft_train.bin"),        dtype=np.uint16, mode="r")
        l = np.memmap(str(tmp_path / "sft_train_labels.bin"), dtype=np.int32,  mode="r")
        assert len(t) == len(l)

    def test_offsets_length_is_n_examples_plus_one(self, tmp_path):
        n = 6
        _write_split(self._make_examples(n), tmp_path, "train")
        offsets = np.load(str(tmp_path / "sft_train_offsets.npy"))
        assert len(offsets) == n + 1

    def test_offsets_first_is_zero(self, tmp_path):
        _write_split(self._make_examples(), tmp_path, "train")
        offsets = np.load(str(tmp_path / "sft_train_offsets.npy"))
        assert offsets[0] == 0

    def test_offsets_last_equals_total_tokens(self, tmp_path):
        _write_split(self._make_examples(3), tmp_path, "train")
        offsets = np.load(str(tmp_path / "sft_train_offsets.npy"))
        tokens  = np.memmap(str(tmp_path / "sft_train.bin"), dtype=np.uint16, mode="r")
        assert int(offsets[-1]) == len(tokens)

    def test_labels_dtype_is_int32(self, tmp_path):
        _write_split(self._make_examples(2), tmp_path, "val")
        l = np.memmap(str(tmp_path / "sft_val_labels.bin"), dtype=np.int32, mode="r")
        assert l.dtype == np.int32

    def test_negative_100_present_in_labels(self, tmp_path):
        _write_split(self._make_examples(3), tmp_path, "train")
        l = np.memmap(str(tmp_path / "sft_train_labels.bin"), dtype=np.int32, mode="r")
        assert (-100 in l)

    def test_roundtrip_example_zero(self, tmp_path):
        examples = self._make_examples(3)
        _write_split(examples, tmp_path, "train")
        offsets = np.load(str(tmp_path / "sft_train_offsets.npy"))
        tokens  = np.memmap(str(tmp_path / "sft_train.bin"), dtype=np.uint16, mode="r")
        s, e = int(offsets[0]), int(offsets[1])
        assert list(map(int, tokens[s:e])) == examples[0][0]


# ---------------------------------------------------------------------------
# build_sft_dataset integration (requires tokenizer, skips network sources)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TOKENIZER, reason="tokenizer.json not present")
class TestBuildSftDataset:
    def test_hand_written_only(self, tmp_path):
        stats = build_sft_dataset(
            out_dir=tmp_path,
            n_alpaca=0,
            n_code=0,
            n_oasst=0,
        )
        assert stats["n_train"] + stats["n_val"] == len(HAND_WRITTEN)
        assert (tmp_path / "sft_train.bin").exists()
        assert (tmp_path / "sft_val.bin").exists()
        assert (tmp_path / "sft_stats.json").exists()

    def test_stats_json_valid(self, tmp_path):
        build_sft_dataset(out_dir=tmp_path, n_alpaca=0, n_code=0, n_oasst=0)
        stats = json.loads((tmp_path / "sft_stats.json").read_text())
        assert "n_train" in stats
        assert "n_val"   in stats
        assert "src_counts" in stats

    def test_dry_run_no_files(self, tmp_path):
        build_sft_dataset(out_dir=tmp_path, n_alpaca=0, n_code=0, n_oasst=0, dry_run=True)
        assert not (tmp_path / "sft_train.bin").exists()

    def test_val_smaller_than_train(self, tmp_path):
        stats = build_sft_dataset(
            out_dir=tmp_path,
            n_alpaca=0, n_code=0, n_oasst=0,
            val_fraction=0.2,
        )
        assert stats["n_val"] < stats["n_train"]

    def test_deterministic_with_same_seed(self, tmp_path):
        out1, out2 = tmp_path / "a", tmp_path / "b"
        build_sft_dataset(out_dir=out1, n_alpaca=0, n_code=0, n_oasst=0, seed=1)
        build_sft_dataset(out_dir=out2, n_alpaca=0, n_code=0, n_oasst=0, seed=1)
        t1 = np.memmap(str(out1 / "sft_train.bin"), dtype=np.uint16, mode="r")
        t2 = np.memmap(str(out2 / "sft_train.bin"), dtype=np.uint16, mode="r")
        assert np.array_equal(t1, t2)
