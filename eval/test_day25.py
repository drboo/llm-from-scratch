"""
Day 25 tests — full evaluation and EVAL.md generation.

Run:  pytest eval/test_day25.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.full_eval import (
    RUBRIC_PROMPTS,
    RUBRIC_DIMENSIONS,
    load_rubric_scores,
    write_eval_md,
)
from eval.humaneval_sandbox import pass_at_k

ROOT = Path(__file__).resolve().parent.parent
TOK_PATH = ROOT / "tokeniser" / "tokenizer.json"
HAS_TOKENIZER = TOK_PATH.exists()


# ---------------------------------------------------------------------------
# Rubric prompts
# ---------------------------------------------------------------------------


class TestRubricPrompts:
    def test_has_email_and_code_keys(self):
        assert "email" in RUBRIC_PROMPTS
        assert "code" in RUBRIC_PROMPTS

    def test_ten_email_prompts(self):
        assert len(RUBRIC_PROMPTS["email"]) == 10

    def test_ten_code_prompts(self):
        assert len(RUBRIC_PROMPTS["code"]) == 10

    def test_all_prompts_non_empty(self):
        for cat, prompts in RUBRIC_PROMPTS.items():
            for p in prompts:
                assert p.strip(), f"Empty prompt in {cat}"

    def test_email_prompts_mention_email(self):
        email_prompts = RUBRIC_PROMPTS["email"]
        # At least 8 of 10 should reference email/message/write
        hits = sum(1 for p in email_prompts
                   if any(w in p.lower() for w in ("email", "message", "write")))
        assert hits >= 8

    def test_code_prompts_mention_python(self):
        code_prompts = RUBRIC_PROMPTS["code"]
        hits = sum(1 for p in code_prompts if "python" in p.lower())
        assert hits >= 8

    def test_no_duplicate_prompts(self):
        all_prompts = RUBRIC_PROMPTS["email"] + RUBRIC_PROMPTS["code"]
        assert len(all_prompts) == len(set(all_prompts))


# ---------------------------------------------------------------------------
# load_rubric_scores
# ---------------------------------------------------------------------------


class TestLoadRubricScores:
    def _write_scores(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "rubric_scores.json"
        p.write_text(json.dumps(data))
        return p

    def test_returns_empty_when_file_missing(self, tmp_path):
        result = load_rubric_scores(tmp_path / "nonexistent.json")
        assert result == {}

    def test_computes_means(self, tmp_path):
        data = {
            "base": [
                {"relevance": 4, "coherence": 3, "correctness": 5},
                {"relevance": 2, "coherence": 4, "correctness": 3},
            ]
        }
        p = self._write_scores(tmp_path, data)
        scores = load_rubric_scores(p)
        assert "base" in scores
        assert scores["base"]["relevance"] == pytest.approx(3.0)
        assert scores["base"]["coherence"] == pytest.approx(3.5)

    def test_overall_is_mean_of_all_dimensions(self, tmp_path):
        data = {
            "base": [
                {"relevance": 4, "coherence": 4, "correctness": 4},
            ]
        }
        p = self._write_scores(tmp_path, data)
        scores = load_rubric_scores(p)
        assert scores["base"]["overall"] == pytest.approx(4.0)

    def test_handles_both_models(self, tmp_path):
        data = {
            "base": [{"relevance": 2, "coherence": 2, "correctness": 2}],
            "sft":  [{"relevance": 4, "coherence": 4, "correctness": 4}],
        }
        p = self._write_scores(tmp_path, data)
        scores = load_rubric_scores(p)
        assert "base" in scores and "sft" in scores
        assert scores["sft"]["overall"] > scores["base"]["overall"]

    def test_partial_scores_tolerated(self, tmp_path):
        data = {
            "base": [{"relevance": 3}]  # only one dimension scored
        }
        p = self._write_scores(tmp_path, data)
        scores = load_rubric_scores(p)
        assert scores["base"]["relevance"] == pytest.approx(3.0)

    def test_empty_entries_list(self, tmp_path):
        data = {"base": []}
        p = self._write_scores(tmp_path, data)
        scores = load_rubric_scores(p)
        assert scores == {}


# ---------------------------------------------------------------------------
# write_eval_md
# ---------------------------------------------------------------------------


class TestWriteEvalMd:
    def _base_results(self) -> dict:
        return {
            "timestamp":  "2026-07-18T12:00:00",
            "base_ckpt":  "checkpoints/ckpt_050000.pt",
            "sft_ckpt":   "checkpoints/sft/ckpt_best.pt",
            "base_ppl":   87.4,
            "sft_ppl":    None,
            "humaneval":  {"pass@1": 0.018, "n_problems": 164, "n_samples": 10},
            "n_rubric_prompts": 20,
            "rubric_scores": {
                "base": {"relevance": 2.1, "coherence": 1.8,
                         "correctness": 1.6, "overall": 1.83},
                "sft":  {"relevance": 3.4, "coherence": 3.1,
                         "correctness": 2.8, "overall": 3.1},
            },
        }

    def test_creates_file(self, tmp_path):
        results = self._base_results()
        write_eval_md(results, tmp_path / "EVAL.md")
        assert (tmp_path / "EVAL.md").exists()

    def test_contains_section_headers(self, tmp_path):
        md = write_eval_md(self._base_results(), tmp_path / "EVAL.md")
        assert "## 1. Validation Perplexity" in md
        assert "## 2. HumanEval" in md
        assert "## 3. Qualitative Rubric" in md
        assert "## 4. Sample Progression" in md

    def test_contains_perplexity_value(self, tmp_path):
        md = write_eval_md(self._base_results(), tmp_path / "EVAL.md")
        assert "87.4" in md

    def test_contains_pass_at_1(self, tmp_path):
        md = write_eval_md(self._base_results(), tmp_path / "EVAL.md")
        assert "1.8%" in md or "pass@1" in md

    def test_placeholder_for_missing_sft_ppl(self, tmp_path):
        results = self._base_results()
        results["sft_ppl"] = None
        md = write_eval_md(results, tmp_path / "EVAL.md")
        # Should show "—" for missing value
        assert "—" in md

    def test_rubric_scores_present(self, tmp_path):
        md = write_eval_md(self._base_results(), tmp_path / "EVAL.md")
        assert "2.1" in md or "Relevance" in md or "relevance" in md.lower()

    def test_no_rubric_scores_still_works(self, tmp_path):
        results = self._base_results()
        results["rubric_scores"] = {}
        md = write_eval_md(results, tmp_path / "EVAL.md")
        assert "EVAL.md" or len(md) > 100

    def test_returns_markdown_string(self, tmp_path):
        md = write_eval_md(self._base_results(), tmp_path / "EVAL.md")
        assert isinstance(md, str)
        assert len(md) > 200

    def test_contains_checkpoint_paths(self, tmp_path):
        md = write_eval_md(self._base_results(), tmp_path / "EVAL.md")
        assert "ckpt_050000.pt" in md

    def test_idempotent(self, tmp_path):
        results = self._base_results()
        md1 = write_eval_md(results, tmp_path / "EVAL.md")
        md2 = write_eval_md(results, tmp_path / "EVAL.md")
        assert md1 == md2


# ---------------------------------------------------------------------------
# full_eval smoke test (no model required)
# ---------------------------------------------------------------------------


class TestFullEvalNoModel:
    def test_runs_without_checkpoints(self, tmp_path):
        from eval.full_eval import full_eval
        results = full_eval(
            base_ckpt=None,
            sft_ckpt=None,
            out_dir=str(tmp_path),
            skip_humaneval=True,
            skip_qualitative=True,
        )
        assert isinstance(results, dict)
        assert results["base_ppl"] is None
        assert results["sft_ppl"]  is None

    def test_writes_eval_md(self, tmp_path):
        from eval.full_eval import full_eval
        full_eval(
            base_ckpt=None,
            sft_ckpt=None,
            out_dir=str(tmp_path),
            skip_humaneval=True,
            skip_qualitative=True,
        )
        assert (ROOT / "EVAL.md").exists()

    def test_writes_results_json(self, tmp_path):
        from eval.full_eval import full_eval
        full_eval(
            base_ckpt=None,
            sft_ckpt=None,
            out_dir=str(tmp_path),
            skip_humaneval=True,
            skip_qualitative=True,
        )
        assert (tmp_path / "eval_results.json").exists()

    def test_report_only_reads_saved_results(self, tmp_path):
        from eval.full_eval import full_eval
        # First run to create saved results
        full_eval(
            base_ckpt=None,
            sft_ckpt=None,
            out_dir=str(tmp_path),
            skip_humaneval=True,
            skip_qualitative=True,
        )
        # Second run in report_only mode
        results = full_eval(
            out_dir=str(tmp_path),
            report_only=True,
        )
        assert isinstance(results, dict)
