"""
Day 24 tests — HumanEval sandbox.

Run:  pytest eval/test_day24.py -v

Network-dependent tests (load_problems) are skipped when offline.
Model-dependent tests are skipped when tokenizer.json is absent.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.humaneval_sandbox import (
    run_in_sandbox,
    make_test_program,
    pass_at_k,
    evaluate,
)

ROOT = Path(__file__).resolve().parent.parent
TOK_PATH = ROOT / "tokeniser" / "tokenizer.json"
HAS_TOKENIZER = TOK_PATH.exists()


# ---------------------------------------------------------------------------
# run_in_sandbox
# ---------------------------------------------------------------------------


class TestRunInSandbox:
    def test_passing_code_returns_true(self):
        passed, _ = run_in_sandbox("x = 1 + 1\nassert x == 2\n")
        assert passed

    def test_failing_code_returns_false(self):
        passed, _ = run_in_sandbox("assert 1 == 2\n")
        assert not passed

    def test_syntax_error_returns_false(self):
        passed, _ = run_in_sandbox("def broken(\n")
        assert not passed

    def test_exception_returns_false(self):
        passed, msg = run_in_sandbox("raise ValueError('oops')\n")
        assert not passed
        assert "ValueError" in msg or msg  # some error message present

    def test_timeout_returns_false(self):
        passed, msg = run_in_sandbox("while True: pass\n", timeout=1.0)
        assert not passed
        assert "Timeout" in msg or "timeout" in msg.lower()

    def test_empty_code_passes(self):
        passed, _ = run_in_sandbox("")
        assert passed

    def test_correct_function_passes(self):
        code = (
            "def add(a, b):\n"
            "    return a + b\n\n"
            "assert add(2, 3) == 5\n"
            "assert add(-1, 1) == 0\n"
        )
        passed, _ = run_in_sandbox(code)
        assert passed

    def test_wrong_function_fails(self):
        code = (
            "def add(a, b):\n"
            "    return a - b\n\n"  # wrong implementation
            "assert add(2, 3) == 5\n"
        )
        passed, _ = run_in_sandbox(code)
        assert not passed

    def test_infinite_import_blocked(self):
        # socket is available, but subprocess has restricted env
        code = (
            "import socket\n"
            "s = socket.socket()\n"
            "# just creating a socket should not crash\n"
        )
        # This should at least not hang — pass or fail is acceptable
        passed, msg = run_in_sandbox(code, timeout=2.0)
        assert isinstance(passed, bool)

    def test_returns_error_message_on_failure(self):
        _, msg = run_in_sandbox("raise RuntimeError('test error')\n")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_multiline_output_ignored(self):
        # stdout output should not affect pass/fail
        code = "for i in range(10):\n    print(i)\n"
        passed, _ = run_in_sandbox(code)
        assert passed


# ---------------------------------------------------------------------------
# make_test_program
# ---------------------------------------------------------------------------


class TestMakeTestProgram:
    def test_assembles_runnable_program(self):
        prompt = "def add(a, b):\n"
        completion = "    return a + b\n"
        test = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
        entry_point = "add"
        program = make_test_program(prompt, completion, test, entry_point)
        passed, _ = run_in_sandbox(program)
        assert passed

    def test_bad_completion_fails(self):
        prompt = "def add(a, b):\n"
        completion = "    return a * b\n"  # wrong
        test = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
        program = make_test_program(prompt, completion, test, "add")
        passed, _ = run_in_sandbox(program)
        assert not passed

    def test_program_contains_all_parts(self):
        program = make_test_program(
            "def f():\n", "    pass\n",
            "def check(c): pass\n", "f"
        )
        assert "def f():" in program
        assert "def check" in program
        assert "check(f)" in program


# ---------------------------------------------------------------------------
# pass_at_k
# ---------------------------------------------------------------------------


class TestPassAtK:
    def test_all_pass(self):
        assert pass_at_k(10, 10, 1) == pytest.approx(1.0)

    def test_none_pass(self):
        assert pass_at_k(10, 0, 1) == pytest.approx(0.0)

    def test_pass_at_1_half(self):
        # With n=2, c=1, k=1: 1 - C(1,1)/C(2,1) = 1 - 1/2 = 0.5
        assert pass_at_k(2, 1, 1) == pytest.approx(0.5)

    def test_n_minus_c_less_than_k_returns_1(self):
        # If we have more passes than needed to guarantee k passes, result = 1
        assert pass_at_k(5, 4, 3) == pytest.approx(1.0)

    def test_increases_with_more_passes(self):
        p1 = pass_at_k(10, 2, 1)
        p2 = pass_at_k(10, 5, 1)
        p3 = pass_at_k(10, 8, 1)
        assert p1 < p2 < p3

    def test_pass_at_k_increases_with_k(self):
        # More chances should give higher (or equal) pass rate
        p1 = pass_at_k(10, 3, 1)
        p3 = pass_at_k(10, 3, 3)
        assert p3 >= p1

    def test_returns_float(self):
        result = pass_at_k(10, 5, 1)
        assert isinstance(result, float)

    def test_between_zero_and_one(self):
        for n in range(1, 20):
            for c in range(n + 1):
                for k in range(1, n + 1):
                    r = pass_at_k(n, c, k)
                    assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# evaluate (uses synthetic problems, no network)
# ---------------------------------------------------------------------------


class TestEvaluate:
    def _make_problems(self, n: int = 3) -> list[dict]:
        """Synthetic HumanEval-shaped problems for offline testing."""
        return [
            {
                "task_id":          f"Test/{i}",
                "prompt":           f"def add_{i}(a, b):\n",
                "entry_point":      f"add_{i}",
                "canonical_solution": f"    return a + b\n",
                "test": (
                    f"def check(candidate):\n"
                    f"    assert candidate({i}, 1) == {i+1}\n"
                    f"    assert candidate(0, 0) == 0\n"
                ),
            }
            for i in range(n)
        ]

    def test_canonical_solutions_pass(self):
        problems = self._make_problems(3)
        def get_correct(prompt):
            for p in problems:
                if p["prompt"] == prompt:
                    return p["canonical_solution"]
            return "    pass\n"

        metrics = evaluate(problems, get_correct, n_samples=1, k_values=[1])
        assert metrics["pass@1"] == pytest.approx(1.0)

    def test_wrong_solutions_fail(self):
        problems = self._make_problems(3)
        def get_wrong(_prompt):
            return "    return None\n"

        metrics = evaluate(problems, get_wrong, n_samples=1, k_values=[1])
        assert metrics["pass@1"] == pytest.approx(0.0)

    def test_returns_per_problem_results(self):
        problems = self._make_problems(2)
        def get_correct(prompt):
            for p in problems:
                if p["prompt"] == prompt:
                    return p["canonical_solution"]
            return "    pass\n"

        metrics = evaluate(problems, get_correct, n_samples=1)
        assert "results" in metrics
        assert len(metrics["results"]) == 2

    def test_n_problems_in_results(self):
        problems = self._make_problems(4)
        metrics = evaluate(problems, lambda _: "    return None\n",
                           n_samples=1, k_values=[1])
        assert metrics["n_problems"] == 4

    def test_multiple_k_values(self):
        problems = self._make_problems(5)
        def get_half(_prompt):
            import random
            return "    return a + b\n" if random.random() > 0.5 else "    pass\n"

        metrics = evaluate(problems, get_half, n_samples=5, k_values=[1, 5])
        assert "pass@1" in metrics
        assert "pass@5" in metrics

    def test_pass_at_k_not_computed_when_n_lt_k(self):
        problems = self._make_problems(2)
        metrics = evaluate(problems, lambda _: "    return None\n",
                           n_samples=1, k_values=[1, 10])
        assert "pass@1"  in metrics
        assert "pass@10" not in metrics   # n=1 < k=10, should be skipped

    def test_generator_exception_handled(self):
        problems = self._make_problems(2)
        def exploding(_prompt):
            raise RuntimeError("model broken")

        metrics = evaluate(problems, exploding, n_samples=1, k_values=[1])
        assert metrics["pass@1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Canonical solution smoke test (requires network)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not pytest.importorskip("datasets", reason="datasets not installed"),
    reason="datasets not installed",
)
class TestCanonicalSolutions:
    def test_first_problem_canonical_passes(self):
        try:
            from eval.humaneval_sandbox import load_problems
            problems = load_problems(n=1)
        except Exception:
            pytest.skip("network unavailable")

        prob = problems[0]
        program = make_test_program(
            prob["prompt"],
            prob["canonical_solution"],
            prob["test"],
            prob["entry_point"],
        )
        passed, msg = run_in_sandbox(program)
        assert passed, f"Canonical solution failed: {msg}"

    def test_empty_solution_fails(self):
        try:
            from eval.humaneval_sandbox import load_problems
            problems = load_problems(n=1)
        except Exception:
            pytest.skip("network unavailable")

        prob = problems[0]
        program = make_test_program(
            prob["prompt"],
            "    pass\n",       # trivially wrong
            prob["test"],
            prob["entry_point"],
        )
        passed, _ = run_in_sandbox(program)
        assert not passed
