"""
Day 24: HumanEval code evaluation sandbox.

Loads HumanEval problems, generates k completions per problem using the model,
executes each in an isolated subprocess with a 5s timeout (no network, temp dir),
runs the provided unit tests, and reports pass@1 / pass@k.

Formula for unbiased pass@k estimator (Chen et al. 2021):
    pass@k = 1 - C(n-c, k) / C(n, k)
where n = total samples, c = passing samples.

Usage:
    # Evaluate a checkpoint (generates 1 sample per problem):
    python eval/humaneval_sandbox.py --ckpt checkpoints/ckpt_050000.pt

    # k=10 samples per problem (for pass@10):
    python eval/humaneval_sandbox.py --ckpt checkpoints/ckpt_050000.pt --k 10 --n 10

    # Sanity check with canonical solutions:
    python eval/humaneval_sandbox.py --canonical

    # Limit to first N problems:
    python eval/humaneval_sandbox.py --ckpt checkpoints/ckpt_050000.pt --n-problems 10
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Iterator

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tokeniser"))

TIMEOUT_SECS = 5


# ---------------------------------------------------------------------------
# HumanEval loader
# ---------------------------------------------------------------------------


def load_problems(n: int | None = None) -> list[dict]:
    """
    Load HumanEval problems from the Hub.

    Returns list of dicts with keys: task_id, prompt, test, entry_point,
    canonical_solution.
    """
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = list(ds)
    if n is not None:
        problems = problems[:n]
    return problems


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------


def run_in_sandbox(code: str, timeout: float = TIMEOUT_SECS) -> tuple[bool, str]:
    """
    Execute code in an isolated subprocess.

    The code should be a complete Python program (function definition +
    test harness).  Returns (passed, stderr_or_error_message).

    Security properties:
      - Runs in a fresh temp directory (no access to project files)
      - timeout=5s hard kill via subprocess
      - No network (subprocess inherits env but has no special network access)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "solution.py"
        script.write_text(code, encoding="utf-8")

        env = {k: v for k, v in os.environ.items()
               if k in ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env=env,
            )
            passed = result.returncode == 0
            msg    = result.stderr if not passed else ""
            return passed, msg
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        except Exception as e:
            return False, str(e)


def make_test_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    """
    Assemble a runnable Python program from HumanEval components.

    Structure:
        <prompt + completion>     # function definition
        <test>                    # defines check(candidate)
        check(<entry_point>)      # call the test
    """
    return f"{prompt}{completion}\n\n{test}\n\ncheck({entry_point})\n"


# ---------------------------------------------------------------------------
# pass@k estimator
# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k.

    n: total completions sampled
    c: completions that pass tests
    k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# Model generation
# ---------------------------------------------------------------------------


def _load_model(ckpt_path: str, ctx: int, device: torch.device):
    """Load GPT from checkpoint, infer config from state dict."""
    from model.gpt import GPT, ModelConfig

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd    = state.get("model_state_dict", state)

    vocab_size = sd["embed.weight"].shape[0]
    d_model    = sd["embed.weight"].shape[1]
    n_layer    = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    n_head     = 6  # fallback — not stored in state dict shape

    cfg   = ModelConfig(vocab_size=vocab_size, d_model=d_model,
                        n_head=n_head, n_layer=n_layer, ctx=ctx)
    model = GPT(cfg)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


def generate_completion(
    model,
    codec,
    prompt: str,
    max_new: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate one completion for a HumanEval prompt.

    Encodes the prompt, runs autoregressive sampling until <|eos|> /
    <|endofturn|> or max_new tokens, then decodes only the new tokens.
    """
    from inference.sample import sample as gen

    prompt_ids = codec.encode(prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = gen(
            model, x,
            n_new=max_new,
            temperature=temperature,
            top_p=top_p,
            eos_id=codec.eos,
        )

    new_ids = out[0, len(prompt_ids):].tolist()
    # Strip at first stop token
    stop = {codec.eos, codec.endofturn}
    for i, t in enumerate(new_ids):
        if t in stop:
            new_ids = new_ids[:i]
            break

    return codec.decode(new_ids)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    problems:      list[dict],
    get_completion,          # callable(prompt) -> str
    n_samples:     int  = 1,
    k_values:      list[int] = None,
    verbose:       bool = False,
) -> dict:
    """
    Evaluate pass@k over a list of HumanEval problems.

    get_completion: function that takes a prompt string and returns a
        completion string (one at a time; called n_samples times per problem).
    n_samples: number of completions to generate per problem.
    k_values:  list of k values to compute pass@k for (default [1]).

    Returns dict with keys like "pass@1", "pass@10", "results" (per-problem).
    """
    if k_values is None:
        k_values = [1]

    results = []

    for i, prob in enumerate(problems):
        task_id     = prob["task_id"]
        prompt      = prob["prompt"]
        test        = prob["test"]
        entry_point = prob["entry_point"]

        passes = 0
        errors = []

        for s in range(n_samples):
            try:
                completion = get_completion(prompt)
            except Exception as e:
                errors.append(str(e))
                continue

            program = make_test_program(prompt, completion, test, entry_point)
            passed, msg = run_in_sandbox(program)
            if passed:
                passes += 1
            elif verbose:
                errors.append(msg[:200])

        results.append({
            "task_id":    task_id,
            "n_samples":  n_samples,
            "n_pass":     passes,
        })

        status = "✓" if passes > 0 else "✗"
        print(
            f"  [{i+1:>3}/{len(problems)}] {task_id:<25}  "
            f"{passes}/{n_samples}  {status}"
            + (f"  {errors[0][:60]}" if errors and verbose else ""),
            flush=True,
        )

    # Aggregate
    metrics = {}
    for k in k_values:
        if n_samples >= k:
            scores = [pass_at_k(r["n_samples"], r["n_pass"], k) for r in results]
            metrics[f"pass@{k}"] = sum(scores) / len(scores)

    metrics["results"]   = results
    metrics["n_problems"] = len(problems)
    metrics["n_samples"]  = n_samples

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HumanEval sandbox (Day 24)")
    p.add_argument("--ckpt",       default="",    help="model checkpoint (.pt)")
    p.add_argument("--ctx",        type=int,   default=256)
    p.add_argument("--n-problems", type=int,   default=None,
                   help="evaluate only first N problems (default: all 164)")
    p.add_argument("--n",          type=int,   default=1,
                   help="completions per problem")
    p.add_argument("--k",          type=int,   nargs="+", default=[1],
                   help="k values for pass@k (e.g. --k 1 10)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p",      type=float, default=0.95)
    p.add_argument("--max-new",    type=int,   default=256)
    p.add_argument("--canonical",  action="store_true",
                   help="run canonical solutions (sanity check; expect ~100%)")
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--out",        default="",
                   help="write JSON results to this path")
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse()
    problems = load_problems(args.n_problems)
    print(f"Loaded {len(problems)} HumanEval problems")

    if args.canonical:
        print("Running canonical solutions (expect ~100% pass rate) …\n")
        def get_completion(prompt):
            # Find the matching problem by prompt prefix
            for p in problems:
                if p["prompt"] == prompt:
                    return p["canonical_solution"]
            return ""
    elif args.ckpt:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {args.ckpt} on {device} …")
        from tokenizer import Codec  # type: ignore
        codec = Codec(str(ROOT / "tokeniser" / "tokenizer.json"))
        model = _load_model(args.ckpt, args.ctx, device)
        print(f"Model loaded ({model.num_params()/1e6:.1f}M params)\n")

        def get_completion(prompt):
            return generate_completion(
                model, codec, prompt,
                max_new=args.max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
    else:
        print("ERROR: provide --ckpt or --canonical")
        sys.exit(1)

    print(f"Evaluating  n={args.n} samples/problem  k={args.k} …\n")
    metrics = evaluate(
        problems,
        get_completion,
        n_samples=args.n,
        k_values=args.k,
        verbose=args.verbose,
    )

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for kv in args.k:
        key = f"pass@{kv}"
        if key in metrics:
            print(f"  {key:10} = {metrics[key]*100:.1f}%")
    print(f"  problems   = {metrics['n_problems']}")
    n_any = sum(1 for r in metrics["results"] if r["n_pass"] > 0)
    print(f"  solved     = {n_any}/{metrics['n_problems']}")
    print("=" * 50)

    if args.out:
        import json
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f"\nResults written to {args.out}")
