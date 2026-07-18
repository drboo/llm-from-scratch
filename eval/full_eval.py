"""
Day 25: Full evaluation + report.

Orchestrates all evaluation dimensions and writes EVAL.md:

  1. Val perplexity   — base pretrain + SFT checkpoint (via eval/perplexity.py)
  2. pass@1 / pass@10 — HumanEval subset via sandbox (eval/humaneval_sandbox.py)
  3. Qualitative rubric — 20 prompts (10 email, 10 code), responses saved to
     eval/rubric_outputs.json for human review; scores in eval/rubric_scores.json
     feed back into EVAL.md automatically when present

Usage:
    # Full run (assumes training is done):
    python eval/full_eval.py \\
        --base-ckpt  checkpoints/ckpt_050000.pt \\
        --sft-ckpt   checkpoints/sft/ckpt_best.pt \\
        --data-dir   data/real \\
        --out-dir    eval/results

    # Skip heavy evaluations to regenerate EVAL.md from saved results:
    python eval/full_eval.py --report-only --out-dir eval/results

    # Dry-run with tiny HumanEval subset:
    python eval/full_eval.py --base-ckpt checkpoints/ckpt_050000.pt \\
        --n-he-problems 5 --n-he-samples 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tokeniser"))

# ---------------------------------------------------------------------------
# Qualitative rubric prompts
# ---------------------------------------------------------------------------

RUBRIC_PROMPTS: dict[str, list[str]] = {
    "email": [
        "Write a short professional email declining a Friday afternoon meeting.",
        "Write an email requesting a one-week deadline extension on a project report.",
        "Write a follow-up email to a client who hasn't responded in two weeks.",
        "Write a thank-you email to a colleague who helped debug a critical issue.",
        "Write a brief email announcing a team lunch next Thursday at noon.",
        "Write an email to introduce a new engineer joining the team on Monday.",
        "Write an apology email for sending incorrect data in a previous report.",
        "Write an email requesting feedback on a design document you shared.",
        "Write an out-of-office auto-reply for a two-week vacation.",
        "Write an email notifying users of a scheduled 2-hour maintenance window.",
    ],
    "code": [
        "Write a Python function to compute the greatest common divisor of two numbers.",
        "Write a Python function that returns True if a string contains only unique characters.",
        "Write a Python function to rotate a list left by k positions.",
        "Write a Python context manager that suppresses a specific exception type.",
        "Write a Python function implementing a simple LRU cache using an OrderedDict.",
        "Write a Python function to parse a URL into its components (scheme, host, path, query).",
        "Write a Python generator that yields prime numbers up to n.",
        "Write a Python function to compute the edit distance between two strings.",
        "Write a Python class for a min-heap with push and pop operations.",
        "Write a Python function that retries a callable up to n times on exception.",
    ],
}

RUBRIC_DIMENSIONS = ["relevance", "coherence", "correctness"]


# ---------------------------------------------------------------------------
# Qualitative response generation
# ---------------------------------------------------------------------------


def generate_qualitative(
    base_ckpt: str | None,
    sft_ckpt:  str | None,
    ctx:       int = 256,
    max_new:   int = 300,
    temperature: float = 0.8,
    top_p:     float = 0.95,
) -> dict[str, list[dict]]:
    """
    Generate model responses to the 20 rubric prompts.

    Returns {"base": [...], "sft": [...]} where each entry is
    {prompt, category, response, model}.  If a checkpoint is unavailable,
    response is left as "".
    """
    results: dict[str, list[dict]] = {"base": [], "sft": []}

    for model_key, ckpt_path in [("base", base_ckpt), ("sft", sft_ckpt)]:
        if not ckpt_path or not Path(ckpt_path).exists():
            for cat, prompts in RUBRIC_PROMPTS.items():
                for prompt in prompts:
                    results[model_key].append(
                        {"prompt": prompt, "category": cat,
                         "response": "", "model": model_key}
                    )
            continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[qualitative] Loading {model_key} from {ckpt_path} …")

        from tokenizer import Codec  # type: ignore
        from eval.humaneval_sandbox import _load_model
        from inference.sample import sample as gen

        codec = Codec(str(ROOT / "tokeniser" / "tokenizer.json"))
        model = _load_model(ckpt_path, ctx, device)

        for cat, prompts in RUBRIC_PROMPTS.items():
            for i, prompt in enumerate(prompts):
                print(f"  [{model_key}] {cat} {i+1}/10 …", end="\r", flush=True)
                try:
                    if model_key == "sft":
                        # Use chat template for SFT model
                        head = [codec.bos, codec.user,
                                *codec.encode(prompt),
                                codec.endofturn, codec.assistant]
                        x = torch.tensor([head], dtype=torch.long, device=device)
                        stop_ids = {codec.endofturn, codec.eos}
                    else:
                        # Base model: just encode the prompt text
                        prompt_ids = codec.encode(prompt)
                        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                        stop_ids = {codec.eos}

                    with torch.no_grad():
                        out = gen(model, x, n_new=max_new,
                                  temperature=temperature, top_p=top_p,
                                  eos_id=codec.eos)

                    new_ids = out[0, x.shape[1]:].tolist()
                    for j, t in enumerate(new_ids):
                        if t in stop_ids:
                            new_ids = new_ids[:j]
                            break
                    response = codec.decode(new_ids)
                except Exception as e:
                    response = f"[ERROR: {e}]"

                results[model_key].append(
                    {"prompt": prompt, "category": cat,
                     "response": response, "model": model_key}
                )

        print(f"\n  {model_key}: {len(results[model_key])} responses generated")

    return results


# ---------------------------------------------------------------------------
# Perplexity helper (wraps eval/perplexity.py)
# ---------------------------------------------------------------------------


def compute_ppl(ckpt_path: str, data_dir: str, ctx: int = 256,
                n_batches: int = 50) -> float | None:
    if not ckpt_path or not Path(ckpt_path).exists():
        return None
    try:
        from eval.perplexity import compute_perplexity, _load_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = _load_model(ckpt_path, n_head=6, ctx=ctx, device=device)
        ppl = compute_perplexity(model, data_dir, "val", ctx,
                                  batch_size=8, n_batches=n_batches, device=device)
        return round(ppl, 2)
    except Exception as e:
        print(f"  [ppl] {e}")
        return None


# ---------------------------------------------------------------------------
# HumanEval helper
# ---------------------------------------------------------------------------


def compute_humaneval(
    ckpt_path:  str,
    ctx:        int  = 256,
    n_problems: int  = 164,
    n_samples:  int  = 1,
    k_values:   list[int] = None,
) -> dict | None:
    if not ckpt_path or not Path(ckpt_path).exists():
        return None
    if k_values is None:
        k_values = [v for v in [1, 10] if v <= n_samples]
    try:
        from eval.humaneval_sandbox import load_problems, evaluate, _load_model
        from tokenizer import Codec  # type: ignore
        from eval.humaneval_sandbox import generate_completion

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[humaneval] Loading model from {ckpt_path} …")
        codec  = Codec(str(ROOT / "tokeniser" / "tokenizer.json"))
        model  = _load_model(ckpt_path, ctx, device)
        probs  = load_problems(n_problems)

        def get_completion(prompt):
            return generate_completion(model, codec, prompt, device=device)

        return evaluate(probs, get_completion, n_samples=n_samples,
                        k_values=k_values)
    except Exception as e:
        print(f"  [humaneval] {e}")
        return None


# ---------------------------------------------------------------------------
# Load rubric scores (human-supplied)
# ---------------------------------------------------------------------------


def load_rubric_scores(scores_path: Path) -> dict[str, dict[str, float]]:
    """
    Load manually-entered rubric scores from a JSON file.

    Expected format:
        {
          "base":  [{"prompt": "...", "relevance": 3, "coherence": 2, "correctness": 2}, ...],
          "sft":   [...]
        }

    Returns {"base": {"relevance": mean, "coherence": mean, "correctness": mean, "overall": mean},
             "sft":  {...}}
    """
    if not scores_path.exists():
        return {}
    raw = json.loads(scores_path.read_text())
    out = {}
    for model_key, entries in raw.items():
        if not entries:
            continue
        agg: dict[str, list[float]] = {d: [] for d in RUBRIC_DIMENSIONS}
        for e in entries:
            for d in RUBRIC_DIMENSIONS:
                if d in e:
                    agg[d].append(float(e[d]))
        means = {d: (sum(vs)/len(vs) if vs else 0.0) for d, vs in agg.items()}
        all_scores = [v for vs in agg.values() for v in vs]
        means["overall"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
        out[model_key] = means
    return out


# ---------------------------------------------------------------------------
# EVAL.md writer
# ---------------------------------------------------------------------------


def write_eval_md(
    results: dict,
    out_path: Path,
) -> str:
    """
    Write EVAL.md from a results dict.  Placeholder text for missing values.
    """
    def _fmt(v, fmt=".2f"):
        return f"{v:{fmt}}" if v is not None else "—"

    base_ppl = results.get("base_ppl")
    sft_ppl  = results.get("sft_ppl")
    he       = results.get("humaneval") or {}
    rubric   = results.get("rubric_scores") or {}
    rubric_b = rubric.get("base", {})
    rubric_s = rubric.get("sft",  {})
    n_rubric = results.get("n_rubric_prompts", 20)
    ts       = results.get("timestamp", datetime.now().isoformat()[:19])
    base_ckpt = results.get("base_ckpt", "—")
    sft_ckpt  = results.get("sft_ckpt",  "—")

    pass1  = he.get("pass@1")
    pass10 = he.get("pass@10")
    n_he_probs  = he.get("n_problems", "—")
    n_he_samples = he.get("n_samples", "—")

    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {ts}",
        "",
        "## Checkpoints",
        "",
        f"| Role | Path |",
        f"|------|------|",
        f"| Base pretrain | `{base_ckpt}` |",
        f"| SFT           | `{sft_ckpt}` |",
        "",
        "---",
        "",
        "## 1. Validation Perplexity",
        "",
        "Lower is better. Random-init baseline ≈ vocab_size (~32,000).",
        "",
        "| Model | Val PPL |",
        "|-------|---------|",
        f"| Base  | {_fmt(base_ppl)} |",
        f"| SFT   | {_fmt(sft_ppl)} |",
        "",
        "> SFT val PPL is computed on the SFT val split, not the pretrain val split.",
        "",
        "---",
        "",
        "## 2. HumanEval Code Evaluation",
        "",
        f"Problems: {n_he_probs}  |  Samples per problem: {n_he_samples}",
        "",
        "| Metric   | Score |",
        "|----------|-------|",
        f"| pass@1   | {_fmt(pass1, '.1%') if pass1 is not None else '—'} |",
        f"| pass@10  | {_fmt(pass10, '.1%') if pass10 is not None else '—'} |",
        "",
        "> Nano-scale baseline: pass@1 of 0–3% is normal. GPT-2 class achieves ~2–5%.",
        "",
        "---",
        "",
        "## 3. Qualitative Rubric",
        "",
        f"{n_rubric} prompts (10 email, 10 code) scored 1–5 on three dimensions.",
        "Responses saved in `eval/results/rubric_outputs.json`.",
        "To add scores: edit `eval/results/rubric_scores.json`, then re-run with `--report-only`.",
        "",
        "| Dimension    | Base | SFT |",
        "|--------------|------|-----|",
    ]

    for dim in RUBRIC_DIMENSIONS + ["overall"]:
        b = rubric_b.get(dim)
        s = rubric_s.get(dim)
        lines.append(
            f"| {dim.capitalize():<12} | {_fmt(b, '.2f')} | {_fmt(s, '.2f')} |"
        )

    lines += [
        "",
        "Scale: 1 = poor, 3 = adequate, 5 = excellent.",
        "",
        "---",
        "",
        "## 4. Sample Progression",
        "",
        "Generated text at training checkpoints shows the noise→words→grammar arc:",
        "",
        "| Step   | Sample excerpt |",
        "|--------|---------------|",
    ]

    samples_dir = ROOT / "checkpoints" / "samples"
    if samples_dir.exists():
        sample_files = sorted(samples_dir.glob("step_*.txt"))
        checkpoints = [0, len(sample_files)//4, len(sample_files)//2,
                       3*len(sample_files)//4, len(sample_files)-1]
        seen = set()
        for idx in checkpoints:
            if idx < 0 or idx >= len(sample_files) or idx in seen:
                continue
            seen.add(idx)
            sf = sample_files[idx]
            step = sf.stem.replace("step_", "")
            text = sf.read_text()
            # Extract just the generated portion (after the header line)
            body = "\n".join(text.splitlines()[1:]).strip()
            excerpt = body[:120].replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {step} | {excerpt}… |")
    else:
        lines.append("| — | *(sample files not found — run pretrain with --sample-prompt)* |")

    lines += [
        "",
        "---",
        "",
        "## 5. Key Takeaways",
        "",
        "- **Perplexity**: fill in once checkpoints are available.",
        "- **HumanEval**: nano-scale; pass@k is expected to be low.",
        "  Infrastructure (sandbox, unbiased estimator) is the deliverable.",
        "- **SFT effect**: compare base vs SFT on the same email prompt —",
        "  the SFT model should produce structured emails and stop at `<|endofturn|>`.",
        "- **Next**: KV cache (Day 26) for real-time generation speed.",
        "",
    ]

    md = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    return md


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def full_eval(
    base_ckpt:    str | None = None,
    sft_ckpt:     str | None = None,
    data_dir:     str = "data/real",
    out_dir:      str = "eval/results",
    n_he_problems: int = 164,
    n_he_samples:  int = 1,
    k_values:      list[int] = None,
    ctx:           int = 256,
    report_only:   bool = False,
    skip_humaneval: bool = False,
    skip_qualitative: bool = False,
) -> dict:
    if k_values is None:
        k_values = [v for v in [1, 10] if v <= n_he_samples]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "eval_results.json"
    outputs_path = out_dir / "rubric_outputs.json"
    scores_path  = out_dir / "rubric_scores.json"

    if report_only and results_path.exists():
        results = json.loads(results_path.read_text())
        print("Loaded saved results — regenerating EVAL.md …")
    else:
        results: dict = {
            "timestamp": datetime.now().isoformat()[:19],
            "base_ckpt": base_ckpt or "—",
            "sft_ckpt":  sft_ckpt  or "—",
        }

        # 1. Perplexity
        print("\n[1/3] Val perplexity …")
        if base_ckpt:
            print(f"  base: {base_ckpt}")
            results["base_ppl"] = compute_ppl(base_ckpt, data_dir, ctx)
            print(f"  base PPL = {results['base_ppl']}")
        else:
            results["base_ppl"] = None

        if sft_ckpt:
            # SFT val perplexity uses SFT data dir
            sft_data = str(Path(sft_ckpt).parent.parent / "sft") \
                if not Path("data/sft").exists() else "data/sft"
            print(f"  sft:  {sft_ckpt}")
            results["sft_ppl"] = compute_ppl(sft_ckpt, sft_data, ctx)
            print(f"  sft PPL  = {results['sft_ppl']}")
        else:
            results["sft_ppl"] = None

        # 2. HumanEval
        if not skip_humaneval and base_ckpt:
            print(f"\n[2/3] HumanEval ({n_he_problems} problems, {n_he_samples} samples) …")
            he = compute_humaneval(base_ckpt, ctx, n_he_problems, n_he_samples, k_values)
            results["humaneval"] = he
        else:
            results["humaneval"] = None

        # 3. Qualitative
        if not skip_qualitative:
            print("\n[3/3] Qualitative rubric (20 prompts) …")
            rubric_outputs = generate_qualitative(base_ckpt, sft_ckpt, ctx)
            results["n_rubric_prompts"] = sum(
                len(v) for v in RUBRIC_PROMPTS.values()
            )
            outputs_path.write_text(json.dumps(rubric_outputs, indent=2))
            print(f"  Responses saved to {outputs_path}")
            print(f"  Score them and save to {scores_path} to include in EVAL.md")
        else:
            results["n_rubric_prompts"] = sum(len(v) for v in RUBRIC_PROMPTS.values())

        results_path.write_text(json.dumps(results, indent=2))

    # Load rubric scores if available
    results["rubric_scores"] = load_rubric_scores(scores_path)

    # Write EVAL.md
    eval_md_path = ROOT / "EVAL.md"
    md = write_eval_md(results, eval_md_path)
    print(f"\nEVAL.md written to {eval_md_path}")
    print(md[:600] + "\n…")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full evaluation + report (Day 25)")
    p.add_argument("--base-ckpt",     default="", help="base pretrain checkpoint")
    p.add_argument("--sft-ckpt",      default="", help="SFT checkpoint")
    p.add_argument("--data-dir",      default="data/real")
    p.add_argument("--out-dir",       default="eval/results")
    p.add_argument("--n-he-problems", type=int, default=164)
    p.add_argument("--n-he-samples",  type=int, default=1)
    p.add_argument("--k",             type=int, nargs="+", default=[1])
    p.add_argument("--ctx",           type=int, default=256)
    p.add_argument("--report-only",   action="store_true",
                   help="skip evals, regenerate EVAL.md from saved results")
    p.add_argument("--skip-humaneval",    action="store_true")
    p.add_argument("--skip-qualitative",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    full_eval(
        base_ckpt      = args.base_ckpt or None,
        sft_ckpt       = args.sft_ckpt  or None,
        data_dir       = args.data_dir,
        out_dir        = args.out_dir,
        n_he_problems  = args.n_he_problems,
        n_he_samples   = args.n_he_samples,
        k_values       = args.k,
        ctx            = args.ctx,
        report_only    = args.report_only,
        skip_humaneval    = args.skip_humaneval,
        skip_qualitative  = args.skip_qualitative,
    )
