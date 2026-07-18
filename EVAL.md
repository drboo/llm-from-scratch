# Evaluation Report

Generated: 2026-07-18T11:41:36

## Checkpoints

| Role | Path |
|------|------|
| Base pretrain | `—` |
| SFT           | `—` |

---

## 1. Validation Perplexity

Lower is better. Random-init baseline ≈ vocab_size (~32,000).

| Model | Val PPL |
|-------|---------|
| Base  | — |
| SFT   | — |

> SFT val PPL is computed on the SFT val split, not the pretrain val split.

---

## 2. HumanEval Code Evaluation

Problems: —  |  Samples per problem: —

| Metric   | Score |
|----------|-------|
| pass@1   | — |
| pass@10  | — |

> Nano-scale baseline: pass@1 of 0–3% is normal. GPT-2 class achieves ~2–5%.

---

## 3. Qualitative Rubric

20 prompts (10 email, 10 code) scored 1–5 on three dimensions.
Responses saved in `eval/results/rubric_outputs.json`.
To add scores: edit `eval/results/rubric_scores.json`, then re-run with `--report-only`.

| Dimension    | Base | SFT |
|--------------|------|-----|
| Relevance    | — | — |
| Coherence    | — | — |
| Correctness  | — | — |
| Overall      | — | — |

Scale: 1 = poor, 3 = adequate, 5 = excellent.

---

## 4. Sample Progression

Generated text at training checkpoints shows the noise→words→grammar arc:

| Step   | Sample excerpt |
|--------|---------------|
| — | *(sample files not found — run pretrain with --sample-prompt)* |

---

## 5. Key Takeaways

- **Perplexity**: fill in once checkpoints are available.
- **HumanEval**: nano-scale; pass@k is expected to be low.
  Infrastructure (sandbox, unbiased estimator) is the deliverable.
- **SFT effect**: compare base vs SFT on the same email prompt —
  the SFT model should produce structured emails and stop at `<|endofturn|>`.
- **Next**: KV cache (Day 26) for real-time generation speed.
