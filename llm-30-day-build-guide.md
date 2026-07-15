# LLM From Scratch — 30-Day Build Guide

A day-by-day tutorial companion to `llm-from-scratch-spec.md`. Each day has a **goal**,
**tasks**, a **deliverable**, and a **checkpoint test** — don't move on until the
checkpoint passes. Days assume ~2–4 focused hours. If a day runs long, let it spill over;
the order matters more than the calendar.

**Gaps this guide plugs vs. the spec:** environment setup, code-level detail (RoPE,
loss masking, memmap dataloaders, KV cache), debugging checkpoints at every stage,
compute budgeting, and troubleshooting.

---

## WEEK 1 — Foundations, Tokenizer, and the Model Skeleton

### Day 1 — Environment + repo setup

**Goal:** A working dev environment and empty-but-structured repo.

- Create the repo with the layout from the spec (`data/`, `tokenizer/`, `model/`, `train/`, `eval/`, `inference/`, `configs/`).
- Python 3.11 venv; install: `torch`, `tokenizers`, `datasets`, `numpy`, `wandb` (or use TensorBoard), `pytest`.
- Verify GPU: `torch.cuda.is_available()`, note VRAM (`nvidia-smi`). Record it — it sets your batch sizes later.
- Write `configs/nano.yaml` from the spec's ModelConfig (n_layer=6, n_head=6, d_model=384, ctx=256).
- Git init, first commit, `.gitignore` for data/checkpoints.

**Deliverable:** repo skeleton + `python -c "import torch; print(torch.cuda.get_device_name())"` works.
**Checkpoint:** a `pytest` run passes on one trivial placeholder test.

### Day 2 — Theory day (yes, really)

**Goal:** Know what you're about to build before building it.

- Watch Karpathy's "Let's build GPT from scratch" (2h) — take notes, don't copy code.
- Skim _Attention Is All You Need_ §3 (architecture only) and the LLaMA-1 paper §2 (RoPE/RMSNorm/SwiGLU rationale).
- On paper, draw the full forward pass for one token batch: shapes at every step, from `(B, T)` int IDs to `(B, T, vocab)` logits.

**Deliverable:** one page of hand-drawn shape-annotated architecture.
**Checkpoint:** you can explain _why_ attention is O(T²) and what the causal mask does, out loud, without notes.

### Day 3 — Tokenizer

**Goal:** Train a byte-level BPE tokenizer.

- Grab a small mixed corpus (~100–500 MB): a slice of OpenWebText/FineWeb + some Python files from The Stack sample + a few thousand Enron emails.
- Train byte-level BPE with HuggingFace `tokenizers`, vocab_size=32000. Add special tokens now: `<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>` — retro-fitting them later is painful.
- Write `tokenizer/train_tokenizer.py` and save `tokenizer.json`.

**Deliverable:** `tokenizer.json` + round-trip test.
**Checkpoint:** `decode(encode(s)) == s` for prose, Python code (check indentation survives!), and an email with a URL. Inspect tokens of a Python snippet — indentation should not explode into dozens of tokens.

### Day 4 — Embeddings, RMSNorm, and the causal mask

**Goal:** First model components, each unit-tested.

- `model/norm.py`: RMSNorm — `x * w / sqrt(mean(x²) + eps)`. Test: output has ~unit RMS.
- `model/model.py`: token embedding `(vocab, d_model)`, init normal(0, 0.02).
- Causal mask: `torch.tril(torch.ones(T, T))` → additive `-inf` mask. Test: position i can't see position j > i.

**Deliverable:** components + passing unit tests.
**Checkpoint:** all tests green; you can state why pre-norm beats post-norm (gradient flow through residuals).

### Day 5 — RoPE + attention

**Goal:** The heart of the model.

- `model/rope.py`: precompute `theta_i = 10000^(-2i/d_head)`; rotate Q,K per-position by angle `m·theta_i` (pairwise dimension rotation). Test: RoPE of position 0 is identity; relative property `⟨RoPE(q,m), RoPE(k,n)⟩` depends only on `m−n`.
- `model/attention.py`: causal multi-head attention. Project to Q,K,V, reshape to `(B, n_head, T, d_head)`, apply RoPE to Q,K, scaled dot-product with causal mask, merge heads, output projection. Use `F.scaled_dot_product_attention` (flash attention for free) but ALSO write the manual version once — you're here to learn.

**Deliverable:** attention module, both implementations agree within 1e-5.
**Checkpoint:** shape test `(B,T,d_model) → (B,T,d_model)`; causality test — perturbing token t+1 does not change output at token t.

### Day 6 — SwiGLU FFN + the transformer block

**Goal:** Complete block: `x = x + attn(norm(x)); x = x + ffn(norm(x))`.

- SwiGLU: `W_down( silu(W_gate·x) ⊙ (W_up·x) )` with hidden dim ≈ (2/3)·4·d_model to match GELU param count.
- Assemble `model/block.py` with the two residual connections.

**Deliverable:** working block.
**Checkpoint:** stack 6 blocks, forward a random batch — no NaNs, activations don't blow up layer to layer (print RMS per layer).

### Day 7 — Full model + weight tying

**Goal:** End-to-end forward pass and loss.

- Assemble: embed → blocks×N → final RMSNorm → linear head (weight tied to embedding).
- Loss: `F.cross_entropy(logits.view(-1, vocab), targets.view(-1))` where targets are inputs shifted by one.
- Param count function. Scaled residual init: multiply output-proj weights by `1/sqrt(2·n_layer)`.

**Deliverable:** `model.py` producing loss on random data.
**Checkpoint:** **initial loss ≈ ln(32000) ≈ 10.4.** If it's far off, your head/loss wiring is wrong. Param count ≈ 10–15M for nano.

---

## WEEK 2 — Training Loop and Nano Pretraining

### Day 8 — Dataloader (memmap)

**Goal:** Fast, simple data feeding.

- `data/prepare_toy.py`: tokenize ~50–100 MB of text into one long uint16 array, save `train.bin` / `val.bin` (90/10 split) via `np.memmap`.
- Batch sampler: pick random offsets `i`, take `x = data[i:i+T]`, `y = data[i+1:i+T+1]`. No padding, no attention masks needed — packed sequences.

**Deliverable:** `get_batch(split)` returning GPU tensors.
**Checkpoint:** decode one batch back to text — it should read as real (concatenated) text, and `y` is `x` shifted by exactly one.

### Day 9 — THE OVERFIT TEST (do not skip)

**Goal:** Prove the model can learn.

- Minimal training loop: AdamW(lr=3e-4), single fixed batch of ~64 sequences, train 500–1000 steps.
- Loss must fall from ~10.4 to **< 0.1**. Then sample from the model — it should regurgitate the memorized batch verbatim.

**Deliverable:** overfit loss curve (screenshot it — it goes in your writeup).
**Checkpoint:** loss < 0.1 and verbatim regurgitation. If not, debug now: commonest bugs are off-by-one targets, mask applied wrong, RoPE on V (it's Q,K only), or forgetting `optimizer.zero_grad()`.

### Day 10 — Real training loop

**Goal:** Production-quality loop.

- Add: cosine LR schedule with linear warmup (2000 steps), grad clip 1.0, **gradient accumulation** (aim ~250k tokens/step: `batch × ctx × accum_steps`), bf16 autocast, periodic val loss, W&B/TensorBoard logging.
- `checkpoint.py`: save/load model + optimizer + step + RNG state. Test resume: train 100 steps, save, restart, loss continues smoothly.

**Deliverable:** `train/pretrain.py` with config-driven everything.
**Checkpoint:** resume-from-checkpoint produces an unbroken loss curve.

### Day 11 — Sampling / generation

**Goal:** Talk to your model.

- `inference/sample.py`: autoregressive loop — forward, take last-position logits, apply temperature, top-k, top-p, sample, append, repeat. (No KV cache yet — naive is fine for now.)
- Generate from your Day-9 overfit checkpoint to verify.

**Deliverable:** `sample.py --prompt "..." --temperature 0.8 --top_k 50`.
**Checkpoint:** temperature 0.1 vs 1.5 visibly changes output diversity.

### Days 12–13 — Nano pretrain run #1

**Goal:** First real training run.

- Train nano on your ~100 MB toy corpus. On a 3090/4090 this is several hours to overnight. Watch: train/val loss both falling, val not diverging (if it does, you're overfitting the small corpus — fine at this stage, note it).
- While it trains: write `eval/perplexity.py` (exp of mean val loss) and start Day 15's data scripts.
- Sample every few thousand steps — watch it go from noise → words → grammar. Save these samples; they're the most satisfying artifact of the project.

**Deliverable:** trained nano checkpoint + loss curves + sample progression.
**Checkpoint:** val perplexity meaningfully below random (< ~100 vs 32000 at init); samples are locally grammatical English.

### Day 14 — Buffer / review day

- Fix whatever's rough. Refactor. Write README so far. If ahead of schedule, read the GQA section of the LLaMA-2 paper and Chinchilla scaling laws (you'll want ~20 tokens per parameter as a rule of thumb — for 15M params, ~300M tokens; for 124M, ~2.5B tokens).

---

## WEEK 3 — The Real Data Pipeline + Scaled Pretraining

### Day 15 — Data acquisition

**Goal:** Assemble the real corpus.

- Download via `datasets` (streaming where possible): a FineWeb/OpenWebText slice (general text backbone, ~60–70% of tokens), Enron corpus (email, ~10%), The Stack **Python-only, permissive licenses** (~20–30%).
- Size target: whatever your Chinchilla math + disk allows. For nano, 300M–1B tokens is plenty; for the 124M stretch, aim 2B+.

**Deliverable:** raw shards on disk, per-source token estimates.
**Checkpoint:** you can state your planned mixture ratios and why.

### Day 16 — Cleaning + dedup

**Goal:** Quality filtering.

- Clean: UTF-8 enforcement, strip HTML boilerplate, drop docs < 100 chars, language-ID filter (fasttext) for the web slice.
- Dedup: exact dedup via hash of normalized text; near-dedup via MinHash/LSH (`datasketch`) on the web+code data. Log how much you removed (often 10–30% — that's normal and good).

**Deliverable:** `data/clean.py`, `data/dedup.py`, cleaned shards.
**Checkpoint:** spot-read 20 random surviving docs — they should all look like content you'd want the model to imitate.

### Day 17 — PII scrubbing + code filtering

**Goal:** Make the email data safe, the code data good.

- Emails: regex-scrub emails addresses, phone numbers, SSNs; replace with placeholder tokens. Strip signature blocks and quoted-reply chains (they're mostly duplication anyway). Consider `presidio` or `scrubadub` for names.
- Code: keep files 50–50k chars, drop generated/minified files (very long lines, low entropy), drop files that are mostly data literals.

**Deliverable:** scrubbed email shard + filtered Python shard.
**Checkpoint:** grep the scrubbed emails for `@` and phone-number patterns — near-zero hits.

### Day 18 — Tokenize + pack the full corpus

**Goal:** Final training binaries.

- Re-train tokenizer on the _real_ mixture (your Day-3 one was on toy data) — or keep it if coverage looks fine; check compression ratio (bytes/token ~4 is healthy).
- Tokenize everything, interleave by your mixture ratios, write `train.bin`/`val.bin`. Hold out val _by document_, not by token offset, to avoid leakage.

**Deliverable:** final `.bin` files + a datasheet (sources, sizes, filters applied, mixture).
**Checkpoint:** decode random windows from `train.bin` — you see prose, then email, then code at roughly the expected ratios.

### Days 19–21 — Main pretraining run

**Goal:** The big nano run (or 124M if you've rented compute).

- Compute budget check first: tokens/sec on your GPU × seconds available ≥ target tokens. If not, shrink the target — a fully-trained nano beats a half-trained small.
- Launch. Babysit the first hour (loss falling smoothly, no spikes), then check twice daily. Loss spikes → lower LR 2× or check for a bad data shard.
- During the run: build Day 22–23 material (SFT data prep, eval sandbox).

**Deliverable:** the base model checkpoint.
**Checkpoint:** samples show domain awareness — prompt with `def ` and it writes Python-shaped code; prompt with `Subject:` and it writes email-shaped text.

---

## WEEK 4 — SFT, Evaluation, Inference

### Day 22 — SFT dataset

**Goal:** Instruction data in your chat template.

- Pull Alpaca + CodeAlpaca (+ a slice of OpenAssistant). Filter junk (empty outputs, refusals, non-English).
- Render into your template: `<|user|>\n{instruction}\n<|assistant|>\n{response}<|end|>`.
- Hand-write 30–50 examples yourself for email-writing and short Python functions — small but disproportionately effective for your two target skills.
- Tokenize with a **loss mask**: label = token id on assistant tokens, `-100` on user/template tokens.

**Deliverable:** `sft_train.bin` + labels, ~20–80k examples.
**Checkpoint:** decode one example and visually verify exactly which tokens carry loss.

### Day 23 — SFT training

**Goal:** From continuer to assistant.

- `train/sft.py`: same loop, but per-example sequences (pad + attention mask, or pack multiple examples with mask boundaries), LR ~1–3e-5, 2–3 epochs, watch val loss for overfitting.
- This is fast — an hour or two at nano scale.

**Deliverable:** instruct checkpoint.
**Checkpoint:** "Write a short email declining a Friday meeting." → it produces an email, and **stops at `<|end|>`** (make sure sampling treats it as EOS). Compare base vs SFT on the same prompt — the difference is the whole lesson.

### Day 24 — Code eval sandbox

**Goal:** Honest code measurement.

- `eval/humaneval_sandbox.py`: load HumanEval problems, generate k completions each, execute in `subprocess` with 5s timeout, no network, temp dir; run the provided tests; report pass@1 / pass@10.
- Expect humbling numbers at nano scale (pass@1 of 0–3% is normal; GPT-2 class gets a few %). The _infrastructure_ is the deliverable — and watching pass@k move when you improve data is the payoff.

**Deliverable:** eval script + baseline numbers.
**Checkpoint:** sandbox correctly passes a known-good solution and fails a known-bad one.

### Day 25 — Full evaluation + report

**Goal:** The writeup evidence.

- Run: val perplexity (base + SFT), pass@k, and a 20-prompt qualitative rubric (10 email, 10 code) scored 1–5 on relevance/coherence/correctness.
- Write `EVAL.md` with tables, loss curves, and the sample-progression story.

**Deliverable:** `EVAL.md`.
**Checkpoint:** every claim in it is backed by a number or artifact.

### Day 26 — KV cache + fast inference

**Goal:** Real-time generation.

- Add KV cache: each attention layer stores past K,V; each new step feeds only the newest token; RoPE must use the _absolute_ position index, not 0.
- Benchmark tokens/sec with vs. without cache at 200-token generations.

**Deliverable:** cached generation, numerically identical outputs to naive.
**Checkpoint:** identical outputs (greedy decoding) and a large speedup at long contexts.

### Day 27 — Serve it

**Goal:** A demo.

- FastAPI endpoint `/generate` (prompt, max_tokens, temperature) wrapping your cached sampler; optionally token streaming. Or a simple CLI chat loop applying the chat template.

**Deliverable:** `inference/serve.py` + a curl example in the README.
**Checkpoint:** you can chat with your model from another terminal.

### Day 28 — Documentation + writeup

**Goal:** Portfolio polish.

- README: architecture diagram, data datasheet, training curves, eval results, honest limitations, "what I learned."
- Clean the repo, pin `requirements.txt`, tag `v1.0`.

**Deliverable:** a repo you'd happily show an interviewer.
**Checkpoint:** a stranger could reproduce the nano run from the README alone.

### Days 29–30 — Stretch (pick one)

- **Scale**: rent an A100, train the 124M config on your full pipeline.
- **Multi-language**: add Rust + C++ from The Stack, re-mix, continue pretraining; re-run pass@k.
- **GQA**: convert attention to grouped-query, measure inference speedup.
- **DPO**: preference-tune on a small preference set — the gentlest intro to alignment.

---

## Troubleshooting quick reference

| Symptom                         | Likely cause                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------- | --- | -------------------------------------- |
| Initial loss ≠ ~10.4            | Head/loss wiring, wrong vocab size, or softmax applied twice                    |
| Overfit test won't go < 0.1     | Off-by-one targets, causal mask wrong, RoPE on V, zero_grad missing             |
| Loss spikes mid-run             | LR too high, bad data shard, fp16 instead of bf16                               |
| NaNs                            | LR too high, missing grad clip, division in RMSNorm without eps                 |
| Val ≫ train loss                | Data leakage check first, then genuine overfitting (more data or smaller model) |
| Code samples ignore indentation | Tokenizer mangled whitespace — retrain byte-level                               |
| SFT model won't stop generating | `<                                                                              | end | >` not treated as EOS in sampling loop |
| OOM                             | Halve micro-batch, raise grad-accum steps; check no `retain_graph`              |

## Daily rhythm

1. Re-read yesterday's checkpoint result.
2. Build today's deliverable.
3. Run the checkpoint test.
4. Commit with a message stating what the checkpoint proved.
5. One-line log: what worked, what surprised you. (This log becomes your writeup.)
