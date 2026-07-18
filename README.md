# LLM from Scratch — 30-Day Build

A GPT-style language model built entirely from scratch in Python/PyTorch over 30 days. Every component is hand-rolled — no `transformers`, no pre-built model classes — following a structured guide from tokenizer training to live inference serving.

**Nano model:** 22.9M parameters · 32k BPE vocab · ctx=256 · 6 layers, 6 heads, d_model=384

---

## Architecture

A decoder-only transformer following the LLaMA design (pre-norm, RoPE, SwiGLU, weight tying):

```
Input token ids  (B, T)
        │
        ▼
┌─────────────────┐
│  TokenEmbedding │  (B, T) → (B, T, d_model)
└─────────────────┘
        │
        ▼  ×N
┌─────────────────────────────────────────────────┐
│  TransformerBlock                               │
│  ┌─────────────────────────────────────────┐   │
│  │ RMSNorm → CausalSelfAttention           │   │
│  │           ├─ fused QKV projection       │   │
│  │           ├─ RoPE on Q and K            │   │
│  │           └─ Flash attention (causal)   │   │
│  └─────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────┐   │
│  │ RMSNorm → SwiGLU FFN                    │   │
│  │           hidden = ⌈8 d_model / 3⌉     │   │
│  └─────────────────────────────────────────┘   │
│  Both sub-layers connected by residual stream   │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────┐
│  RMSNorm        │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  Linear head    │  weights tied to embed (saves 32k × 384 params)
└─────────────────┘
        │
        ▼
Logits  (B, T, vocab_size)
```

**Key design choices:**

| Choice | Why |
|--------|-----|
| Pre-norm (RMSNorm before each sub-layer) | Cleaner gradient flow; training more stable than post-norm at depth |
| RoPE on Q and K only | Relative position encoding, no learned positional params, extrapolates to longer ctx |
| SwiGLU FFN | ~10% lower loss vs ReLU at same parameter budget (PaLM ablation) |
| Weight tying (head = embed^T) | Saves `vocab × d_model` params, enforces consistency between token input/output spaces |
| Scaled residual init | `out_proj` and `w_down` at `std = 0.02 / √(2N)` keeps residual stream variance ~1 at init regardless of depth |
| Fused QKV projection | Single `Linear(d, 3d)` vs three separate — one GEMM, better GPU utilization |
| GQA (optional) | K and V use fewer heads than Q — reduces KV cache memory proportionally with no change to forward math |

### Grouped Query Attention (GQA)

`CausalSelfAttention` supports three attention modes via `ModelConfig.n_kv_head`:

| Mode | `n_kv_head` | KV cache vs MHA | Used by |
|------|-------------|-----------------|---------|
| MHA (default) | `0` or `n_head` | 1× | GPT-2, this build by default |
| GQA | `1 < n_kv_head < n_head` | `n_kv_head / n_head` | LLaMA-2-70B (8 KV / 64 Q heads) |
| MQA | `1` | `1 / n_head` | PaLM, Falcon |

The fused QKV projection becomes `Linear(d_model, (n_head + 2·n_kv_head)·d_head)` — Q keeps `n_head` channels, K and V each use `n_kv_head`. Before `scaled_dot_product_attention`, the KV heads are expanded via `repeat_interleave` to match Q. The KV cache allocates `(1, n_kv_head, max_seq_len, d_head)` per layer instead of `(1, n_head, …)`, giving a direct memory saving.

```python
# Example: 3× smaller KV cache
cfg = ModelConfig(n_head=6, n_kv_head=2, d_model=384, ...)
model = GPT(cfg)
```

`n_head` and `n_kv_head` are both inferred automatically when loading a checkpoint — no flags needed.

---

## Data Pipeline

The full pipeline runs in sequence: `acquire → clean → dedup → scrub → filter → tokenize → pack`.

### Datasets

| Source | Type | Raw tokens (est.) |
|--------|------|-------------------|
| FineWeb (HuggingFace) | Web text | ~2B/shard (sampled 1 shard) |
| Enron email corpus (`SetFit/enron_spam`) | Email | ~50M |
| The Stack (`code-search-net/code_search_net`) | Code (Python + others) | ~200M |

### Pipeline stages

```
data/acquire.py       — stream from HuggingFace Hub into raw shards
data/clean.py         — deduplicate, length filter, remove near-empty docs
data/dedup.py         — MinHash near-deduplication across shards
data/scrub.py         — PII scrubbing (email address regex)
data/filter_code.py   — code quality filter (syntax check, min/max length)
data/prepare_real.py  — tokenize with BPE Codec, pack to .bin (uint16 tokens)
data/prepare_toy.py   — lightweight version for smoke tests
```

Final output: `data/real/train.bin` and `data/real/val.bin` (np.memmap, uint16).

---

## Training

### Pretraining

```bash
python train/pretrain.py \
    --data-dir data/real \
    --out-dir checkpoints \
    --max-steps 50000 \
    --batch-size 16 \
    --accum-steps 4 \
    --ctx 256 \
    --lr 3e-4 \
    --eval-every 500 \
    --ckpt-every 1000 \
    --sample-prompt "Once upon a time" \
    --sample-steps 2000
```

Effective batch size = `batch_size × ctx × accum_steps` = 16 × 256 × 4 ≈ 16k tokens/step.

Training loop features:
- Cosine LR schedule with linear warmup (2% of steps)
- Gradient clipping (`max_norm=1.0`)
- Mixed-precision (`torch.autocast`)
- Checkpoint save on val-loss improvement
- Resume from checkpoint (`--resume`)
- Optional W&B logging (`--wandb-project`)

### SFT (Supervised Fine-Tuning)

```bash
python train/sft.py \
    --base-ckpt checkpoints/ckpt_050000.pt \
    --data-dir data/sft \
    --out-dir checkpoints/sft
```

SFT specifics:
- Chat template: `<|bos|><|user|>{instruction}<|endofturn|><|assistant|>{response}<|endofturn|><|eos|>`
- Loss masking: `ignore_index=-100` on prompt and padding tokens — only assistant tokens contribute to loss
- Datasets: Alpaca (20k) + CodeAlpaca (10k) + OpenAssistant (5k) + 30 hand-written examples
- Early stopping on val loss (patience=3 evaluations)
- Per-example padded batching: pad to longest-in-batch, labels=-100 on padding

Build SFT dataset:
```bash
python -c "from sft.data import build_sft_dataset; build_sft_dataset('data/sft')"
```

### DPO (Direct Preference Optimization)

DPO (Rafailov et al. 2023) aligns the model with human preferences without a reward model. Starting from a frozen SFT reference, it trains a policy to prefer chosen responses over rejected ones using the loss:

```
L_DPO = −E [ log σ( β · (log π/π_ref|chosen − log π/π_ref|rejected) ) ]
```

The implicit reward for a response is `β · (log π(y|x) − log π_ref(y|x))`. After training, `r_chosen > r_rejected` — the policy assigns more probability to preferred completions relative to the reference.

```bash
python train/dpo.py \
    --ref-ckpt checkpoints/sft/ckpt_best.pt \
    --out-dir  checkpoints/dpo \
    --beta 0.1 --lr 5e-7 --n-steps 500
```

DPO details:
- Preference data: 40 hand-written pairs (20 professional email, 20 Python code) with quality-demonstrating chosen vs. terse/buggy rejected responses (`dpo/data.py`)
- Loss masking: identical to SFT — `-100` on prompt tokens so only response tokens contribute to `sequence_logprobs`
- Reference model is deep-copied from the SFT checkpoint and frozen (`requires_grad=False`) for the entire run
- `β=0.1` controls how far the policy can deviate from the reference; lower = stays closer to SFT
- Works with GQA checkpoints: `n_head` and `n_kv_head` are inferred automatically from the state dict

---

## Tokenizer

Byte-level BPE, 32k vocab, trained on FineWeb + code_search_net (~60M tokens).

Special tokens:

| Token | Role |
|-------|------|
| `<\|bos\|>` | Beginning of sequence |
| `<\|eos\|>` | End of sequence |
| `<\|pad\|>` | Padding (SFT batching) |
| `<\|user\|>` / `<\|assistant\|>` / `<\|endofturn\|>` | Chat template |
| `<\|fim_prefix\|>` / `<\|fim_middle\|>` / `<\|fim_suffix\|>` | Fill-in-the-middle |
| `<\|email\|>` / `<\|py\|>` / `<\|rs\|>` / `<\|cpp\|>` | Domain markers |

Train the tokenizer (~10 min, requires HuggingFace datasets):
```bash
python tokeniser/tokeniser.py --max-web 50000 --max-code 10000
```

---

## Evaluation

### Perplexity
```bash
python eval/perplexity.py --ckpt-dir checkpoints --data data/real
```
Random-init baseline: PPL ≈ vocab_size (~32,000). Target after full pretraining: PPL < 50.

### HumanEval (code generation)
```bash
python eval/full_eval.py \
    --base-ckpt checkpoints/ckpt_050000.pt \
    --sft-ckpt  checkpoints/sft/ckpt_best.pt \
    --out-dir   eval/results
```

Uses the unbiased pass@k estimator: `1 - C(n-c, k) / C(n, k)`.
Each solution is executed in a sandboxed subprocess with a 5-second timeout and restricted environment.
Nano-scale pass@1 of 0–3% is typical; the infrastructure (sandbox, estimator) is the deliverable.

### Full evaluation report
Generates `EVAL.md` covering perplexity, HumanEval, qualitative rubric scores, and sample progression:
```bash
python eval/full_eval.py --base-ckpt <path> --sft-ckpt <path> --out-dir eval/results
# To regenerate the report from saved results:
python eval/full_eval.py --report-only --out-dir eval/results
```

---

## Inference

### FastAPI server
```bash
python inference/serve.py --ckpt checkpoints/ckpt_050000.pt --port 8000
```

**Endpoints:**

```bash
# Health check
curl http://localhost:8000/health

# Generate (blocking)
curl -X POST http://localhost:8000/generate \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"Once upon a time","max_tokens":80,"temperature":0.8,"top_k":50}'

# Stream tokens via SSE
curl -N 'http://localhost:8000/generate/stream?prompt=Hello&max_tokens=40&temperature=0.8'
```

Request schema for `POST /generate`:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `prompt` | string | — | Required, min 1 char |
| `max_tokens` | int | 200 | 1–2048 |
| `temperature` | float | 1.0 | 0 = greedy |
| `top_k` | int | 0 | 0 = disabled |
| `top_p` | float | 1.0 | nucleus filter |

### CLI chat loop (SFT model)
```bash
python inference/serve.py --ckpt checkpoints/sft/ckpt_best.pt --chat
```

### Naive sampler (no server)
```bash
python inference/sample.py \
    --ckpt checkpoints/ckpt_050000.pt \
    --prompt "Once upon a time" \
    --temperature 0.8 --top-k 50 --n-new 200
```

### KV-cache benchmark
```bash
python inference/benchmark.py --ckpt checkpoints/ckpt_050000.pt --n-new 200
```
Compares tokens/sec with vs. without KV cache. On GPU with ctx ≥ 512, expect 5–20× speedup.

---

## Reproducing the nano run from scratch

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Train tokenizer (needs HuggingFace login)
python tokeniser/tokeniser.py --max-web 50000 --max-code 10000

# 2. Acquire + clean corpus (can take 30–60 min)
python data/acquire.py --out-dir data/raw_shards
python data/clean.py   --in-dir  data/raw_shards  --out-dir data/clean_shards
python data/dedup.py   --in-dir  data/clean_shards --out-dir data/dedup_shards
python data/scrub.py   --in-dir  data/dedup_shards  --out-dir data/scrub_shards
python data/filter_code.py --in-dir data/scrub_shards --out-dir data/filtered_shards
python data/prepare_real.py --in-dir data/filtered_shards --out-dir data/real

# 3. Pretrain (GPU recommended; ~6h on a single A10)
python train/pretrain.py \
    --data-dir data/real --out-dir checkpoints \
    --max-steps 50000 --batch-size 16 --accum-steps 4 \
    --lr 3e-4 --sample-prompt "Once upon a time"

# 4. Build SFT data + fine-tune
python -c "from sft.data import build_sft_dataset; build_sft_dataset('data/sft')"
python train/sft.py --base-ckpt checkpoints/ckpt_050000.pt --data-dir data/sft

# 5. DPO alignment (optional — fine-tunes the SFT model on preference pairs)
python train/dpo.py \
    --ref-ckpt checkpoints/sft/ckpt_best.pt \
    --out-dir  checkpoints/dpo \
    --beta 0.1 --lr 5e-7 --n-steps 500

# 6. Evaluate
python eval/full_eval.py \
    --base-ckpt checkpoints/ckpt_050000.pt \
    --sft-ckpt  checkpoints/sft/ckpt_best.pt \
    --out-dir   eval/results

# 7. Serve
python inference/serve.py --ckpt checkpoints/dpo/ckpt_best.pt
```

For a quick smoke test on CPU (no GPU, no data download):
```bash
python train/overfit.py --tiny   # proves backprop works, ~30s
python -m pytest model/ train/ inference/ eval/ sft/ data/ -v  # full test suite
```

---

## Tests

```bash
# All tests (no network, ~25s on CPU):
python -m pytest model/ train/ inference/ eval/ sft/ data/ dpo/ -v

# Tokenizer tests (requires tokenizer.json):
python -m pytest tokeniser/ -v

# HumanEval sandbox (requires network to pull the dataset):
python -m pytest eval/test_day24.py::TestCanonicalSolutions -v
```

**Test counts by day:**

| Day | File | Tests |
|-----|------|-------|
| 4 | `model/test_day4.py` | 8 |
| 5 | `model/test_day5.py` | 7 |
| 6 | `model/test_day6.py` | 8 |
| 7 | `model/test_day7.py` | 9 |
| 8 | `data/test_day8.py` | 6 |
| 9 | `train/test_day9.py` | 5 |
| 10 | `train/test_day10.py` | 8 |
| 11 | `inference/test_day11.py` | 8 |
| 12 | `eval/test_day12.py` | 7 |
| 13 | `eval/test_day13.py` | 6 |
| 15 | `data/test_day15.py` | 8 |
| 16 | `data/test_day16.py` | 10 |
| 17 | `data/test_day17.py` | 9 |
| 18 | `data/test_day18.py` | 8 |
| 22 | `sft/test_day22.py` | 35 |
| 23 | `train/test_day23.py` | 18 |
| 24 | `eval/test_day24.py` | 31 |
| 25 | `eval/test_day25.py` | 27 |
| 26 | `model/test_day26.py` | 17 |
| 27 | `inference/test_day27.py` | 22 |
| 29 | `model/test_day29.py` | 25 |
| 30 | `dpo/test_day30.py` | 36 |

---

## Directory structure

```
.
├── configs/
│   └── nano.yaml                 # nano model hyperparameters
├── data/
│   ├── acquire.py                # stream FineWeb + Enron + Stack from HuggingFace
│   ├── clean.py                  # dedup + length filter
│   ├── dedup.py                  # MinHash near-dedup
│   ├── scrub.py                  # PII scrubbing (email regex)
│   ├── filter_code.py            # code quality filter
│   ├── prepare_real.py           # tokenize + pack into .bin
│   ├── prepare_toy.py            # lightweight smoke-test corpus
│   └── dataloader.py             # np.memmap get_batch()
├── eval/
│   ├── perplexity.py             # PPL evaluator
│   ├── sample_progression.py     # sweep checkpoints, show output arc
│   ├── humaneval_sandbox.py      # HumanEval runner + pass@k estimator
│   └── full_eval.py              # orchestrator → EVAL.md
├── inference/
│   ├── sample.py                 # temperature / top-k / top-p sampler (no cache)
│   ├── benchmark.py              # KV-cache speedup benchmark
│   └── serve.py                  # FastAPI server + CLI chat loop
├── model/
│   ├── norm.py                   # RMSNorm
│   ├── rope.py                   # RoPE (precompute + apply)
│   ├── model.py                  # TokenEmbedding, causal_mask
│   ├── attention.py              # CausalSelfAttention (flash + cached)
│   ├── ffn.py                    # SwiGLU FFN
│   ├── block.py                  # TransformerBlock (forward + forward_cached)
│   ├── gpt.py                    # GPT model (forward + generate_cached)
│   └── kv_cache.py               # KVCache — pre-allocated K/V tensors
├── dpo/
│   └── data.py                   # 40 hand-written preference pairs + encoder
├── sft/
│   └── data.py                   # SFT dataset builder (Alpaca + OASST + hand-written)
├── tokeniser/
│   ├── tokeniser.py              # BPE training script (HuggingFace tokenizers)
│   └── tokenizer.py              # Codec: encode_document, encode_chat, encode_fim
└── train/
    ├── checkpoint.py             # save/load with full RNG state
    ├── overfit.py                # single-batch sanity check
    ├── pretrain.py               # production pretraining loop
    ├── sft.py                    # SFT loop with loss masking + early stopping
    └── dpo.py                    # DPO loop (frozen ref + policy, β-scaled margin loss)
```

---

## Honest limitations

- **Scale**: the nano model (22.9M params, ctx=256) demonstrates all the mechanics correctly but sits well below the quality threshold for useful text generation. Chinchilla-optimal training would require ~460M tokens; the current corpus is closer to 10M for the toy run.
- **HumanEval pass@k**: expect ~0% pass@1 at nano scale on zero-shot HumanEval. The infrastructure (sandbox, unbiased estimator) is the deliverable; real numbers require at least a GPT-2-small-scale model.
- **SFT data size**: 35k examples is enough to see the chat template applied correctly and refusal behavior disappear, but not enough for robust instruction following.
- **DPO data size**: 40 hand-written preference pairs demonstrates the mechanics (gradient sign, implicit reward direction, ref model frozen) but is far too small for meaningful alignment. A production run needs thousands of diverse pairs from a human labelling pipeline or a stronger teacher model.
- **Tokenizer training corpus**: the BPE tokenizer was trained on a small sample; fertility (tokens per word) is higher than production tokenizers like GPT-4's cl100k.
- **ctx=256**: short context window means the model cannot handle most real documents. Increasing to 1024+ requires proportionally more memory and is a straightforward config change.

---

## What I learned

**Mechanics that only clicked by implementing them:**

- **RoPE absolute-vs-relative**: the KV cache correctness invariant — new K vectors must be rotated by their *absolute* sequence position, not position-within-chunk — is invisible in the math but breaks generation silently if wrong. The identity test caught this.

- **Loss masking is a two-line change with a large effect**: setting `ignore_index=-100` on prompt tokens in `F.cross_entropy` is trivial to code but fundamental — without it, the model wastes capacity learning to predict the instruction it was already given.

- **Prefill vs. decode in cached attention**: using `is_causal=(T > 1)` in `scaled_dot_product_attention` feels like a minor detail but is load-bearing: prefill needs the causal mask (T tokens attending to each other), decode does not (single query, full context already attended).

- **Gradient accumulation correctness**: you must divide the loss by `accum_steps` *before* calling `.backward()`, not after. Scaling after accumulation gives the same total gradient but incorrect gradient norms for clipping.

- **BPE merge order matters**: a BPE vocabulary with the same final merges but different merge-order encoding can produce the same token strings but different IDs — the tokenizer is only consistent with itself.

- **SFT data quality > quantity**: the hand-written examples (30 prompts) were more impactful per example than streaming thousands of Alpaca completions that include refusals, which train the model to say "I cannot."

- **Data pipeline is the slow path**: the model trains in hours; building, cleaning, and deduplicating a diverse corpus of millions of documents took more engineering time than the model itself.

- **GQA saves memory, not compute**: with `n_kv_head=2` and `n_head=6`, the KV cache shrinks 3×, enabling a larger batch or longer context at the same memory budget. On a tiny CPU model the wall-clock speedup is negligible because the bottleneck is matrix multiply, not memory bandwidth. On GPU at ctx ≥ 512 the bandwidth saving becomes the dominant effect.

- **DPO's β is the only real knob**: too high (≥ 1.0) and the policy barely moves; too low (≤ 0.01) and it drifts far from the reference and collapses. The sweet spot for a small model on a small preference set is typically 0.05–0.2. The implicit reward margin sign is the clearest training signal — a positive margin for chosen vs. rejected means the update is going in the right direction.

- **Reference model identity is everything in DPO**: the reference and policy must start from identical weights (deep copy, not re-load from disk) or you get a spurious constant offset in the log-ratio that corrupts the loss. The test `test_ref_model_frozen_during_training` catches this class of bug before it silently degrades training.
