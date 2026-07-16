# LLM from Scratch — 30-Day Build

A GPT-style language model built entirely from scratch in Python/PyTorch, following a 30-day guide. Every component is implemented by hand — no `transformers`, no pre-built model classes.

---

## What's been built (Days 1–13)

| Day | Deliverable | Key file(s) |
|-----|-------------|-------------|
| 1–2 | Repo scaffolding, dev environment | — |
| 3   | Byte-level BPE tokenizer (32k vocab) | `tokeniser/tokeniser.py` |
| 4   | Codec: encode_document, encode_chat, encode_fim, write_bin | `tokeniser/tokenizer.py` |
| 5   | RMSNorm, causal mask, token embedding | `model/norm.py`, `model/model.py` |
| 5   | RoPE (rotary position embeddings) | `model/rope.py` |
| 6   | Causal self-attention (flash + manual) | `model/attention.py` |
| 6   | SwiGLU FFN, transformer block | `model/ffn.py`, `model/block.py` |
| 7   | Full GPT model, weight tying, scaled residual init | `model/gpt.py` |
| 8   | np.memmap corpus loader | `data/dataloader.py`, `data/prepare_toy.py` |
| 9   | Overfit test — proves the model can learn | `train/overfit.py` |
| 10  | Production training loop: cosine LR, grad accum, checkpointing | `train/pretrain.py`, `train/checkpoint.py` |
| 11  | Autoregressive sampler: temperature, top-k, top-p | `inference/sample.py` |
| 12  | Perplexity evaluator | `eval/perplexity.py` |
| 13  | Sample progression — snapshot model output at each checkpoint | `eval/sample_progression.py` |

---

## Architecture

A decoder-only transformer matching the LLaMA/GPT-2 design:

```
token embed
    ↓
N × TransformerBlock
    ├─ RMSNorm (pre-norm)
    ├─ CausalSelfAttention  (fused QKV, RoPE on Q+K, flash attention)
    ├─ RMSNorm
    └─ SwiGLU FFN  (hidden = ⌈8d/3⌉ rounded to nearest 64)
    ↓
RMSNorm
    ↓
Linear head  (weights tied to embed)
```

**Nano config** (`configs/nano.yaml`): 6 layers, 6 heads, d_model=384, ctx=256 → ~23M parameters.

Key design choices:
- **Pre-norm** (RMSNorm before each sub-layer) — clean gradient flow
- **RoPE** applied to Q and K only — relative position encoding with no learned params
- **Weight tying** (head = embed^T) — saves `vocab_size × d_model` parameters
- **Scaled residual init** — `out_proj` and `w_down` initialised at `std = 0.02 / sqrt(2 × n_layer)` to keep residual stream variance bounded at init

---

## Tokenizer

Byte-level BPE (HuggingFace `tokenizers` library), 32k vocab, trained on FineWeb + code_search_net.

Special tokens: `<|pad|>` `<|bos|>` `<|eos|>` `<|user|>` `<|assistant|>` `<|endofturn|>` `<|fim_prefix|>` `<|fim_middle|>` `<|fim_suffix|>` `<|text|>` `<|email|>` `<|py|>` `<|rs|>` `<|cpp|>`

To train the tokenizer (takes ~10 min, writes `tokeniser/tokenizer.json`):
```bash
python tokeniser/tokeniser.py --max-web 50000 --max-code 10000
```

---

## Running

### 1. Prepare toy corpus

Downloads ~7k documents, tokenizes, writes `data/toy/train.bin` and `data/toy/val.bin`:

```bash
python data/prepare_toy.py --web 5000 --code 2000 --out data/toy
```

For a larger corpus (better training signal):
```bash
python data/prepare_toy.py --web 20000 --code 5000 --out data/toy
```

### 2. Train

```bash
python train/pretrain.py \
    --data-dir data/toy \
    --out-dir checkpoints \
    --max-steps 5000 \
    --sample-prompt "Once upon a time" \
    --sample-steps 500
```

Key flags:
- `--resume checkpoints/ckpt_002500.pt` — resume from checkpoint
- `--wandb-project my-llm` — enable W&B logging
- `--sample-prompt "..."` — save generated text at each checkpoint
- `--batch-size 8 --accum-steps 4` → effective batch = 8 × 256 × 4 = 8192 tokens/step

### 3. Monitor training

```bash
# Perplexity at each saved checkpoint
python eval/perplexity.py --ckpt-dir checkpoints --data data/toy

# View sample progression (noise → words → grammar)
python eval/sample_progression.py --samples-dir checkpoints/samples
```

At random init, perplexity ≈ 32000 (= vocab size). Target after a full run on toy data: PPL < 100.

### 4. Sample from a checkpoint

```bash
python inference/sample.py \
    --ckpt checkpoints/ckpt_005000.pt \
    --prompt "Once upon a time" \
    --temperature 0.8 \
    --top-k 50 \
    --n-new 200
```

### 5. Overfit sanity check (no data needed)

```bash
python train/overfit.py --tiny   # tiny model, ~30s on CPU
python train/overfit.py          # nano model
```

---

## Tests

```bash
# All non-tokenizer tests (fast, ~10s):
python3 -m pytest data/ model/ train/ inference/ eval/ -v

# Tokenizer tests (requires tokenizer.json):
python3 -m pytest tokeniser/ -v
```

---

## Scaling intuition (Chinchilla)

Rule of thumb: ~20 tokens per parameter for compute-optimal training.

| Model | Params | Tokens needed |
|-------|--------|---------------|
| Nano (this) | ~23M | ~460M |
| GPT-2 small | 124M | ~2.5B |
| LLaMA-7B | 7B | ~140B |

The toy corpus (~10M tokens) is enough to see loss fall and grammar emerge, but far below compute-optimal for the nano model. Week 3 of the guide builds the full data pipeline.

---

## Directory structure

```
.
├── configs/
│   └── nano.yaml            # nano model hyperparameters
├── data/
│   ├── dataloader.py        # np.memmap get_batch()
│   ├── prepare_toy.py       # download + tokenize toy corpus
│   └── test_day8.py
├── eval/
│   ├── perplexity.py        # exp(mean CE loss) evaluator
│   ├── sample_progression.py# sweep checkpoints, show output evolution
│   ├── test_day12.py
│   └── test_day13.py
├── inference/
│   ├── sample.py            # temperature / top-k / top-p sampling
│   └── test_day11.py
├── model/
│   ├── attention.py         # CausalSelfAttention (RoPE + flash)
│   ├── block.py             # TransformerBlock (pre-norm)
│   ├── ffn.py               # SwiGLUFFN
│   ├── gpt.py               # GPT + ModelConfig
│   ├── model.py             # TokenEmbedding, causal_mask
│   ├── norm.py              # RMSNorm
│   ├── rope.py              # precompute_rope_freqs, apply_rope
│   └── test_day{4-7}.py
├── tokeniser/
│   ├── tokeniser.py         # BPE training script
│   ├── tokenizer.py         # Codec (encode_document, encode_chat, encode_fim)
│   ├── test_tokenizer.py
│   └── test_codec.py
└── train/
    ├── checkpoint.py        # save/load with full RNG state
    ├── overfit.py           # single-batch overfit sanity check
    ├── pretrain.py          # production training loop
    ├── test_day9.py
    └── test_day10.py
```
