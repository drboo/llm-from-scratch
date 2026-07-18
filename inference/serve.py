"""
Day 27: serve the model via FastAPI or an interactive CLI chat loop.

──────────────────────────────────────────────────────────────────────
FastAPI server (wraps the KV-cache sampler):

    python inference/serve.py --ckpt checkpoints/ckpt_050000.pt

    # In another terminal:
    curl -s -X POST http://localhost:8000/generate \\
         -H 'Content-Type: application/json' \\
         -d '{"prompt":"Once upon a time","max_tokens":80,"temperature":0.8}' \\
         | python -m json.tool

    # Token streaming (SSE):
    curl -N http://localhost:8000/generate/stream?prompt=Hello&max_tokens=40

──────────────────────────────────────────────────────────────────────
CLI chat loop (applies the SFT chat template if a tokeniser is found):

    python inference/serve.py --ckpt checkpoints/sft/ckpt_best.pt --chat

──────────────────────────────────────────────────────────────────────
No checkpoint — random model (good for smoke-testing the server):

    python inference/serve.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncIterator

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPT, ModelConfig
from inference.sample import top_k_filter, top_p_filter

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(ckpt_path: str | None, ctx: int, device: torch.device) -> GPT:
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd    = state.get("model_state_dict", state.get("model", state))
        vocab_size = sd["embed.weight"].shape[0]
        d_model    = sd["embed.weight"].shape[1]
        n_layer    = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
        n_head     = 6
        cfg = ModelConfig(vocab_size=vocab_size, d_model=d_model,
                          n_head=n_head, n_layer=n_layer, ctx=ctx)
        model = GPT(cfg)
        model.load_state_dict(sd, strict=True)
        print(f"[serve] Loaded {Path(ckpt_path).name} — "
              f"{model.num_params()/1e6:.1f}M params", flush=True)
    else:
        cfg   = ModelConfig()
        model = GPT(cfg)
        print("[serve] No checkpoint — using random-init nano model", flush=True)

    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Codec / tokenizer helpers
# ---------------------------------------------------------------------------


def _load_codec():
    """Return a Codec instance or None if the tokeniser isn't present."""
    tok_path = ROOT / "tokeniser" / "tokenizer.json"
    if not tok_path.exists():
        return None
    try:
        sys.path.insert(0, str(ROOT / "tokeniser"))
        from tokenizer import Codec  # type: ignore
        return Codec(str(tok_path))
    except Exception as e:
        print(f"[serve] Codec unavailable: {e}", flush=True)
        return None


def _encode(text: str, codec, vocab_size: int, device: torch.device) -> torch.Tensor:
    if codec and vocab_size > 256:
        ids = codec.encode(text)
    else:
        ids = list(text.encode("utf-8"))
    return torch.tensor([ids], dtype=torch.long, device=device)


def _decode(ids: list[int], codec, vocab_size: int) -> str:
    if codec and vocab_size > 256:
        return codec.decode(ids)
    return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def _encode_chat(instruction: str, codec, vocab_size: int,
                 device: torch.device) -> torch.Tensor:
    """Encode just the prompt half of the chat template (no response yet)."""
    if codec and hasattr(codec, "encode_chat") and vocab_size > 256:
        ids, _ = codec.encode_chat(instruction, "")
        # Keep only the prompt portion (up to and including <|assistant|>)
        text = f"<|bos|><|user|>{instruction}<|endofturn|><|assistant|>"
        ids  = codec.encode(text)
    else:
        text = f"<|bos|><|user|>{instruction}<|endofturn|><|assistant|>"
        ids  = list(text.encode("utf-8"))
    return torch.tensor([ids], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Generation wrapper (token streaming via async generator)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_tokens(
    model:       GPT,
    prompt_ids:  torch.Tensor,
    max_tokens:  int,
    temperature: float = 1.0,
    top_k:       int   = 0,
    top_p:       float = 1.0,
    eos_id:      int | None = None,
) -> list[int]:
    """Generate up to max_tokens new tokens; return list of generated ids."""
    return list(_iter_tokens(model, prompt_ids, max_tokens,
                             temperature, top_k, top_p, eos_id))


def _iter_tokens(
    model:       GPT,
    prompt_ids:  torch.Tensor,
    max_tokens:  int,
    temperature: float,
    top_k:       int,
    top_p:       float,
    eos_id:      int | None,
):
    """Synchronous generator yielding one token id at a time via KV cache."""
    from model.kv_cache import KVCache

    device     = prompt_ids.device
    prompt_len = prompt_ids.shape[1]
    max_len    = prompt_len + max_tokens

    cache  = KVCache.for_model(model, max_len, device,
                                dtype=next(model.parameters()).dtype)
    logits = model.forward_cached(prompt_ids, start_pos=0, cache=cache)
    next_logits = logits[:, -1, :]

    for step in range(max_tokens):
        if temperature == 0.0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            scaled = next_logits / temperature
            if top_k:
                scaled = top_k_filter(scaled, top_k)
            if top_p < 1.0:
                scaled = top_p_filter(scaled, top_p)
            probs   = F.softmax(scaled, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        tok = next_id.item()
        yield tok

        if eos_id is not None and tok == eos_id:
            break
        if prompt_len + step + 1 >= max_len:
            break

        pos        = prompt_len + step
        logits     = model.forward_cached(next_id, start_pos=pos, cache=cache)
        next_logits = logits[:, -1, :]


# ---------------------------------------------------------------------------
# Pydantic models (module-level so Pydantic v2 can resolve forward refs)
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field  # noqa: E402


class GenerateRequest(BaseModel):
    prompt:      str   = Field(default="Once upon a time", min_length=1)
    max_tokens:  int   = Field(default=200, ge=1, le=2048)
    temperature: float = Field(default=1.0,  ge=0.0, le=4.0)
    top_k:       int   = Field(default=0,    ge=0)
    top_p:       float = Field(default=1.0,  ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    text:             str
    tokens_generated: int


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def build_app(model: GPT, codec, device: torch.device):
    """Build and return the FastAPI app (imported lazily to keep startup fast)."""
    from fastapi import Body, FastAPI, Query
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="LLM-from-scratch", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "params_M": round(model.num_params() / 1e6, 1)}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest = Body(...)):
        prompt_ids = _encode(req.prompt, codec, model.cfg.vocab_size, device)
        new_ids    = generate_tokens(
            model, prompt_ids, req.max_tokens,
            req.temperature, req.top_k, req.top_p,
        )
        text = _decode(new_ids, codec, model.cfg.vocab_size)
        return GenerateResponse(text=text, tokens_generated=len(new_ids))

    @app.get("/generate/stream")
    def generate_stream(
        prompt:      str   = Query(default="Once upon a time"),
        max_tokens:  int   = Query(default=200, ge=1, le=2048),
        temperature: float = Query(default=1.0, ge=0.0),
        top_k:       int   = Query(default=0,   ge=0),
        top_p:       float = Query(default=1.0, ge=0.0, le=1.0),
    ):
        """Server-sent events stream — each event is one token as JSON."""
        prompt_ids = _encode(prompt, codec, model.cfg.vocab_size, device)

        def event_stream():
            for tok_id in _iter_tokens(model, prompt_ids, max_tokens,
                                       temperature, top_k, top_p, eos_id=None):
                token_text = _decode([tok_id], codec, model.cfg.vocab_size)
                yield f"data: {json.dumps({'token': token_text})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


# ---------------------------------------------------------------------------
# CLI chat loop
# ---------------------------------------------------------------------------


def chat_loop(model: GPT, codec, device: torch.device,
              max_tokens: int = 256, temperature: float = 0.8,
              top_k: int = 50, eos_id: int | None = None) -> None:
    """Interactive CLI chat loop applying the SFT chat template."""
    vocab_size = model.cfg.vocab_size
    has_template = codec is not None and vocab_size > 256

    print("\n" + "═" * 60)
    print("  LLM-from-scratch — chat mode")
    if has_template:
        print("  Chat template active (SFT format)")
    else:
        print("  Raw byte encoding (no tokenizer found)")
    print("  Type 'quit' or press Ctrl-C to exit.")
    print("═" * 60 + "\n")

    stop_tokens = {"<|endofturn|>", "<|eos|>"}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("[bye]")
            break

        if has_template:
            prompt_ids = _encode_chat(user_input, codec, vocab_size, device)
        else:
            prompt_ids = _encode(user_input, codec, vocab_size, device)

        print("Assistant: ", end="", flush=True)
        generated: list[int] = []

        for tok_id in _iter_tokens(model, prompt_ids, max_tokens,
                                   temperature, top_k, 1.0, eos_id):
            generated.append(tok_id)
            piece = _decode([tok_id], codec, vocab_size)
            # Stop if the decoded piece is a stop token
            if piece in stop_tokens or (eos_id is not None and tok_id == eos_id):
                break
            print(piece, end="", flush=True)

        print()   # newline after response
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve or chat with your LLM (Day 27)")
    p.add_argument("--ckpt",        default="",  help="checkpoint path")
    p.add_argument("--ctx",         type=int,   default=256)
    p.add_argument("--host",        default="0.0.0.0")
    p.add_argument("--port",        type=int,   default=8000)
    p.add_argument("--chat",        action="store_true",
                   help="run interactive CLI chat loop instead of API server")
    p.add_argument("--max-tokens",  type=int,   default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k",       type=int,   default=50)
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_model(args.ckpt or None, args.ctx, device)
    codec  = _load_codec()

    if args.chat:
        chat_loop(model, codec, device,
                  max_tokens=args.max_tokens,
                  temperature=args.temperature,
                  top_k=args.top_k)
    else:
        import uvicorn
        app = build_app(model, codec, device)
        print(f"[serve] Listening on http://{args.host}:{args.port}", flush=True)
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
