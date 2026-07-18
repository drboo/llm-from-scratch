"""
Day 27 tests — inference server (FastAPI) and chat helpers.

Runs against a random-init nano model; no checkpoint needed.

    pytest inference/test_day27.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import GPT, ModelConfig
from inference.serve import (
    _load_model,
    _encode,
    _decode,
    generate_tokens,
    _iter_tokens,
    build_app,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nano() -> GPT:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, d_model=64, n_head=2, n_layer=2, ctx=64)
    return GPT(cfg).eval()


def _client(model=None):
    m = model or _nano()
    app = build_app(m, codec=None, device=torch.device("cpu"))
    return TestClient(app), m


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------


class TestGenerateTokens:
    def test_returns_list_of_ints(self):
        model  = _nano()
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out    = generate_tokens(model, prompt, max_tokens=10, temperature=0.0)
        assert isinstance(out, list)
        assert all(isinstance(t, int) for t in out)

    def test_length(self):
        model  = _nano()
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out    = generate_tokens(model, prompt, max_tokens=15, temperature=0.0)
        assert len(out) == 15

    def test_eos_stops_early(self):
        model  = _nano()
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        # Find what first token would be
        full   = generate_tokens(model, prompt, max_tokens=20, temperature=0.0)
        eos    = full[0]
        out    = generate_tokens(model, prompt, max_tokens=20,
                                 temperature=0.0, eos_id=eos)
        assert len(out) == 1

    def test_greedy_deterministic(self):
        model  = _nano()
        prompt = torch.tensor([[5, 6, 7]], dtype=torch.long)
        a = generate_tokens(model, prompt, max_tokens=12, temperature=0.0)
        b = generate_tokens(model, prompt, max_tokens=12, temperature=0.0)
        assert a == b

    def test_stochastic_differs(self):
        """Stochastic sampling should (virtually always) produce different runs."""
        torch.manual_seed(0)
        model  = _nano()
        prompt = torch.tensor([[5, 6, 7]], dtype=torch.long)
        torch.manual_seed(1)
        a = generate_tokens(model, prompt, max_tokens=20, temperature=2.0)
        torch.manual_seed(99)
        b = generate_tokens(model, prompt, max_tokens=20, temperature=2.0)
        # With vocab=256 and temp=2.0, runs almost certainly differ
        assert a != b


# ---------------------------------------------------------------------------
# Encode / decode helpers (byte mode, no codec)
# ---------------------------------------------------------------------------


class TestEncodeDecode:
    def test_encode_returns_tensor(self):
        ids = _encode("hello", codec=None, vocab_size=256,
                      device=torch.device("cpu"))
        assert ids.shape[0] == 1
        assert ids.dtype == torch.long

    def test_roundtrip(self):
        text   = "Hello, world!"
        prompt = _encode(text, codec=None, vocab_size=256,
                         device=torch.device("cpu"))
        back   = _decode(prompt[0].tolist(), codec=None, vocab_size=256)
        assert back == text

    def test_encode_non_ascii(self):
        text = "café"
        ids  = _encode(text, codec=None, vocab_size=256,
                       device=torch.device("cpu"))
        assert ids.shape[1] > 0


# ---------------------------------------------------------------------------
# FastAPI — /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_ok(self):
        client, _ = _client()
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_returns_params(self):
        client, model = _client()
        r = client.get("/health")
        assert "params_M" in r.json()
        expected = round(model.num_params() / 1e6, 1)
        assert r.json()["params_M"] == expected


# ---------------------------------------------------------------------------
# FastAPI — POST /generate
# ---------------------------------------------------------------------------


class TestGenerateEndpoint:
    def test_basic_request(self):
        client, _ = _client()
        r = client.post("/generate", json={"prompt": "hello", "max_tokens": 10})
        assert r.status_code == 200
        body = r.json()
        assert "text" in body
        assert "tokens_generated" in body

    def test_tokens_generated_count(self):
        client, _ = _client()
        r = client.post("/generate",
                        json={"prompt": "hi", "max_tokens": 20,
                              "temperature": 0.0})
        assert r.json()["tokens_generated"] == 20

    def test_empty_prompt_rejected(self):
        client, _ = _client()
        r = client.post("/generate", json={"prompt": "", "max_tokens": 5})
        assert r.status_code == 422  # prompt min_length=1

    def test_temperature_zero_deterministic(self):
        client, _ = _client()
        payload = {"prompt": "once", "max_tokens": 15, "temperature": 0.0}
        a = client.post("/generate", json=payload).json()["text"]
        b = client.post("/generate", json=payload).json()["text"]
        assert a == b

    def test_max_tokens_validated(self):
        client, _ = _client()
        r = client.post("/generate", json={"prompt": "x", "max_tokens": 0})
        assert r.status_code == 422   # pydantic validation error

    def test_response_is_string(self):
        client, _ = _client()
        r = client.post("/generate", json={"prompt": "test", "max_tokens": 5})
        assert isinstance(r.json()["text"], str)


# ---------------------------------------------------------------------------
# FastAPI — GET /generate/stream
# ---------------------------------------------------------------------------


class TestStreamEndpoint:
    def test_stream_returns_sse(self):
        client, _ = _client()
        with client.stream("GET", "/generate/stream",
                           params={"prompt": "hi", "max_tokens": 10,
                                   "temperature": "0.0"}) as r:
            assert r.status_code == 200
            chunks = list(r.iter_lines())

        # Each "data:" line should be valid JSON or "[DONE]"
        data_lines = [c for c in chunks if c.startswith("data:")]
        assert len(data_lines) > 0
        done_line = data_lines[-1]
        assert done_line == "data: [DONE]"

    def test_stream_tokens_are_json(self):
        client, _ = _client()
        with client.stream("GET", "/generate/stream",
                           params={"prompt": "hello", "max_tokens": 5,
                                   "temperature": "0.0"}) as r:
            chunks = list(r.iter_lines())

        data_lines = [c[len("data: "):] for c in chunks
                      if c.startswith("data:") and c != "data: [DONE]"]
        for line in data_lines:
            obj = json.loads(line)
            assert "token" in obj

    def test_stream_token_count(self):
        client, _ = _client()
        n = 8
        with client.stream("GET", "/generate/stream",
                           params={"prompt": "x", "max_tokens": str(n),
                                   "temperature": "0.0"}) as r:
            chunks = list(r.iter_lines())

        token_lines = [c for c in chunks
                       if c.startswith("data:") and c != "data: [DONE]"]
        assert len(token_lines) == n


# ---------------------------------------------------------------------------
# _load_model helper
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_no_checkpoint_returns_model(self, tmp_path):
        m = _load_model(None, ctx=64, device=torch.device("cpu"))
        assert isinstance(m, GPT)

    def test_nonexistent_checkpoint_returns_model(self, tmp_path):
        m = _load_model("/nonexistent/path.pt", ctx=64,
                        device=torch.device("cpu"))
        assert isinstance(m, GPT)

    def test_model_is_eval(self):
        m = _load_model(None, ctx=64, device=torch.device("cpu"))
        assert not m.training
