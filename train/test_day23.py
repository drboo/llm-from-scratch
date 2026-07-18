"""
Day 23 tests — SFT training loop.

Run:  pytest train/test_day23.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tokeniser"))

ROOT = Path(__file__).resolve().parent.parent
TOK_PATH = ROOT / "tokeniser" / "tokenizer.json"
HAS_TOKENIZER = TOK_PATH.exists()

from train.sft import (
    SFTConfig,
    sft_get_batch,
    sft_loss,
    get_lr,
    make_optimizer,
    eval_loss,
    count_examples,
)
from model.gpt import GPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sft_data(tmp_path: Path, n: int = 20, ctx: int = 32) -> Path:
    """Write minimal sft_train and sft_val splits to tmp_path."""
    if not HAS_TOKENIZER:
        pytest.skip("tokenizer.json not present")
    from tokenizer import Codec  # type: ignore
    from sft.data import encode_example, _write_split

    codec = Codec(str(TOK_PATH))
    examples = []
    for i in range(n):
        result = encode_example(
            f"Question number {i}.",
            f"This is answer number {i}.",
            codec,
            max_tokens=ctx,
        )
        if result:
            examples.append(result)

    n_val = max(1, len(examples) // 5)
    _write_split(examples[n_val:], tmp_path, "train")
    _write_split(examples[:n_val],  tmp_path, "val")
    return tmp_path


def _tiny_cfg(data_dir: str, ctx: int = 32) -> SFTConfig:
    cfg = SFTConfig()
    cfg.data_dir   = data_dir
    cfg.ctx        = ctx
    cfg.vocab_size = 32_000
    cfg.d_model    = 64
    cfg.n_head     = 2
    cfg.n_layer    = 2
    cfg.batch_size = 4
    cfg.accum_steps = 1
    cfg.lr         = 1e-4
    cfg.min_lr     = 1e-5
    cfg.eval_batches = 3
    cfg.patience   = 999
    return cfg


# ---------------------------------------------------------------------------
# sft_get_batch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TOKENIZER, reason="tokenizer.json not present")
class TestSFTGetBatch:
    def test_output_shapes(self, tmp_path):
        data = _make_sft_data(tmp_path)
        cfg  = _tiny_cfg(str(data))
        device = torch.device("cpu")
        x, y = sft_get_batch("train", data, cfg.ctx, cfg.batch_size, device)
        assert x.shape[0] == cfg.batch_size
        assert y.shape[0] == cfg.batch_size
        assert x.shape == y.shape

    def test_same_seq_length(self, tmp_path):
        data = _make_sft_data(tmp_path)
        x, y = sft_get_batch("train", data, 32, 4, torch.device("cpu"))
        assert x.shape[1] == y.shape[1]

    def test_token_dtype_long(self, tmp_path):
        data = _make_sft_data(tmp_path)
        x, y = sft_get_batch("train", data, 32, 4, torch.device("cpu"))
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_labels_contain_minus100(self, tmp_path):
        data = _make_sft_data(tmp_path)
        _, y = sft_get_batch("train", data, 32, 8, torch.device("cpu"))
        assert (y == -100).any()

    def test_labels_contain_positive(self, tmp_path):
        data = _make_sft_data(tmp_path)
        _, y = sft_get_batch("train", data, 32, 8, torch.device("cpu"))
        assert (y >= 0).any()

    def test_token_ids_within_vocab(self, tmp_path):
        data = _make_sft_data(tmp_path)
        x, _ = sft_get_batch("train", data, 32, 4, torch.device("cpu"))
        assert x.min() >= 0
        assert x.max() < 32_000

    def test_val_split_works(self, tmp_path):
        data = _make_sft_data(tmp_path)
        x, y = sft_get_batch("val", data, 32, 2, torch.device("cpu"))
        assert x.shape[0] == 2


# ---------------------------------------------------------------------------
# sft_loss
# ---------------------------------------------------------------------------


class TestSFTLoss:
    def test_ignores_minus100(self):
        B, T, V = 2, 8, 100
        logits = torch.randn(B, T, V)
        # All labels = -100 → loss should be NaN or handled gracefully
        labels_all_masked = torch.full((B, T), -100, dtype=torch.long)
        # PyTorch returns NaN when all positions are ignored
        loss = sft_loss(logits, labels_all_masked, V)
        assert torch.isnan(loss) or loss.item() == 0.0

    def test_different_from_full_loss(self):
        """Loss with masking should differ from unmasked loss."""
        torch.manual_seed(0)
        B, T, V = 2, 8, 100
        logits = torch.randn(B, T, V)
        # Half tokens masked
        labels = torch.randint(0, V, (B, T))
        labels_masked = labels.clone()
        labels_masked[:, :T // 2] = -100

        full_loss   = sft_loss(logits, labels, V)
        masked_loss = sft_loss(logits, labels_masked, V)
        assert abs(full_loss.item() - masked_loss.item()) > 1e-6

    def test_loss_is_finite_on_valid_labels(self):
        torch.manual_seed(1)
        B, T, V = 2, 8, 100
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        loss = sft_loss(logits, labels, V)
        assert torch.isfinite(loss)

    def test_loss_gradient_flows(self):
        """Gradients must reach model parameters on loss tokens."""
        torch.manual_seed(2)
        B, T, V = 2, 8, 100
        logits = torch.randn(B, T, V, requires_grad=True)
        labels = torch.randint(0, V, (B, T))
        labels[:, :4] = -100
        loss = sft_loss(logits, labels, V)
        loss.backward()
        assert logits.grad is not None
        # Gradients for masked positions must be zero
        assert (logits.grad[:, :4, :] == 0).all()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


class TestGetLR:
    def test_warmup_linear(self):
        cfg = SFTConfig()
        cfg.lr = 1e-5; cfg.min_lr = 1e-6; cfg.warmup_frac = 0.1
        total = 1000
        lr_start = get_lr(1,  total, cfg)
        lr_mid   = get_lr(50, total, cfg)   # middle of warmup
        assert lr_start < lr_mid

    def test_decays_after_warmup(self):
        cfg = SFTConfig()
        cfg.lr = 1e-5; cfg.min_lr = 1e-6; cfg.warmup_frac = 0.1
        total = 1000
        warmup = int(total * cfg.warmup_frac)
        lr_peak = get_lr(warmup, total, cfg)
        lr_late = get_lr(900, total, cfg)
        assert lr_late < lr_peak

    def test_approaches_min_lr(self):
        cfg = SFTConfig()
        cfg.lr = 1e-5; cfg.min_lr = 1e-6; cfg.warmup_frac = 0.1
        total = 1000
        lr_end = get_lr(total, total, cfg)
        assert abs(lr_end - cfg.min_lr) < 1e-9


# ---------------------------------------------------------------------------
# count_examples
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TOKENIZER, reason="tokenizer.json not present")
class TestCountExamples:
    def test_correct_count(self, tmp_path):
        data = _make_sft_data(tmp_path, n=20)
        n = count_examples(data, "train")
        assert n > 0
        assert n <= 20


# ---------------------------------------------------------------------------
# Training smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TOKENIZER, reason="tokenizer.json not present")
class TestSFTTrainSmoke:
    def test_loss_decreases_on_overfit(self, tmp_path):
        """10 gradient steps on a tiny dataset should reduce loss."""
        data = _make_sft_data(tmp_path, n=8, ctx=32)
        cfg  = _tiny_cfg(str(data), ctx=32)
        cfg.epochs = 20

        device = torch.device("cpu")
        model  = GPT(cfg.model_config()).to(device)
        opt    = make_optimizer(model, cfg)
        total_steps = 20

        losses = []
        for step in range(1, total_steps + 1):
            opt.zero_grad()
            x, y = sft_get_batch("train", data, cfg.ctx, cfg.batch_size, device)
            logits, _ = model(x, targets=None)
            loss = sft_loss(logits, y, cfg.vocab_size)
            if torch.isnan(loss):
                continue
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "loss did not decrease"

    def test_eval_loss_runs(self, tmp_path):
        data = _make_sft_data(tmp_path, n=10, ctx=32)
        cfg  = _tiny_cfg(str(data), ctx=32)
        device = torch.device("cpu")
        model  = GPT(cfg.model_config()).to(device)
        v = eval_loss(model, cfg, device)
        assert isinstance(v, float)
        assert v > 0

    def test_checkpoint_roundtrip(self, tmp_path):
        from train.checkpoint import save as ckpt_save, load as ckpt_load
        data = _make_sft_data(tmp_path, n=8, ctx=32)
        cfg  = _tiny_cfg(str(data), ctx=32)
        device = torch.device("cpu")
        model  = GPT(cfg.model_config()).to(device)
        opt    = make_optimizer(model, cfg)

        ckpt = tmp_path / "sft_test.pt"
        ckpt_save(ckpt, model, opt, step=5, loss=1.23)
        step, loss = ckpt_load(ckpt, model, opt)
        assert step == 5
        assert abs(loss - 1.23) < 1e-5
