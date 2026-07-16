"""
Day 10 checkpoint tests — production training loop.

Run:  pytest train/test_day10.py -v -s

All tests use tiny models and synthetic data so the suite stays fast.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import GPT, ModelConfig
from train.pretrain import TrainConfig, get_lr, make_optimizer
from train.checkpoint import save as ckpt_save, load as ckpt_load
from train.overfit import fixed_batch

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------

TINY_MODEL = ModelConfig(vocab_size=256, d_model=64, n_head=4, n_layer=2, ctx=32)
TINY_TRAIN = TrainConfig(
    max_steps=60, warmup_steps=10,
    lr=3e-3, min_lr=3e-4,
    batch_size=2, accum_steps=2, ctx=32,
)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def test_lr_zero_at_step_zero():
    assert get_lr(0, TINY_TRAIN) == pytest.approx(0.0)


def test_lr_max_at_warmup_end():
    assert get_lr(TINY_TRAIN.warmup_steps, TINY_TRAIN) == pytest.approx(TINY_TRAIN.lr)


def test_lr_min_at_end():
    assert get_lr(TINY_TRAIN.max_steps, TINY_TRAIN) == pytest.approx(TINY_TRAIN.min_lr)


def test_lr_monotone_during_warmup():
    lrs = [get_lr(s, TINY_TRAIN) for s in range(TINY_TRAIN.warmup_steps + 1)]
    assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))


def test_lr_monotone_during_decay():
    lrs = [get_lr(s, TINY_TRAIN) for s in range(TINY_TRAIN.warmup_steps, TINY_TRAIN.max_steps + 1)]
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


def test_lr_between_min_and_max():
    for step in range(0, TINY_TRAIN.max_steps + 1, 5):
        lr = get_lr(step, TINY_TRAIN)
        # During warmup LR grows from 0; after warmup it stays in [min_lr, lr]
        assert 0.0 <= lr <= TINY_TRAIN.lr + 1e-9
        if step >= TINY_TRAIN.warmup_steps:
            assert lr >= TINY_TRAIN.min_lr - 1e-9


# ---------------------------------------------------------------------------
# Optimizer param groups
# ---------------------------------------------------------------------------


def test_weight_decay_param_groups():
    """2D params get weight_decay; 1D params (norms) do not."""
    model = GPT(TINY_MODEL)
    opt   = make_optimizer(model, TINY_TRAIN)
    assert len(opt.param_groups) == 2
    wd_group    = next(g for g in opt.param_groups if g["weight_decay"] > 0)
    no_wd_group = next(g for g in opt.param_groups if g["weight_decay"] == 0)
    for p in wd_group["params"]:
        assert p.dim() >= 2
    for p in no_wd_group["params"]:
        assert p.dim() < 2


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


def test_grad_clip():
    """After clip_grad_norm_, total norm must be ≤ max_norm."""
    torch.manual_seed(0)
    model  = GPT(TINY_MODEL)
    x, y   = fixed_batch(None, TINY_MODEL.ctx, 4, TINY_MODEL.vocab_size, torch.device("cpu"))
    _, loss = model(x, y)
    loss.backward()

    max_norm = 1.0
    norm_before = sum(p.grad.norm() ** 2 for p in model.parameters()
                      if p.grad is not None).sqrt().item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    norm_after = sum(p.grad.norm() ** 2 for p in model.parameters()
                     if p.grad is not None).sqrt().item()

    assert norm_after <= max_norm + 1e-5, f"Grad norm {norm_after:.4f} > {max_norm}"
    if norm_before > max_norm:
        assert norm_after < norm_before  # clipping actually happened


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------


def test_grad_accumulation_equals_full_batch():
    """Gradient from 2 accumulated micro-steps == gradient from 1 double batch."""
    torch.manual_seed(1)
    model = GPT(TINY_MODEL)

    # Full batch
    B, T = 4, TINY_MODEL.ctx
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(0, TINY_MODEL.vocab_size, (B, T + 1), generator=g)
    x_full, y_full = seq[:, :-1], seq[:, 1:]
    _, loss_full = model(x_full, y_full)
    loss_full.backward()
    grads_full = [p.grad.clone() for p in model.parameters()]

    # Two micro-steps of B//2
    model.zero_grad()
    for i in range(2):
        xi = x_full[i * B // 2: (i + 1) * B // 2]
        yi = y_full[i * B // 2: (i + 1) * B // 2]
        _, loss_micro = model(xi, yi)
        (loss_micro / 2).backward()   # divide by accum_steps=2
    grads_accum = [p.grad.clone() for p in model.parameters()]

    for gf, ga in zip(grads_full, grads_accum):
        assert torch.allclose(gf, ga, atol=1e-5), \
            f"Accumulated grad differs from full-batch grad (max diff {(gf-ga).abs().max():.2e})"


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip(tmp_path):
    """Save and reload: state_dicts and step must be identical."""
    torch.manual_seed(0)
    model = GPT(TINY_MODEL)
    opt   = make_optimizer(model, TINY_TRAIN)

    # Dummy step so optimizer has momentum state
    x, y = fixed_batch(None, TINY_MODEL.ctx, 2, TINY_MODEL.vocab_size, torch.device("cpu"))
    _, loss = model(x, y)
    loss.backward()
    opt.step()
    opt.zero_grad()

    ckpt_path = tmp_path / "test.pt"
    ckpt_save(ckpt_path, model, opt, step=1, loss=loss.item())

    # New model + optimizer — load into them
    model2 = GPT(TINY_MODEL)
    opt2   = make_optimizer(model2, TINY_TRAIN)
    step, loaded_loss = ckpt_load(ckpt_path, model2, opt2)

    assert step == 1
    assert abs(loaded_loss - loss.item()) < 1e-6

    # Model weights must match
    for (n, p1), (_, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1, p2), f"Parameter mismatch after load: {n}"


def test_checkpoint_resume_unbroken_loss(tmp_path):
    """Train 60 steps == train 30 + save + resume + train 30 (identical losses)."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    x, y   = fixed_batch(None, TINY_MODEL.ctx, 2, TINY_MODEL.vocab_size, device)

    def run(model, opt, n_steps):
        losses = []
        for _ in range(n_steps):
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        return losses

    # ── Run A: 60 uninterrupted steps ───────────────────────────────────────
    torch.manual_seed(0)
    model_A = GPT(TINY_MODEL)
    opt_A   = make_optimizer(model_A, TINY_TRAIN)
    losses_A = run(model_A, opt_A, 60)

    # ── Run B: 30 steps, checkpoint, resume, 30 more ────────────────────────
    torch.manual_seed(0)
    model_B = GPT(TINY_MODEL)
    opt_B   = make_optimizer(model_B, TINY_TRAIN)
    losses_B1 = run(model_B, opt_B, 30)

    ckpt_path = tmp_path / "mid.pt"
    ckpt_save(ckpt_path, model_B, opt_B, step=30, loss=losses_B1[-1])

    model_C = GPT(TINY_MODEL)
    opt_C   = make_optimizer(model_C, TINY_TRAIN)
    ckpt_load(ckpt_path, model_C, opt_C)
    losses_B2 = run(model_C, opt_C, 30)

    losses_B = losses_B1 + losses_B2

    # All 60 losses must match the uninterrupted run
    for i, (a, b) in enumerate(zip(losses_A, losses_B)):
        assert abs(a - b) < 1e-5, f"Loss diverged at step {i+1}: {a:.6f} vs {b:.6f}"
