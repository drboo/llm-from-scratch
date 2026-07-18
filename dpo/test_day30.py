"""
Day 30 tests — DPO (Direct Preference Optimization).

Tests cover:
  - DPO loss math (zero-margin case, gradient direction, log-sigmoid formula)
  - sequence_logprobs (shape, masking, always negative)
  - Preference dataset encoding
  - Training: ref model stays frozen, policy log-probs diverge from ref
  - Implicit reward is positive for chosen over rejected after training

Run:  pytest dpo/test_day30.py -v
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import GPT, ModelConfig
from train.dpo import (
    sequence_logprobs, dpo_loss, DPOConfig, _eval,
    _infer_n_head, _infer_n_kv_head, _build_nano, _load_model,
)
from dpo.data import (
    PREFERENCE_PAIRS,
    TRAIN_PAIRS,
    VAL_PAIRS,
    PreferencePair,
    encode_preference_pair,
    load_split_inmemory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nano() -> GPT:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, d_model=64, n_head=2, n_layer=2, ctx=64)
    return GPT(cfg).eval()


def _nano_gqa(n_kv_head: int = 1) -> GPT:
    """Nano model with GQA (n_head=2, n_kv_head=1 = MQA by default)."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, d_model=64, n_head=2,
                      n_kv_head=n_kv_head, n_layer=2, ctx=64)
    return GPT(cfg).eval()


def _ids_labels(prompt_len: int = 4, response_len: int = 6,
                vocab: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, labels) with prompt positions masked."""
    torch.manual_seed(7)
    total = prompt_len + response_len
    ids    = torch.randint(1, vocab, (1, total), dtype=torch.long)
    labels = ids.clone()
    labels[:, :prompt_len] = -100
    return ids, labels


# ---------------------------------------------------------------------------
# DPO loss math
# ---------------------------------------------------------------------------


class TestDPOLoss:
    def test_loss_is_positive_scalar(self):
        B = 3
        pol_c = torch.tensor([-1.5, -2.0, -1.0])
        pol_r = torch.tensor([-2.5, -3.0, -2.0])
        ref_c = torch.tensor([-2.0, -2.5, -1.5])
        ref_r = torch.tensor([-3.0, -3.5, -2.5])
        loss, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        assert loss.item() > 0
        assert loss.shape == ()

    def test_zero_margin_loss_equals_log2(self):
        """When chosen and rejected are equal, loss = log(2) ≈ 0.6931."""
        pol_c = torch.tensor([-1.0])
        pol_r = torch.tensor([-1.0])
        ref_c = torch.tensor([-1.0])
        ref_r = torch.tensor([-1.0])
        loss, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        assert abs(loss.item() - 0.6931) < 1e-3

    def test_loss_decreases_with_larger_margin(self):
        """Bigger chosen-over-rejected margin → lower DPO loss."""
        ref_c = torch.tensor([-1.0])
        ref_r = torch.tensor([-1.0])
        loss_small, _ = dpo_loss(
            torch.tensor([-0.9]), torch.tensor([-1.1]), ref_c, ref_r, beta=0.1)
        loss_large, _ = dpo_loss(
            torch.tensor([-0.5]), torch.tensor([-1.5]), ref_c, ref_r, beta=0.1)
        assert loss_large < loss_small

    def test_beta_scales_margin(self):
        """Higher beta amplifies the margin, reducing loss further."""
        pol_c = torch.tensor([-0.5])
        pol_r = torch.tensor([-1.5])
        ref_c = torch.tensor([-1.0])
        ref_r = torch.tensor([-1.0])
        loss_lo, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.01)
        loss_hi, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=1.0)
        assert loss_hi < loss_lo

    def test_margin_is_positive_for_preferred_chosen(self):
        """Margin should be positive when the policy prefers chosen over rejected."""
        pol_c = torch.tensor([-0.5, -1.0])
        pol_r = torch.tensor([-2.5, -3.0])
        ref_c = torch.tensor([-1.5, -2.0])
        ref_r = torch.tensor([-2.0, -2.5])
        _, margin = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        assert margin.item() > 0

    def test_gradient_flows_through_loss(self):
        """Gradients must flow back to the policy log-probs."""
        pol_c = torch.tensor([-1.0], requires_grad=True)
        pol_r = torch.tensor([-2.0], requires_grad=True)
        ref_c = torch.tensor([-1.5])
        ref_r = torch.tensor([-2.5])
        loss, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        loss.backward()
        assert pol_c.grad is not None
        assert pol_r.grad is not None

    def test_chosen_grad_negative_rejected_grad_positive(self):
        """
        Loss decreases when chosen logps go up and rejected logps go down.
        So d_loss/d_pol_chosen < 0 and d_loss/d_pol_rejected > 0.
        """
        pol_c = torch.tensor([-1.5], requires_grad=True)
        pol_r = torch.tensor([-2.5], requires_grad=True)
        ref_c = torch.tensor([-1.5])
        ref_r = torch.tensor([-2.5])
        loss, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        loss.backward()
        assert pol_c.grad.item() < 0
        assert pol_r.grad.item() > 0

    def test_batch_loss_aggregation(self):
        """Loss with B=4 must be a scalar."""
        B = 4
        pol_c = torch.randn(B)
        pol_r = torch.randn(B)
        ref_c = torch.randn(B)
        ref_r = torch.randn(B)
        loss, margin = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        assert loss.shape == ()
        assert margin.shape == ()


# ---------------------------------------------------------------------------
# sequence_logprobs
# ---------------------------------------------------------------------------


class TestSequenceLogprobs:
    def test_returns_scalar(self):
        model = _nano()
        ids, labels = _ids_labels()
        lp = sequence_logprobs(model, ids, labels)
        assert lp.shape == ()

    def test_always_negative(self):
        """Log-probabilities are always ≤ 0."""
        torch.manual_seed(0)
        model = _nano()
        for _ in range(5):
            ids, labels = _ids_labels(prompt_len=3, response_len=8)
            lp = sequence_logprobs(model, ids, labels)
            assert lp.item() <= 0.0

    def test_masked_positions_excluded(self):
        """Masking out response tokens should give 0 (or near-0) logp."""
        model = _nano()
        ids = torch.randint(1, 256, (1, 10), dtype=torch.long)
        labels_all_masked = torch.full_like(ids, -100)
        lp = sequence_logprobs(model, ids, labels_all_masked)
        assert abs(lp.item()) < 1e-5

    def test_more_response_tokens_lower_logp(self):
        """Summing log-probs over more tokens gives a more negative value."""
        model = _nano()
        ids_short, labels_short = _ids_labels(prompt_len=6, response_len=2)
        ids_long,  labels_long  = _ids_labels(prompt_len=2, response_len=6)
        lp_short = sequence_logprobs(model, ids_short, labels_short)
        lp_long  = sequence_logprobs(model, ids_long,  labels_long)
        assert lp_long.item() < lp_short.item()

    def test_gradient_flows(self):
        model = _nano()
        model.train()
        ids, labels = _ids_labels()
        lp = sequence_logprobs(model, ids, labels)
        lp.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# Preference dataset
# ---------------------------------------------------------------------------


class TestPreferenceData:
    def test_dataset_size(self):
        assert len(PREFERENCE_PAIRS) == 40
        assert len(TRAIN_PAIRS) + len(VAL_PAIRS) == 40

    def test_val_is_nonempty(self):
        assert len(VAL_PAIRS) >= 1

    def test_encode_returns_ids_and_labels(self):
        pair = PREFERENCE_PAIRS[0]
        result = encode_preference_pair(pair, codec=None, max_tokens=512)
        assert result is not None
        (c_ids, c_lab), (r_ids, r_lab) = result
        assert len(c_ids) == len(c_lab)
        assert len(r_ids) == len(r_lab)

    def test_labels_have_neg100_on_prompt(self):
        pair = PREFERENCE_PAIRS[0]
        result = encode_preference_pair(pair, codec=None, max_tokens=512)
        (_, c_lab), _ = result
        assert -100 in c_lab
        # At least one response token must NOT be -100
        assert any(l != -100 for l in c_lab)

    def test_chosen_longer_than_rejected(self):
        """The chosen response is generally longer (higher quality)."""
        pair = PREFERENCE_PAIRS[0]
        result = encode_preference_pair(pair, codec=None, max_tokens=512)
        (c_ids, _), (r_ids, _) = result
        assert len(c_ids) >= len(r_ids)

    def test_max_tokens_filter(self):
        pair = PREFERENCE_PAIRS[0]
        # Tiny max_tokens should cause None return
        result = encode_preference_pair(pair, codec=None, max_tokens=5)
        assert result is None

    def test_load_split_inmemory(self):
        examples = load_split_inmemory(TRAIN_PAIRS, codec=None)
        assert len(examples) > 0
        for (c_ids, c_lab), (r_ids, r_lab) in examples:
            assert len(c_ids) > 0
            assert len(r_ids) > 0

    def test_all_pairs_have_prompt_chosen_rejected(self):
        for pair in PREFERENCE_PAIRS:
            assert pair.prompt
            assert pair.chosen
            assert pair.rejected

    def test_chosen_differs_from_rejected(self):
        for pair in PREFERENCE_PAIRS:
            assert pair.chosen != pair.rejected


# ---------------------------------------------------------------------------
# Training dynamics
# ---------------------------------------------------------------------------


class TestDPOTrainingDynamics:
    def _make_pair_tensors(self, model: GPT, device: torch.device):
        """Create a synthetic short (chosen, rejected) pair for testing.

        We build tensors directly rather than encoding real preference pairs
        so the dynamics tests are independent of context length constraints.
        prompt_len=4, response_len=8 fits comfortably in ctx=64.
        """
        ctx      = model.cfg.ctx
        vocab    = model.cfg.vocab_size
        torch.manual_seed(42)

        def _make(resp_seed: int):
            torch.manual_seed(resp_seed)
            prompt_len  = 4
            resp_len    = 8
            prompt_ids  = torch.randint(1, vocab, (prompt_len,), dtype=torch.long)
            resp_ids    = torch.randint(1, vocab, (resp_len,),   dtype=torch.long)
            ids    = torch.cat([prompt_ids, resp_ids]).unsqueeze(0).to(device)
            labels = ids.clone()
            labels[:, :prompt_len] = -100
            return ids, labels

        c_ids_t, c_lab_t = _make(1)
        r_ids_t, r_lab_t = _make(2)   # different response tokens
        return c_ids_t, c_lab_t, r_ids_t, r_lab_t

    def test_ref_model_frozen_during_training(self):
        device    = torch.device("cpu")
        ref_model = _nano().to(device)
        policy    = copy.deepcopy(ref_model)
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        ref_params_before = [p.clone() for p in ref_model.parameters()]

        c_ids, c_lab, r_ids, r_lab = self._make_pair_tensors(ref_model, device)

        with torch.no_grad():
            ref_c = sequence_logprobs(ref_model, c_ids, c_lab)
            ref_r = sequence_logprobs(ref_model, r_ids, r_lab)

        pol_c = sequence_logprobs(policy, c_ids, c_lab)
        pol_r = sequence_logprobs(policy, r_ids, r_lab)
        loss, _ = dpo_loss(pol_c.unsqueeze(0), pol_r.unsqueeze(0),
                           ref_c.unsqueeze(0), ref_r.unsqueeze(0), beta=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Reference model must NOT have changed
        for before, after in zip(ref_params_before, ref_model.parameters()):
            assert torch.equal(before, after), "Reference model was modified!"

    def test_policy_diverges_from_ref_after_steps(self):
        """Policy weights must change after a few DPO gradient steps."""
        device    = torch.device("cpu")
        ref_model = _nano().to(device)
        policy    = copy.deepcopy(ref_model)
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        policy_before = copy.deepcopy(policy.state_dict())

        c_ids, c_lab, r_ids, r_lab = self._make_pair_tensors(ref_model, device)

        for _ in range(5):
            with torch.no_grad():
                ref_c = sequence_logprobs(ref_model, c_ids, c_lab)
                ref_r = sequence_logprobs(ref_model, r_ids, r_lab)
            pol_c = sequence_logprobs(policy, c_ids, c_lab)
            pol_r = sequence_logprobs(policy, r_ids, r_lab)
            loss, _ = dpo_loss(pol_c.unsqueeze(0), pol_r.unsqueeze(0),
                               ref_c.unsqueeze(0), ref_r.unsqueeze(0), beta=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        changed = any(
            not torch.equal(policy_before[k], policy.state_dict()[k])
            for k in policy_before
        )
        assert changed, "Policy weights did not change after DPO steps"

    def test_implicit_reward_moves_positive_after_training(self):
        """
        After training on a pair, the implicit reward for the chosen
        response should be larger than for the rejected response:
            r_chosen > r_rejected  ↔  (pol_c - ref_c) > (pol_r - ref_r)
        """
        device    = torch.device("cpu")
        ref_model = _nano().to(device)
        policy    = copy.deepcopy(ref_model)
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)

        c_ids, c_lab, r_ids, r_lab = self._make_pair_tensors(ref_model, device)

        with torch.no_grad():
            ref_c0 = sequence_logprobs(ref_model, c_ids, c_lab).item()
            ref_r0 = sequence_logprobs(ref_model, r_ids, r_lab).item()

        # Train for several steps
        for _ in range(30):
            with torch.no_grad():
                ref_c = sequence_logprobs(ref_model, c_ids, c_lab)
                ref_r = sequence_logprobs(ref_model, r_ids, r_lab)
            pol_c = sequence_logprobs(policy, c_ids, c_lab)
            pol_r = sequence_logprobs(policy, r_ids, r_lab)
            loss, _ = dpo_loss(pol_c.unsqueeze(0), pol_r.unsqueeze(0),
                               ref_c.unsqueeze(0), ref_r.unsqueeze(0), beta=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check implicit reward margin
        policy.eval()
        with torch.no_grad():
            ref_c_f = sequence_logprobs(ref_model, c_ids, c_lab).item()
            ref_r_f = sequence_logprobs(ref_model, r_ids, r_lab).item()
            pol_c_f = sequence_logprobs(policy, c_ids, c_lab).item()
            pol_r_f = sequence_logprobs(policy, r_ids, r_lab).item()

        reward_chosen   = pol_c_f - ref_c_f
        reward_rejected = pol_r_f - ref_r_f
        assert reward_chosen > reward_rejected, (
            f"Implicit reward: chosen={reward_chosen:.4f} "
            f"rejected={reward_rejected:.4f} — margin should be positive"
        )

    def test_loss_decreases_over_steps(self):
        """Training loss should decrease (or at least not systematically increase)."""
        device    = torch.device("cpu")
        ref_model = _nano().to(device)
        policy    = copy.deepcopy(ref_model)
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)

        c_ids, c_lab, r_ids, r_lab = self._make_pair_tensors(ref_model, device)

        losses = []
        for _ in range(20):
            with torch.no_grad():
                ref_c = sequence_logprobs(ref_model, c_ids, c_lab)
                ref_r = sequence_logprobs(ref_model, r_ids, r_lab)
            pol_c = sequence_logprobs(policy, c_ids, c_lab)
            pol_r = sequence_logprobs(policy, r_ids, r_lab)
            loss, _ = dpo_loss(pol_c.unsqueeze(0), pol_r.unsqueeze(0),
                               ref_c.unsqueeze(0), ref_r.unsqueeze(0), beta=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # First quarter should be higher than last quarter on average
        first_mean = sum(losses[:5])  / 5
        last_mean  = sum(losses[-5:]) / 5
        assert last_mean < first_mean, (
            f"Loss did not decrease: first={first_mean:.4f} last={last_mean:.4f}"
        )


# ---------------------------------------------------------------------------
# Day 29 × Day 30: DPO on GQA models
# ---------------------------------------------------------------------------


class TestDPOWithGQA:
    """GQA models (Day 29) must work end-to-end with DPO (Day 30)."""

    def test_sequence_logprobs_gqa(self):
        """sequence_logprobs must work on a GQA model."""
        model = _nano_gqa(n_kv_head=1)
        ids, labels = _ids_labels()
        lp = sequence_logprobs(model, ids, labels)
        assert lp.shape == ()
        assert lp.item() <= 0.0

    def test_dpo_loss_gqa_gradient_flows(self):
        """Gradients from DPO loss must flow through a GQA policy."""
        model = _nano_gqa(n_kv_head=1)
        model.train()
        ids, labels = _ids_labels()
        ids2, labels2 = _ids_labels(prompt_len=3, response_len=7)

        pol_c = sequence_logprobs(model, ids,  labels)
        pol_r = sequence_logprobs(model, ids2, labels2)

        with torch.no_grad():
            ref_c = sequence_logprobs(_nano_gqa(), ids,  labels)
            ref_r = sequence_logprobs(_nano_gqa(), ids2, labels2)

        loss, _ = dpo_loss(pol_c.unsqueeze(0), pol_r.unsqueeze(0),
                           ref_c.unsqueeze(0), ref_r.unsqueeze(0), beta=0.1)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_gqa_ref_frozen(self):
        """Reference GQA model must not change during a policy update."""
        device    = torch.device("cpu")
        ref_model = _nano_gqa().to(device)
        policy    = copy.deepcopy(ref_model)
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        ref_params_before = [p.clone() for p in ref_model.parameters()]

        ids, labels   = _ids_labels()
        ids2, labels2 = _ids_labels(prompt_len=3, response_len=7)
        ids   = ids.to(device);   labels   = labels.to(device)
        ids2  = ids2.to(device);  labels2  = labels2.to(device)

        with torch.no_grad():
            ref_c = sequence_logprobs(ref_model, ids,  labels)
            ref_r = sequence_logprobs(ref_model, ids2, labels2)
        pol_c = sequence_logprobs(policy, ids,  labels)
        pol_r = sequence_logprobs(policy, ids2, labels2)
        loss, _ = dpo_loss(pol_c.unsqueeze(0), pol_r.unsqueeze(0),
                           ref_c.unsqueeze(0), ref_r.unsqueeze(0), beta=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for before, after in zip(ref_params_before, ref_model.parameters()):
            assert torch.equal(before, after), "GQA ref model was modified!"

    def test_infer_n_head_mha(self):
        """_infer_n_head recovers n_head from MHA rope buffer."""
        model = _nano()   # n_head=2, d_model=64
        sd    = model.state_dict()
        assert _infer_n_head(sd, d_model=64) == 2

    def test_infer_n_head_gqa(self):
        """_infer_n_head recovers n_head from GQA model (same rope, n_head unchanged)."""
        model = _nano_gqa(n_kv_head=1)   # n_head=2, n_kv_head=1
        sd    = model.state_dict()
        assert _infer_n_head(sd, d_model=64) == 2

    def test_infer_n_kv_head_mha(self):
        """_infer_n_kv_head returns n_head for a standard MHA model."""
        model = _nano()   # MHA: n_head=2, n_kv_head=0 → kv_heads=2
        sd    = model.state_dict()
        inferred = _infer_n_kv_head(sd, n_head=2, d_model=64)
        assert inferred == 2

    def test_infer_n_kv_head_gqa(self):
        """_infer_n_kv_head correctly recovers n_kv_head from GQA state dict."""
        model = _nano_gqa(n_kv_head=1)   # MQA
        sd    = model.state_dict()
        inferred = _infer_n_kv_head(sd, n_head=2, d_model=64)
        assert inferred == 1

    def test_load_model_gqa_from_tempfile(self, tmp_path):
        """_load_model must reconstruct a GQA model from a saved checkpoint."""
        model = _nano_gqa(n_kv_head=1)
        ckpt_path = tmp_path / "gqa.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        loaded = _load_model(str(ckpt_path), ctx=64, device=torch.device("cpu"))
        assert loaded.cfg.kv_heads == 1
        assert isinstance(loaded, GPT)

    def test_gqa_implicit_reward_positive_after_training(self):
        """Implicit reward margin should be positive after DPO on a GQA model."""
        device    = torch.device("cpu")
        ref_model = _nano_gqa(n_kv_head=1).to(device)
        policy    = copy.deepcopy(ref_model)
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)

        torch.manual_seed(0)
        c_ids, c_lab = _ids_labels(prompt_len=4, response_len=8)
        torch.manual_seed(99)
        r_ids, r_lab = _ids_labels(prompt_len=4, response_len=6)
        c_ids = c_ids.to(device); c_lab = c_lab.to(device)
        r_ids = r_ids.to(device); r_lab = r_lab.to(device)

        for _ in range(30):
            with torch.no_grad():
                ref_c = sequence_logprobs(ref_model, c_ids, c_lab)
                ref_r = sequence_logprobs(ref_model, r_ids, r_lab)
            pol_c = sequence_logprobs(policy, c_ids, c_lab)
            pol_r = sequence_logprobs(policy, r_ids, r_lab)
            loss, _ = dpo_loss(pol_c.unsqueeze(0), pol_r.unsqueeze(0),
                               ref_c.unsqueeze(0), ref_r.unsqueeze(0), beta=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        policy.eval()
        with torch.no_grad():
            rc  = sequence_logprobs(ref_model, c_ids, c_lab).item()
            rr  = sequence_logprobs(ref_model, r_ids, r_lab).item()
            pc  = sequence_logprobs(policy,    c_ids, c_lab).item()
            pr  = sequence_logprobs(policy,    r_ids, r_lab).item()

        assert (pc - rc) > (pr - rr), (
            f"GQA implicit reward: chosen={pc-rc:.4f} rejected={pr-rr:.4f}"
        )

    def test_gqa_kv_cache_smaller_during_dpo_inference(self):
        """GQA model has smaller KV cache than MHA for DPO generation."""
        from model.kv_cache import KVCache
        mha = _nano()
        gqa = _nano_gqa(n_kv_head=1)
        cache_mha = KVCache.for_model(mha, max_seq_len=64, device=torch.device("cpu"))
        cache_gqa = KVCache.for_model(gqa, max_seq_len=64, device=torch.device("cpu"))
        mha_kv_bytes = cache_mha.k[0].numel() * cache_mha.k[0].element_size()
        gqa_kv_bytes = cache_gqa.k[0].numel() * cache_gqa.k[0].element_size()
        assert gqa_kv_bytes < mha_kv_bytes
