# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
PR-1 — aggregator data-integrity tests.

Three independent assertions, each pinning a behaviour the synthesised
review identified as a load-bearing correctness gap:

  1. ``aggregate()`` drops or sanitises non-finite deltas instead of
     letting one corrupt miner's NaN/Inf poison the mean (gradient.py).
  2. ``compute_loss(mask_nonfinite_rows=True)`` masks WHOLE rows where any
     field is non-finite, not just per-field (model.py). Prevents partial
     poisoning of cross-field aggregates during validator scoring.
  3. ``evaluate_gradient`` returns a strongly-negative score (so the
     gradient is rejected downstream) without running the val loop when
     ``apply_gradient`` leaves any model parameter non-finite (distributed.py).
"""
import math

import pytest
import torch

from GenTRX.src.gradient import (
    CompressedGradient,
    GradientDelta,
    GradientMetadata,
    aggregate,
    compress,
)
from GenTRX.src.model import compute_loss


def _make_cg(values: dict, miner_uid: int = 1) -> CompressedGradient:
    """Build a dense CompressedGradient from a {param_name: tensor} dict."""
    delta = GradientDelta(
        delta=dict(values),
        metadata=GradientMetadata(
            window_id=0,
            miner_uid=miner_uid,
            steps_trained=10,
            loss_before=1.0,
            loss_after=0.9,
            loss_trajectory=[1.0, 0.9],
            model_v_trained=0,
        ),
    )
    return compress(delta, top_k_frac=1.0)


# ── (1) aggregate() NaN/Inf handling ──────────────────────────────────────


def test_aggregate_drops_polluted_delta_when_clean_present():
    clean = _make_cg({"w": torch.tensor([1.0, 2.0, 3.0])}, miner_uid=1)
    nan_polluted = _make_cg(
        {"w": torch.tensor([float("nan"), 100.0, 100.0])}, miner_uid=2
    )
    inf_polluted = _make_cg(
        {"w": torch.tensor([float("inf"), 100.0, 100.0])}, miner_uid=3
    )

    agg = aggregate([clean, nan_polluted, inf_polluted])
    # `sparse` stores fully-dense values after aggregate; pull them out by
    # reconstructing from (indices, values, shape).
    indices, values, shape = agg.sparse["w"]
    dense = torch.zeros(shape.numel())
    dense[indices] = values
    dense = dense.reshape(shape)
    # Only the clean delta should contribute.
    assert torch.allclose(dense, torch.tensor([1.0, 2.0, 3.0]))


def test_aggregate_sanitises_when_all_polluted():
    """When every miner sent at least one non-finite entry, we still produce
    a finite aggregate by zeroing the bad positions rather than failing
    the whole round."""
    a = _make_cg({"w": torch.tensor([float("nan"), 2.0])}, miner_uid=1)
    b = _make_cg({"w": torch.tensor([5.0, float("inf")])}, miner_uid=2)
    agg = aggregate([a, b])
    indices, values, shape = agg.sparse["w"]
    dense = torch.zeros(shape.numel())
    dense[indices] = values
    dense = dense.reshape(shape)
    # Position 0: (0 + 5) / 2 = 2.5; position 1: (2 + 0) / 2 = 1.0
    assert torch.allclose(dense, torch.tensor([2.5, 1.0]))
    assert torch.isfinite(dense).all()


def test_aggregate_single_gradient_passthrough_unchanged():
    """Single-input aggregate is the identity — no scrub even if non-finite."""
    g = _make_cg({"w": torch.tensor([1.0, float("nan")])}, miner_uid=1)
    agg = aggregate([g])
    assert agg is g


def test_aggregate_all_clean_baseline_unchanged():
    """Two clean gradients average to the elementwise mean — confirms the
    sanitise path doesn't silently activate when nothing's wrong."""
    a = _make_cg({"w": torch.tensor([1.0, 2.0, 3.0])}, miner_uid=1)
    b = _make_cg({"w": torch.tensor([3.0, 4.0, 5.0])}, miner_uid=2)
    agg = aggregate([a, b])
    indices, values, shape = agg.sparse["w"]
    dense = torch.zeros(shape.numel())
    dense[indices] = values
    dense = dense.reshape(shape)
    assert torch.allclose(dense, torch.tensor([2.0, 3.0, 4.0]))


# ── (2) compute_loss whole-row mask ───────────────────────────────────────

# Field sizes MUST match the real schema: order_type has 3 classes (bid/ask/
# cancel) and _ORDER_TYPE_WEIGHTS is sized (3,). Mismatching n_types would
# fail inside F.cross_entropy(weight=...), not in our masking logic.
_N_TYPES = 3
_N_BINS = 16


def _toy_logits_labels(n_rows: int):
    return (
        {
            "order_type": torch.randn(n_rows, _N_TYPES),
            "price": torch.randn(n_rows, _N_BINS),
            "vol_int": torch.randn(n_rows, _N_BINS),
            "vol_dec": torch.randn(n_rows, _N_BINS),
            "interval": torch.randn(n_rows, _N_BINS),
        },
        {
            "order_type": torch.randint(0, _N_TYPES, (n_rows,)),
            "price": torch.randint(0, _N_BINS, (n_rows,)),
            "vol_int": torch.randint(0, _N_BINS, (n_rows,)),
            "vol_dec": torch.randint(0, _N_BINS, (n_rows,)),
            "interval": torch.randint(0, _N_BINS, (n_rows,)),
        },
    )


def test_compute_loss_whole_row_mask_drops_partial_rows():
    """When mask_nonfinite_rows=True, a row where ONE field has non-finite
    per-row loss must be dropped from EVERY field's reduction. Otherwise
    fields with finite values for that row still poison their mean."""
    torch.manual_seed(0)
    logits, labels = _toy_logits_labels(8)
    # Inject a NaN logit in the `price` field at row 3. This makes the
    # per-row loss at row 3 NaN for `price`; without the whole-row mask
    # the OTHER fields would still average row 3 normally.
    logits["price"][3] = float("nan")

    total, details = compute_loss(logits, labels, mask_nonfinite_rows=True)
    assert math.isfinite(total.item()), f"total loss must be finite, got {total}"
    for name, val in details.items():
        assert math.isfinite(val), f"field {name} loss must be finite (got {val})"


def test_compute_loss_all_nonfinite_returns_nan_per_field():
    """If EVERY row has at least one non-finite field, no rows survive the
    mask and each field's loss is NaN. Matches the prior fallback."""
    logits, labels = _toy_logits_labels(4)
    logits["order_type"][:] = float("nan")
    total, details = compute_loss(logits, labels, mask_nonfinite_rows=True)
    for name, val in details.items():
        assert math.isnan(val), f"field {name} should be NaN (got {val})"
    assert math.isnan(total.item())


def test_compute_loss_no_mask_is_bit_exact_training_path():
    """mask_nonfinite_rows=False is the training path — its behaviour MUST
    NOT change. Sanity-check that a clean batch produces finite, non-neg
    per-field losses."""
    torch.manual_seed(42)
    logits, labels = _toy_logits_labels(4)
    total, details = compute_loss(logits, labels, mask_nonfinite_rows=False)
    assert math.isfinite(total.item())
    for _name, val in details.items():
        assert math.isfinite(val) and val >= 0.0


# ── (3) evaluate_gradient post-apply finite check ─────────────────────────


def _make_metadata(miner_uid: int = 7) -> GradientMetadata:
    return GradientMetadata(
        window_id=0,
        miner_uid=miner_uid,
        steps_trained=0,
        loss_before=0.0,
        loss_after=0.0,
        loss_trajectory=[],
        model_v_trained=0,
    )


def test_evaluate_gradient_rejects_nonfinite_apply(monkeypatch):
    """If apply_gradient leaves any model param NaN/Inf, evaluate_gradient
    must short-circuit to loss_before - _MAX_EVAL_LOSS without invoking
    _eval_loss a second time (no risk of running scoring on a corrupt
    model). The model must be rolled back to its original state."""
    from GenTRX.src import distributed as dist_mod

    class _StubModel:
        """Minimal stand-in for the OrderModel surface evaluate_gradient
        uses: named_parameters(), state_dict(), load_state_dict()."""

        def __init__(self):
            self._params = {"w": torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))}

        def named_parameters(self):
            return list(self._params.items())

        def state_dict(self):
            return {k: v.detach().clone() for k, v in self._params.items()}

        def load_state_dict(self, sd):
            for k, v in sd.items():
                self._params[k] = torch.nn.Parameter(v.clone())

    model = _StubModel()

    # Track number of _eval_loss calls — the post-apply finite check must
    # short-circuit BEFORE the second call would happen.
    calls = {"n": 0}

    def fake_eval_loss(*a, **k):
        calls["n"] += 1
        return 5.0  # baseline loss_before

    monkeypatch.setattr(dist_mod, "_eval_loss", fake_eval_loss)
    monkeypatch.setattr(
        dist_mod,
        "decompress",
        lambda _: GradientDelta(
            delta={"w": torch.tensor([float("nan"), 0.0, 0.0])},
            metadata=_make_metadata(),
        ),
    )
    monkeypatch.setattr(
        dist_mod,
        "apply_gradient",
        lambda m, d: m._params.__setitem__(
            "w", torch.nn.Parameter(m._params["w"] + d.delta["w"])
        ),
    )

    score = dist_mod.evaluate_gradient(
        model=model,
        gradient=_make_cg({"w": torch.tensor([float("nan"), 0.0, 0.0])}, miner_uid=7),
        val_loader=None,  # never reached on the failure path
        device="cpu",
    )

    # Only the baseline _eval_loss should have been called.
    assert calls["n"] == 1, "post-apply path called _eval_loss more than once"
    # Score = loss_before - _MAX_EVAL_LOSS, strongly negative → rejected.
    assert score < -1000.0, f"non-finite apply must produce strongly-negative score, got {score}"
    # Rollback: model param restored to original values.
    assert torch.allclose(model._params["w"].detach(), torch.tensor([1.0, 2.0, 3.0]))


def test_evaluate_gradient_happy_path_calls_eval_twice(monkeypatch):
    """Sanity: when the apply produces a finite model, both baseline and
    post-apply _eval_loss are called and the score is the difference."""
    from GenTRX.src import distributed as dist_mod

    class _StubModel:
        def __init__(self):
            self._params = {"w": torch.nn.Parameter(torch.tensor([1.0, 2.0]))}

        def named_parameters(self):
            return list(self._params.items())

        def state_dict(self):
            return {k: v.detach().clone() for k, v in self._params.items()}

        def load_state_dict(self, sd):
            for k, v in sd.items():
                self._params[k] = torch.nn.Parameter(v.clone())

    losses = [10.0, 7.5]  # before, after
    calls = {"n": 0}

    def fake_eval(*a, **k):
        v = losses[calls["n"]]
        calls["n"] += 1
        return v

    monkeypatch.setattr(dist_mod, "_eval_loss", fake_eval)
    monkeypatch.setattr(
        dist_mod,
        "decompress",
        lambda _: GradientDelta(
            delta={"w": torch.tensor([0.1, 0.1])},
            metadata=_make_metadata(),
        ),
    )
    monkeypatch.setattr(
        dist_mod,
        "apply_gradient",
        lambda m, d: m._params.__setitem__(
            "w", torch.nn.Parameter(m._params["w"] + d.delta["w"])
        ),
    )

    score = dist_mod.evaluate_gradient(
        model=_StubModel(),
        gradient=_make_cg({"w": torch.tensor([0.1, 0.1])}, miner_uid=7),
        val_loader=None,
        device="cpu",
    )
    assert calls["n"] == 2
    assert score == pytest.approx(2.5, abs=1e-6)
