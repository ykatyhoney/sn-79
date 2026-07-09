# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Window-based distributed training primitives.

Wraps the existing training infrastructure to support the distributed protocol:
  1. train_window()  — train for N steps, return a GradientDelta
  2. apply_gradient() — apply an aggregated delta to the model
  3. evaluate_gradient() — score a gradient by loss improvement (for validators)

Usage:
    from GenTRX.src.distributed import train_window, WindowConfig

    # Miner side: train a window and get the gradient
    delta = train_window(model, dataloader, WindowConfig(n_steps=100), device)

    # Compress and share
    compressed = compress(delta, top_k_frac=0.05)
    data = serialize(compressed)  # → bytes for upload

    # Validator side: score a gradient
    score = evaluate_gradient(model, compressed, val_loader, device)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from GenTRX.src.gradient import (
    GradientDelta,
    GradientMetadata,
    CompressedGradient,
    snapshot_state,
    extract_delta,
    decompress,
    apply_gradient,
)
from GenTRX.src.model import OrderModel, compute_loss
from GenTRX.src.train import _forward_batch
from GenTRX.src.bt_log import gtx_log

logger = logging.getLogger(__name__)

# Returned instead of inf when an eval has no usable (finite) batches. A large
# finite value keeps comparisons/serialization normal — it sorts as "very bad"
# so scoring rejects and aggregation rolls back, without any inf in the system.
_MAX_EVAL_LOSS = 1e4


@dataclass
class WindowConfig:
    """Configuration for a single training window."""

    n_steps: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # Gradient accumulation: micro-batches summed per optimizer step. Effective
    # batch = dataloader batch_size * accum_steps. Lets weak hardware reach a
    # large effective batch (the overfit fix) without the memory of a real one.
    # 1 = no accumulation (step every micro-batch).
    accum_steps: int = 1
    window_id: int = 0
    miner_uid: int = 0
    model_version: int = 0
    label_smooth_sigma: float = 0.0
    # Wall-clock training budget (seconds). Used by train_incremental to stop
    # mid-run when the round deadline is near. None = no time limit.
    budget_s: float | None = None


def _train_step(model, batch, optimizer, device, label_smooth_sigma, grad_clip):
    """One optimizer step. Returns the loss, or None on a non-finite loss."""
    logits, labels = _forward_batch(model, batch, device)
    loss, _ = compute_loss(logits, labels, label_smooth_sigma=label_smooth_sigma)
    if not math.isfinite(loss.item()):
        return None
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


def train_window(
    model: OrderModel,
    dataloader: DataLoader,
    config: WindowConfig,
    device: str = "cpu",
) -> GradientDelta:
    """Train model for N steps and return the parameter delta.

    Snapshots θ_before, trains N steps on the dataloader, computes
    Δθ = θ_after - θ_before. Model is left in the trained state.

    Args:
        model: The model to train (will be modified in-place).
        dataloader: Training data. Will cycle if fewer batches than n_steps.
        config: Window training configuration.
        device: Device to train on.

    Returns:
        GradientDelta with the parameter change and training metadata.
    """
    theta_before = snapshot_state(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    model.train()
    loss_trajectory: list[float] = []
    step = 0
    accum = max(1, config.accum_steps)
    t0 = time.perf_counter()

    data_iter = iter(dataloader)
    optimizer.zero_grad()
    micro_losses: list[float] = []
    micro = 0
    # One "step" = one optimizer update over `accum` micro-batches (effective
    # batch = loader batch * accum). accum=1 reproduces per-batch stepping.
    while step < config.n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            try:
                batch = next(data_iter)
            except StopIteration:
                # Empty loader (no usable pages this window) — stop cleanly
                # rather than crash the miner; the tiny/empty delta is rejected
                # downstream by held-out scoring.
                logger.warning("window %d: empty dataloader, stopping", config.window_id)
                break

        logits, labels = _forward_batch(model, batch, device)
        loss, _ = compute_loss(logits, labels, label_smooth_sigma=config.label_smooth_sigma)
        lv = loss.item()
        if not math.isfinite(lv):
            logger.error("NaN/Inf loss at window step %d — stopping window", step)
            break
        (loss / accum).backward()
        micro_losses.append(lv)
        micro += 1
        if micro % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            loss_trajectory.append(sum(micro_losses) / len(micro_losses))
            micro_losses = []
            step += 1

    elapsed = time.perf_counter() - t0
    theta_after = snapshot_state(model)

    loss_before = loss_trajectory[0] if loss_trajectory else 0.0
    loss_after = loss_trajectory[-1] if loss_trajectory else 0.0

    metadata = GradientMetadata(
        window_id=config.window_id,
        miner_uid=config.miner_uid,
        steps_trained=step,
        loss_before=loss_before,
        loss_after=loss_after,
        loss_trajectory=loss_trajectory,
        model_v_trained=config.model_version,
    )

    delta = extract_delta(theta_before, theta_after, metadata)

    # Three full state-dict copies (theta_before, theta_after, delta) plus
    # the optimizer's Adam state buffers (2× model params) all live
    # simultaneously here. Drop the two snapshots and the optimizer before
    # returning so the caller's compress() works against a lower
    # high-water mark.
    del theta_before, theta_after, optimizer

    logger.info(
        "Window %d (miner %d): %d steps in %.1fs, loss %.4f → %.4f, Δnorm=%.4f",
        config.window_id,
        config.miner_uid,
        step,
        elapsed,
        loss_before,
        loss_after,
        delta.norm,
    )

    return delta


def train_incremental(
    model: OrderModel,
    dataloaders: list[DataLoader],
    config: WindowConfig,
    device: str = "cpu",
) -> GradientDelta:
    """Train one epoch per loader, in order, until the wall-clock budget runs out.

    Snapshots θ_before once, trains each loader for a full epoch on a shared
    optimizer, and stops as soon as `config.budget_s` elapses (checked every
    step). Loaders should be ordered most-relevant first (recent pages first),
    so the freshest data is always trained. The budget makes hardware
    self-limit: a fast GPU clears several loaders, a slow CPU does a partial
    pass of the first. `config.n_steps` (> 0) is an optional total-step cap;
    `config.budget_s = None` disables the time limit. Returns one GradientDelta
    over everything trained.
    """
    theta_before = snapshot_state(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    model.train()

    loss_trajectory: list[float] = []
    step = 0
    pages = 0
    budget = config.budget_s
    cap = config.n_steps if config.n_steps and config.n_steps > 0 else None
    t0 = time.perf_counter()
    stop = False

    for loader in dataloaders:
        if stop:
            break
        trained_any = False
        for batch in loader:
            if budget is not None and (time.perf_counter() - t0) >= budget:
                stop = True
                break
            if cap is not None and step >= cap:
                stop = True
                break
            loss_val = _train_step(
                model, batch, optimizer, device,
                config.label_smooth_sigma, config.grad_clip,
            )
            if loss_val is None:
                logger.error("NaN/Inf loss at step %d — stopping", step)
                stop = True
                break
            loss_trajectory.append(loss_val)
            step += 1
            trained_any = True
        if trained_any:
            pages += 1

    elapsed = time.perf_counter() - t0
    theta_after = snapshot_state(model)
    loss_before = loss_trajectory[0] if loss_trajectory else 0.0
    loss_after = loss_trajectory[-1] if loss_trajectory else 0.0

    metadata = GradientMetadata(
        window_id=config.window_id,
        miner_uid=config.miner_uid,
        steps_trained=step,
        loss_before=loss_before,
        loss_after=loss_after,
        loss_trajectory=loss_trajectory,
        model_v_trained=config.model_version,
    )
    delta = extract_delta(theta_before, theta_after, metadata)
    del theta_before, theta_after, optimizer

    logger.info(
        "Incremental (miner %d): %d/%d pages, %d steps in %.1fs, loss %.4f → %.4f, Δnorm=%.4f",
        config.miner_uid,
        pages,
        len(dataloaders),
        step,
        elapsed,
        loss_before,
        loss_after,
        delta.norm,
    )
    return delta


def apply_version_deltas(model, store, agg_uid, from_v: int, to_v: int, expected_shapes=None) -> int:
    """Apply canonical version deltas (from_v+1 .. to_v) to `model` in place.

    Returns the highest version reached: to_v on full success, or the last
    version applied before a delta went missing (e.g. pruned). Mutates the
    model, so callers that hit a gap should reload a baseline checkpoint and
    replay from there rather than trust a partial advance.
    """
    from GenTRX.src.gradient import deserialize

    reached = from_v
    for v in range(from_v + 1, to_v + 1):
        data = store.get_version_delta(agg_uid, v)
        if data is None:
            break
        comp = deserialize(data, expected_shapes=expected_shapes)
        apply_gradient(model, decompress(comp))
        reached = v
    return reached


def model_state_hash(model) -> str:
    """Deterministic sha256 of a model's parameters (sorted key order, CPU bytes).

    Used for the optional drift check: a miner that advanced via deltas can
    compare against the server-published hash and resync if they diverge.
    """
    import hashlib

    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate_gradient(
    model: OrderModel,
    gradient: CompressedGradient,
    val_loader: DataLoader,
    device: str = "cpu",
    max_batches: int = 50,
    label_smooth_sigma: float = 0.0,
    loss_before: float | None = None,
) -> float:
    """Score a gradient by the loss improvement it produces.

    Computes loss_before on the current model, applies the gradient,
    computes loss_after, then **rolls back** the model to its original state.
    `label_smooth_sigma` must match what the miner trained under, or the
    before/after comparison is across two different losses.

    `loss_before` may be supplied to skip the pre-apply eval: under the shared
    held-out window every miner is scored against the same base model on the
    same val loader, so the baseline is identical and can be computed once per
    round (single-pass scoring).

    Returns:
        score = loss_before - loss_after (positive = improvement)
    """
    # Snapshot current state for rollback
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    if loss_before is None:
        loss_before = _eval_loss(model, val_loader, device, max_batches, label_smooth_sigma)

    # Apply gradient
    delta = decompress(gradient)
    apply_gradient(model, delta)

    # Post-apply finite check on the model. If any parameter tensor went
    # non-finite (NaN/Inf from the apply), the val loss would be garbage
    # noise. Short-circuit to _MAX_EVAL_LOSS and roll back immediately.
    bad_params = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad_params:
        logger.warning(
            "evaluate_gradient (miner %d, window %d): non-finite model "
            "params after apply (%d tensors affected, e.g. %s) — rolling "
            "back, scoring as _MAX_EVAL_LOSS",
            gradient.metadata.miner_uid,
            gradient.metadata.window_id,
            len(bad_params),
            bad_params[:3],
        )
        model.load_state_dict(original_state)
        return loss_before - _MAX_EVAL_LOSS

    loss_after = _eval_loss(model, val_loader, device, max_batches, label_smooth_sigma)

    # Rollback
    model.load_state_dict(original_state)

    score = loss_before - loss_after
    logger.info(
        "Gradient eval (miner %d, window %d): loss %.4f → %.4f, score = %+.4f",
        gradient.metadata.miner_uid,
        gradient.metadata.window_id,
        loss_before,
        loss_after,
        score,
    )
    return score


def _log_bad_batch(i, batch, logits, labels, details):
    """Diagnostic dump for a val batch whose loss is non-finite on every row.
    Shows which field, whether logits overflowed (forward issue) vs not (label/
    input issue), label ranges, and input magnitudes — so the cause is visible
    without reproducing offline."""
    try:
        nonfinite_logits = {
            k: int((~torch.isfinite(v)).sum().item()) for k, v in logits.items()
        }
        logit_absmax = {
            k: round(float(v.detach().float().abs().max().item()), 3)
            for k, v in logits.items()
        }
        label_range = {
            k: (int(v.min().item()), int(v.max().item())) for k, v in labels.items()
        }
        input_absmax = {}
        if hasattr(batch, "items"):
            for k, v in batch.items():
                if torch.is_tensor(v) and v.numel():
                    input_absmax[k] = round(float(v.detach().float().abs().max().item()), 3)
        gtx_log.warning(
            "BAD VAL BATCH i=%d (all rows non-finite): per_field_loss=%s "
            "nonfinite_logits=%s logit_absmax=%s label_range=%s input_absmax=%s"
            % (i, details, nonfinite_logits, logit_absmax, label_range, input_absmax)
        )
    except Exception as exc:
        gtx_log.warning("bad-batch dump failed (i=%d): %s" % (i, exc))


def _eval_loss(
    model: OrderModel,
    loader: DataLoader,
    device: str,
    max_batches: int,
    label_smooth_sigma: float = 0.0,
) -> float:
    """Average loss over up to max_batches. Non-finite rows are masked within a
    batch; a batch is skipped (and dumped) only if every row is non-finite."""
    model.eval()
    total = 0.0
    n = 0
    n_bad = 0
    n_salvaged = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            logits, labels = _forward_batch(model, batch, device)
            loss, details = compute_loss(
                logits, labels, label_smooth_sigma=label_smooth_sigma,
                mask_nonfinite_rows=True,
            )
            lv = loss.item()
            if not math.isfinite(lv):
                n_bad += 1
                _log_bad_batch(i, batch, logits, labels, details)
                continue
            if not all(bool(torch.isfinite(v).all()) for v in logits.values()):
                n_salvaged += 1
            total += lv
            n += 1
    model.train()
    if n_bad or n_salvaged:
        gtx_log.warning(
            "eval: %d/%d batches non-finite, skipped; %d salvaged via row-masking"
            % (n_bad, n_bad + n, n_salvaged)
        )
    if n == 0:
        return _MAX_EVAL_LOSS if n_bad else 0.0
    return total / n


def _eval_loss_per_field(
    model: OrderModel,
    loader: DataLoader,
    device: str,
    max_batches: int,
    label_smooth_sigma: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Like `_eval_loss` but also returns the per-field CE breakdown."""
    model.eval()
    total = 0.0
    n = 0
    n_bad = 0
    n_salvaged = 0
    per_batch_losses: list[float] = []
    per_field_sum: dict[str, float] = {}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            logits, labels = _forward_batch(model, batch, device)
            loss, details = compute_loss(
                logits, labels, label_smooth_sigma=label_smooth_sigma,
                mask_nonfinite_rows=True,
            )
            lv = loss.item()
            # Always-on per-batch trace at DEBUG-level so a finer-grained
            # picture is one --logging.debug flip away. INFO summary below.
            per_batch_losses.append(lv)
            gtx_log.debug(
                "eval(per-field) batch %d: lv=%r finite=%s details=%s",
                i, lv, math.isfinite(lv), {k: round(v, 4) for k, v in details.items()},
            )
            if not math.isfinite(lv):
                n_bad += 1
                _log_bad_batch(i, batch, logits, labels, details)
                continue
            if not all(bool(torch.isfinite(v).all()) for v in logits.values()):
                n_salvaged += 1
            total += lv
            for name, val in details.items():
                per_field_sum[name] = per_field_sum.get(name, 0.0) + val
            n += 1
    model.train()
    # Always log a one-line trace summary so the eval's internal state is
    # visible even when nothing's flagged as bad. Useful when the rollback
    # log says ↑ inf but no BAD VAL BATCH was emitted — the per-batch and
    # cumulative-total values surface the actual progression.
    finite_batches = [v for v in per_batch_losses if math.isfinite(v)]
    summary_max = max(finite_batches) if finite_batches else float("nan")
    summary_min = min(finite_batches) if finite_batches else float("nan")
    gtx_log.info(
        "eval(per-field): n=%d n_bad=%d n_salvaged=%d total=%r batch_max=%.4g batch_min=%.4g per_batch=%s",
        n, n_bad, n_salvaged, total, summary_max, summary_min,
        [round(v, 4) if math.isfinite(v) else repr(v) for v in per_batch_losses],
    )
    if n_bad or n_salvaged:
        gtx_log.warning(
            "eval(per-field): %d/%d batches non-finite, skipped; %d salvaged via row-masking"
            % (n_bad, n_bad + n, n_salvaged)
        )
    if n == 0:
        return (_MAX_EVAL_LOSS if n_bad else 0.0), {}
    per_field_avg = {name: s / n for name, s in per_field_sum.items()}
    avg = total / n
    gtx_log.info(
        "eval(per-field) returning avg=%r (total=%r / n=%d)", avg, total, n,
    )
    return avg, per_field_avg
