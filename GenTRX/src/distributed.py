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
    window_id: int = 0
    miner_uid: int = 0
    model_version: int = 0
    label_smooth_sigma: float = 0.0


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
    t0 = time.perf_counter()

    data_iter = iter(dataloader)
    while step < config.n_steps:
        # Cycle through dataloader if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        logits, labels = _forward_batch(model, batch, device)
        loss, _ = compute_loss(logits, labels, label_smooth_sigma=config.label_smooth_sigma)

        if not math.isfinite(loss.item()):
            logger.error("NaN/Inf loss at window step %d — stopping window", step)
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        loss_trajectory.append(loss.item())
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


def evaluate_gradient(
    model: OrderModel,
    gradient: CompressedGradient,
    val_loader: DataLoader,
    device: str = "cpu",
    max_batches: int = 50,
    label_smooth_sigma: float = 0.0,
) -> float:
    """Score a gradient by the loss improvement it produces.

    Computes loss_before on the current model, applies the gradient,
    computes loss_after, then **rolls back** the model to its original state.
    `label_smooth_sigma` must match what the miner trained under, or the
    before/after comparison is across two different losses.

    Returns:
        score = loss_before - loss_after (positive = improvement)
    """
    # Snapshot current state for rollback
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    loss_before = _eval_loss(model, val_loader, device, max_batches, label_smooth_sigma)

    # Apply gradient
    delta = decompress(gradient)
    apply_gradient(model, delta)

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
