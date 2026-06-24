# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Tests for the gradient-acceptance predicate (_is_gradient_acceptable).

This gate decides which scored gradients enter the aggregation that produces
the published proposal — training-integrity critical. It rejects below-threshold
scores and (outside warmup) gradients trained against a stale model version.
"""
from types import SimpleNamespace

from GenTRX.src.gradient_server import GradientAggregator

_accept = GradientAggregator._is_gradient_acceptable


def _agg(min_score, in_warmup):
    return SimpleNamespace(_effective_min_score=min_score, _in_warmup=in_warmup)


def test_below_or_at_threshold_rejected():
    a = _agg(0.5, in_warmup=False)
    assert _accept(a, 0.5, {}) is False
    assert _accept(a, 0.4, {}) is False


def test_above_threshold_accepted():
    a = _agg(0.5, in_warmup=False)
    assert _accept(a, 0.6, {}) is True


def test_version_mismatch_rejected_outside_warmup():
    a = _agg(0.5, in_warmup=False)
    assert _accept(a, 0.9, {"_version_mismatched": True}) is False


def test_version_mismatch_accepted_during_warmup():
    a = _agg(0.5, in_warmup=True)
    assert _accept(a, 0.9, {"_version_mismatched": True}) is True
