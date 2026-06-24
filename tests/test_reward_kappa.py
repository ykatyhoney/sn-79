# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Contract tests for the reward core.

The incentive mechanism had zero test coverage. These pin load-bearing
invariants of the Kappa-3 risk-adjusted-return maths (`kappa_3`) and the
score orchestrator's early-exit, exercising the real functions (no stubs).
"""
import pytest

from taos.im.utils.kappa import kappa_3
from taos.im.validator.reward import calculate_kappa_score

# kappa_3 gating knobs kept permissive so the maths, not the gates, is under test.
_K = dict(
    tau=0.0,
    lookback=100,
    norm_min=0.0,
    norm_max=2.0,
    min_lookback=1,
    min_realized_observations=3,
    grace_period=0,
    deregistered_uids=[],
    book_count=1,
)


def _series(pnls, step=1_000_000):
    """Build {timestamp: {book0: pnl}} from a list of per-period P&Ls."""
    return {i * step: {0: p} for i, p in enumerate(pnls)}


def test_kappa3_none_on_no_data():
    assert kappa_3(1, {}, **_K) is None


def test_kappa3_none_on_deregistered():
    k = dict(_K)
    k["deregistered_uids"] = [1]
    assert kappa_3(1, _series([1.0, 2.0, 3.0, 4.0]), **k) is None


def test_kappa3_returns_book_structure_for_sufficient_data():
    out = kappa_3(1, _series([1.0, 2.0, 1.5, 2.5, 1.0]), **_K)
    assert out is not None
    assert "books" in out and 0 in out["books"]


def test_kappa3_rewards_steady_over_volatile_for_same_mean():
    """Risk adjustment: a steady positive series must out-score a volatile one
    with the same mean (lower downside → higher Kappa-3)."""
    steady = kappa_3(1, _series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]), **_K)["books"][0]
    volatile = kappa_3(1, _series([8.0, -4.0, 8.0, -4.0, 8.0, -4.0]), **_K)["books"][0]
    assert steady is not None and volatile is not None
    assert steady > volatile


def test_calculate_kappa_score_full_path_regression():
    """Locks the orchestrator's full path (active + idle + None-kappa books, decay
    and P&L weighting enabled) to a known value. Guards the helper extraction in
    calculate_kappa_score against any change in calculation outcome."""
    S = 1_000_000_000
    ts = 50_000 * S
    uid = 7
    cfg = {
        "interval": 5_000_000_000,
        "max_inactive_books_ratio": 0.375,
        "kappa": {"normalization_min": -2.5, "normalization_max": 2.5,
                  "lookback": 10800_000_000_000, "pnl": {"impact": 0.5}},
        "activity": {"capital_turnover_cap": 10.0, "trade_volume_sampling_interval": 600_000_000_000,
                     "decay_grace_period": 600_000_000_000, "impact": 0.33, "decay_rate": 1.0},
    }
    score = calculate_kappa_score(
        uid=uid,
        kappa_values={uid: {"books": {0: 1.8, 1: -0.5, 2: None, 3: 0.9}}},
        activity_factors={uid: {0: 1.0, 1: 0.8, 2: 1.2, 3: 0.5}},
        pnl_factors={uid: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}},
        roundtrip_volumes={uid: {0: {ts - 100 * S: 3000.0}, 1: {ts - 30000 * S: 50.0}, 3: {ts - 700 * S: 1200.0}}},
        realized_pnl_history={uid: {ts - 200 * S: {0: 300.0, 1: -150.0, 3: 80.0}}},
        config=cfg,
        simulation_config={"miner_wealth": 50000.0, "publish_interval": 5 * S, "volumeDecimals": 4},
        simulation_timestamp=ts,
    )
    assert float(score) == pytest.approx(0.6848940067839999, abs=1e-12)


def test_calculate_kappa_score_unknown_uid_is_zero():
    """Score orchestrator early-exits to 0.0 for a miner with no kappa data."""
    score = calculate_kappa_score(
        uid=42,
        kappa_values={},
        activity_factors={},
        pnl_factors={},
        roundtrip_volumes={},
        realized_pnl_history={},
        config={"kappa": {"normalization_min": 0.0, "normalization_max": 2.0, "lookback": 10}},
        simulation_config={"miner_wealth": 1000.0, "publish_interval": 60, "volumeDecimals": 2},
        simulation_timestamp=1_000_000_000,
    )
    assert score == 0.0
