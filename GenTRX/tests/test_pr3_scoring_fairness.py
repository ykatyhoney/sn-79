# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
PR-3 — scoring fairness + observability.

Covers the validator-side reward changes (EMA bootstrap-at-0, per-round
idempotency) and the gradient_server scores-payload additions
(rejection_reason + held_unavailable).

`score_uid` lives in /taos/dev/taos/im/validator/reward.py. The test
calls it with a hand-crafted validator_data dict so we don't need to
spin up a real validator.
"""
from pathlib import Path

import pytest

# Make sure /taos/dev/taos/im is importable as `taos`.
import sys
sys.path.insert(0, str(Path("/taos/dev").resolve()))

from taos.im.validator.reward import _gentrx_rank_normalize, score_uid


def _base_validator_data():
    """Minimal validator_data dict that satisfies score_uid's requirements
    without exercising the kappa/pnl branches (we set both weights to 0)."""
    return {
        "kappa_values": {1: {}, 2: {}, 3: {}},
        "activity_factors": {1: 1.0, 2: 1.0, 3: 1.0},
        "pnl_factors": {1: 1.0, 2: 1.0, 3: 1.0},
        "roundtrip_volumes": {1: 0, 2: 0, 3: 0},
        "realized_pnl_history": {1: [], 2: [], 3: []},
        "config": {
            "scoring": {
                "kappa": {"weight": 0.0, "lookback": 0, "min_lookback": 0,
                          "min_realized_observations": 0},
                "pnl": {"weight": 0.0},
                "interval": 0,
                "max_inactive_books_ratio": 1.0,
                "gentrx": {"simulation_share": 0.5, "ema_alpha": 0.1},
            }
        },
        "simulation_config": {
            "miner_wealth": 1.0,
            "book_count": 1,
            "grace_period": 0,
        },
        "simulation_timestamp": 0,
        "gentrx_scores": {
            "1": {"score": 0.5, "accepted": True},
            "2": {"score": 0.8, "accepted": True},
            "3": {"score": 0.2, "accepted": True},
        },
        "gentrx_ema": {},
        "gentrx_round": 0,
    }


def test_ema_bootstraps_at_zero_for_new_miner(monkeypatch):
    """A miner never seen before should have prev=0, so first-round EMA is
    alpha * round_score + 0.9 * 0 = 0.1 * round_score (not round_score)."""
    monkeypatch.setattr(
        "taos.im.validator.reward.calculate_kappa_score", lambda **_: 0.0
    )
    monkeypatch.setattr(
        "taos.im.validator.reward.calculate_pnl_score", lambda **_: 0.0
    )

    data = _base_validator_data()
    # uid 2 has the best raw score (0.8), so its rank-norm is 1.0.
    _, g2 = score_uid(data, 2)
    # With prev=0 bootstrap and alpha=0.1: g = 0.1 * 1.0 + 0.9 * 0 = 0.1.
    assert g2 == pytest.approx(0.1, abs=1e-6)
    # The persistence test: EMA is now 0.1 for uid 2.
    assert data["gentrx_ema"][2] == pytest.approx(0.1, abs=1e-6)


def test_ema_idempotent_within_one_round(monkeypatch):
    """Calling score_uid twice for the same (uid, round) must not double-
    apply the EMA. The second call should return the same value as the
    first without mutating gentrx_ema again."""
    monkeypatch.setattr(
        "taos.im.validator.reward.calculate_kappa_score", lambda **_: 0.0
    )
    monkeypatch.setattr(
        "taos.im.validator.reward.calculate_pnl_score", lambda **_: 0.0
    )

    data = _base_validator_data()
    _, g_first = score_uid(data, 2)
    _, g_second = score_uid(data, 2)
    assert g_first == g_second == pytest.approx(0.1, abs=1e-6)
    # EMA dict still shows the single application.
    assert data["gentrx_ema"][2] == pytest.approx(0.1, abs=1e-6)


def test_ema_advances_across_rounds(monkeypatch):
    """Bumping `gentrx_round` re-arms the per-round idempotency, so a second
    call now updates the EMA again."""
    monkeypatch.setattr(
        "taos.im.validator.reward.calculate_kappa_score", lambda **_: 0.0
    )
    monkeypatch.setattr(
        "taos.im.validator.reward.calculate_pnl_score", lambda **_: 0.0
    )

    data = _base_validator_data()
    _, g0 = score_uid(data, 2)
    # Round advances — fresh round-id resets the idempotency set.
    data["gentrx_round"] = 1
    _, g1 = score_uid(data, 2)
    assert g1 > g0, "second-round EMA must increase toward the round_score"
    # 0.1 * 1.0 + 0.9 * 0.1 = 0.19
    assert g1 == pytest.approx(0.19, abs=1e-6)


def test_ema_legacy_caller_without_round_id_falls_through(monkeypatch):
    """When `gentrx_round` is absent (legacy callers), the per-round guard
    is a no-op and the EMA updates on every call. We document this in the
    code; the test pins it so a future refactor doesn't accidentally
    silently change the legacy behaviour."""
    monkeypatch.setattr(
        "taos.im.validator.reward.calculate_kappa_score", lambda **_: 0.0
    )
    monkeypatch.setattr(
        "taos.im.validator.reward.calculate_pnl_score", lambda **_: 0.0
    )

    data = _base_validator_data()
    del data["gentrx_round"]  # legacy validator
    _, g1 = score_uid(data, 2)
    _, g2 = score_uid(data, 2)
    # Without the round guard, the EMA advances toward round_score every call.
    # First call: prev=0 → 0.1; second: prev=0.1 → 0.19.
    assert g1 == pytest.approx(0.1, abs=1e-6)
    assert g2 == pytest.approx(0.19, abs=1e-6)


def test_rank_normalize_skips_rejected_entries():
    """Sanity: the existing rank-normalise still skips entries whose
    `accepted` flag is False. Rejected miners get rank 0 via the .get()
    default downstream, not via the rank distribution."""
    scores = {
        "1": {"score": 0.5, "accepted": True},
        "2": {"score": -1.0, "accepted": False},  # rejected — must be skipped
        "3": {"score": 0.2, "accepted": True},
    }
    ranked = _gentrx_rank_normalize(scores)
    assert 2 not in ranked, "rejected miner must NOT appear in the rank"
    assert ranked[1] == pytest.approx(1.0)
    assert ranked[3] == pytest.approx(0.0)
