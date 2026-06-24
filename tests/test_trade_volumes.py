# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Characterization tests for update_trade_volumes.

This reward-feeding function (volume sums, realized P&L, round-trip volume,
self/maker/taker split) had no test coverage. These pin its observable output
on a faithful fixture exercising taker/self volume, a FIFO open+close cycle,
and pruning — both as a regression guard and as the safety net for refactoring.

pagerduty_alert is wired to raise so any fixture-fidelity gap surfaces instead
of being swallowed by the function's per-uid try/except.
"""
from collections import defaultdict, deque
from types import SimpleNamespace

import pytest

from taos.im.validator.trade import update_trade_volumes

S = 1_000_000_000


def _make_self():
    def boom(msg, **k):
        raise AssertionError(f"pagerduty_alert hit (fixture infidelity): {msg}")

    return SimpleNamespace(
        simulation=SimpleNamespace(volumeDecimals=4),
        config=SimpleNamespace(scoring=SimpleNamespace(
            activity=SimpleNamespace(
                trade_volume_sampling_interval=600 * S,
                trade_volume_assessment_period=3600 * S,
            ),
            kappa=SimpleNamespace(lookback=10800 * S),
        )),
        step=1,
        effective_max_uids=3,
        trade_volumes={},
        volume_sums=defaultdict(lambda: defaultdict(float)),
        maker_volume_sums=defaultdict(lambda: defaultdict(float)),
        taker_volume_sums=defaultdict(lambda: defaultdict(float)),
        self_volume_sums=defaultdict(lambda: defaultdict(float)),
        roundtrip_volume_sums=defaultdict(lambda: defaultdict(float)),
        fee_sums=defaultdict(lambda: defaultdict(float)),
        recent_trades={},
        recent_miner_trades=defaultdict(dict),
        open_positions=defaultdict(lambda: defaultdict(lambda: {"longs": deque(), "shorts": deque()})),
        realized_pnl_history=defaultdict(dict),
        roundtrip_volumes=defaultdict(lambda: defaultdict(lambda: defaultdict(float))),
        inventory_history=defaultdict(dict),
        initial_balances=defaultdict(dict),
        pagerduty_alert=boom,
        _last_prune_timestamp=None,
    )


def _trade(Ma, Ta, b, q, p, side, Mf=0.1, Tf=0.2):
    return {"y": "ET", "Ma": Ma, "Ta": Ta, "b": b, "q": q, "p": p, "s": side, "Mf": Mf, "Tf": Tf}


def _state(ts, notices, accounts=None, book_ids=(0, 1)):
    return SimpleNamespace(
        books={b: {"e": []} for b in book_ids}, timestamp=ts,
        accounts=accounts or {}, notices=notices,
    )


def test_taker_and_self_volume_split():
    s = _make_self()
    update_trade_volumes(s, _state(1000 * S, {
        1: [_trade(Ma=5, Ta=1, b=0, q=10.0, p=100.0, side=0)],   # uid1 taker
        2: [_trade(Ma=2, Ta=2, b=1, q=3.0, p=50.0, side=0)],     # uid2 self (Ma==Ta)
    }, accounts={1: {0: {"WEALTH": 1000.0}}, 2: {1: {"WEALTH": 500.0}}}))
    assert s.taker_volume_sums[1][0] == 10.0 * 100.0
    assert s.self_volume_sums[2][1] == 3.0 * 50.0
    assert s.volume_sums[1][0] == 1000.0


def test_fifo_close_realizes_pnl_and_roundtrip():
    s = _make_self()
    # open long (taker buy), then close (taker sell) higher
    update_trade_volumes(s, _state(1000 * S, {
        1: [_trade(Ma=5, Ta=1, b=0, q=10.0, p=100.0, side=0)],
    }, accounts={1: {0: {"WEALTH": 1000.0}}}))
    update_trade_volumes(s, _state(1100 * S, {
        1: [_trade(Ma=5, Ta=1, b=0, q=10.0, p=110.0, side=1)],
    }, accounts={1: {0: {"WEALTH": 1100.0}}}))
    # (110-100)*10 = 100 gross, minus taker fees 0.2+0.2 → 99.6
    assert s.realized_pnl_history[1][1100 * S][0] == pytest.approx(99.6, abs=1e-9)
    assert s.roundtrip_volume_sums[1][0] == pytest.approx(10.0 * 110.0, abs=1e-9)


def test_pruning_runs_without_error_on_far_future_tick():
    s = _make_self()
    update_trade_volumes(s, _state(1000 * S, {1: [_trade(5, 1, 0, 10.0, 100.0, 0)]},
                                   accounts={1: {0: {"WEALTH": 1000.0}}}))
    # far-future tick (> assessment period + prune interval) triggers the prune path
    update_trade_volumes(s, _state(6000 * S, {1: [_trade(5, 1, 1, 2.0, 80.0, 0)]},
                                   accounts={1: {1: {"WEALTH": 900.0}}}))
    assert s._last_prune_timestamp == 6000 * S
