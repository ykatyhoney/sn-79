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
        agent_pnl_by_book=defaultdict(lambda: defaultdict(float)),
        agent_pnl_total=defaultdict(float),
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


def _recompute_pnl_from_history(s):
    """Ground-truth: what _prepare_reporting_data's old O(N*T*B) loop produced —
    a fresh sum over realized_pnl_history. The running totals must match this."""
    by_book = defaultdict(lambda: defaultdict(float))
    total = defaultdict(float)
    for uid, hist in s.realized_pnl_history.items():
        for ts_d in hist.values():
            for book_id, pnl in ts_d.items():
                by_book[uid][book_id] += pnl
                total[uid] += pnl
    return by_book, total


def _assert_running_totals_match_history(s):
    truth_by_book, truth_total = _recompute_pnl_from_history(s)
    # Every uid with history must have running totals equal to a fresh sum.
    for uid in s.realized_pnl_history:
        for book_id, v in truth_by_book[uid].items():
            assert s.agent_pnl_by_book[uid][book_id] == pytest.approx(v, abs=1e-9), (
                f"agent_pnl_by_book[{uid}][{book_id}]={s.agent_pnl_by_book[uid][book_id]} "
                f"!= history sum {v}"
            )
        assert s.agent_pnl_total[uid] == pytest.approx(truth_total[uid], abs=1e-9), (
            f"agent_pnl_total[{uid}]={s.agent_pnl_total[uid]} != history sum {truth_total[uid]}"
        )


def test_running_totals_track_history_across_writes():
    """The MVTRX-push running totals (agent_pnl_by_book / agent_pnl_total) must
    equal a fresh sum over realized_pnl_history after every mutation — this is the
    invariant _prepare_reporting_data now relies on instead of re-summing."""
    s = _make_self()
    # open long
    update_trade_volumes(s, _state(1000 * S, {
        1: [_trade(Ma=5, Ta=1, b=0, q=10.0, p=100.0, side=0)],
    }, accounts={1: {0: {"WEALTH": 1000.0}}}))
    _assert_running_totals_match_history(s)
    # close long higher on same book (realizes pnl into history[1][ts][0])
    update_trade_volumes(s, _state(1100 * S, {
        1: [_trade(Ma=5, Ta=1, b=0, q=10.0, p=110.0, side=1)],
    }, accounts={1: {0: {"WEALTH": 1100.0}}}))
    _assert_running_totals_match_history(s)
    # a second uid trades a different book, and uid1 opens+closes book 1 at a loss
    update_trade_volumes(s, _state(1200 * S, {
        1: [_trade(Ma=5, Ta=1, b=1, q=4.0, p=200.0, side=0)],
        2: [_trade(Ma=5, Ta=2, b=0, q=6.0, p=50.0, side=0)],
    }, accounts={1: {1: {"WEALTH": 1100.0}}, 2: {0: {"WEALTH": 500.0}}}))
    update_trade_volumes(s, _state(1300 * S, {
        1: [_trade(Ma=5, Ta=1, b=1, q=4.0, p=180.0, side=1)],
    }, accounts={1: {1: {"WEALTH": 1020.0}}}))
    _assert_running_totals_match_history(s)


def test_running_totals_stay_consistent_through_prune():
    """After a prune drops out-of-lookback timestamps, the running totals must
    still equal a fresh sum over the *remaining* history — i.e. the prune
    compensation subtracted exactly the pnl it removed, no more, no less."""
    s = _make_self()
    # Realize pnl at an early ts, then trade again far in the future so the early
    # ts falls outside kappa.lookback (10800s) and gets pruned.
    update_trade_volumes(s, _state(1000 * S, {
        1: [_trade(Ma=5, Ta=1, b=0, q=10.0, p=100.0, side=0)],
    }, accounts={1: {0: {"WEALTH": 1000.0}}}))
    update_trade_volumes(s, _state(1100 * S, {
        1: [_trade(Ma=5, Ta=1, b=0, q=10.0, p=110.0, side=1)],
    }, accounts={1: {0: {"WEALTH": 1100.0}}}))
    _assert_running_totals_match_history(s)
    # far-future tick: > lookback (10800s) past the 1100s realize -> prune path
    # subtracts the pruned pnl from the running totals; open a fresh position too.
    update_trade_volumes(s, _state(20000 * S, {
        1: [_trade(Ma=5, Ta=1, b=1, q=5.0, p=300.0, side=0)],
    }, accounts={1: {1: {"WEALTH": 1100.0}}}))
    _assert_running_totals_match_history(s)
    # close the fresh position -> new realized pnl inside the (shifted) window
    update_trade_volumes(s, _state(20100 * S, {
        1: [_trade(Ma=5, Ta=1, b=1, q=5.0, p=320.0, side=1)],
    }, accounts={1: {1: {"WEALTH": 1200.0}}}))
    _assert_running_totals_match_history(s)


def test_prune_tail_drop_removes_expired_head_keeps_retained():
    """The prune deletes the expired HEAD in place and stops at the first
    retained ts (O(expired), not a full-dict rebuild). Pre-seed a mix of
    out-of-lookback head timestamps followed by in-lookback ones, fire a prune,
    and assert only the head is dropped, the tail survives, and the running
    totals equal a fresh sum over the remaining history."""
    from taos.im.validator.trade import bootstrap_pnl_totals

    s = _make_self()
    # lookback = 10800s. At the prune tick below (ts=20000s) the threshold is
    # 20000-10800 = 9200s: 1000/1100/1200 are the expired head; 15000/15100 stay.
    s.realized_pnl_history = defaultdict(dict, {
        1: {
            1000 * S: {0: 10.0, 1: -3.0},
            1100 * S: {0: 5.0},
            1200 * S: {1: 2.0},
            15000 * S: {0: 7.5},
            15100 * S: {0: -1.5, 1: 4.0},
        },
    })
    bootstrap_pnl_totals(s)
    _assert_running_totals_match_history(s)  # seeded totals cover the full history

    # No new trades for uid 1 -> isolates the prune's effect. First call fires
    # the prune (_last_prune_timestamp is None).
    update_trade_volumes(s, _state(20000 * S, {}))

    # Expired head dropped; retained tail preserved in order. (The current ts,
    # 20000, is seeded as an empty dict by the update path — a no-op for totals.)
    keys = list(s.realized_pnl_history[1].keys())
    assert 1000 * S not in keys and 1100 * S not in keys and 1200 * S not in keys
    assert keys[:2] == [15000 * S, 15100 * S]
    _assert_running_totals_match_history(s)
    assert s.agent_pnl_total[1] == pytest.approx(7.5 - 1.5 + 4.0, abs=1e-9)
    assert s.agent_pnl_by_book[1][0] == pytest.approx(7.5 - 1.5, abs=1e-9)
    assert s.agent_pnl_by_book[1][1] == pytest.approx(4.0, abs=1e-9)


def test_bootstrap_pnl_totals_matches_history():
    """bootstrap_pnl_totals (startup / sim-restart rebuild path) must reproduce
    exactly what a fresh sum over history yields."""
    from taos.im.validator.trade import bootstrap_pnl_totals

    s = _make_self()
    s.realized_pnl_history = {
        1: {1000 * S: {0: 99.6, 1: -40.0}, 2000 * S: {0: 12.5}},
        2: {1500 * S: {3: 7.25}},
        7: {},  # empty history -> zero totals, no KeyError
    }
    bootstrap_pnl_totals(s)
    _assert_running_totals_match_history(s)
    assert s.agent_pnl_total[1] == pytest.approx(99.6 - 40.0 + 12.5, abs=1e-9)
    assert s.agent_pnl_by_book[1][0] == pytest.approx(99.6 + 12.5, abs=1e-9)
    assert s.agent_pnl_by_book[1][1] == pytest.approx(-40.0, abs=1e-9)
    assert s.agent_pnl_total[2] == pytest.approx(7.25, abs=1e-9)
    assert s.agent_pnl_total[7] == 0.0
