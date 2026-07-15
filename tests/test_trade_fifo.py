# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Tests for FIFO trade matching / realized-P&L (match_trade_fifo).

This is reward-critical: realized P&L and round-trip volume feed the activity
and P&L factors in the Kappa score. The review flagged the incentive maths as
unverified; these pin the FIFO accounting (open, full/partial close, fees,
long & short) so it can be refactored safely later.
"""
from collections import deque
from types import SimpleNamespace

from taos.im.validator.trade import match_trade_fifo


def _self():
    # open_positions[uid][book] = {'longs': deque, 'shorts': deque}
    return SimpleNamespace(open_positions={0: {0: {"longs": deque(), "shorts": deque()}}})


def test_open_long_when_flat_returns_zero_and_stores_position():
    s = _self()
    pnl, rt = match_trade_fifo(s, 0, 0, is_buy=True, quantity=10.0, price=100.0, fee=1.0, timestamp=1)
    assert (pnl, rt) == (0.0, 0.0)
    longs = s.open_positions[0][0]["longs"]
    assert list(longs) == [(1, 10.0, 100.0, 1.0)]


def test_full_close_long_realizes_pnl_net_of_fees():
    s = _self()
    match_trade_fifo(s, 0, 0, is_buy=True, quantity=10.0, price=100.0, fee=1.0, timestamp=1)
    pnl, rt = match_trade_fifo(s, 0, 0, is_buy=False, quantity=10.0, price=110.0, fee=1.0, timestamp=2)
    # (110-100)*10 = 100 gross; minus open_fee(1) and close_fee(1) = 98 net.
    assert pnl == 98.0
    assert rt == 10.0
    assert len(s.open_positions[0][0]["longs"]) == 0


def test_partial_close_long_leaves_residual_position():
    s = _self()
    match_trade_fifo(s, 0, 0, is_buy=True, quantity=10.0, price=100.0, fee=2.0, timestamp=1)
    pnl, rt = match_trade_fifo(s, 0, 0, is_buy=False, quantity=4.0, price=120.0, fee=1.0, timestamp=2)
    # close 4 of 10: (120-100)*4=80; open_fee=2*4/10=0.8; close_fee=1 → 80-0.8-1=78.2
    assert round(pnl, 10) == 78.2
    assert rt == 4.0
    residual = list(s.open_positions[0][0]["longs"])
    assert len(residual) == 1 and residual[0][1] == 6.0  # 6 qty remains


def test_short_then_buy_to_close_realizes_pnl():
    s = _self()
    match_trade_fifo(s, 0, 0, is_buy=False, quantity=5.0, price=200.0, fee=0.0, timestamp=1)  # open short
    pnl, rt = match_trade_fifo(s, 0, 0, is_buy=True, quantity=5.0, price=190.0, fee=0.0, timestamp=2)
    # short at 200, cover at 190 → (200-190)*5 = 50 profit
    assert pnl == 50.0
    assert rt == 5.0
    assert len(s.open_positions[0][0]["shorts"]) == 0


def test_buy_exceeding_short_closes_then_opens_long():
    s = _self()
    match_trade_fifo(s, 0, 0, is_buy=False, quantity=3.0, price=100.0, fee=0.0, timestamp=1)  # short 3
    pnl, rt = match_trade_fifo(s, 0, 0, is_buy=True, quantity=8.0, price=100.0, fee=0.0, timestamp=2)
    assert rt == 3.0                       # only the 3 covered counts as round-trip
    longs = list(s.open_positions[0][0]["longs"])
    assert len(longs) == 1 and longs[0][1] == 5.0   # 5 remaining opens a long


def test_sell_closing_multiple_longs_ending_partial_prorates_fee():
    """A single sell that fully closes one long and partially closes the next must
    allocate the fill fee proportionally across BOTH portions. The partial branch
    used to charge the full fee again, double-counting the fee already taken by the
    fully-closed lot and understating realized P&L."""
    s = _self()
    match_trade_fifo(s, 0, 0, is_buy=True, quantity=6.0, price=100.0, fee=0.0, timestamp=1)  # long 6
    match_trade_fifo(s, 0, 0, is_buy=True, quantity=6.0, price=100.0, fee=0.0, timestamp=2)  # long 6
    pnl, rt = match_trade_fifo(s, 0, 0, is_buy=False, quantity=10.0, price=110.0, fee=1.0, timestamp=3)
    # gross (110-100)*10 = 100; total close fee = the fill's 1.0 (0.6 on the 6 fully
    # closed + 0.4 on the 4 partially closed), open fees 0 → 99.0.
    assert round(pnl, 10) == 99.0
    assert rt == 10.0
    residual = list(s.open_positions[0][0]["longs"])
    assert len(residual) == 1 and residual[0][1] == 2.0  # 2 qty remains on lot B


def test_buy_covering_multiple_shorts_ending_partial_prorates_fee():
    """Short-cover mirror of the above: full close of one short + partial close of the
    next must prorate the fill fee, not re-charge it in full on the partial lot."""
    s = _self()
    match_trade_fifo(s, 0, 0, is_buy=False, quantity=6.0, price=100.0, fee=0.0, timestamp=1)  # short 6
    match_trade_fifo(s, 0, 0, is_buy=False, quantity=6.0, price=100.0, fee=0.0, timestamp=2)  # short 6
    pnl, rt = match_trade_fifo(s, 0, 0, is_buy=True, quantity=10.0, price=90.0, fee=1.0, timestamp=3)
    # gross (100-90)*10 = 100; total close fee = the fill's 1.0 → 99.0.
    assert round(pnl, 10) == 99.0
    assert rt == 10.0
    residual = list(s.open_positions[0][0]["shorts"])
    assert len(residual) == 1 and residual[0][1] == 2.0  # 2 qty remains on short B
