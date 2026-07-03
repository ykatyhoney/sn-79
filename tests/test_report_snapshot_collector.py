# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Tests for the reporting SnapshotCollector.

Covers the migration of the clear-and-rebuild gauge families (trades / books /
miner_trades) from eager Gauges to replace-mode SnapshotCollectors:

  * exposition byte-identity vs an eager Gauge (same series, values, HELP/TYPE),
  * replace semantics (a series dropped this cycle disappears — no carry-forward
    leak of stale rolling-buffer slots),
  * carry-forward semantics unchanged for agent_gauges (a series persists at its
    last value until re-emitted or evicted).
"""
from prometheus_client import CollectorRegistry, Gauge, generate_latest

from taos.im.validator.report import _SnapshotCollector

TRADE_LABELS = ["wallet", "netuid", "sim_id", "book_id", "slot", "trade_gauge_name"]
HELP = "Gauge summaries for trade metrics."


def _sample_lines(text):
    return sorted(line for line in text.decode().splitlines() if line and not line.startswith("#"))


def _meta_lines(text):
    return [line for line in text.decode().splitlines() if line.startswith("#")]


def _make_series(n_books=8, n_slots=5):
    names = ["timestamp", "price", "volume", "maker_fee", "taker_fee", "side"]
    series = []
    v = 0.0
    for b in range(n_books):
        for slot in range(n_slots):
            for nm in names:
                v += 1.5
                series.append((("wAddr", "79", "simX", b, slot, nm), v))
    return series


def test_replace_collector_exposition_matches_eager_gauge():
    """A replace-mode collector must expose byte-identical series/values and the
    same HELP/TYPE header as the eager Gauge it replaces."""
    series = _make_series()

    r_eager = CollectorRegistry()
    g = Gauge("trades", HELP, TRADE_LABELS, registry=r_eager)
    for labels, val in series:
        g.labels(*labels).set(val)
    eager = generate_latest(r_eager)

    r_coll = CollectorRegistry()
    c = _SnapshotCollector("trades", HELP, TRADE_LABELS, carry_forward=False)
    c.update({labels: val for labels, val in series})
    r_coll.register(c)
    coll = generate_latest(r_coll)

    assert _sample_lines(eager) == _sample_lines(coll)
    assert _meta_lines(eager) == _meta_lines(coll)


def test_replace_drops_series_not_reemitted():
    """Replace mode fully swaps each cycle: a label tuple present last cycle but
    absent this cycle must NOT survive (mirrors gauge.clear() + rebuild). This is
    the correctness guard against stale rolling-buffer slots leaking forward."""
    c = _SnapshotCollector("trades", HELP, TRADE_LABELS, carry_forward=False)
    c.update({("w", "79", "s", 0, 0, "price"): 100.0, ("w", "79", "s", 0, 1, "price"): 99.0})
    samples = list(next(c.collect()).samples)
    assert len(samples) == 2
    # Next cycle emits only one of them; the other must vanish.
    c.update({("w", "79", "s", 0, 0, "price"): 101.0})
    samples = list(next(c.collect()).samples)
    assert len(samples) == 1
    assert samples[0].value == 101.0
    assert samples[0].labels["slot"] == "0"


def test_carry_forward_persists_until_evicted():
    """Carry-forward mode (agent_gauges) keeps a series at its last value across
    cycles where it isn't re-emitted, and drops it only on explicit evict()."""
    c = _SnapshotCollector("agent_gauges", HELP, TRADE_LABELS, carry_forward=True)
    k = ("w", "79", "s", 0, 0, "price")
    c.update({k: 100.0})
    # A cycle that emits nothing must retain the prior series at its last value.
    c.update({})
    samples = list(next(c.collect()).samples)
    assert len(samples) == 1
    assert samples[0].value == 100.0
    # Explicit eviction removes it at the next update.
    c.evict(k)
    c.update({})
    assert list(next(c.collect()).samples) == []


def test_clear_empties_snapshot():
    c = _SnapshotCollector("trades", HELP, TRADE_LABELS, carry_forward=False)
    c.update({("w", "79", "s", 0, 0, "price"): 1.0})
    assert list(next(c.collect()).samples)
    c.clear()
    assert list(next(c.collect()).samples) == []
