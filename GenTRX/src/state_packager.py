# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""State extractor for the validator/proxy.

Walks a `MarketSimulationStateUpdate` (or compatible dict) and produces a
serialization-friendly tick packet that the gradient server can ingest.

State packet format:
    {
      "step": 0,
      "ts": 1234567890000,
      "books": {
        0: {
          "bids": [[price, qty], ...],
          "asks": [[price, qty], ...],
          "events": [
            {"y": "o", "s": 0, "i": 42, "p": 100.5, "q": 1.0, "t": ...},
            {"y": "t", "Ti": 42, "Mi": 10, "p": 100.5, "q": 0.5, "t": ..., "s": 0},
            {"y": "c", "s": 0, "i": 10, "p": 100.5, "q": 0.0, "t": ...},
          ]
        },
      }
    }

Event types: "o" = order (quantity is REMAINING after fills),
"t" = trade (Ti=taker_id, Mi=maker_id, q=filled quantity),
"c" = cancellation (i=orderId).

To reconstruct original order size: order.q + sum(trade.q where trade.Ti == order.i)

Transport: GenTRXService.push_state msgpack-encodes the packet and POSTs it
to the gradient server's `/gentrx/state` endpoint. The gradient server runs
as a separate process — single-machine deployments use a loopback URL.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class OrderEvent(TypedDict):
    y: str  # "o"
    s: int
    i: int
    p: float
    q: float
    t: int


class TradeEvent(TypedDict):
    y: str  # "t"
    Ti: int
    Mi: int
    p: float
    q: float
    t: int
    s: int


class CancelEvent(TypedDict):
    y: str  # "c"
    s: int
    i: int
    p: float
    q: float
    t: int


Event = OrderEvent | TradeEvent | CancelEvent


class BookPacket(TypedDict):
    bids: list[list[float]]
    asks: list[list[float]]
    events: list[Event]


class ConfigPacket(TypedDict, total=False):
    priceDecimals: int
    volumeDecimals: int
    simulation_id: str | None


class _TickBase(TypedDict):
    step: int
    ts: int
    books: dict[int, BookPacket]


class TickPacket(_TickBase, total=False):
    sim_events: list[str]
    config: ConfigPacket
    sim_id: str


def _val(obj: Any, key: str, default: Any = None) -> Any:
    """Get value from dict or object.

    Handles both pydantic objects (property accessors) and raw dicts
    (which may use short keys from the taos protocol).
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    v = getattr(obj, key, None)
    return v if v is not None else default


# Short-key aliases used by the taos protocol wire format.
# parse_dict() sets state.books to raw dicts with short keys (b/a/e/p/q/i/s/t).
# Pydantic objects use long-name @property accessors (bids/asks/events/price/...).
_SHORT = {
    "bids": "b",
    "asks": "a",
    "events": "e",
    "price": "p",
    "quantity": "q",
    "id": "i",
    "side": "s",
    "timestamp": "t",
    "orderId": "i",
    "taker_id": "Ti",
    "maker_id": "Mi",
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get value trying long name first, then short alias."""
    v = _val(obj, key)
    if v is not None:
        return v
    short = _SHORT.get(key)
    if short is not None:
        return _val(obj, short, default)
    return default


def _get_price(level: Any) -> float:
    return _get(level, "price", 0)


def _get_qty(level: Any) -> float:
    return _get(level, "quantity", 0)


def _is_trade(event: Any) -> bool:
    if isinstance(event, dict):
        return "taker_id" in event or "takerAgentId" in event or "Ti" in event
    return hasattr(event, "taker_id") or hasattr(event, "takerAgentId")


def _is_cancellation(event: Any) -> bool:
    if isinstance(event, dict):
        return event.get("y") == "c" or "orderId" in event
    return hasattr(event, "orderId")


_SIM_EVENT_CODES = {"ESS", "ESE"}


def _extract_sim_events(state: Any) -> list[str]:
    """Return abbreviated sim-lifecycle event codes present on this tick.

    `state.notices` is `dict[uid] -> list[event]` (ESS/ESE carry no agentId
    so the protocol broadcasts them into every uid's list); a flat list is
    tolerated too. Returns any 'ESS' (SimulationStartEvent) or 'ESE'
    (SimulationEndEvent) codes found, deduped. Empty on the vast majority of
    ticks.
    """
    notices = _val(state, "notices", None) or {}
    groups = notices.values() if isinstance(notices, dict) else [notices]
    out: list[str] = []
    for group in groups:
        for n in group or []:
            code = _val(n, "y") or _val(n, "type")
            if code in _SIM_EVENT_CODES and code not in out:
                out.append(code)
    return out


class StatePackager:
    """Extract a tick packet from the validator's state object.

    Pure transformation — no I/O. The returned dict is serialized to msgpack
    by GenTRXService.push_state and POSTed to the gradient server.
    """

    def __init__(self) -> None:
        self._step: int = 0
        self._config_saved: bool = False
        self._current_sim_id: str | None = None

    def extract_state(self, state: Any) -> TickPacket:
        """Extract books + events + timestamp from state into a tick packet."""
        books = _val(state, "books") or {}
        packed_books: dict[int, BookPacket] = {}

        # Exchange event timestamps arrive as Unix SECONDS (chain wall-clock),
        # but GenTRX pages on a monotonic ns clock like the simulator's. Left as
        # seconds they all collapse to ~1s (identical ddHHMMSS tag), the sim-time
        # interval flush never advances, and every event lands in its own 1-row
        # parquet — so no page ever reaches seq_len rows and miners can't train.
        # Naive *1e9 would push it to Unix-ns and overflow the ddHHMMSS tag, so
        # instead stamp exchange rows with the tick's block-relative ns timestamp
        # (monotonic + tag-safe). The simulator already emits proper sim-ns and is
        # left untouched (is_exchange=False).
        tick_ts = int(_get(state, "timestamp", 0))
        _cfg = _val(state, "config", None)
        is_exchange = str(_get(_cfg, "simulation_id", "") or "") == "exchange"

        for book_id, book in books.items():
            bids = [
                [_get_price(level), _get_qty(level)]
                for level in (_get(book, "bids") or [])
            ]
            asks = [
                [_get_price(level), _get_qty(level)]
                for level in (_get(book, "asks") or [])
            ]
            events = self._extract_events(book, tick_ts, is_exchange)
            packed_books[int(book_id)] = {
                "bids": bids,
                "asks": asks,
                "events": events,
            }

        packet: TickPacket = {
            "step": self._step,
            "ts": tick_ts,
            "books": packed_books,
        }

        # Sim lifecycle markers. The simulator emits SimulationStartEvent
        # ('ESS') on a fresh sim and SimulationEndEvent ('ESE') when the
        # active sim finishes; both arrive via state.notices. The gradient
        # server uses these as the primary signal to reset in-memory tick
        # buffers and schedule data/ cleanup on S3; a backwards-time
        # heuristic in the gradient server handles the case where no ESE is emitted.
        sim_events = _extract_sim_events(state)
        if sim_events:
            packet["sim_events"] = sim_events
            if "ESS" in sim_events:
                self._config_saved = False
                self._current_sim_id = None

        if not self._config_saved:
            config = self._extract_config(state)
            if config is not None:
                packet["config"] = config
                sim_id = config.get("simulation_id")
                if sim_id is not None:
                    self._config_saved = True
                    self._current_sim_id = sim_id

        if self._current_sim_id is None:
            cur = _get(_val(state, "config", None), "simulation_id", None)
            if cur is not None:
                self._current_sim_id = cur

        if self._current_sim_id is not None:
            packet["sim_id"] = self._current_sim_id

        self._step += 1
        return packet

    def _extract_config(self, state: Any) -> ConfigPacket | None:
        """Build the config block from state, or None when state has no config.

        Never fabricates decimal precision. The simulator's config carries
        priceDecimals/volumeDecimals as required fields, so a tick missing them
        is a real misconfiguration: those keys are omitted (not defaulted) and
        logged, leaving the gradient server to decide rather than silently
        binding a wrong price/volume scale.
        """
        config = _val(state, "config", None)
        if config is None:
            return None

        out: ConfigPacket = {}
        pd = _get(config, "priceDecimals", None)
        vd = _get(config, "volumeDecimals", None)
        sim_id = _get(config, "simulation_id", None)
        if pd is not None:
            out["priceDecimals"] = int(pd)
        if vd is not None:
            out["volumeDecimals"] = int(vd)
        if pd is None or vd is None:
            logger.warning(
                "state config present but missing decimals "
                "(priceDecimals=%s, volumeDecimals=%s); omitting from packet",
                pd,
                vd,
            )
        out["simulation_id"] = sim_id
        return out

    def _extract_events(
        self, book: Any, tick_ts: int = 0, is_exchange: bool = False
    ) -> list[Event]:
        """Extract raw event dicts from book (no matching engine).

        Includes trade events (y="t") so the gradient server can reconstruct
        original order sizes: Order.quantity is the REMAINING size after fills,
        so we need Trade.taker_id + Trade.quantity to recover the full volume.

        `t` (row timestamp) is the event's own sim-ns timestamp for the
        simulator; for the exchange (whose events carry Unix seconds) it is the
        tick's block-relative ns timestamp so GenTRX paging stays on a monotonic,
        tag-safe ns clock. See extract_state for the rationale.
        """
        events = _get(book, "events") or []
        out: list[Event] = []
        for e in events:
            _t = tick_ts if is_exchange else int(_get(e, "timestamp", 0))
            if _is_trade(e):
                out.append(
                    {
                        "y": "t",
                        "Ti": _get(e, "taker_id", 0),
                        "Mi": _get(e, "maker_id", 0),
                        "p": _get(e, "price", 0),
                        "q": float(_get(e, "quantity", 0)),
                        "t": _t,
                        "s": _get(e, "side", 0),
                    }
                )
            elif _is_cancellation(e):
                out.append(
                    {
                        "y": "c",
                        "s": _get(e, "side", 0),
                        "i": _get(e, "orderId", 0),
                        "p": _get(e, "price", 0),
                        "q": float(_get(e, "quantity", 0)),
                        "t": _t,
                    }
                )
            else:
                out.append(
                    {
                        "y": "o",
                        "s": _get(e, "side", 0),
                        "i": _get(e, "id", 0),
                        "p": _get(e, "price", 0),
                        "q": float(_get(e, "quantity", 0)),
                        "t": _t,
                    }
                )
        return out
