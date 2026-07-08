"""
validator/engines/base.py

Canonical state envelope and protocol abstraction for the unified validator.
Both SimulationEngine and ExchangeEngine produce NormalizedState; all
scoring logic in validator.py consumes only NormalizedState and never
branches on mode below handle_state().
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Canonical trade event
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizedTradeEvent:
    """
    Single canonical trade event in the format _update_trade_volumes() expects.

    Matches the simulation's ET notice dict exactly so that function needs
    zero changes. to_notice_dict() produces the {'y':'ET', 'b':..., ...}
    format directly.

    book_id  — simulation: book index  |  exchange: netuid
    side     — 0 = buy, 1 = sell
    maker_uid / taker_uid — both set to uid for AMM fills (no counterparty)
    """
    book_id:   int
    quantity:  float
    price:     float
    side:      int
    maker_uid: int
    taker_uid: int
    maker_fee: float
    taker_fee: float
    timestamp:           int     = field(default_factory=lambda: int(time.time_ns()))
    order_id:            Optional[str] = field(default=None)
    close_reason:        Optional[str] = field(default=None)   # 'SL' | 'TP' | None
    linked_order_id:     Optional[int] = field(default=None)   # originating LOB order id

    def to_notice_dict(self) -> dict:
        d = {
            'y':  'ET',
            't':  self.timestamp,
            'a':  self.taker_uid,
            'b':  self.book_id,
            'i':  0,
            'c':  None,
            'Ti': 0,
            'Mi': 0,
            'q':  self.quantity,
            'p':  self.price,
            's':  self.side,
            'Ma': self.maker_uid,
            'Ta': self.taker_uid,
            'Mf': self.maker_fee,
            'Tf': self.taker_fee,
        }
        if self.close_reason is not None:
            d['cr'] = self.close_reason
        if self.linked_order_id is not None:
            d['Toi'] = self.linked_order_id
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Canonical state envelope
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizedState:
    """
    Single canonical state type the validator loop operates on.

    Field mapping:
        simulation                       exchange
        ───────────────────────────────  ────────────────────────────────────
        timestamp  ← state.timestamp     block × block_time_ns
        block      ← ts // block_time_ns current chain block
        books      ← state.books         per-netuid pool state
        accounts   ← state.accounts      per-uid TAO/alpha balances from chain
        notices    ← state.notices        empty on receive; filled by execute()
        config     ← state.config        ExchangeConfig
        version    ← state.version       SPEC_VERSION

    The field names and dtypes are identical to what _update_trade_volumes(),
    _reward(), and all other validator methods already read, so those methods
    require no changes.
    """
    timestamp: int          # nanoseconds
    block:     int          # chain block number
    books:     dict         # {book_id/netuid: market state dict}
    accounts:  dict         # {uid: {book_id/netuid: balance dict}}
    notices:   dict         # {uid: [ET notice dicts]}
    config:    Any          # MarketSimulationConfig or ExchangeConfig
    version:   int
    logDir:    Any = None   # simulation log directory (preserved from raw message dict)
    pools:     dict | None = None  # {netuid: {price, tao_in, alpha_in, ...}} — exchange only


# ─────────────────────────────────────────────────────────────────────────────
# Protocol ABC
# ─────────────────────────────────────────────────────────────────────────────

class MarketEngine(ABC):
    """
    Abstracts the three mode-specific operations in the validator loop.

    The validator calls these three methods and nothing else that is
    mode-specific.  All scoring, persistence, and weight-setting is above
    this boundary and runs identically in both modes.

    receive()  — wait for the next tick; return NormalizedState
    execute()  — act on miner responses; return trade events ([] for sim)
    respond()  — write response back (IPC write for sim, no-op for exchange)
    """

    @abstractmethod
    async def receive(self) -> tuple[Any, Optional[NormalizedState], float]:
        """
        Blocking: wait for and return the next state tick.

        Returns:
            raw_message    — original message object (passed back to respond())
            normalized     — NormalizedState, or None for lifecycle-only ticks
                             (e.g. simulation END notice — skip handle_state)
            receive_start  — time.time() when message arrived
        """

    @abstractmethod
    async def execute(
        self,
        state: NormalizedState,
        miner_responses: list,
    ) -> list[NormalizedTradeEvent]:
        """
        Act on miner responses and return canonical trade events.

        Simulation: no-op. Simulator injects trade events into the next
                    tick's state.notices itself. Returns [].
        Exchange:   send instructions to LOB engine, execute on-chain,
                    convert ExecutionResult[] → NormalizedTradeEvent[].
        """

    @abstractmethod
    def respond(self, raw_message: Any, response: dict) -> None:
        """
        Send validator response back to whoever is waiting.

        Simulation: msgpack.packb(response) → write to /taosim-res socket.
        Exchange:   no-op (on-chain execution is the response).
        """

    def start(self) -> None:
        """Called once from validator.__init__() after base setup is done."""

    def stop(self) -> None:
        """Called from validator.cleanup()."""

    # ── Shared external-order polling ────────────────────────────────────────
    # UI/wallet-submitted orders are fetched from the data service on a
    # background task, not inline in receive(), so a slow or busy data service
    # never adds latency to the round and the per-order sr25519 verify stays off
    # the critical path. The data-service GET is an atomic destructive pop
    # (exactly-once), so faster polling cannot double-inject. receive() drains
    # the buffer with no await. _fetch_external_orders() is implemented per-engine.

    _EXTERNAL_POLL_INTERVAL = 1.0

    def _ensure_external_poller(self) -> None:
        """Start the background poll task on first call (requires a running loop)."""
        if getattr(self, "_external_poll_task", None) is None:
            self._external_poll_task = asyncio.ensure_future(self._external_orders_loop())

    async def _external_orders_loop(self) -> None:
        while True:
            try:
                instrs = await self._fetch_external_orders()
                if instrs:
                    self._external_buffer.extend(instrs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                import bittensor as bt

                bt.logging.debug(f"external orders poll failed: {exc}")
            await asyncio.sleep(self._EXTERNAL_POLL_INTERVAL)

    def _drain_external_orders(self) -> list:
        """Return buffered external instructions and clear the buffer.

        Sync with no await between the read and the reset, so it is atomic
        against the background poller under single-threaded asyncio.
        """
        instrs = self._external_buffer
        self._external_buffer = []
        return instrs

    def _stop_external_poller(self) -> None:
        task = getattr(self, "_external_poll_task", None)
        if task is not None:
            task.cancel()
            self._external_poll_task = None

    @property
    @abstractmethod
    def book_ids(self) -> list[int]:
        """All active book IDs. Simulation: range(book_count). Exchange: netuids."""

    @property
    def book_count(self) -> int:
        return len(self.book_ids)

    @property
    @abstractmethod
    def mode(self) -> str:
        """'simulation' or 'exchange'."""

    # ── Lifecycle hooks (called by validator.onStart / onEnd) ────────────────

    def on_start(self, state: Any) -> None:
        """
        Called when the engine signals a new run start.
        Simulation: triggered by 'START' system notice in receive().
        Exchange:   triggered by first valid chain state block.
        Default no-op; SimulationEngine overrides with timestamp-shift logic.
        """

    def on_end(self, state: Any) -> None:
        """
        Called when the engine signals a run end.
        Simulation: triggered by 'END' system notice in receive().
        Exchange:   triggered by graceful shutdown.
        Default no-op; SimulationEngine overrides with save + update_repo.
        """

    # ── Reset detection (called by validator.process_resets) ─────────────────

    def collect_resets(self, state: NormalizedState, pending: set) -> None:
        """
        Populate `pending` with UIDs that need scoring state reset this tick.

        Simulation: scans state.notices for RDRA/ERDRA system notices.
        Exchange:   no-op — deregistrations are detected in resync_metagraph()
                    and placed directly into validator.deregistered_uids.
        """

    # ── New engine interface methods ─────────────────────────────────────────

    @property
    @abstractmethod
    def effective_max_uids(self) -> int:
        """Total addressable UID count. Simulation: max_uids + benchmark. Exchange: max_uids."""

    @abstractmethod
    def initialize_structures(self) -> None:
        """Initialize all validator state structures for this mode. Called from start()."""

    @abstractmethod
    def build_simulation_state(self) -> dict:
        """Return mode-specific state dict for persistence (the 'simulation' save file)."""

    @abstractmethod
    def restore_simulation_state(self, data: dict) -> None:
        """Restore mode-specific state from dict loaded from disk."""

    @abstractmethod
    def handle_deregistration(self, uid: int) -> None:
        """Mode-specific deregistration: zero score, flag for reset, etc."""

    @abstractmethod
    def apply_resets(self, pending: set) -> None:
        """Zero all scoring state for each UID in pending."""

    def get_extended_metagraph(self) -> Any:
        """Return metagraph extended with mode-specific virtual UIDs.
        Default: return real metagraph unchanged."""
        return self.validator.metagraph

    def on_resync_metagraph(self, old_size: int, new_size: int) -> None:
        """Called when metagraph size changes during resync. Default: no-op."""

    def on_tick(self, state: "NormalizedState") -> None:
        """
        Called once per state tick from handle_state(), before volume injection.
        Simulation: updates simulation_timestamp, detects logDir changes, fires
                    periodic compression and update_repo.
        Exchange:   default no-op.
        """

from taos.im.validator.engines.simulation import SimulationEngine
# ExchangeEngine is exchange-mode-only; consumers must import it explicitly:
#   from taos.im.validator.engines.exchange import ExchangeEngine, ExchangeConfig
