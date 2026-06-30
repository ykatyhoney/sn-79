# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

if __name__ != "__mp_main__":
    import os
    # Must precede `import bittensor` below: bittensor 10.3.2 builds its logging
    # singleton at import time and returns an empty config (disabling logging)
    # unless BT_NO_PARSE_CLI_ARGS is "false" in the env at that point. The run
    # scripts also export this, but set it here so a direct `python validator.py`
    # launch is equally safe. No-op on bittensor <10.3.2.
    os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "false")
    import json
    import signal
    import sys
    import platform
    import time
    import argparse
    import torch
    import traceback
    import msgspec
    import asyncio
    import posix_ipc
    import mmap
    import msgpack
    import atexit
    import subprocess
    import struct
    import select
    import copy
    from typing import TYPE_CHECKING, Any
    from ypyjson import YpyObject

    if TYPE_CHECKING:
        from taos.im.protocol.models import MarketSimulationConfig

    import bittensor as bt

    from GenTRX.src.bt_log import gtx_log

    import uvicorn
    from fastapi import FastAPI, APIRouter
    from fastapi import Request
    import threading
    from threading import Thread, Lock, Event
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    from git import Repo
    from pathlib import Path

    from taos.common.neurons.validator import BaseValidatorNeuron
    from taos.im.utils import duration_from_timestamp
    from taos.im.utils.affinity import get_core_allocation

    from taos.im.config import add_im_validator_args
    from taos.im.protocol.simulator import SimulatorResponseBatch
    from taos.im.protocol import MarketSimulationStateUpdate, FinanceEventNotification
    from taos.im.protocol.events import SimulationStartEvent
    from taos.im.validator.engines import NormalizedState, SimulationEngine

    async def _push_fill_notifications(trade_events: list, url: str) -> None:
        """Fire-and-forget: push fill notifications directly to the data service
        immediately after on-chain execution, bypassing the heavy ingest pipeline."""
        if not url or not trade_events:
            return
        _secret = os.environ.get("INGEST_SECRET", "")
        _headers = {"Content-Type": "application/json"}
        if _secret:
            _headers["x-ingest-secret"] = _secret

        async def _post_with_retry(client, endpoint: str, body: bytes, label: str) -> bool:
            """POST with up to 3 attempts on ConnectError (service restart window)."""
            _delays = [1.0, 3.0]
            for _attempt in range(3):
                try:
                    _resp = await client.post(endpoint, content=body, headers=_headers)
                    if _resp.status_code == 200:
                        bt.logging.info(f"_push_fill_notifications: {label}")
                    else:
                        bt.logging.warning(f"_push_fill_notifications: {label} HTTP {_resp.status_code}: {_resp.text[:120]}")
                    return True
                except httpx.ConnectError as _exc:
                    if _attempt < len(_delays):
                        bt.logging.warning(f"_push_fill_notifications: {label} ConnectError (attempt {_attempt+1}/3), retrying in {_delays[_attempt]}s: {_exc!r}")
                        await asyncio.sleep(_delays[_attempt])
                    else:
                        bt.logging.warning(f"_push_fill_notifications: {label} failed after 3 attempts: {_exc!r}")
                        return False
                except Exception as _exc:
                    bt.logging.warning(f"_push_fill_notifications: {label} failed: {_exc!r}")
                    return False
            return False

        try:
            import json as _json
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, write=5.0, read=5.0, pool=5.0)) as client:
                for _te in trade_events:
                    _taker_uid = getattr(_te, 'taker_uid', None)
                    _maker_uid = getattr(_te, 'maker_uid', None)
                    _nid      = getattr(_te, 'book_id', 0)
                    _p        = getattr(_te, 'price', 0.0)
                    _q        = getattr(_te, 'quantity', 0.0)
                    _side     = getattr(_te, 'side', 0)
                    _ts           = getattr(_te, 'timestamp', int(time.time() * 1e9))
                    _order_id     = getattr(_te, 'order_id', None)
                    _close_reason = getattr(_te, 'close_reason', None)
                    _linked_oid   = getattr(_te, 'linked_order_id', None)

                    # Push for taker
                    if _taker_uid is not None and _taker_uid != 0:
                        _notif = {
                            "type":         "fill",
                            "uid":          _taker_uid,
                            "netuid":       _nid,
                            "price":        _p,
                            "alpha_volume": _q,
                            "qty":          _q,
                            "direction":    _side,
                            "side":         "buy" if _side == 0 else "sell",
                            "role":         "taker",
                            "taker_uid":    _taker_uid,
                            "maker_uid":    _maker_uid,
                            "timestamp":    _ts,
                            "order_id":     _order_id,
                        }
                        if _close_reason is not None:
                            _notif["close_reason"]    = _close_reason
                        if _linked_oid is not None:
                            _notif["linked_order_id"] = _linked_oid
                        await _post_with_retry(
                            client,
                            f"{url}/api/v1/agents/{_taker_uid}/notify",
                            _json.dumps(_notif).encode('utf-8'),
                            f"taker uid={_taker_uid} netuid={_nid} side={_side} qty={_q:.4f} price={_p:.6f}",
                        )

                    # Push for maker when it's a distinct counterparty (not same as taker)
                    if _maker_uid is not None and _maker_uid != 0 and _maker_uid != _taker_uid:
                        _maker_side = 1 - _side  # maker is opposite of taker
                        _maker_notif = {
                            "type":         "fill",
                            "uid":          _maker_uid,
                            "netuid":       _nid,
                            "price":        _p,
                            "alpha_volume": _q,
                            "qty":          _q,
                            "direction":    _maker_side,
                            "side":         "buy" if _maker_side == 0 else "sell",
                            "role":         "maker",
                            "taker_uid":    _taker_uid,
                            "maker_uid":    _maker_uid,
                            "timestamp":    _ts,
                            "order_id":     _order_id,
                        }
                        if _close_reason is not None:
                            _maker_notif["close_reason"]    = _close_reason
                        if _linked_oid is not None:
                            _maker_notif["linked_order_id"] = _linked_oid
                        await _post_with_retry(
                            client,
                            f"{url}/api/v1/agents/{_maker_uid}/notify",
                            _json.dumps(_maker_notif).encode('utf-8'),
                            f"maker uid={_maker_uid} netuid={_nid} side={_maker_side} qty={_q:.4f} price={_p:.6f}",
                        )
        except Exception as _exc:
            bt.logging.warning(f"_push_fill_notifications: setup failed: {_exc!r}")

    async def _push_mvtrx(payload: dict, url: str) -> None:
        """Delegate to the optional data-service push module when present."""
        try:
            from taos.im.validator.mvtrx_push import push as _push
        except ImportError:
            return
        await _push(payload, url)

    class Validator(BaseValidatorNeuron):
        """
        Intelligent market simulation validator implementation.

        The validator is run as a FastAPI client in order to receive messages from the simulator engine for processing and forwarding to miners.
        Metagraph maintenance, weight setting, state persistence and other general bittensor routines are executed in a separate thread.
        The validator also handles publishing of metrics via Prometheus for visualization and analysis, as well as retrieval and recording of seed data for simulation price process generation.
        """

        # Instance state populated at init/load and by the delegated engine,
        # persistence, and trade modules (which operate on `self: Validator`).
        # Annotation-only declarations — no runtime effect; they document the
        # validator's state surface so static analysis resolves these attributes.
        simulation: "MarketSimulationConfig"
        roundtrip_volumes: dict
        roundtrip_volume_sums: dict
        volume_sums: dict
        maker_volume_sums: dict
        taker_volume_sums: dict
        self_volume_sums: dict
        fee_sums: dict
        inventory_history: dict
        realized_pnl_history: dict
        open_positions: dict
        pending_notices: dict
        kappa_cache: dict
        _gentrx_ema: dict
        seed_process: Any
        sltp_enabled: bool
        xml_config: str
        simulation_state_file: str
        validator_state_file: str
        simulator_config_file: str

        @classmethod
        def add_args(cls, parser: argparse.ArgumentParser) -> None:
            """
            Registers Intelligent-Markets-specific CLI configuration parameters.

            Args:
                parser (argparse.ArgumentParser): The main argument parser to extend.

            Returns:
                None
            """
            add_im_validator_args(cls, parser)

        def _setup_signal_handlers(self):
            """
            Registers OS signal handlers for graceful shutdown.

            Behavior:
                - Captures SIGINT, SIGTERM, and SIGHUP (if available).
                - Logs the received signal.
                - Triggers full validator cleanup.
                - Exits the process cleanly.

            Returns:
                None
            """
            def signal_handler(signum, frame):
                signal_name = signal.Signals(signum).name
                bt.logging.info(f"Received {signal_name}, initiating graceful shutdown...")
                self.cleanup()
                sys.exit(0)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, signal_handler)

        def _start_query_service(self):
            """
            Launches the validator's query service and initializes POSIX IPC resources.

            Responsibilities:
                - Spawns the query subprocess with correct wallet and network parameters.
                - Waits for creation of shared memory blocks and message queues.
                - Initializes memory maps for request/response communication.
                - Verifies that the query service is alive during startup.
                - Raises a RuntimeError if IPC initialization fails or service dies.

            Returns:
                None

            Raises:
                RuntimeError: If IPC endpoints are not ready within timeout
                            or if the subprocess exits unexpectedly.
            """
            import fcntl as _fcntl

            bt.logging.info("Starting query service from: ../validator/query.py")

            # Reap any existing query subprocess before spawning a new one.
            # Without this, every restart orphans the old process (reparented to
            # init); it keeps reading the shared named IPC queue, stealing
            # query/deliver messages and notifying its now-stale pipe fd — which
            # the live validator reads back as b'' / notification timeouts.
            _old_q = getattr(self, 'query_process', None)
            if _old_q is not None and _old_q.poll() is None:
                bt.logging.warning(f"Reaping existing query service (pid {_old_q.pid}) before restart")
                try:
                    _old_q.terminate()
                    try:
                        _old_q.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        _old_q.kill()
                        _old_q.wait(timeout=2.0)
                except Exception as _e:
                    bt.logging.warning(f"Failed to reap old query service: {_e}")

            # Recreate the notify pipe on every call (required on restarts because the
            # write end was closed after the previous start, and a fresh subprocess needs
            # a live write fd to pass via pass_fds).
            try:
                os.close(self.query_notify_read)
            except OSError:
                pass
            try:
                os.close(self.query_notify_write)
            except OSError:
                pass
            self.query_notify_read, self.query_notify_write = os.pipe()
            flags = _fcntl.fcntl(self.query_notify_read, _fcntl.F_GETFL)
            _fcntl.fcntl(self.query_notify_read, _fcntl.F_SETFL, flags | os.O_NONBLOCK)
            bt.logging.info(f"Notification pipe: read_fd={self.query_notify_read}, write_fd={self.query_notify_write}")

            core_allocation = get_core_allocation(
                sltp_cores_count=int(os.environ.get("SLTP_CORES_COUNT", "0")),
                grad_server_cores=int(os.environ.get("GRAD_CORES_COUNT", "0")),
            )
            _engine = getattr(self.config, 'engine', 'simulation')
            self._query_ipc_prefix = 'exchange' if _engine == 'exchange' else 'validator'
            cmd = [
                sys.executable,
                '-u',
                '../validator/query.py',
                '--logging.trace' if self.config.logging.trace else '--logging.debug' if self.config.logging.debug else '--logging.info',
                '--wallet.path', self.config.wallet.path,
                '--wallet.name', self.config.wallet.name,
                '--wallet.hotkey', self.config.wallet.hotkey,
                '--subtensor.network', self.config.subtensor.network,
                '--netuid', str(self.config.netuid),
                '--neuron.timeout', str(self.config.neuron.timeout),
                '--neuron.global_query_timeout', str(self.config.neuron.global_query_timeout),
                '--compression.level', str(self.config.compression.level),
                '--compression.engine', self.config.compression.engine,
                '--compression.parallel_workers', str(self.config.compression.parallel_workers),
                '--cpu-cores', ','.join(map(str, core_allocation['query'])),
                '--notify-fd', str(self.query_notify_write),
                '--ipc-prefix', self._query_ipc_prefix,
            ]
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            self.query_process = subprocess.Popen(
                cmd,
                stderr=sys.stderr,
                env=env,
                pass_fds=(self.query_notify_write,)
            )
            bt.logging.info(f"Query service PID: {self.query_process.pid}")
            os.close(self.query_notify_write)
            self.query_notify_write = -1  # mark as closed

            queue_name = f"/{self._query_ipc_prefix}_query_{self.config.wallet.hotkey}"

            # Wait for the ready signal BEFORE opening IPC resources.
            # The query subprocess sends b'R' only after setup_ipc() finishes — which
            # unlinks any stale queues/shm and creates fresh ones.  Connecting to IPC
            # before this signal risks grabbing resources left over from a previous
            # (still-running) query service that will be unlinked moments later.
            bt.logging.info("Waiting for query service ready signal...")
            ready = False
            for _attempt in range(150):  # 15s max (150 × 0.1s)
                if self.query_process.poll() is not None:
                    raise RuntimeError(f"Query service died with exit code {self.query_process.returncode}")
                readable, _, _ = select.select([self.query_notify_read], [], [], 0.1)
                if readable:
                    ready_signal = os.read(self.query_notify_read, 1)
                    if ready_signal == b'R':
                        bt.logging.success("Query service ready!")
                        ready = True
                        break
                    elif ready_signal != b'':
                        bt.logging.warning(f"Unexpected ready signal: {ready_signal!r}")
                        ready = True
                        break
            if not ready:
                raise RuntimeError("Query service did not send ready signal within 15s")

            # Now connect to the fresh IPC resources.
            bt.logging.info("Connecting to query service IPC resources...")
            max_retries = 30  # 30 seconds max
            for attempt in range(max_retries):
                try:
                    self.request_queue = posix_ipc.MessageQueue(f"{queue_name}_req")
                    self.response_queue = posix_ipc.MessageQueue(f"{queue_name}_res")
                    self.request_shm = posix_ipc.SharedMemory(f"{queue_name}_req_shm")
                    self.response_shm = posix_ipc.SharedMemory(f"{queue_name}_res_shm")

                    self.request_mem = mmap.mmap(self.request_shm.fd, self.request_shm.size)
                    self.response_mem = mmap.mmap(self.response_shm.fd, self.response_shm.size)

                    bt.logging.info(f"Query service IPC connected (request_shm: {self.request_shm.size / 1024 / 1024:.0f}MB, response_shm: {self.response_shm.size / 1024 / 1024:.0f}MB)")
                    return
                except posix_ipc.ExistentialError:
                    if attempt == 0:
                        bt.logging.debug("IPC resources not ready yet, waiting...")
                    time.sleep(1)
                    if self.query_process.poll() is not None:
                        raise RuntimeError(f"Query service died with exit code {self.query_process.returncode}")
            raise RuntimeError("Timeout waiting for query service IPC resources")

        def _start_reporting_service(self):
            bt.logging.info("Starting reporting service from: ../validator/report.py")

            # Reap any existing reporting subprocess before spawning a new one —
            # same orphan hazard as the query service (a stale report.py keeps
            # holding the prometheus port + shared report IPC; see _start_query_service).
            _old_r = getattr(self, 'reporting_process', None)
            if _old_r is not None and _old_r.poll() is None:
                bt.logging.warning(f"Reaping existing reporting service (pid {_old_r.pid}) before restart")
                try:
                    _old_r.terminate()
                    try:
                        _old_r.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        _old_r.kill()
                        _old_r.wait(timeout=2.0)
                except Exception as _e:
                    bt.logging.warning(f"Failed to reap old reporting service: {_e}")

            self._reporting = False
            core_allocation = get_core_allocation(
                sltp_cores_count=int(os.environ.get("SLTP_CORES_COUNT", "0")),
                grad_server_cores=int(os.environ.get("GRAD_CORES_COUNT", "0")),
            )
            # Use mode-specific IPC names so sim and exchange validators don't share queues.
            _engine = getattr(self.config, 'engine', 'simulation')
            self._reporting_ipc_prefix = 'exchange' if _engine == 'exchange' else 'validator'
            cmd = [
                sys.executable,
                '-u',
                '../validator/report.py',
                '--logging.trace' if self.config.logging.trace else '--logging.debug' if self.config.logging.debug else '--logging.info',
                '--wallet.path', self.config.wallet.path,
                '--wallet.name', self.config.wallet.name,
                '--wallet.hotkey', self.config.wallet.hotkey,
                '--subtensor.network', self.config.subtensor.network,
                '--netuid', str(self.config.netuid),
                '--prometheus.port', str(self.config.prometheus.port),
                '--prometheus.level', str(self.config.prometheus.level),
                '--cpu-cores', ','.join(map(str, core_allocation['reporting'])),
                '--ipc-prefix', self._reporting_ipc_prefix,
            ]
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            self.reporting_process = subprocess.Popen(cmd, stderr=sys.stderr, env=env)
            bt.logging.info(f"Reporting service PID: {self.reporting_process.pid}")

            _pfx = self._reporting_ipc_prefix
            bt.logging.info(f"Waiting for reporting service IPC resources (prefix={_pfx!r})...")
            max_retries = 30
            for attempt in range(max_retries):
                try:
                    self.reporting_request_queue = posix_ipc.MessageQueue(f"/{_pfx}-report-req")
                    self.reporting_response_queue = posix_ipc.MessageQueue(f"/{_pfx}-report-res")
                    self.reporting_request_shm = posix_ipc.SharedMemory(f"/{_pfx}-report-data")
                    self.reporting_response_shm = posix_ipc.SharedMemory(f"/{_pfx}-report-response-data")

                    self.reporting_request_mem = mmap.mmap(self.reporting_request_shm.fd, self.reporting_request_shm.size)
                    self.reporting_response_mem = mmap.mmap(self.reporting_response_shm.fd, self.reporting_response_shm.size)

                    bt.logging.info(f"Reporting service ready (shm: {self.reporting_request_shm.size / 1024 / 1024:.0f}MB)")
                    return

                except posix_ipc.ExistentialError:
                    if attempt == 0:
                        bt.logging.debug("IPC resources not ready yet, waiting...")
                    time.sleep(1)

                    if self.reporting_process.poll() is not None:
                        raise RuntimeError(f"Reporting service died with exit code {self.reporting_process.returncode}")

            raise RuntimeError("Timeout waiting for reporting service IPC resources")

        def monitor(self) -> None:
            """
            Periodically checks simulator health and restarts if needed with timeout protection.

            Runs in a blocking loop:
                - Sleeps 5 minutes between checks.
                - Logs simulator availability.
                - Handles and logs unexpected exceptions with timeout protection.

            Returns:
                None
            """
            last_restart_time = 0
            restart_cooldown = 60.0
            check_timeout = 30.0
            restart_timeout = 120.0
            observe = bool(getattr(getattr(self.config, 'neuron', None), 'observe', False))

            while True:
                try:
                    time.sleep(300)
                    if observe:
                        # Observe mode reads chain state via the provider — there's
                        # no local simulator/exchange (LOB) engine to health-check
                        # or restart. Skip to avoid false PagerDuty alerts and the
                        # no-op "restart" log noise. (Thread stays alive on purpose:
                        # returning would trip the post-launch is_alive() check.)
                        continue
                    bt.logging.info("Checking simulator state...")
                    check_start = time.time()
                    try:
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(check_simulator, self)
                            try:
                                is_healthy = future.result(timeout=check_timeout)
                            except concurrent.futures.TimeoutError:
                                bt.logging.error(f"Simulator health check timed out after {check_timeout}s")
                                is_healthy = False
                        check_elapsed = time.time() - check_start
                        if not is_healthy:
                            time_since_restart = time.time() - last_restart_time
                            if time_since_restart < restart_cooldown:
                                bt.logging.warning(
                                    f"Simulator unhealthy but restart on cooldown "
                                    f"({time_since_restart:.1f}s < {restart_cooldown}s)"
                                )
                                continue
                            bt.logging.warning(f"Simulator unhealthy (check took {check_elapsed:.1f}s), restarting...")
                            restart_start = time.time()
                            try:
                                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                    restart_future = executor.submit(restart_simulator, self)
                                    try:
                                        restart_future.result(timeout=restart_timeout)
                                        last_restart_time = time.time()
                                        restart_elapsed = time.time() - restart_start
                                        bt.logging.info(f"Simulator restarted successfully ({restart_elapsed:.1f}s)")
                                    except concurrent.futures.TimeoutError:
                                        bt.logging.error(f"Simulator restart timed out after {restart_timeout}s")
                                        self.pagerduty_alert(
                                            f"Simulator restart timeout after {restart_timeout}s"
                                        )
                                        last_restart_time = time.time()
                            except Exception as restart_ex:
                                bt.logging.error(f"Simulator restart failed: {restart_ex}")
                                bt.logging.error(traceback.format_exc())
                                self.pagerduty_alert(
                                    f"Simulator restart failed: {restart_ex}",
                                    details={"trace": traceback.format_exc()}
                                )
                                last_restart_time = time.time()
                        else:
                            bt.logging.info(f"Simulator online! (check took {check_elapsed:.1f}s)")

                    except Exception as check_ex:
                        bt.logging.error(f"Simulator health check failed: {check_ex}")
                        bt.logging.error(traceback.format_exc())

                except Exception:
                    bt.logging.error(f"Failure in simulator monitor : {traceback.format_exc()}")
                    time.sleep(10)

        def monitor_sltp(self) -> None:
            """Delegate to the optional SL/TP monitor loop when installed."""
            try:
                from taos.im.validator.exetrx import monitor_sltp as _monitor
            except ImportError:
                return
            _monitor(self)

        def update_repo(self, end=False) -> bool:
            """
            Checks for source or config changes in the repository and reloads components.

            Behavior:
                - Pulls latest remote changes.
                - Rebuilds simulator when C++ or Python sources change.
                - Restarts simulator on configuration changes.
                - Updates validator process when its own Python source changes.
                - Handles special behavior on simulation end.

            Args:
                end (bool): Whether the update is performed during simulation shutdown.

            Returns:
                bool: True if update steps completed successfully, False on error.
            """
            endpoint = getattr(getattr(self.config, "subtensor", None), "chain_endpoint", "") or ""
            if "localhost" in endpoint or "127.0.0.1" in endpoint:
                # Auto-update assumes pm2 supervision; localnet runs raw python.
                return True
            try:
                validator_py_files_changed, simulator_config_changed, simulator_py_files_changed, simulator_cpp_files_changed = check_repo(self)
                remote = self.repo.remotes[self.config.repo.remote]

                exchange_mode = getattr(getattr(self, 'engine', None), 'mode', 'simulation') == 'exchange'

                if not end:
                    if validator_py_files_changed and not (simulator_cpp_files_changed or simulator_py_files_changed):
                        bt.logging.warning("VALIDATOR LOGIC UPDATED - PULLING AND DEPLOYING.")
                        remote.pull()
                        update_validator(self)
                else:
                    try:
                        remote.pull()
                    except Exception as ex:
                        self.pagerduty_alert(f"Failed to pull changes from repo on simulation end : {ex}")
                    if not exchange_mode:
                        if simulator_cpp_files_changed or simulator_py_files_changed:
                            bt.logging.warning("SIMULATOR SOURCE CHANGED")
                            rebuild_simulator(self)
                        if simulator_config_changed:
                            bt.logging.warning("SIMULATOR CONFIG CHANGED")
                        restart_simulator(self, end)
                    if validator_py_files_changed:
                        update_validator(self)
                return True
            except Exception as ex:
                self.pagerduty_alert(f"Failed to update repo : {ex}", details={"traceback" : traceback.format_exc()})
                return False

        # NB: compress_outputs / load_simulation_config now live on the simulation engine
        # (taos/im/validator/engines/simulation.py).  Retained here as a comment to make
        # the move explicit during the GenTRX merge.

        def __init__(self, config=None) -> None:
            """
            Initializes the Intelligent Markets validator node.

            Responsibilities:
                - Loads simulation configuration from XML.
                - Initializes metagraph and subnet info.
                - Sets up executors, event loops, state locks, and signal handlers.
                - Loads prior simulation/validator state if available.
                - Initializes metrics, reporting, and query service.
                - Starts IPC-backed query subprocess.

            Args:
                config: Validator configuration object.

            Raises:
                Exception: If the simulation config XML file is missing.
            """
            super(Validator, self).__init__(config=config)

            kappa_w = self.config.scoring.kappa.weight
            pnl_w = self.config.scoring.pnl.weight
            gentrx_sim_share = getattr(getattr(self.config.scoring, 'gentrx', None), 'simulation_share', 0.0) or 0.0
            burn_ratio = getattr(self.config.neuron, 'burn_ratio', 0.0) or 0.0
            tolerance = 1e-6

            trading_sum = kappa_w + pnl_w
            if abs(trading_sum - 1.0) > tolerance:
                error_msg = (
                    f"Trading-pool weights must sum to 1.0, got "
                    f"kappa={kappa_w:.6f} + pnl={pnl_w:.6f} = {trading_sum:.6f}."
                )
                bt.logging.error(error_msg)
                raise ValueError(error_msg)

            if not (0.0 <= gentrx_sim_share <= 1.0):
                error_msg = (
                    f"--scoring.gentrx.simulation_share must be in [0, 1], "
                    f"got {gentrx_sim_share:.6f}."
                )
                bt.logging.error(error_msg)
                raise ValueError(error_msg)

            bt.logging.info(
                f"Scoring weights validated: trading=(kappa={kappa_w:.4f}, pnl={pnl_w:.4f}), "
                f"gentrx.simulation_share={gentrx_sim_share:.4f}, burn_ratio={burn_ratio:.4f}"
            )

            # Initialize subnet info and other basic validator/simulation properties
            self.subnet_info = self.subtensor.get_metagraph_info(self.config.netuid)

            self.benchmark_agents = []
            self.benchmark_start_uid = self.subnet_info.max_uids
            if self.config.benchmark.agents:
                self._load_benchmark_agents()
            max_bm_uid = max((a['uid'] for a in self.benchmark_agents), default=self.subnet_info.max_uids - 1)
            self.effective_max_uids = max(self.subnet_info.max_uids, max_bm_uid + 1)
            self.scores = torch.zeros(
                self.effective_max_uids, dtype=torch.float32, device=self.device
            )
            self.gentrx_scores = torch.zeros(
                self.effective_max_uids, dtype=torch.float32, device=self.device
            )

            self.last_state = None
            self.last_response = None
            self.msgpack_error_counter = 0
            self.simulation_timestamp = 0
            self.start_time = None
            self.start_timestamp = None
            self.last_state_time = None
            self.step_rates = []
            self._last_defrag_hour = -1
            self._last_prune_timestamp = None

            self.main_loop = asyncio.new_event_loop()
            self._main_loop_ready = Event()

            core_allocation = get_core_allocation(
                sltp_cores_count=int(os.environ.get("SLTP_CORES_COUNT", "0")),
                grad_server_cores=int(os.environ.get("GRAD_CORES_COUNT", "0")),
            )
            ipc_cores = core_allocation['ipc']
            if len(ipc_cores) >= 2:
                mid = len(ipc_cores) // 2
                query_cores = ipc_cores[:mid]
                reporting_cores = ipc_cores[mid:]
            else:
                query_cores = ipc_cores
                reporting_cores = ipc_cores
            self.query_ipc_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix='query_ipc',
                initializer=lambda: os.sched_setaffinity(0, set(query_cores))
            )
            self.reporting_ipc_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix='reporting_ipc',
                initializer=lambda: os.sched_setaffinity(0, set(reporting_cores))
            )
            bt.logging.info(f"Query IPC executor assigned to cores: {query_cores}")
            bt.logging.info(f"Reporting IPC executor assigned to cores: {reporting_cores}")
            validator_cores = core_allocation['validator']
            os.sched_setaffinity(0, set(validator_cores))
            bt.logging.info(f"Validator assigned to cores: {validator_cores}")
            self.reward_cores = core_allocation['reward']
            self.reward_executor = ThreadPoolExecutor(max_workers=len(self.reward_cores),initializer=lambda: os.sched_setaffinity(0, set(self.reward_cores)))
            bt.logging.info(f"Reward executor assigned to cores: {self.reward_cores}")
            self.save_state_executor = ThreadPoolExecutor(max_workers=1)
            self.maintenance_executor = ThreadPoolExecutor(max_workers=1)
            self.maintenance_subtensor = bt.Subtensor(self.config.subtensor.chain_endpoint)

            self.maintaining = False
            self.compressing = False
            self.querying = False
            # Mutual exclusion between the miner state-update query (forward)
            # and GTX assignment delivery (deliver_gentrx): they share the IPC
            # request queue/notify pipe and the miner network, so they must
            # never run concurrently. asyncio.Lock — both run on the Listen loop.
            self.miner_net_lock = asyncio.Lock()
            # Background GTX round-check/delivery task (single-flight).
            self._gentrx_task = None
            self._receiving_state = False
            self._rewarding = False
            self._pending_reward_tasks = 0
            self._saving = False
            self._reporting = False
            self._rewarding_lock = Lock()
            self._saving_lock = Lock()
            self._reporting_lock = Lock()
            self._reward_lock = asyncio.Lock()
            self._setup_signal_handlers()
            self._cleanup_done = False
            atexit.register(self.cleanup)


            self.router = APIRouter()
            self.router.add_api_route("/orderbook", self.orderbook, methods=["GET"])
            self.router.add_api_route("/account", self.account, methods=["GET"])
            self.router.add_api_route("/sltp", self.sltp_records, methods=["POST", "GET"])
            self.router.add_api_route("/sltp/levels", self.sltp_levels, methods=["GET"])

            # ---- GenTRX distributed training (optional) ----
            # Log via the `gtx_log` shim (imported at module top) — forwards
            # to bt.logging with the `[GTX]` prefix baked in, so every record
            # surfaces through the same stream the rest of the validator uses.
            self._gentrx = None
            _gentrx_enabled = getattr(getattr(self.config, "gentrx", None), "enabled", False)
            gtx_log.info(f"init: enabled={_gentrx_enabled}")
            if _gentrx_enabled:
                try:
                    from GenTRX.src.service import GenTRXService
                    # Miner UIDs are computed dynamically — metagraph changes
                    # over time as miners register/dereg, and miners may not be
                    # registered yet at validator init.
                    my_hotkey = self.wallet.hotkey.ss58_address

                    def _current_miner_uids():
                        uids = [
                            uid for uid in range(self.metagraph.n)
                            if self.metagraph.axons[uid].hotkey != my_hotkey
                            and self.metagraph.axons[uid].ip != "0.0.0.0"
                            and self.metagraph.axons[uid].port != 0
                        ]
                        # Include benchmark agents that have active axons (scored,
                        # not rewarded — same pattern as trading benchmarks).
                        for i, agent in enumerate(self.benchmark_agents):
                            axon = agent['axon']
                            if axon.ip != "0.0.0.0" and axon.port != 0:
                                uids.append(self.benchmark_start_uid + i)
                        return uids

                    gtx_log.info(f"init: server_url={self.config.gentrx.gradient_server_url}, current miner_uids={_current_miner_uids()}")

                    def _get_current_block():
                        try:
                            with self._subtensor_lock:
                                return self.subtensor.get_current_block()
                        except Exception:
                            return 0

                    self._gentrx = GenTRXService.from_config(
                        self.config,
                        deliver_fn=self._deliver_gentrx_assignments,
                        miner_uids_fn=_current_miner_uids,
                        get_block_fn=_get_current_block,
                        validator_uid=self.uid,
                    )
                    if self._gentrx:
                        if self._gentrx.is_healthy():
                            gtx_log.info(f"init: gradient server healthy at {self.config.gentrx.gradient_server_url}")
                        else:
                            gtx_log.warning(f"init: gradient server unreachable at {self.config.gentrx.gradient_server_url} — will retry on each tick")
                        # Commit validator bucket read credentials to chain.
                        # Miners and the aggregator use this to discover the bucket
                        # (data, scores, checkpoints) without pre-configured env vars.
                        try:
                            from GenTRX.src.chain import BucketInfo, GenTRXChain
                            val_bucket = BucketInfo.from_validator_env()
                            if val_bucket:
                                gtx_chain = GenTRXChain(self.subtensor, self.config.netuid, self.metagraph)
                                gtx_chain.commit_bucket(self.wallet, val_bucket)
                                gtx_log.info("validator bucket committed to chain")
                            else:
                                gtx_log.debug("GENTRX_VALIDATOR_S3_* not set — skipping chain commitment")
                        except Exception as exc:
                            import traceback
                            gtx_log.warning(f"validator chain commit failed: {exc}")
                            gtx_log.warning(traceback.format_exc())
                        # Register benchmark miner buckets with the gradient server.
                        # These miners can't commit to chain (not registered), so the
                        # validator injects their bucket info via the REST API.
                        for bm in self.benchmark_agents:
                            bkt = bm.get('gentrx_bucket')
                            if bkt:
                                ok = self._gentrx.register_benchmark_bucket(bm['uid'], bkt)
                                if ok:
                                    gtx_log.info(f"registered benchmark bucket: uid={bm['uid']} bucket={bkt.get('bucket_name')}")
                                else:
                                    gtx_log.warning(f"failed to register benchmark bucket: uid={bm['uid']}")
                    else:
                        gtx_log.warning("init: from_config returned None — check that --gentrx.gradient_server_url is set")
                except ImportError as exc:
                    gtx_log.warning(f"init: import failed — {exc}")
                except Exception as exc:
                    gtx_log.error(f"init: failed — {exc}")
                    import traceback
                    gtx_log.error(traceback.format_exc())
            gtx_log.info(f"init: {'ACTIVE' if self._gentrx else 'DISABLED'}")

            self.repo_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
            self.repo = Repo(self.repo_path)
            self.update_repo()

            self.query_process = None
            self.query_notify_read = -1
            self.query_notify_write = -1
            if not getattr(getattr(self.config, 'neuron', None), 'observe', False):
                self._start_query_service()
            else:
                bt.logging.info("Observe mode — skipping query service")
            self.report_process = None
            self._start_reporting_service()
            engine_mode = getattr(self.config, 'engine', 'simulation')
            if engine_mode == 'exchange':
                bt.logging.info("Starting validator in EXCHANGE engine mode")
                from taos.im.validator.engines.exchange import ExchangeEngine
                self.engine = ExchangeEngine(self.config, self)
                self.engine._push_fill_notify = _push_fill_notifications
            else:
                self.engine = SimulationEngine(self.config, self)
            self.engine.start()

        def _load_benchmark_agents(self):
            """Load benchmark agent configurations"""
            try:
                with open(self.config.benchmark.agents, 'r') as f:
                    benchmark_config = json.load(f)                
                for idx, agent in enumerate(benchmark_config['agents']):
                    uid = agent.get('uid', self.benchmark_start_uid + idx)
                    entry = {
                        'uid': uid,
                        'name': agent['name'],
                        'ip': agent['ip'],
                        'port': agent['port'],
                        'hotkey': agent['hotkey'],
                        'axon': bt.AxonInfo(
                            ip=agent['ip'],
                            port=agent['port'],
                            hotkey=agent['hotkey'],
                            coldkey='benchmark',
                            version=self.metagraph.axons[0].version,
                            ip_type=self.metagraph.axons[0].ip_type,
                            protocol=self.metagraph.axons[0].protocol,
                        )
                    }
                    if 'gentrx_bucket' in agent:
                        entry['gentrx_bucket'] = agent['gentrx_bucket']
                    self.benchmark_agents.append(entry)
                    bt.logging.info(f"Loaded benchmark agent: {agent['name']} (UID {uid}) at {agent['ip']}:{agent['port']}")            
            except Exception as ex:
                bt.logging.error(f"Failed to load benchmark agents: {ex}")
                bt.logging.error(traceback.format_exc())
                self.benchmark_agents = []
                
        def _initialize_all_structures(self):
            """
            Initialize all UID-indexed structures to effective_max_uids.
            """
            bt.logging.info(
                f"Initializing structures for {self.effective_max_uids} UIDs "
                f"(network: {self.subnet_info.max_uids}, benchmarks: {len(self.benchmark_agents)}) "
                f"with {self.simulation.book_count} books"
            )
            book_count = self.simulation.book_count
            if not hasattr(self, 'activity_factors') or len(self.activity_factors) < self.effective_max_uids:
                self.activity_factors = {
                    uid: {bookId: 0.0 for bookId in range(book_count)}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'pnl_factors') or len(self.pnl_factors) < self.effective_max_uids:
                self.pnl_factors = {
                    uid: {bookId: 1.0 for bookId in range(book_count)}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'kappa_values') or len(self.kappa_values) < self.effective_max_uids:
                self.kappa_values = {
                    uid: {
                        'books': {bookId: None for bookId in range(book_count)},
                        'books_weighted': {bookId: 0.0 for bookId in range(book_count)},
                        'total': None,
                        'average': None,
                        'median': None,
                        'normalized_average': 0.0,
                        'normalized_median': 0.0,
                        'normalized_total': 0.0,
                        'activity_weighted_normalized_median': 0.0,
                        'penalty': 0.0,
                        'score': 0.0,
                    }
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'unnormalized_scores') or len(self.unnormalized_scores) < self.effective_max_uids:
                self.unnormalized_scores = {uid: 0.0 for uid in range(self.effective_max_uids)}
            if not hasattr(self, 'trade_volumes') or len(self.trade_volumes) < self.effective_max_uids:
                self.trade_volumes = {
                    uid: {
                        bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                        for bookId in range(book_count)
                    }
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'initial_balances') or len(self.initial_balances) < self.effective_max_uids:
                self.initial_balances = {
                    uid: {
                        bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                        for bookId in range(book_count)
                    }
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'recent_miner_trades') or len(self.recent_miner_trades) < self.effective_max_uids:
                self.recent_miner_trades = {
                    uid: {bookId: [] for bookId in range(book_count)}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'miner_stats') or len(self.miner_stats) < self.effective_max_uids:
                self.miner_stats = {
                    uid: {'requests': 0, 'timeouts': 0, 'failures': 0, 'rejections': 0, 'call_time': []}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'recent_trades'):
                self.recent_trades = {bookId: [] for bookId in range(book_count)}
            if not hasattr(self, 'fundamental_price'):
                self.fundamental_price = {bookId: None for bookId in range(book_count)}
            
            bt.logging.success(f"All structures initialized for {self.effective_max_uids} UIDs")

        def get_extended_metagraph(self):
            return self.engine.get_extended_metagraph()

        def __enter__(self):
            """
            Enables use of Validator as a context manager.

            Returns:
                Validator: The active validator instance.
            """
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Ensures cleanup is triggered when exiting a context manager block.

            Args:
                exc_type: Exception type, if any.
                exc_val: Exception instance, if any.
                exc_tb: Traceback, if any.

            Returns:
                bool: False to propagate exceptions.
            """
            self.cleanup()
            return False

        @property
        def shared_state_rewarding(self):
            with self._rewarding_lock:
                return self._rewarding

        @shared_state_rewarding.setter
        def shared_state_rewarding(self, value):
            with self._rewarding_lock:
                self._rewarding = value

        @property
        def shared_state_saving(self):
            with self._saving_lock:
                return self._saving

        @shared_state_saving.setter
        def shared_state_saving(self, value):
            with self._saving_lock:
                self._saving = value

        @property
        def shared_state_reporting(self):
            with self._reporting_lock:
                return self._reporting

        @shared_state_reporting.setter
        def shared_state_reporting(self, value):
            with self._reporting_lock:
                self._reporting = value

        async def wait_for(self, check_fn: callable, message: str, interval: float = 0.01):
            """
            Asynchronously waits for a condition to become False.

            Behavior:
                - Logs a message once per second while waiting.
                - Returns immediately if condition is already False.
                - Provides debug timing on completion.

            Args:
                check_fn (callable): Function returning a boolean condition.
                message (str): Log message describing the wait condition.
                interval (float): Interval between checks in seconds.

            Returns:
                None
            """
            if not check_fn():
                return

            start_time = time.time()
            last_log_time = start_time

            bt.logging.info(message)

            while check_fn():
                await asyncio.sleep(interval)

                current_time = time.time()
                elapsed = current_time - start_time

                if current_time - last_log_time >= 1.0:
                    bt.logging.info(f"{message} (waited {elapsed:.1f}s)")
                    last_log_time = current_time

            total_wait = time.time() - start_time
            bt.logging.debug(f"Wait completed after {total_wait:.1f}s")

        async def wait_for_event(self, event: asyncio.Event, wait_process: str, run_process: str):
            """
            Waits for an asyncio.Event to be set before continuing execution.

            Provides periodic logging while waiting, and measures the total wait
            duration for operational visibility.

            Args:
                event (asyncio.Event): The event that must be completed.
                wait_process (str): Name of the process being waited on.
                run_process (str): Name of the process to run afterward.

            Returns:
                None
            """
            if not event.is_set():
                bt.logging.debug(f"Waiting for {wait_process} to complete before {run_process}...")
                start_wait = time.time()
                while not event.is_set():
                    try:
                        await asyncio.wait_for(event.wait(), timeout=1.0)
                        break
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0)
                        elapsed = time.time() - start_wait
                        if int(elapsed) % 5 == 0 and elapsed > 0:
                            bt.logging.debug(f"Still waiting for {wait_process}... ({elapsed:.1f}s)")
                total_wait = time.time() - start_wait
                bt.logging.debug(f"Waited {total_wait:.1f}s for {wait_process}")

        async def _write_ipc_nonblocking(self, mem, queue, data_bytes, operation="IPC", executor=None):
            """
            Write to shared memory in executor thread.
            """
            if executor is None:
                executor = self.query_ipc_executor
            write_timeout = 10.0

            def write_worker():
                try:
                    mem.seek(0)
                    mem.write(struct.pack('Q', len(data_bytes)))
                    mem.write(data_bytes)
                    return True
                except Exception as e:
                    self.pagerduty_alert(f"{operation} write failed: {e}")
                    return False

            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        executor,
                        write_worker
                    ),
                    timeout=write_timeout
                )
                return result
            except asyncio.TimeoutError:
                self.pagerduty_alert(f"{operation} write timeout after {write_timeout}s")
                return False

        def onStart(self, timestamp, event : SimulationStartEvent) -> None:
            self.engine.on_start(timestamp, event)

        def onEnd(self) -> None:
            self.engine.on_end()


        def _build_validator_state(
            self,
            inventory_snapshot,
            realized_pnl_snapshot,
            volume_sums_snapshots,
            trade_volumes_snapshot,
            roundtrip_volumes_snapshot,
            open_positions_snapshot
        ):
            """Assemble validator state dict from all snapshots."""
            return build_validator_state(
                self, inventory_snapshot, realized_pnl_snapshot, volume_sums_snapshots,
                trade_volumes_snapshot, roundtrip_volumes_snapshot, open_positions_snapshot
            )


        def _snapshot_inventory_history(self):
            return snapshot_inventory_history(self)


        def _snapshot_realized_pnl_history(self):
            return snapshot_realized_pnl_history(self)


        def _snapshot_2_level_dict(self, source_dict):
            return snapshot_2_level_dict(self, source_dict)


        def _snapshot_volume_sums(self):
            return snapshot_volume_sums(self)


        def _snapshot_trade_volumes(self):
            return snapshot_trade_volumes(self)


        def _snapshot_roundtrip_volumes(self):
            return snapshot_roundtrip_volumes(self)


        def _snapshot_open_positions(self):
            return snapshot_open_positions(self)


        def _construct_save_data_sync(self):
            """Synchronously build all state data for saving."""
            return construct_save_data_sync(self)


        def _defragment_histories(self):
            """Rebuild history dicts to eliminate memory fragmentation."""
            defragment_histories(self)


        def save_state(self) -> None:
            """Schedules the asynchronous state-saving coroutine on the main event loop."""
            schedule_save(self)

        async def _save_state_sync(self):
            """Save validator state synchronously from an async context."""
            await save_state_sync(self)

        def migrate_sampling_interval(self, old_interval: int, new_interval: int):
            """Re-align trade volumes from old sampling interval to new interval."""
            migrate_sampling_interval(self, old_interval, new_interval)


        def load_state(self) -> None:
            """Loads validator and simulation state from msgpack or legacy PyTorch files."""
            load_state(self)


        def get_n_target_miners(self) -> int:
            """
            Count registered miner UIDs in the metagraph (validators excluded).

            Used by `prepare_weights` to size the GenTRX pool's participation
            denominator. A miner is identified by having a reachable axon
            endpoint and not being the running validator itself.
            """
            try:
                my_hotkey = self.wallet.hotkey.ss58_address
            except Exception:
                my_hotkey = None
            count = 0
            for uid in range(self.metagraph.n):
                axon = self.metagraph.axons[uid]
                if my_hotkey is not None and axon.hotkey == my_hotkey:
                    continue
                if axon.ip == "0.0.0.0" or axon.port == 0:
                    continue
                count += 1
            return count

        def handle_deregistration(self, uid) -> None:
            """Engine handles primary deregistration; we also reset GenTRX state here."""
            self.engine.handle_deregistration(uid)
            # Reset GenTRX EMA + service score cache so a new miner at the
            # same UID slot doesn't inherit the old miner's score history.
            if hasattr(self, 'gentrx_scores'):
                self.gentrx_scores[uid] = 0.0
            if hasattr(self, '_gentrx_ema') and self._gentrx_ema:
                self._gentrx_ema.pop(uid, None)
            if getattr(self, '_gentrx', None) is not None:
                self._gentrx._scores.pop(uid, None)

        def process_resets(self, state: NormalizedState) -> None:
            """
            Processes reset notices delivered by the simulator.
            Collects UIDs needing reset via the engine, then applies them.
            """
            pending = set()
            self.engine.collect_resets(state, pending)
            self.engine.apply_resets(pending)


        def resync_metagraph(self):
            """Resyncs the metagraph and updates hotkeys and scores."""
            bt.logging.trace("resync_metagraph()")
            previous_metagraph = copy.deepcopy(self.metagraph)
            bt.logging.debug("Syncing metagraph...")
            self.metagraph.sync(subtensor=self.subtensor)
            if previous_metagraph.axons == self.metagraph.axons and len(self.hotkeys) == len(self.metagraph.hotkeys):
                bt.logging.debug("No axon changes!")
                # Re-register benchmark buckets even when the metagraph is unchanged,
                # so a gradient server restart self-heals within one resync cycle.
                _gtx = getattr(self, '_gentrx', None)
                if _gtx is not None:
                    for bm in self.benchmark_agents:
                        bkt = bm.get('gentrx_bucket')
                        if bkt:
                            ok = _gtx.register_benchmark_bucket(bm['uid'], bkt)
                            if ok:
                                gtx_log.debug("re-registered benchmark bucket: uid=%d", bm['uid'])
                            else:
                                gtx_log.warning("failed to re-register benchmark bucket: uid=%d", bm['uid'])
                return

            bt.logging.info(
                "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
            )
            for uid, hotkey in enumerate(self.hotkeys):
                if uid < len(self.metagraph.hotkeys) and hotkey != self.metagraph.hotkeys[uid]:
                    self.handle_deregistration(uid)

            old_metagraph_size = len(self.hotkeys)
            new_metagraph_size = len(self.metagraph.hotkeys)
            if old_metagraph_size != new_metagraph_size:
                bt.logging.info(f"Metagraph size changed: {old_metagraph_size} -> {new_metagraph_size}")
                # `super().__init__` calls sync() → resync_metagraph() before `self.engine`
                # gets assigned in the subclass init body. During that first pass there is
                # nothing for the engine to resize yet, so skip the engine hook safely.
                if hasattr(self, 'engine'):
                    self.engine.on_resync_metagraph(old_metagraph_size, new_metagraph_size)

                # Expand per-UID dicts to cover any UIDs not yet present (new network
                # registrations, or holes left after benchmark shifting above).
                _bc = self.simulation.book_count
                _kappa_default = {
                    'books': {b: None for b in range(_bc)},
                    'books_weighted': {b: 0.0 for b in range(_bc)},
                    'total': None, 'average': None, 'median': None,
                    'normalized_average': 0.0, 'normalized_median': 0.0,
                    'normalized_total': 0.0,
                    'activity_weighted_normalized_median': 0.0,
                    'penalty': 0.0, 'score': 0.0,
                }
                for _uid in range(self.effective_max_uids):
                    if _uid not in self.miner_stats:
                        self.miner_stats[_uid] = {
                            'requests': 0, 'timeouts': 0, 'failures': 0,
                            'rejections': 0, 'call_time': [],
                        }
                    if _uid not in self.activity_factors:
                        self.activity_factors[_uid] = {b: 0.0 for b in range(_bc)}
                    if _uid not in self.pnl_factors:
                        self.pnl_factors[_uid] = {b: 1.0 for b in range(_bc)}
                    if _uid not in self.kappa_values:
                        self.kappa_values[_uid] = dict(_kappa_default)
                        self.kappa_values[_uid]['books'] = {b: None for b in range(_bc)}
                        self.kappa_values[_uid]['books_weighted'] = {b: 0.0 for b in range(_bc)}
                    if _uid not in self.unnormalized_scores:
                        self.unnormalized_scores[_uid] = 0.0
                    if _uid not in self.initial_balances:
                        self.initial_balances[_uid] = {
                            b: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                            for b in range(_bc)
                        }
                    if _uid not in self.initial_balances_published:
                        self.initial_balances_published[_uid] = False
                    if _uid not in self.inventory_history:
                        self.inventory_history[_uid] = {}
                    if _uid not in self.recent_miner_trades:
                        self.recent_miner_trades[_uid] = {b: [] for b in range(_bc)}

            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
            bt.logging.success(f"Metagraph resync complete: {len(self.hotkeys)} network hotkeys")
            # Create proxy wallets for any newly registered UIDs and refresh
            # executor uid→coldkey maps.  Must run after hotkeys is updated.
            if hasattr(self.engine, 'on_metagraph_synced'):
                self.engine.on_metagraph_synced(self.metagraph)

            # Re-register benchmark buckets after every resync so that a gradient
            # server restart self-heals without requiring a validator restart.
            _gtx = getattr(self, '_gentrx', None)
            if _gtx is not None:
                for bm in self.benchmark_agents:
                    bkt = bm.get('gentrx_bucket')
                    if bkt:
                        _gtx.register_benchmark_bucket(bm['uid'], bkt)

        async def _maintain(self) -> None:
            """
            Executes metagraph sync and maintenance operations asynchronously.

            Actions:
                - Marks the validator as in maintenance mode.
                - Runs synchronous maintenance work in an executor thread.
                - Logs timing and reports issues via PagerDuty.

            Returns:
                None
            """
            try:
                self.maintaining = True
                bt.logging.info(f"Synchronizing at Step {self.step}...")
                start = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.maintenance_executor,
                    self._sync_and_check
                )
                bt.logging.info(f"Synchronized ({time.time()-start:.4f}s)")

            except Exception as ex:
                self.pagerduty_alert(f"Failed to sync: {ex}", details={"trace": traceback.format_exc()})
            finally:
                self.maintaining = False

        async def _deliver_gentrx_assignments(self, assignments: dict) -> None:
            """Build delivery list from assignments and call deliver_gentrx."""
            try:
                from taos.im.protocol.gentrx import GenTRXAssignment

                my_hotkey = self.wallet.hotkey.ss58_address
                try:
                    my_uid = self.metagraph.hotkeys.index(my_hotkey)
                except ValueError:
                    my_uid = -1
                deliveries = []
                for uid, assignment in assignments.items():
                    if uid >= len(self.metagraph.axons):
                        bm_idx = uid - self.benchmark_start_uid
                        if bm_idx < 0 or bm_idx >= len(self.benchmark_agents):
                            continue
                        axon = self.benchmark_agents[bm_idx]['axon']
                    else:
                        axon = self.metagraph.axons[uid]
                    if axon.hotkey == my_hotkey:
                        continue
                    if axon.ip == "0.0.0.0" or axon.port == 0:
                        continue
                    try:
                        deliveries.append((
                            uid,
                            axon,
                            GenTRXAssignment(
                                round=assignment.get("round", 0),
                                model_version=assignment.get("model_version", 0),
                                books=assignment.get("books", []),
                                ts_start=assignment.get("ts_start", 0),
                                ts_end=assignment.get("ts_end", 0),
                                data=assignment.get("data", []),
                                data_source=assignment.get("data_source", "s3"),
                                data_endpoint=assignment.get("data_endpoint", ""),
                                data_bucket=assignment.get("data_bucket", ""),
                                data_access_key=assignment.get("data_access_key", ""),
                                data_secret_key=assignment.get("data_secret_key", ""),
                                validator_uid=my_uid,
                            ),
                        ))
                    except Exception as exc:
                        import traceback
                        gtx_log.warning(f"build assignment for uid {uid} failed: {exc}")
                        gtx_log.warning(traceback.format_exc())

                if not deliveries:
                    return

                round_id = next(iter(assignments.values())).get("round", "?")
                gtx_log.info(f"round={round_id}: delivering to uids={[u for u,_,_ in deliveries]}")
                # One atomic send to ALL miners (never chunked), but never
                # overlapping a state-update query: the lock is held by
                # forward() for its query window and by us for the delivery.
                async with self.miner_net_lock:
                    await deliver_gentrx(self, deliveries)
            except Exception as exc:
                gtx_log.error(f"delivery failed: {exc}")
                import traceback
                gtx_log.error(traceback.format_exc())

        def _sync_and_check(self):
            """
            Performs synchronous metagraph maintenance and simulator health checks.

            Steps:
                - Runs Bittensor sync (without saving state).
                - Verifies simulator health.
                - Restarts simulator if unhealthy.

            Returns:
                None
            """
            # Use a dedicated subtensor connection so this thread doesn't
            # share a websocket with the chain data provider running concurrently.
            primary, self.subtensor = self.subtensor, self.maintenance_subtensor
            try:
                self.sync(save_state=False)
            finally:
                self.subtensor = primary
            if not check_simulator(self):
                restart_simulator(self)

        def maintain(self) -> None:
            """
            Schedules asynchronous maintenance work from the maintenance thread.

            Behavior:
                - Ensures maintenance is not already running.
                - Triggers only at specific simulation timestamps.
                - Sends a coroutine to the main event loop thread-safely.

            Returns:
                None
            """
            if not self.maintaining and self.last_state and self.last_state.timestamp % self.config.scoring.interval == 2_000_000_000:
                bt.logging.debug(f"[MAINT] Scheduling from thread: {threading.current_thread().name}")
                bt.logging.debug(f"[MAINT] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
                self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._maintain()))

        def _prepare_reporting_data(self):
            bt.logging.debug("Retrieving fundamental prices...")
            start = time.time()
            if hasattr(self.engine, 'load_fundamental'):
                self.engine.load_fundamental()
            bt.logging.debug(f"Retrieved fundamental prices ({time.time()-start:.4f}s).")

            book_count = self.simulation.book_count
            bt.logging.debug("Computing realized P&L totals...")
            pnl_start = time.time()
            total_realized_pnl = {}
            realized_pnl_by_book = {}

            for uid, hist in self.realized_pnl_history.items():
                if not hist:
                    total_realized_pnl[uid] = 0.0
                    realized_pnl_by_book[uid] = {book_id: 0.0 for book_id in range(book_count)}
                    continue

                book_totals = [0.0] * book_count
                for _ts, timestamp_data in hist.items():
                    for book_id, pnl in timestamp_data.items():
                        if book_id < len(book_totals):
                            book_totals[book_id] += pnl

                realized_pnl_by_book[uid] = {book_id: book_totals[book_id] for book_id in range(book_count)}
                total_realized_pnl[uid] = sum(book_totals)

            bt.logging.debug(f"Computed realized P&L totals ({time.time()-pnl_start:.4f}s)")

            bt.logging.debug("Building minimal inventory...")
            inv_start = time.time()
            minimal_inventory = {}
            for uid, hist in self.inventory_history.items():
                if not hist:
                    continue
                timestamps = sorted(hist.keys())
                n = len(timestamps)
                if n >= 3:
                    minimal_inventory[uid] = {
                        timestamps[0]: hist[timestamps[0]],
                        timestamps[-2]: hist[timestamps[-2]],
                        timestamps[-1]: hist[timestamps[-1]]
                    }
                elif n > 0:
                    minimal_inventory[uid] = {ts: hist[ts] for ts in timestamps}
            bt.logging.debug(f"Built minimal inventory ({time.time()-inv_start:.4f}s)")

            bt.logging.debug("Serializing volume sums...")
            serialize_start = time.time()

            def serialize_nested_dict(d):
                return {
                    f"{uid}:{book_id}": vol
                    for uid, books in d.items()
                    for book_id, vol in books.items()
                }

            volume_sums_flat = serialize_nested_dict(self.volume_sums)
            maker_volume_sums_flat = serialize_nested_dict(self.maker_volume_sums)
            taker_volume_sums_flat = serialize_nested_dict(self.taker_volume_sums)
            self_volume_sums_flat = serialize_nested_dict(self.self_volume_sums)
            fee_sums_flat = serialize_nested_dict(getattr(self, 'fee_sums', {}))
            roundtrip_volume_sums_flat = serialize_nested_dict(self.roundtrip_volume_sums)

            bt.logging.debug(f"Serialized volume sums ({time.time()-serialize_start:.4f}s)")

            bt.logging.debug("Building metagraph data...")
            meta_start = time.time()
            metagraph_data = {
                'hotkeys': [str(hk) for hk in self.metagraph.hotkeys],
                'coldkeys': [str(ck) for ck in self.metagraph.coldkeys] if hasattr(self.metagraph, 'coldkeys') else [],
                'stake': self.metagraph.stake.tolist(),
                'trust': self.metagraph.trust.tolist() if hasattr(self.metagraph, 'trust') else [],
                'consensus': self.metagraph.consensus.tolist(),
                'incentive': self.metagraph.incentive.tolist(),
                'emission': self.metagraph.emission.tolist(),
                'validator_trust': self.metagraph.validator_trust.tolist(),
                'validator_permit': self.metagraph.validator_permit.tolist() if hasattr(self.metagraph, 'validator_permit') else [],
                'dividends': self.metagraph.dividends.tolist(),
                'active': self.metagraph.active.tolist(),
                'last_update': self.metagraph.last_update.tolist(),
            }
            bt.logging.debug(f"Built metagraph data ({time.time()-meta_start:.4f}s)")

            bt.logging.debug("Building recent trades...")
            trades_start = time.time()
            recent_trades = {
                bookId: [t.model_dump() for t in trades[-25:]]
                for bookId, trades in self.recent_trades.items()
            }

            recent_miner_trades = {
                uid: {
                    bookId: [
                        {'trade': miner_trade.model_dump(), 'role': role}
                        for miner_trade, role in trades[-5:]
                    ]
                    for bookId, trades in book_trades.items()
                }
                for uid, book_trades in self.recent_miner_trades.items()
            }
            bt.logging.debug(f"Built minimal recent trades ({time.time()-trades_start:.4f}s)")

            def _cap_notices_per_book(notices, per_book=20):
                _NOTICE_TYPES = frozenset({
                    'EVENT_TRADE', 'ET',
                    'RDPOL', 'ERDPOL', 'RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT', 'ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT',
                    'RDPOM', 'ERDPOM', 'RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET', 'ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET',
                    'RDCO', 'ERDCO', 'RESPONSE_DISTRIBUTED_CANCEL_ORDERS', 'ERROR_RESPONSE_DISTRIBUTED_CANCEL_ORDERS',
                })
                by_book: dict = {}
                for n in notices:
                    if n.get('y') not in _NOTICE_TYPES:
                        continue
                    bk = n.get('b') or n.get('bookId') or 0
                    by_book.setdefault(bk, []).append(n)
                return [n for ns in by_book.values() for n in ns[-per_book:]]

            bt.logging.debug("Building minimal state...")
            state_start = time.time()
            minimal_state = {
                'accounts': self.last_state.accounts,
                'pools': self.last_state.pools,
                'books': {
                    bookId: {
                        'b': book['b'][:5] if book['b'] else [],
                        'a': book['a'][:5] if book['a'] else [],
                        # Include all event types (orders, fills, cancels, chain events)
                        # not just fills — required for the full L3 stream in the data service.
                        'e': (book.get('e') or [])[-50:],
                        'r': book.get('r', book.get('mtr', 0.0))
                    }
                    for bookId, book in self.last_state.books.items()
                },
                'notices': {
                    uid: _cap_notices_per_book(notices)
                    for uid, notices in self.last_state.notices.items()
                }
            }
            bt.logging.debug(f"Built minimal state ({time.time()-state_start:.4f}s)")

            bt.logging.debug("Building position summary...")
            pos_start = time.time()
            open_positions = {
                uid: {
                    book_id: {
                        'longs_count': len(pos['longs']),
                        'shorts_count': len(pos['shorts'])
                    }
                    for book_id, pos in books.items()
                }
                for uid, books in self.open_positions.items()
            }
            bt.logging.debug(f"Built position summary ({time.time()-pos_start:.4f}s)")

            bt.logging.debug("Assembling final data structure...")
            final_start = time.time()

            data = {
                'metagraph_data': metagraph_data,
                'simulation': self.simulation.model_dump(),
                'last_state': minimal_state,
                'simulation_timestamp': self.simulation_timestamp,
                'step': self.step,
                'step_rates': list(self.step_rates[-100:]),
                'volume_sums': volume_sums_flat,
                'maker_volume_sums': maker_volume_sums_flat,
                'taker_volume_sums': taker_volume_sums_flat,
                'self_volume_sums': self_volume_sums_flat,
                'fee_sums': fee_sums_flat,
                'roundtrip_volume_sums': roundtrip_volume_sums_flat,
                'inventory_history': minimal_inventory,
                'total_realized_pnl': total_realized_pnl,
                'realized_pnl_by_book': realized_pnl_by_book,
                'book_count': book_count,
                'activity_factors': self.activity_factors,
                'pnl_factors': self.pnl_factors,
                'kappa_values': self.kappa_values,
                'unnormalized_scores': self.unnormalized_scores,
                'scores': {i: score.item() for i, score in enumerate(self.scores)},
                'gentrx_scores': {i: score.item() for i, score in enumerate(self.gentrx_scores)},
                'gentrx_enabled': self._gentrx is not None,
                'gentrx_training': self._gentrx.get_training_stats() if self._gentrx is not None else {},
                'gentrx_scores_detailed': self._gentrx.get_scores() if self._gentrx is not None else {},
                'gentrx_config': self._gentrx.get_config() if self._gentrx is not None else {},
                'miner_stats': self.miner_stats,
                'initial_balances': self.initial_balances,
                'initial_balances_published': self.initial_balances_published,
                'recent_trades': recent_trades,
                'recent_miner_trades': recent_miner_trades,
                'open_positions': open_positions,
                'fundamental_price': self.fundamental_price,
                'shared_state_rewarding': self.shared_state_rewarding,
                'current_block': self.current_block,
                'uid': self.uid,
                'reconciliation': dict(getattr(getattr(self, 'engine', None), '_pending_reconciliation', None) or {}),
                # Exchange engine: include chain-level events and delegates for the data service.
                # block_events feeds the L3 stream (StakeAdded/StakeRemoved etc.).
                # delegates feeds the Redis cache used for order-placement delegate defaults.
                'block_events': list((getattr(getattr(self, 'engine', None), '_last_chain_state', None) or {}).get('block_events', [])),
                'delegates':    dict((getattr(getattr(self, 'engine', None), '_last_chain_state', None) or {}).get('delegates', {})),
                'validator_config' : {
                    'observe': bool(getattr(getattr(self.config, 'neuron', None), 'observe', False)),
                    'scoring': {
                        'interval': self.config.scoring.interval,
                        'max_instructions_per_book': self.config.scoring.max_instructions_per_book,
                        'min_delay': self.config.scoring.min_delay,
                        'max_delay': self.config.scoring.max_delay,
                        'min_instruction_delay': self.config.scoring.min_instruction_delay,
                        'max_instruction_delay': self.config.scoring.max_instruction_delay,
                        'kappa_weight': self.config.scoring.kappa.weight,
                        'kappa_lookback': self.config.scoring.kappa.lookback,
                        'kappa_min_lookback': self.config.scoring.kappa.min_lookback,
                        'kappa_tau': self.config.scoring.kappa.tau,
                        'kappa_min_realized_observations': self.config.scoring.kappa.min_realized_observations,
                        'kappa_normalization_min': self.config.scoring.kappa.normalization_min,
                        'kappa_normalization_max': self.config.scoring.kappa.normalization_max,
                        'kappa_pnl_impact': self.config.scoring.kappa.pnl.impact,
                        'pnl_weight': self.config.scoring.pnl.weight,
                        'gentrx_simulation_share': getattr(getattr(self.config.scoring, 'gentrx', None), 'simulation_share', 0.0) or 0.0,
                        'pnl_normalization_min_daily_return': self.config.scoring.pnl.min_daily_return,
                        'pnl_normalization_max_daily_return': self.config.scoring.pnl.max_daily_return,
                        'activity_impact': self.config.scoring.activity.impact,
                        'activity_trade_volume_sampling_interval': self.config.scoring.activity.trade_volume_sampling_interval,
                        'activity_trade_volume_assessment_period': self.config.scoring.activity.trade_volume_assessment_period,
                        'activity_decay_grace_period': self.config.scoring.activity.decay_grace_period,
                        'activity_decay_rate': self.config.scoring.activity.decay_rate,
                        'activity_capital_turnover_cap': self.config.scoring.activity.capital_turnover_cap,
                        'activity_max_volume': self.config.scoring.activity.capital_turnover_cap * self.simulation.miner_wealth,
                    }
                }
            }

            bt.logging.debug(f"Assembled final structure ({time.time()-final_start:.4f}s)")

            return data

        async def _report(self):
            if self._reporting:
                bt.logging.warning(f"Previous reporting still in progress, skipping step {self.step}")
                return

            if self.reporting_process.poll() is not None:
                bt.logging.error(f"Reporting service died with exit code {self.reporting_process.returncode}")
                bt.logging.error("Attempting to restart reporting service...")
                self._start_reporting_service()
                if self.reporting_process.poll() is not None:
                    self.pagerduty_alert("Failed to restart reporting service")
                    return

            self._reporting = True
            reporting_step = self.step
            bt.logging.info(f"Starting Reporting at step {reporting_step}...")
            start = time.time()
            try:
                drain_start = time.time()
                drained = 0
                while True:
                    try:
                        await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                self.reporting_ipc_executor,
                                lambda: self.reporting_response_queue.receive(timeout=0.001)
                            ),
                            timeout=0.05
                        )
                        drained += 1
                    except (posix_ipc.BusyError, asyncio.TimeoutError):
                        break

                if drained > 0:
                    bt.logging.warning(f"Drained {drained} stale messages from reporting response queue ({time.time()-drain_start:.4f}s)")

                prep_start = time.time()
                async with self._reward_lock:
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(
                        self.reporting_ipc_executor,
                        self._prepare_reporting_data
                    )
                prep_time = time.time() - prep_start
                bt.logging.info(f"Prepared reporting data ({prep_time:.4f}s)")

                serialize_start = time.time()
                data_bytes = await asyncio.get_event_loop().run_in_executor(
                    self.reporting_ipc_executor,
                    lambda: msgpack.packb(data, use_bin_type=True)
                )
                serialize_time = time.time() - serialize_start
                data_mb = len(data_bytes) / 1024 / 1024
                bt.logging.info(f"Reporting data: {data_mb:.2f} MB (prep={prep_time:.4f}s, serialize={serialize_time:.4f}s)")

                write_start = time.time()
                write_success = await self._write_ipc_nonblocking(
                    self.reporting_request_mem,
                    self.reporting_request_queue,
                    data_bytes,
                    "Reporting",
                    self.reporting_ipc_executor
                )

                if not write_success:
                    self.pagerduty_alert("Failed to write reporting data")
                    return

                bt.logging.info(f"Wrote reporting data ({time.time()-write_start:.4f}s)")

                await asyncio.get_event_loop().run_in_executor(
                    self.reporting_ipc_executor,
                    self.reporting_request_queue.send,
                    b'publish'
                )

                receive_start = time.time()
                max_wait = 120.0
                elapsed = 0
                message = None
                poll_count = 0
                loop = asyncio.get_event_loop()
                while elapsed < max_wait:
                    poll_count += 1
                    try:
                        # Offload the blocking queue receive to the reporting IPC
                        # executor so the main event loop stays free while the
                        # (separate) reporting process applies its metric updates.
                        # A synchronous receive() on the loop blocks every other
                        # coroutine — including handle_state for incoming simulator
                        # ticks — for the full 30-60s a report takes, inflating
                        # "State update handled" time and causing miner-query
                        # timeouts. The prior form yielded only ~0.001s every 10
                        # polls, leaving the loop ~99% starved for the report's
                        # duration.
                        message, _ = await loop.run_in_executor(
                            self.reporting_ipc_executor,
                            lambda: self.reporting_response_queue.receive(timeout=0.5)
                        )
                        bt.logging.info(f"Received reporting response after {poll_count} polls ({time.time()-receive_start:.4f}s)")
                        break
                    except posix_ipc.BusyError:
                        elapsed = time.time() - receive_start
                        if poll_count % 20 == 0:
                            bt.logging.debug(f"Still polling for reporting response ({elapsed:.1f}s, {poll_count} polls)")
                        continue
                else:
                    self.pagerduty_alert(f"Reporting response timeout after {max_wait}s")
                    return
                if message is None:
                    self.pagerduty_alert(f"Reporting response timeout after {max_wait}s")
                    return

                read_start = time.time()
                self.reporting_response_mem.seek(0)
                size_bytes = self.reporting_response_mem.read(8)
                data_size = struct.unpack('Q', size_bytes)[0]
                result_bytes = self.reporting_response_mem.read(data_size)
                result_mb = len(result_bytes) / 1024 / 1024
                time.time()
                result = msgpack.unpackb(result_bytes, raw=False, strict_map_key=False)
                bt.logging.info(f"Read reporting response data ({time.time()-read_start:.4f}s | {result_mb:.2f}MB)")
                self.initial_balances_published = result['initial_balances_published']
                self.miner_stats = result['miner_stats']

            except Exception as e:
                self.pagerduty_alert(f"Error sending to reporting service: {e}", details={"trace": traceback.format_exc()})
            finally:
                self._reporting = False
                bt.logging.info(f"Completed reporting for step {reporting_step} ({time.time() - start:.4f}s)")

        def report(self) -> None:
            if self.config.reporting.disabled or not self.last_state or self.last_state.timestamp % self.config.scoring.interval != 0:
                return
            if self._reporting:
                bt.logging.warning(f"Skipping reporting at step {self.step} — previous report still running.")
                return
            bt.logging.debug(f"[REPORT] Scheduling from thread: {threading.current_thread().name}")
            self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._report()))

        def _match_trade_fifo(self, uid: int, book_id: int, is_buy: bool, quantity: float,
                            price: float, fee: float, timestamp: int) -> tuple[float, float]:
            """FIFO matching including fee accounting."""
            return match_trade_fifo(self, uid, book_id, is_buy, quantity, price, fee, timestamp)


        def _update_trade_volumes(self, state: MarketSimulationStateUpdate):
            """Updates and maintains all trade volume tracking and position accounting structures."""
            update_trade_volumes(self, state)


        def should_block_queries(self) -> bool:
            """Block queries if reward is lagging."""
            if self._pending_reward_tasks >= 5:
                return True
            return False

        async def _reward(self, state: MarketSimulationStateUpdate):
            """
            Asynchronously perform the full reward computation pipeline.
            """
            with self._rewarding_lock:
                self._pending_reward_tasks += 1

            start_wait = time.time()
            rewarding_step = self.step
            try:
                async with self._reward_lock:
                    waited = time.time() - start_wait
                    if waited > 0:
                        bt.logging.debug(f"Acquired reward lock after waiting {waited:.3f}s")

                    self.shared_state_rewarding = True

                    timestamp = state.timestamp
                    duration = duration_from_timestamp(timestamp)
                    bt.logging.info(f"Starting reward calculation for step {rewarding_step}...")
                    start = time.time()

                    try:
                        self._update_trade_volumes(state)
                        if timestamp % self.config.scoring.interval != 0:
                            bt.logging.info(f"Agent Scores Data Updated for {duration} ({time.time()-start:.4f}s)")
                            return
                        bt.logging.info("Starting reward calculation...")
                        calc_start = time.time()

                        loop = asyncio.get_event_loop()
                        trading_rewards, gentrx_rewards, updated_data, all_uids = await loop.run_in_executor(
                            self.reward_executor,
                            get_rewards,
                            self
                        )

                        bt.logging.info(f"Reward calculation completed ({time.time()-calc_start:.4f}s)")

                        self.kappa_values = updated_data['kappa_values']
                        self.activity_factors = updated_data['activity_factors']
                        self.pnl_factors = updated_data['pnl_factors']

                        bt.logging.debug(
                            f"Agent Rewards Recalculated for {duration} ({time.time()-start:.4f}s):\n"
                            f"trading={trading_rewards}\ngentrx={gentrx_rewards}"
                        )
                        self.update_scores(trading_rewards, all_uids, gentrx_rewards=gentrx_rewards)
                        bt.logging.info(f"Agent Scores Updated for {duration} ({time.time()-start:.4f}s)")
                        self._last_rewarded_sim_timestamp = timestamp

                    except Exception as ex:
                        self.pagerduty_alert(f"Rewarding failed: {ex}", details={"trace": traceback.format_exc()})
                    finally:
                        self.shared_state_rewarding = False
                        bt.logging.debug(f"Completed rewarding (TOTAL {time.time()-start_wait:.4f}s).")
            finally:
                with self._rewarding_lock:
                    self._pending_reward_tasks -= 1

        def reward(self, state : MarketSimulationStateUpdate) -> None:
            """
            Schedule asynchronous reward calculation on the validator's main event loop.
            Offloads work to `_reward()` to ensure that:

            • Reward computation always occurs in the correct asyncio event loop
            • CPU-intensive work does not block the calling thread
            • Reward logic executes with proper async locking

            Args:
                state (MarketSimulationStateUpdate):
                    Simulation state for the current tick, forwarded to `_reward`.
            """
            if getattr(getattr(self.config, 'neuron', None), 'observe', False):
                return
            bt.logging.debug(f"[REWARD] Scheduling from thread: {threading.current_thread().name}")
            bt.logging.debug(f"[REWARD] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
            self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._reward(state)))

        async def handle_state(self, state: NormalizedState, receive_start: float) -> dict:
            """
            Handle a full simulator state update, enrich it with validator data, compute responses,
            update internal validator state, and return instructions back to the simulator.


            This method is the central processing loop for each simulation step. It performs:
            - Periodic validator configuration reloads.
            - Per‑account volume injection.
            - Simulation metadata updates and logging.
            - State forwarding to miners and response aggregation.
            - Reward calculation, scoring, persistence, and metric publication.


            Args:
            state (NormalizedState): Canonical state envelope from the engine.
            receive_start (float): time.time() when the message arrived, used for latency metrics.


            Returns:
            dict: Serialized response batch to be returned to the simulator.
            """
            # Per-tick simulation-specific updates (logDir, simulation_timestamp,
            # simulation_id, periodic compression/update_repo).
            self.engine.on_tick(state)

            start = time.time()
            for uid, accounts in state.accounts.items():
                for book_id in accounts:
                    state.accounts[uid][book_id]['v'] = self.volume_sums.get((uid, book_id), 0.0)
            bt.logging.info(f"Volumes added to state ({time.time()-start:.4f}s).")

            # Update variables
            if not self.start_time:
                self.start_time = time.time()
                self.start_timestamp = state.timestamp

            self.step_rates.append((state.timestamp - (self.last_state.timestamp if self.last_state else self.start_timestamp)) / (time.time() - (self.last_state_time if self.last_state_time else self.start_time)))
            self.last_state = state
            self.step += 1

            # Log received state data
            bt.logging.info(f"STATE UPDATE RECEIVED | VALIDATOR STEP : {self.step} | TIME : {duration_from_timestamp(state.timestamp)} (T={state.timestamp})")
            if self.config.logging.debug or self.config.logging.trace:
                debug_text = ''
                for bookId, book in state.books.items():
                    debug_text += '-' * 50 + "\n"
                    debug_text += f"BOOK {bookId}" + "\n"
                    if book['b'] and book['a']:
                        debug_text += ' | '.join([f"{level['q']:.4f}@{level['p']}" for level in reversed(book['b'][:5])]) + '||' + ' | '.join([f"{level['q']:.4f}@{level['p']}" for level in book['a'][:5]]) + "\n"
                    else:
                        debug_text += "EMPTY" + "\n"
                bt.logging.debug("\n" + debug_text.strip("\n"))

            # Process deregistration notices
            self.process_resets(state)

            # GenTRX: push state to gradient server before the mining query.
            # Offloaded to a thread for the msgpack packing, and time-bounded so
            # a slow/hung gradient server can never stall the validator's hot
            # path (state -> query -> reward -> weights). On timeout the executor
            # thread keeps running harmlessly (push_state is enqueue-only); we
            # just stop awaiting and proceed.
            if self._gentrx is not None:
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self._gentrx.push_state, state
                        ),
                        timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    gtx_log.warning("handle_state push_state exceeded 10s — proceeding without it")
                except Exception as _gex:
                    bt.logging.warning(f"[GTX] handle_state push_state error: {_gex}")

            # Forward state synapse to miners and collect responses
            start = time.time()
            miner_responses = await forward(self, state)
            bt.logging.debug(f"Gathered Response Batch ({time.time()-start}s)")

            # Exchange: send instructions to LOB, execute on-chain, inject trade events
            if self.engine.mode == 'exchange':
                start = time.time()
                _trade_events: list = []  # preserved for nexus book event injection
                try:
                    trade_events = await self.engine.execute(state, miner_responses)
                    if trade_events:
                        _trade_events = trade_events
                        for event in trade_events:
                            _notice = event.to_notice_dict() if hasattr(event, 'to_notice_dict') else event.__dict__
                            for _uid in filter(None, {
                                getattr(event, 'taker_uid', None),
                                getattr(event, 'maker_uid', None),
                            }):
                                state.notices.setdefault(_uid, []).append(_notice)
                        bt.logging.info(
                            f"ExchangeEngine.execute: {len(trade_events)} trade events "
                            f"injected into state ({time.time()-start:.4f}s)"
                        )
                        # Fast-path fill notifications: push directly to data service
                        # without waiting for the 40s ingest pipeline.
                        _fast_url = getattr(getattr(self.config, 'exchange', None), 'data_service_url', '') or ''
                        if _fast_url:
                            asyncio.create_task(_push_fill_notifications(trade_events, _fast_url))
                except Exception as _exc:
                    bt.logging.error(f"ExchangeEngine.execute failed: {_exc}")
                # Refresh state.books/accounts from post-execute LOB state so the
                # ingest payload carries fill events from this block (not just the
                # reconciliation snapshot from receive()).
                try:
                    _cs = getattr(self.engine, '_last_chain_state', None)
                    if _cs is not None and hasattr(self.engine, '_normalize'):
                        _post = self.engine._normalize(_cs)
                        state.books    = _post.books
                        state.accounts = _post.accounts
                except Exception:
                    pass
                # Inject on-chain fill events into state.books['e'] for the nexus
                # event stream. The LOB books from reconciliation only carry order
                # placement/cancel events; actual fill events must come from here.
                try:
                    _fill_ts_ns = int(time.time() * 1e9)
                    for _te in _trade_events:
                        _nid = getattr(_te, 'book_id', None)
                        if _nid is None:
                            continue
                        _ev = {
                            'y':   't',
                            'nid': _nid,
                            'p':   getattr(_te, 'price',     0.0),
                            'q':   getattr(_te, 'quantity',  0.0),
                            's':   getattr(_te, 'side',      0),
                            'Ta':  getattr(_te, 'taker_uid', None),
                            'Ma':  getattr(_te, 'maker_uid', None),
                            't':   getattr(_te, 'timestamp', _fill_ts_ns),
                        }
                        _bk = state.books.get(_nid)
                        if _bk is None:
                            state.books[_nid] = {'b': [], 'a': [], 'e': [_ev]}
                        else:
                            _bk = dict(_bk)
                            _bk['e'] = list(_bk.get('e') or []) + [_ev]
                            state.books[_nid] = _bk
                except Exception:
                    pass
                # Push post-execution state to MVTRX data service on every block
                # Build open-order detail from LOB account state (state.accounts[uid][nid]['o'])
                # which is populated by the C++ engine's packAccounts() on every block.
                _agent_open_orders: dict = {}
                _agent_orders_detail: dict = {}
                _ext_detail = getattr(getattr(self, 'engine', None), '_external_order_detail', {}) or {}
                for _uid_str, _uid_data in (getattr(state, 'accounts', {}) or {}).items():
                    try:
                        _uid = int(_uid_str)
                        if not isinstance(_uid_data, dict):
                            continue
                        for _nid_str, _acct in _uid_data.items():
                            if not isinstance(_acct, dict):
                                continue
                            for _o in (_acct.get('o') or []):
                                if not isinstance(_o, dict):
                                    continue
                                _o_side = int(_o.get('s', _o.get('side', 0)))
                                _o_placed_at = int(_o.get('t') or 0) // 1_000_000
                                if _o_placed_at == 0:
                                    _ext_info = _ext_detail.get((_uid, int(_nid_str), _o_side))
                                    if _ext_info:
                                        _o_placed_at = _ext_info.get("placed_at", 0)
                                _agent_open_orders[_uid] = _agent_open_orders.get(_uid, 0) + 1
                                _agent_orders_detail.setdefault(_uid, []).append({
                                    "order_id":  _o.get('i') or _o.get('id'),
                                    "netuid":    int(_nid_str),
                                    "side":      _o_side,
                                    "price":     float(_o.get('p', _o.get('price', 0))),
                                    "quantity":  float(_o.get('q', _o.get('quantity', 0))),
                                    "placed_at": _o_placed_at,
                                })
                    except Exception:
                        continue
                # Augment from engine's external-order registry (UI-submitted orders)
                # since the C++ LOB doesn't include open orders in account IPC responses.
                try:
                    for (_eod_uid, _eod_nid, _eod_dir), _eod_info in _ext_detail.items():
                        if _eod_uid not in _agent_orders_detail or not any(
                            o.get('netuid') == _eod_nid and o.get('side') == _eod_dir
                            for o in _agent_orders_detail.get(_eod_uid, [])
                        ):
                            _agent_open_orders[_eod_uid] = _agent_open_orders.get(_eod_uid, 0) + 1
                            _agent_orders_detail.setdefault(_eod_uid, []).append({
                                "order_id":  None,
                                "netuid":    _eod_nid,
                                "side":      _eod_dir,
                                "price":     _eod_info.get("price", 0.0),
                                "quantity":  _eod_info.get("quantity", 0.0),
                                "placed_at": _eod_info.get("placed_at", 0),
                            })
                except Exception:
                    pass
                _meta = {}
                try:
                    _meta = {
                        "hotkeys":         [str(hk) for hk in self.metagraph.hotkeys],
                        "coldkeys":        [str(ck) for ck in self.metagraph.coldkeys] if hasattr(self.metagraph, 'coldkeys') else [],
                        "stake":           self.metagraph.stake.tolist(),
                        "emission":        self.metagraph.emission.tolist(),
                        "incentive":       self.metagraph.incentive.tolist(),
                        "validator_trust": self.metagraph.validator_trust.tolist(),
                        "validator_permit": self.metagraph.validator_permit.tolist() if hasattr(self.metagraph, 'validator_permit') else [],
                        "dividends":       self.metagraph.dividends.tolist() if hasattr(self.metagraph, 'dividends') else [],
                        "last_update":     self.metagraph.last_update.tolist() if hasattr(self.metagraph, 'last_update') else [],
                    }
                except Exception:
                    pass
                _ingest_url = getattr(getattr(self.config, 'exchange', None), 'data_service_url', '') or ''
                _ingest_block = getattr(state, 'block', self.current_block)
                _ingest_books = len(getattr(state, 'books', {}) or {})
                bt.logging.info(f"Scheduling ingest push: block={_ingest_block} books={_ingest_books} url={_ingest_url}")
                _sltp_changed = getattr(getattr(self, 'engine', None), '_sltp_changed', False)
                _live_triggers = dict(getattr(getattr(self, 'engine', None), '_live_triggers', {}))
                if hasattr(self.engine, '_sltp_changed'):
                    self.engine._sltp_changed = False
                # Merge ET fill notices from the previous block's _chain_bg into the
                # current ingest push.  _chain_bg populates _pending_et_notices after
                # on-chain execution completes; by the time the next block arrives it is
                # ready.  Pattern mirrors _pending_reconciliation.
                _pending_et = dict(getattr(self.engine, '_pending_et_notices', {}) or {})
                if _pending_et:
                    self.engine._pending_et_notices = {}
                _state_notices = {str(k): list(v) for k, v in (getattr(state, 'notices', None) or {}).items()}
                for _et_uid, _et_evs in _pending_et.items():
                    _key = str(_et_uid)
                    if _key in _state_notices:
                        _state_notices[_key] = _state_notices[_key] + list(_et_evs)
                    else:
                        _state_notices[_key] = list(_et_evs)
                asyncio.create_task(_push_mvtrx({
                    "mode":                "exchange",
                    "network":             getattr(getattr(self.config, 'exchange', None), 'network', '') or '',
                    "timestamp":           int(time.time() * 1e9),
                    "block":               _ingest_block,
                    "books":               getattr(state, 'books', {}) or {},
                    "accounts":            getattr(state, 'accounts', {}) or {},
                    "pools":               getattr(state, 'pools', None),
                    "block_events":        list((getattr(self.engine, '_last_chain_state', None) or {}).get('block_events', [])),
                    "delegates":           dict((getattr(self.engine, '_last_chain_state', None) or {}).get('delegates', {})),
                    "chain_balances":      dict((getattr(self.engine, '_last_chain_state', None) or {}).get('balances', {})),
                    "offex_balances":      dict((getattr(self.engine, '_last_chain_state', None) or {}).get('offex_balances', {})),
                    "reconciliation":      dict(getattr(self.engine, '_pending_reconciliation', None) or {}),
                    "notices":             _state_notices,
                    "agent_open_orders":   _agent_open_orders,
                    "agent_orders_detail": _agent_orders_detail,
                    "metagraph":           _meta,
                    "validator_uid":       self.uid,
                    "benchmark_agents":    [{"uid": _ba["uid"], "coldkey": _ba.get("coldkey", ""), "hotkey": _ba.get("hotkey", ""), "name": _ba.get("name", "")} for _ba in getattr(self, 'benchmark_agents', [])],
                    "sltp_triggers":       _live_triggers,
                }, url=_ingest_url))
                # Yield once so the HTTP POST starts sending while the remaining
                # synchronous work (maintain/reward/save scheduling) runs.
                await asyncio.sleep(0)
                response = {"responses": []}
            else:
                start = time.time()
                response = SimulatorResponseBatch(miner_responses)
                response = response.serialize()
                bt.logging.debug(f"Serialized Response Batch ({time.time()-start}s)")
                # Inject UI-submitted external orders
                _ext = getattr(self.engine, '_external_instructions', [])
                if _ext:
                    response['responses'].extend(_ext)
                    self.engine._external_instructions = []
                    bt.logging.info(f"Injected {len(_ext)} external order(s) into simulation response")
                # Push simulation state to MVTRX data service each tick
                # ── Scan notices for fills (ET) and rejections (ERDPOL/ERDPOM) ──
                _sim_fills: list = []
                _sim_rejects: list = []
                _seen_trade_ids: set = set()
                for _uid_str, _evs in (getattr(state, 'notices', {}) or {}).items():
                    for _ev in (_evs or []):
                        if not isinstance(_ev, dict):
                            continue
                        _ev_type = _ev.get("y") or _ev.get("type")
                        if _ev_type in ("ET", "EVENT_TRADE"):
                            _tid = _ev.get("i") if _ev.get("i") is not None else _ev.get("tradeId")
                            if _tid is None:
                                continue
                            _book_id = _ev.get("b") if _ev.get("b") is not None else _ev.get("bookId")
                            _price   = float(_ev.get("p") or _ev.get("price") or 0)
                            _qty     = float(_ev.get("q") or _ev.get("quantity") or 0)
                            _side    = int(_ev.get("s") if _ev.get("s") is not None else _ev.get("side", 0))
                            _taker   = _ev.get("Ta") if _ev.get("Ta") is not None else _ev.get("takerAgentId")
                            _maker   = _ev.get("Ma") if _ev.get("Ma") is not None else _ev.get("makerAgentId")
                            _ti      = _ev.get("Ti")   # taker order ID
                            _mi      = _ev.get("Mi")   # maker order ID
                            _cr_raw  = _ev.get("cr", 0) or 0
                            _cr      = "SL" if _cr_raw == 1 else ("TP" if _cr_raw == 2 else None)
                            _toi     = _ev.get("Toi") or None  # originating order ID (SL/TP only)
                            for _agent_uid, _is_taker in ((_taker, True), (_maker, False)):
                                if _agent_uid is None:
                                    continue
                                _key = (_tid, int(_agent_uid))
                                if _key in _seen_trade_ids:
                                    continue
                                _seen_trade_ids.add(_key)
                                _dir  = _side if _is_taker else (1 - _side)
                                _oid  = _ti if _is_taker else _mi
                                _fill_cr  = _cr if _is_taker else None
                                _fill_toi = _toi if (_is_taker and _cr) else None
                                _sim_fills.append({
                                    "uid":              int(_agent_uid),
                                    "netuid":           int(_book_id) if _book_id is not None else 0,
                                    "price":            _price,
                                    "qty":              _qty,
                                    "alpha_volume":     _qty,
                                    "tao_volume":       round(_price * _qty, 9),
                                    "direction":        _dir,
                                    "side":             "buy" if _dir == 0 else "sell",
                                    "role":             "taker" if _is_taker else "maker",
                                    "trade_id":         _tid,
                                    "order_id":         int(_oid) if _oid is not None else None,
                                    "close_reason":     _fill_cr,
                                    "linked_order_id":  int(_fill_toi) if _fill_toi is not None else None,
                                    "maker_uid":        int(_maker) if _maker is not None else None,
                                    "is_partial":       False,
                                    "timestamp":        state.timestamp,
                                })
                        elif _ev_type in ("ERDPOL", "ERDPOM"):
                            _agent_uid = _ev.get("a") if _ev.get("a") is not None else _ev.get("agentId")
                            if _agent_uid is None:
                                continue
                            _seen_rej = (_ev_type, int(_agent_uid))
                            if _seen_rej in _seen_trade_ids:
                                continue
                            _seen_trade_ids.add(_seen_rej)
                            _sim_rejects.append({
                                "uid":    int(_agent_uid),
                                "netuid": int(_ev.get("b") if _ev.get("b") is not None else (_ev.get("bookId") or 0)),
                                "reason": _ev.get("m") or _ev.get("message") or "rejected",
                                "side":   int(_ev.get("s") if _ev.get("s") is not None else _ev.get("side", 0)),
                                "qty":    float(_ev.get("q") or _ev.get("quantity") or 0),
                                "price":  float(_ev.get("p") or _ev.get("price") or 0),
                            })
                # ── Extract open orders from accounts ─────────────────────────────
                _sim_open_orders: dict = {}
                _sim_orders_detail: dict = {}
                for _uid_str, _uid_data in (getattr(state, 'accounts', {}) or {}).items():
                    try:
                        _uid = int(_uid_str)
                        if not isinstance(_uid_data, dict):
                            continue
                        for _nid_str, _acct in _uid_data.items():
                            if not isinstance(_acct, dict):
                                continue
                            for _o in (_acct.get('o') or []):
                                if not isinstance(_o, dict):
                                    continue
                                _sim_open_orders[_uid] = _sim_open_orders.get(_uid, 0) + 1
                                _sim_orders_detail.setdefault(_uid, []).append({
                                    "order_id":  _o.get('i') or _o.get('id'),
                                    "netuid":    int(_nid_str),
                                    "side":      int(_o.get('s', _o.get('side', 0))),
                                    "price":     float(_o.get('p', _o.get('price', 0))),
                                    "quantity":  float(_o.get('q', _o.get('quantity', 0))),
                                    "placed_at": int(_o.get('t') or 0) // 1_000_000,
                                })
                    except Exception:
                        continue
                # Build per-agent volume / pnl / score dicts for the data service.
                # volume_sums / maker_volume_sums / taker_volume_sums:
                #   defaultdict(uid → defaultdict(book_id → float)) — sum across books.
                # realized_pnl_history: defaultdict(uid → defaultdict(ts → {book_id: pnl}))
                def _sv(d):
                    return {str(uid): sum(float(v) for v in list(bks.values()))
                            for uid, bks in list(d.items()) if bks}
                _vs  = _sv(getattr(self, 'volume_sums', {}))
                _mvs = _sv(getattr(self, 'maker_volume_sums', {}))
                _tvs = _sv(getattr(self, 'taker_volume_sums', {}))
                _pnl = {str(uid): sum(p for ts_d in list(hist.values()) for p in list(ts_d.values()))
                        for uid, hist in list((getattr(self, 'realized_pnl_history', {}) or {}).items()) if hist}
                _sc  = getattr(self, 'scores', None)
                _sc_dict = ({str(i): float(_sc[i]) for i in range(len(_sc))}
                            if _sc is not None else {})
                _kappa_raw   = {}
                _kappa_score = {}
                _kappa_books = {}        # per-book raw kappa: {uid: {book_id: float}}
                _kappa_books_w = {}      # per-book weighted:  {uid: {book_id: float}}
                for _kuid, _kv in getattr(self, 'kappa_values', {}).items():
                    if _kv and isinstance(_kv, dict):
                        if _kv.get('total') is not None:
                            _kappa_raw[str(_kuid)] = float(_kv['total'])
                        if _kv.get('normalized_total') is not None:
                            _kappa_score[str(_kuid)] = float(_kv['normalized_total'])
                        _bks = {str(bid): float(v) for bid, v in (_kv.get('books') or {}).items() if v is not None}
                        if _bks:
                            _kappa_books[str(_kuid)] = _bks
                        _bw = {str(bid): float(v) for bid, v in (_kv.get('books_weighted') or {}).items() if v is not None}
                        if _bw:
                            _kappa_books_w[str(_kuid)] = _bw
                # Augment pools with per-book 24H volume (sum across all agents)
                _book_vol: dict = {}
                for _uid_bk, _bk_dict in getattr(self, 'volume_sums', {}).items():
                    for _bid, _bvol in _bk_dict.items():
                        _book_vol[int(_bid)] = _book_vol.get(int(_bid), 0.0) + float(_bvol)
                _sim_pools: dict = dict(state.pools or {})
                for _bid, _bvol in _book_vol.items():
                    _bk = _sim_pools.get(_bid) or _sim_pools.get(str(_bid))
                    if _bk is not None:
                        _bk['volume_24h'] = round(_bvol, 4)
                _sim_live_triggers = dict(getattr(getattr(self, 'engine', None), '_live_triggers', {}))
                if hasattr(self.engine, '_sltp_changed'):
                    self.engine._sltp_changed = False
                asyncio.create_task(_push_mvtrx({
                    "mode":                "simulation",
                    "simulation_id":       getattr(self.simulation, 'simulation_id', None),
                    "timestamp":           state.timestamp,
                    "block":               state.block,
                    "books":               state.books or {},
                    "accounts":            state.accounts or {},
                    "pools":               _sim_pools,
                    "benchmark_agents":    [{"uid": _ba["uid"], "coldkey": _ba.get("coldkey", ""), "hotkey": _ba.get("hotkey", ""), "name": _ba.get("name", "")} for _ba in getattr(self, 'benchmark_agents', [])],
                    "reconciliation":      {"fills": _sim_fills, "rejections": _sim_rejects},
                    "notices":             {str(k): list(v) for k, v in (state.notices or {}).items()},
                    "agent_open_orders":   _sim_open_orders,
                    "agent_orders_detail": _sim_orders_detail,
                    "validator_uid":       getattr(self, 'uid', None),
                    "chain_block":         getattr(self, 'current_block', None),
                    "metagraph":           (lambda _m: {
                        "hotkeys":         [str(hk) for hk in _m.hotkeys],
                        "coldkeys":        [str(ck) for ck in _m.coldkeys] if hasattr(_m, 'coldkeys') else [],
                        "stake":           _m.stake.tolist(),
                        "emission":        _m.emission.tolist(),
                        "incentive":       _m.incentive.tolist(),
                        "validator_trust": _m.validator_trust.tolist(),
                        "validator_permit":[bool(v) for v in _m.validator_permit] if hasattr(_m, 'validator_permit') else [],
                        "dividends":       _m.dividends.tolist() if hasattr(_m, 'dividends') else [],
                        "last_update":     _m.last_update.tolist() if hasattr(_m, 'last_update') else [],
                    } if _m is not None else {})(getattr(self, 'metagraph', None)),
                    "agent_scores":        _sc_dict,
                    "agent_kappa":         _kappa_raw,
                    "agent_kappa_score":   _kappa_score,
                    "agent_kappa_books":   _kappa_books,
                    "agent_kappa_books_w": _kappa_books_w,
                    "agent_volume":        _vs,
                    "agent_maker_volume":  _mvs,
                    "agent_taker_volume":  _tvs,
                    "agent_pnl":           _pnl,
                    # NB: dict(...) snapshots the OUTER mapping before iteration. These
                    # dicts (realized_pnl_history, volume_sums, fee_sums, roundtrip_volume_sums,
                    # activity_factors, kappa_values) get mutated from concurrent async tasks
                    # (state-update handler at l.1302, trade.py:417-449, etc.). Without the
                    # snapshot, building this push payload races with key adds and raises
                    # RuntimeError: dictionary changed size during iteration. Inner dicts (bks)
                    # are smaller and only mutated within a single state-update cycle, so the
                    # race window is narrower — fixed here only at the outer level.
                    "agent_pnl_book":      {str(uid): {str(bid): round(float(v), 6)
                                            for bid, v in (
                                                {bid: sum(ts_d.get(bid, 0) for ts_d in hist.values())
                                                 for bid in set(bid for ts_d in hist.values() for bid in ts_d)}
                                            ).items() if v != 0}
                                           for uid, hist in dict(getattr(self, 'realized_pnl_history', {})).items() if hist},
                    "agent_volume_book":   {str(uid): {str(bid): round(float(v), 4)
                                            for bid, v in bks.items() if v}
                                           for uid, bks in dict(getattr(self, 'volume_sums', {})).items() if bks},
                    "agent_fee_book":      {str(uid): {str(bid): round(float(v), 6)
                                            for bid, v in bks.items() if v != 0}
                                           for uid, bks in dict(getattr(self, 'fee_sums', {})).items() if bks},
                    "agent_roundtrip_volume": {str(uid): round(float(sum(bks.values())), 4)
                                               for uid, bks in dict(getattr(self, 'roundtrip_volume_sums', {})).items() if bks},
                    "agent_activity_factor":   {str(uid): round(float(sum(bks.values()) / len(bks)), 4)
                                                for uid, bks in dict(getattr(self, 'activity_factors', {})).items() if bks},
                    "agent_median_kappa":      {str(uid): round(float(kv.get('activity_weighted_normalized_median') or 0), 6)
                                               for uid, kv in dict(getattr(self, 'kappa_values', {})).items() if kv},
                    "fee_policy":              ({"fee_type": self.simulation.fee_policy.fee_type,
                                                **{k: float(v) for k, v in self.simulation.fee_policy.params.items()
                                                   if k in ("targetMTR", "makerFee", "takerFee", "maxMakerRate", "maxTakerRate")}}
                                               if getattr(self.simulation, 'fee_policy', None) else None),
                    "exchange_constraints":    ({"min_order_size":     float(getattr(self.simulation, 'min_order_size', 0.0)),
                                                 "max_open_orders":    int(getattr(self.simulation, 'max_open_orders', 0) or 0),
                                                 "max_leverage":       float(getattr(self.simulation, 'max_leverage', 0)),
                                                 "max_loan":           float(getattr(self.simulation, 'max_loan', 0)),
                                                 "maintenance_margin": float(getattr(self.simulation, 'maintenance_margin', 0)),
                                                 "price_decimals":     int(getattr(self.simulation, 'priceDecimals', 4)),
                                                 "volume_decimals":    int(getattr(self.simulation, 'volumeDecimals', 4)),
                                                 "init_price":         float(getattr(self.simulation, 'init_price', 0)),
                                                 "time_unit":          str(getattr(self.simulation, 'time_unit', 'ns')),
                                                 "grace_period":       int(getattr(self.simulation, 'grace_period', 0)),
                                                }
                                               if getattr(self, 'simulation', None) else None),
                    "sltp_triggers": _sim_live_triggers,
                }, url=getattr(getattr(self.config, 'simulation', None), 'data_service_url', '') or getattr(getattr(self.config, 'exchange', None), 'data_service_url', '')))

            # GenTRX: poll for round advance and deliver assignments after the
            # mining query completes. Round-check stays triggered per state
            # update (no wall-clock polling) but runs as a BACKGROUND task so
            # the grad-server HTTP (data-status / push_round / health check)
            # and the miner delivery never block the round. Exclusion with the
            # state-update query is enforced by miner_net_lock (held by
            # forward() for its query window, and by _deliver_gentrx_assignments
            # around the one atomic deliver to ALL miners) — so queries and
            # deliveries still never overlap, on the IPC channel or the wire.
            if self._gentrx is not None:
                if self._gentrx_task is not None and not self._gentrx_task.done():
                    # Single-flight: one deliver (~4-6s) per ~5-min round means a
                    # previous task still running at the NEXT state update is a
                    # sign something is genuinely stuck — surface it loudly.
                    gtx_log.warning(
                        "previous poll_and_deliver still running at next state "
                        "update — skipping this cycle (investigate if recurring)")
                else:
                    async def _run_gtx_background():
                        try:
                            # Generous hard bound purely as a stuck-task backstop;
                            # nothing in the round awaits this.
                            await asyncio.wait_for(
                                self._gentrx.poll_and_deliver(), timeout=120.0)
                        except asyncio.TimeoutError:
                            gtx_log.warning(
                                "background poll_and_deliver exceeded 120s — abandoned")
                        except Exception as _gex:
                            bt.logging.warning(
                                f"[GTX] background poll_and_deliver error: {_gex}")
                    self._gentrx_task = asyncio.create_task(_run_gtx_background())
            # Log response data, start state serialization and reporting threads, and return miner instructions to the simulator
            if len(response['responses']) > 0:
                bt.logging.trace(f"RESPONSE : {response}")
            bt.logging.info(f"RATE : {(self.step_rates[-1] if self.step_rates != [] else 0) / 1e9:.2f} STEPS/s | AVG : {(sum(self.step_rates) / len(self.step_rates) / 1e9 if self.step_rates != [] else 0):.2f}  STEPS/s")
            self.step_rates = self.step_rates[-10000:]
            self.last_state_time = time.time()

            # Calculate latest rewards, update miner scores, save state and publish metrics
            self.maintain()
            self.reward(state)
            self.save_state()
            self.report()
            bt.logging.info(f"State update handled ({time.time()-receive_start}s)")

            return response

        async def _listen(self):
            """
            Thin event loop: delegates IPC receive/respond to the engine.
            """
            # Startup push — fetch full exchange state (chain + LOB) and push to
            # the data service immediately, before waiting for the first block.
            if self.engine.mode == 'exchange':
                try:
                    bt.logging.info("MVTRX startup push: fetching chain + LOB state...")
                    _t0 = time.time()
                    _cs = await self.engine._fetch_chain_state()
                    if _cs:
                        # Mirror receive()'s auto-discovery so _normalize() has book_ids
                        if not self.engine._book_ids and _cs.get('pools'):
                            self.engine._book_ids = sorted(_cs['pools'].keys())
                            self.engine._update_exchange_config()
                        # Query LOB for current books/accounts/open orders (empty reconciliation
                        # payload triggers a state-only round-trip without on-chain execution).
                        loop = asyncio.get_event_loop()
                        _lob = await loop.run_in_executor(
                            None,
                            lambda: self.engine._send_to_lob(_cs, reconciliation={})
                        )
                        if _lob is not None:
                            _exc_state = _lob.get('state', _lob)
                            if _exc_state:
                                self.engine._last_exchange_state = {
                                    **_exc_state,
                                    'books':    _lob.get('books', {}),
                                    'accounts': _lob.get('accounts', {}),
                                }
                        _ss = self.engine._normalize(_cs)
                        _sm: dict = {}
                        try:
                            _sm = {
                                "hotkeys":         [str(hk) for hk in self.metagraph.hotkeys],
                                "coldkeys":        [str(ck) for ck in self.metagraph.coldkeys] if hasattr(self.metagraph, 'coldkeys') else [],
                                "stake":           self.metagraph.stake.tolist(),
                                "emission":        self.metagraph.emission.tolist(),
                                "incentive":       self.metagraph.incentive.tolist(),
                                "validator_trust": self.metagraph.validator_trust.tolist(),
                                "validator_permit": self.metagraph.validator_permit.tolist() if hasattr(self.metagraph, 'validator_permit') else [],
                            }
                        except Exception:
                            pass
                        # Build open-order detail from the LOB accounts in the startup state
                        _soo: dict = {}
                        _sod: dict = {}
                        for _uid_str, _uid_data in (getattr(_ss, 'accounts', {}) or {}).items():
                            try:
                                _uid = int(_uid_str)
                                if not isinstance(_uid_data, dict):
                                    continue
                                for _nid_str, _acct in _uid_data.items():
                                    if not isinstance(_acct, dict):
                                        continue
                                    for _o in (_acct.get('o') or []):
                                        if not isinstance(_o, dict):
                                            continue
                                        _soo[_uid] = _soo.get(_uid, 0) + 1
                                        _sod.setdefault(_uid, []).append({
                                            "order_id":  _o.get('i') or _o.get('id'),
                                            "netuid":    int(_nid_str),
                                            "side":      int(_o.get('s', _o.get('side', 0))),
                                            "price":     float(_o.get('p', _o.get('price', 0))),
                                            "quantity":  float(_o.get('q', _o.get('quantity', 0))),
                                            "placed_at": int(_o.get('t') or 0) // 1_000_000,
                                        })
                            except Exception:
                                continue
                        bt.logging.info(
                            f"MVTRX startup push: block={_cs.get('block')} "
                            f"books={len(getattr(_ss,'books',{}))} "
                            f"accounts={len(getattr(_ss,'accounts',{}))} "
                            f"pools={len(getattr(_ss,'pools',{}) or {})} "
                            f"open_orders={sum(_soo.values())} "
                            f"({time.time()-_t0:.2f}s)"
                        )
                        asyncio.create_task(_push_mvtrx({
                            "mode":                "exchange",
                            "network":             getattr(getattr(self.config, 'exchange', None), 'network', '') or '',
                            "timestamp":           int(time.time() * 1e9),
                            "block":               _cs.get('block', self.current_block),
                            "books":               getattr(_ss, 'books', {}) or {},
                            "accounts":            getattr(_ss, 'accounts', {}) or {},
                            "pools":               getattr(_ss, 'pools', None),
                            "reconciliation":      {},
                            "agent_open_orders":   _soo,
                            "agent_orders_detail": _sod,
                            "metagraph":           _sm,
                            "validator_uid":       self.uid,
                            "benchmark_agents":    [{"uid": _ba["uid"], "coldkey": _ba.get("coldkey", ""), "hotkey": _ba.get("hotkey", ""), "name": _ba.get("name", "")} for _ba in getattr(self, 'benchmark_agents', [])],
                        }, url=getattr(getattr(self.config, 'exchange', None), 'data_service_url', '')))
                except Exception as _exc:
                    bt.logging.warning(f"MVTRX startup push failed: {_exc}")
            bt.logging.info("Listener loop starting — awaiting engine.receive()...")

            # Liveness probe: if these stop, the event loop is blocked/starved
            # (sync native call on the loop, GIL monopolised, etc.) — if they
            # continue while receive() makes no progress, a specific await hangs.
            async def _listen_heartbeat():
                n = 0
                while True:
                    await asyncio.sleep(30)
                    n += 1
                    bt.logging.debug(f"[hb] listener event loop alive #{n}")
                    if n % 2 == 0:
                        # Dump every task's full await chain (get_stack only reports
                        # one frame for suspended coroutines) down to the awaited
                        # object — pinpoints exactly which await is parked.
                        for t in asyncio.all_tasks():
                            if t is asyncio.current_task():
                                continue
                            try:
                                coro = t.get_coro()
                                chain = []
                                while coro is not None and len(chain) < 12:
                                    fr = getattr(coro, 'cr_frame', None) or getattr(coro, 'gi_frame', None)
                                    if fr is not None:
                                        chain.append(
                                            f"{fr.f_code.co_name}@{fr.f_code.co_filename.rsplit('/', 1)[-1]}:{fr.f_lineno}"
                                        )
                                    nxt = getattr(coro, 'cr_await', None)
                                    if nxt is None:
                                        nxt = getattr(coro, 'gi_yieldfrom', None)
                                    if nxt is None or nxt is coro:
                                        break
                                    if not hasattr(nxt, 'cr_frame') and not hasattr(nxt, 'gi_frame'):
                                        chain.append(f"awaiting {repr(nxt)[:160]}")
                                        break
                                    coro = nxt
                                desc = ' <- '.join(chain)
                            except Exception as _e:
                                desc = f"<chain error: {_e}>"
                            bt.logging.debug(f"[hb] task {t.get_name()}: {desc or '<no frames>'}")
            _hb_task = asyncio.ensure_future(_listen_heartbeat())  # noqa: F841 — keep ref

            try:
                while True:
                    response = {"responses": []}
                    raw_message = None
                    try:
                        raw_message, normalized_state, receive_start = await self.engine.receive()
                        if normalized_state is not None:
                            response = await self.handle_state(normalized_state, receive_start)
                    except Exception as ex:
                        traceback.print_exc()
                        self.pagerduty_alert(
                            f"Exception in listener loop: {ex}",
                            details={"trace": traceback.format_exc()}
                        )
                    finally:
                        self.engine.respond(raw_message, response)
            finally:
                pass

        def listen(self):
            """
            Synchronous wrapper for the asynchronous `_listen` method.
            """
            try:
                os.nice(-10)
            except PermissionError:
                bt.logging.warning("Cannot set process priority (need sudo for negative nice values)")
            try:
                asyncio.run(self._listen())
            except KeyboardInterrupt:
                print("Listening stopped by user.")

        async def orderbook(self, request : Request) -> dict:
            """
            HTTP route endpoint that receives a complete simulator state update over HTTP,
            parses it, and forwards it to `handle_state`.


            This is the HTTP equivalent of the IPC listener used when running a
            distributed or containerized simulator. It performs:
            - Streaming request‑body read.
            - Basic JSON‑structure validation.
            - Construction of a Ypy‑backed state object.
            - Conversion into a typed MarketSimulationStateUpdate model.
            - Delegation to the main validator processing pipeline.


            Args:
            request (Request): Incoming HTTP request containing a JSON‑encoded simulation state update.


            Returns:
            dict: Serialized simulation response batch.
            """
            bt.logging.info("Received state update from simulator.")
            global_start = time.time()
            start = time.time()
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
            bt.logging.info(f"Retrieved request body ({time.time()-start:.4f}s).")
            if not body:
                return {}
            if body[-3:].decode() != "]}}":
                raise Exception("Incomplete JSON!")
            message = YpyObject(body, 1)
            bt.logging.info(f"Constructed YpyObject ({time.time()-start:.4f}s).")
            state = MarketSimulationStateUpdate.from_ypy(message)
            bt.logging.info(f"Synapse populated ({time.time()-start:.4f}s).")
            del body

            normalized = self.engine._normalize(state)
            normalized.logDir = message['logDir']
            response = await self.handle_state(normalized, global_start)

            bt.logging.info(f"State update processed ({time.time()-global_start}s)")
            return response

        async def account(self, request : Request) -> None:
            """
            HTTP route endpoint for receiving event‑level notifications from the simulator
            (e.g., simulation start, simulation end, error reports, market notices).


            Responsibilities:
            - Immediately forward simulation‑start events to miners.
            - Handle simulation‑end markers.
            - Record and persist error‑report batches.
            - Forward all other event notifications to miners.
            - Trigger alerting for msgpack or simulation integrity errors.

            Args:
            request (Request): HTTP request containing a batch of simulator event messages.

            Returns:
            None | dict: `{"continue": True/False}` when error‑report limits are reached.
            Otherwise returns `None`.
            """
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
            batch = msgspec.json.decode(body)
            bt.logging.info(f"NOTICE : {batch}")
            notices = []
            ended = False
            for message in batch['messages']:
                if message['type'] == 'EVENT_SIMULATION_START':
                    self.engine.on_start(message['timestamp'], FinanceEventNotification.from_json(message).event)
                    continue
                elif message['type'] == 'EVENT_SIMULATION_END':
                    ended = True
                elif message['type'] == 'RESPONSES_ERROR_REPORT':
                    dump_file = self.config.neuron.full_path + f"/{self.last_state.config.simulation_id}.{message['timestamp']}.responses.json"
                    with open(dump_file, "w") as f:
                        json.dump(self.last_response, f, indent=4)
                    error_file = self.config.neuron.full_path + f"/{self.last_state.config.simulation_id}.{message['timestamp']}.error.json"
                    with open(error_file, "w") as f:
                        json.dump(message, f, indent=4)
                    self.msgpack_error_counter += len(message) - 3
                    if self.msgpack_error_counter < 10:
                        self.pagerduty_alert(f"{self.msgpack_error_counter} msgpack deserialization errors encountered in simulator - continuing.", details=message)
                        return { "continue": True }
                    else:
                        self.pagerduty_alert(f"{self.msgpack_error_counter} msgpack deserialization errors encountered in simulator - terminating simulation.", details=message)
                        return { "continue": False }
                notice = FinanceEventNotification.from_json(message)
                if not notice:
                    bt.logging.error(f"Unrecognized notification : {message}")
                else:
                    notices.append(notice)
            await notify(self, notices)
            if ended:
                self.engine.on_end()

        async def sltp_records(self, request: Request):
            """Delegate to the active engine; return [] when no extension installed."""
            if hasattr(self, 'engine') and hasattr(self.engine, 'sltp_records'):
                try:
                    return await self.engine.sltp_records(request)
                except Exception:
                    pass
            return []

        async def sltp_levels(self):
            """Return the engine's trigger snapshot if available, else {}."""
            if hasattr(self, 'engine') and hasattr(self.engine, '_live_triggers'):
                return self.engine._live_triggers
            return {}

        def cleanup_ipc(self):
            """
            Shuts down the query service and releases all POSIX IPC resources.
            """
            cleanup_ipc(self)

        def cleanup_executors(self):
            """
            Shuts down thread and process executors used by the validator.
            """
            cleanup_executors(self)

        def cleanup_event_loop(self):
            """
            Gracefully shuts down the main event loop and any pending tasks.
            """
            cleanup_event_loop(self)

        def cleanup(self):
            """
            Performs full resource cleanup for the validator during shutdown.
            """
            cleanup(self)

    from taos.im.validator.trade import match_trade_fifo, update_trade_volumes
    from taos.im.validator.cleanup import (
        cleanup_ipc, cleanup_executors, cleanup_event_loop, cleanup
    )
    from taos.im.validator.persistence import (
        load_state, build_validator_state,
        snapshot_inventory_history, snapshot_realized_pnl_history,
        snapshot_2_level_dict, snapshot_volume_sums, snapshot_trade_volumes,
        snapshot_roundtrip_volumes, snapshot_open_positions,
        construct_save_data_sync, defragment_histories, schedule_save,
        save_state_sync, migrate_sampling_interval,
    )

if __name__ == "__main__":
    from taos.im.validator.update import check_repo, update_validator, check_simulator, rebuild_simulator, restart_simulator
    from taos.im.validator.forward import forward, notify, deliver_gentrx
    from taos.im.validator.reward import get_rewards

    if float(platform.freedesktop_os_release()['VERSION_ID']) < 22.04:
        raise Exception("taos validator requires Ubuntu >= 22.04!")

    # Apply logging config before any bt.logging calls — bt.logging starts in
    # Default/WARNING state at import time and never auto-applies CLI flags.
    # Use direct set_* calls (same pattern as BaseNeuron.__init__ and subprocesses)
    # because bt.logging(config=...) can silently fall through to enable_default()
    # if _extract_logging_config returns the wrong object.
    _cfg_lc = getattr(Validator.config(), 'logging', None)
    if getattr(_cfg_lc, 'trace', False):
        bt.logging.set_trace()
    elif getattr(_cfg_lc, 'debug', False):
        bt.logging.set_debug()
    else:
        bt.logging.set_info()
    bt.logging.info("Initializing validator...")
    app = FastAPI()
    validator = Validator()
    try:
        app.include_router(validator.router)

        bt.logging.info("Starting background threads...")
        threads = []
        engine_mode = getattr(validator.config, 'engine', 'simulation')

        # For exchange mode the block-monitoring loop drives everything;
        # for simulation mode uvicorn drives everything (simulator pushes state).
        # Build the background thread list accordingly.
        background_targets = [('Monitor', validator.monitor)]
        try:
            from taos.im.validator.exetrx import setup_attrs as _setup_sltp_attrs
            _setup_sltp_attrs(validator, engine_mode)
        except ImportError:
            validator.sltp_enabled = False
        # Observe-mode validators don't run agents and never launch the SL/TP
        # service, so don't monitor (and pointlessly try to PM2-restart) a
        # process that doesn't exist — it just spams ERROR logs.
        if bool(getattr(getattr(validator.config, 'neuron', None), 'observe', False)):
            validator.sltp_enabled = False
        if getattr(validator, 'sltp_enabled', False):
            background_targets.append(('SLTPMonitor', validator.monitor_sltp))
        if engine_mode == 'exchange':
            # Absorb POST /MarketSimulationStateUpdate from simulation validators that
            # query this axon. Return 404 (not 200) so the dendrite skips synapse
            # body-parsing and won't raise a ValidationError for missing fields.
            from fastapi.responses import JSONResponse as _JSONResponse
            @app.post("/MarketSimulationStateUpdate")
            async def _sim_stub():
                return _JSONResponse(status_code=404, content={"message": "exchange mode"})

            import logging as _logging
            class _SuppressSimStateLog(_logging.Filter):
                def filter(self, record):
                    return "MarketSimulationStateUpdate" not in record.getMessage()
            _logging.getLogger("uvicorn.access").addFilter(_SuppressSimStateLog())

            # Listen runs in the main thread below; uvicorn is a background thread.
            background_targets.append(('API', lambda: uvicorn.run(app, host="0.0.0.0", port=validator.config.port)))
        else:
            # Simulation mode: Listen is a background thread; uvicorn runs below.
            background_targets.append(('Listen', validator.listen))

        for name, target in background_targets:
            try:
                bt.logging.info(f"Starting {name} thread...")
                thread = Thread(target=target, daemon=True, name=name)
                thread.start()
                threads.append(thread)
            except Exception as ex:
                validator.pagerduty_alert(f"Exception starting {name} thread: {ex}")
                raise

        time.sleep(1)
        for thread in threads:
            if not thread.is_alive():
                validator.pagerduty_alert(f"Failed to start {thread.name} thread!")
                raise RuntimeError(f"Thread '{thread.name}' failed to start")

        bt.logging.info("All threads running. Starting FastAPI server and main event loop...")

        def run_main_loop():
            """Run the pre-created main event loop."""
            async def keep_alive():
                bt.logging.info("Main event loop started for background tasks")
                bt.logging.debug(f"[MAINLOOP] Thread: {threading.current_thread().name}")
                bt.logging.debug(f"[MAINLOOP] Loop: {id(validator.main_loop)}")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    bt.logging.info("Main event loop stopping...")
            loop = validator.main_loop
            asyncio.set_event_loop(loop)
            validator._main_loop_ready.set()
            bt.logging.debug(f"[MAINLOOP] Running loop: {id(loop)}")
            try:
                loop.run_until_complete(keep_alive())
            finally:
                loop.close()

        main_loop_thread = Thread(target=run_main_loop, daemon=True, name='main')
        main_loop_thread.start()
        threads.append(main_loop_thread)
        time.sleep(0.5)

        if engine_mode == 'exchange':
            bt.logging.info("Exchange mode: running block monitoring loop in main thread...")
            validator.listen()   # blocks; errors are visible and process dies on crash
        else:
            bt.logging.info(f"Starting FastAPI server on port {validator.config.port}...")
            uvicorn.run(app, host="0.0.0.0", port=validator.config.port)
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt received")
    except Exception as ex:
        bt.logging.error(f"Fatal error: {ex}")
        bt.logging.debug(traceback.format_exc())
        sys.exit(1)