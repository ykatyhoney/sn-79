# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""GenTRX startup isolation — `_restore_written_parquets` wall-clock budget.

Background: on sim-local-dev with a stale 12 GB minio data-dir, boto3's
paginated `ListObjectsV2` against the validator's bucket hangs in
`socket.readinto`; the Config-level `read_timeout` doesn't reliably fire
under that pagination state, and the gradient server's `start()` never
returns — so `uvicorn.run()` never runs and the validator can't push
state.

PR added a hard wall-clock budget around the call: a one-shot
`ThreadPoolExecutor` with `future.result(timeout=BUDGET_S)` and a
`shutdown(wait=False)` to abandon the stuck worker so startup continues.
Cost on timeout: one round of ~5-min re-accumulation. Correctness
preserved (`_check_data_readiness` rebuilds the registry lazily).

These tests pin the new contract:
  1. Happy path — fast restore completes inside the budget.
  2. Slow restore — exceeds budget; method returns promptly and logs a
     warning; the stuck worker is detached (no deadlock).
  3. `GENTRX_PARQUET_RESTORE_BUDGET_S=0` short-circuits the call entirely.
  4. The env override accepts a custom budget float.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from GenTRX.src.gradient_server import GradientAggregator


def _make_aggregator(tmp_path: Path) -> GradientAggregator:
    return GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path="",
        output_path=str(tmp_path / "out.pt"),
        log_path=str(tmp_path / "agg.jsonl"),
        validator_store=None,
        is_aggregator=False,
        no_startup_cleanup=True,
    )


def test_restore_happy_path_returns_in_budget(tmp_path):
    """A fast `_restore_written_parquets_locked` completes inside the
    budget and the wrapper returns without warning."""
    agg = _make_aggregator(tmp_path)
    called = {"n": 0}

    def _quick():
        called["n"] += 1

    # Monkeypatch the inner body to a near-instant no-op.
    agg._restore_written_parquets_locked = _quick  # type: ignore[method-assign]

    t0 = time.perf_counter()
    agg._restore_written_parquets()
    elapsed = time.perf_counter() - t0
    assert called["n"] == 1
    assert elapsed < 1.0, f"happy-path restore should finish fast, took {elapsed:.2f}s"


def test_restore_timeout_does_not_deadlock(tmp_path, caplog, monkeypatch):
    """A stuck `_restore_written_parquets_locked` (one that never returns)
    must NOT wedge the wrapper. The outer method returns within
    budget+epsilon, logs a warning, and the worker is detached.

    This is the bug we hit live: a `with ThreadPoolExecutor()` block
    would call `shutdown(wait=True)` on exit and re-deadlock. The fix
    is manual `shutdown(wait=False)`.
    """
    caplog.set_level(logging.WARNING, logger="GenTRX.src.gradient_server")
    agg = _make_aggregator(tmp_path)

    # Shorten the budget to keep the test fast.
    monkeypatch.setenv("GENTRX_PARQUET_RESTORE_BUDGET_S", "0.5")

    stuck_started = {"flag": False}

    def _stuck():
        # Simulate the boto3 socket-stuck behaviour: sleep way past the
        # budget. Daemonised worker stays alive but the wrapper must
        # abandon it.
        stuck_started["flag"] = True
        time.sleep(5.0)

    agg._restore_written_parquets_locked = _stuck  # type: ignore[method-assign]

    t0 = time.perf_counter()
    agg._restore_written_parquets()
    elapsed = time.perf_counter() - t0

    assert stuck_started["flag"], "worker thread should have started"
    # Wrapper must return promptly after the 0.5s budget — allow 2.5s
    # slack for GC / scheduling, but NOT 5s (the inner sleep duration).
    assert elapsed < 2.5, (
        f"wrapper should return after the budget timeout, took {elapsed:.2f}s "
        "(if this exceeds 5s, the executor's shutdown(wait=True) is blocking — "
        "regression of the with-block bug)"
    )
    assert any(
        "Parquet restore exceeded" in r.getMessage() for r in caplog.records
    ), "budget-exceeded warning must be logged"


def test_restore_budget_zero_skips_call_entirely(tmp_path, caplog, monkeypatch):
    """Setting `GENTRX_PARQUET_RESTORE_BUDGET_S=0` short-circuits the
    restore without invoking the inner body. Useful for operators who
    don't want the startup S3 walk at all."""
    caplog.set_level(logging.WARNING, logger="GenTRX.src.gradient_server")
    agg = _make_aggregator(tmp_path)
    monkeypatch.setenv("GENTRX_PARQUET_RESTORE_BUDGET_S", "0")

    called = {"n": 0}

    def _should_not_run():
        called["n"] += 1

    agg._restore_written_parquets_locked = _should_not_run  # type: ignore[method-assign]

    t0 = time.perf_counter()
    agg._restore_written_parquets()
    elapsed = time.perf_counter() - t0

    assert called["n"] == 0, "inner body must not run when budget==0"
    assert elapsed < 0.5
    assert any(
        "Parquet restore disabled" in r.getMessage() for r in caplog.records
    ), "disabled warning must be logged"


def test_restore_budget_env_override_parses(tmp_path, monkeypatch):
    """The env var is read fresh each call (operators can tune without a
    restart of the aggregator object). A bad value falls back to the
    default; a good value is used verbatim."""
    agg = _make_aggregator(tmp_path)

    # Garbage value — should fall back to the class default, not crash.
    monkeypatch.setenv("GENTRX_PARQUET_RESTORE_BUDGET_S", "not-a-number")
    agg._restore_written_parquets_locked = lambda: None  # type: ignore[method-assign]
    # No exception — falls back to default budget.
    agg._restore_written_parquets()

    # Good value — parsed as float.
    monkeypatch.setenv("GENTRX_PARQUET_RESTORE_BUDGET_S", "0.1")
    agg._restore_written_parquets()


def test_restore_locked_signature_matches_original(tmp_path, monkeypatch):
    """The body method `_restore_written_parquets_locked` MUST accept the
    same (self,) signature the original `_restore_written_parquets` did,
    so existing code (e.g. tests that monkeypatch the call) can still
    use it. Calls the real body with a stub validator_store."""
    agg = _make_aggregator(tmp_path)

    class _StubStore:
        def list_books(self, _uid):
            return []  # Empty: no books to walk.

    monkeypatch.setattr(agg, "validator_store", _StubStore())
    # Direct call to the locked body — should return cleanly on an
    # empty bucket.
    agg._restore_written_parquets_locked()
    # And via the wrapper — also clean.
    agg._restore_written_parquets()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
