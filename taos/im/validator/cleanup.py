# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
from __future__ import annotations

import time
import asyncio
import traceback
import subprocess
from typing import TYPE_CHECKING

import bittensor as bt

if TYPE_CHECKING:
    from taos.im.neurons.validator import Validator


def cleanup_ipc(self: Validator):
    """
    Shuts down the query service and releases all POSIX IPC resources.

    Behavior:
        - Attempts to send a shutdown message to the query service.
        - Waits for graceful termination, falling back to terminate/kill.
        - Closes memory maps and shared memory file descriptors.
        - Closes message queues.
        - Logs detailed warnings for any partial cleanup failures.

    Returns:
        None
    """
    try:
        if hasattr(self, 'engine'):
            self.engine.stop()

        bt.logging.info("Cleaning up query service...")
        if hasattr(self, 'request_queue'):
            try:
                self.request_queue.send(b'shutdown', timeout=1.0)
                bt.logging.info("Sent shutdown command to query service")
            except Exception as e:
                bt.logging.warning(f"Failed to send shutdown command: {e}")
        if hasattr(self, 'query_process') and self.query_process:
            try:
                self.query_process.wait(timeout=5.0)
                bt.logging.info(f"Query service exited with code {self.query_process.returncode}")
            except subprocess.TimeoutExpired:
                bt.logging.warning("Query service did not exit gracefully, terminating...")
                self.query_process.terminate()
                try:
                    self.query_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    bt.logging.error("Query service did not terminate, killing...")
                    self.query_process.kill()

        if hasattr(self, 'request_mem'):
            try:
                self.request_mem.close()
                bt.logging.debug("Closed request memory map")
            except Exception as e:
                bt.logging.warning(f"Error closing request memory map: {e}")

        if hasattr(self, 'response_mem'):
            try:
                self.response_mem.close()
                bt.logging.debug("Closed response memory map")
            except Exception as e:
                bt.logging.warning(f"Error closing response memory map: {e}")

        if hasattr(self, 'request_shm'):
            try:
                self.request_shm.close_fd()
                bt.logging.debug("Closed request shared memory fd")
            except Exception as e:
                bt.logging.warning(f"Error closing request shared memory fd: {e}")

        if hasattr(self, 'response_shm'):
            try:
                self.response_shm.close_fd()
                bt.logging.debug("Closed response shared memory fd")
            except Exception as e:
                bt.logging.warning(f"Error closing response shared memory fd: {e}")

        if hasattr(self, 'request_queue'):
            try:
                self.request_queue.close()
                bt.logging.debug("Closed request queue")
            except Exception as e:
                bt.logging.warning(f"Error closing request queue: {e}")

        if hasattr(self, 'response_queue'):
            try:
                self.response_queue.close()
                bt.logging.debug("Closed response queue")
            except Exception as e:
                bt.logging.warning(f"Error closing response queue: {e}")

        bt.logging.info("Query service cleanup complete")

        bt.logging.info("Cleaning up reporting service...")

        if hasattr(self, 'reporting_request_queue'):
            try:
                self.reporting_request_queue.send(b'shutdown', timeout=1.0)
                bt.logging.info("Sent shutdown command to reporting service")
            except Exception as e:
                bt.logging.warning(f"Failed to send shutdown command to reporting: {e}")

        if hasattr(self, 'reporting_process') and self.reporting_process:
            try:
                self.reporting_process.wait(timeout=5.0)
                bt.logging.info(f"Reporting service exited with code {self.reporting_process.returncode}")
            except subprocess.TimeoutExpired:
                bt.logging.warning("Reporting service did not exit gracefully, terminating...")
                self.reporting_process.terminate()
                try:
                    self.reporting_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    bt.logging.error("Reporting service did not terminate, killing...")
                    self.reporting_process.kill()

        if hasattr(self, 'reporting_request_mem'):
            try:
                self.reporting_request_mem.close()
                bt.logging.debug("Closed reporting request memory map")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting request memory map: {e}")

        if hasattr(self, 'reporting_response_mem'):
            try:
                self.reporting_response_mem.close()
                bt.logging.debug("Closed reporting response memory map")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting response memory map: {e}")

        if hasattr(self, 'reporting_request_shm'):
            try:
                self.reporting_request_shm.close_fd()
                bt.logging.debug("Closed reporting request shared memory fd")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting request shared memory fd: {e}")

        if hasattr(self, 'reporting_response_shm'):
            try:
                self.reporting_response_shm.close_fd()
                bt.logging.debug("Closed reporting response shared memory fd")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting response shared memory fd: {e}")

        if hasattr(self, 'reporting_request_queue'):
            try:
                self.reporting_request_queue.close()
                bt.logging.debug("Closed reporting request queue")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting request queue: {e}")

        if hasattr(self, 'reporting_response_queue'):
            try:
                self.reporting_response_queue.close()
                bt.logging.debug("Closed reporting response queue")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting response queue: {e}")

        bt.logging.info("Reporting service cleanup complete")

        bt.logging.info("Cleaning up seed service...")
        if hasattr(self, 'seed_process') and self.seed_process:
            try:
                self.seed_process.terminate()
                self.seed_process.wait(timeout=5.0)
                bt.logging.info(f"Seed service exited with code {self.seed_process.returncode}")
            except subprocess.TimeoutExpired:
                bt.logging.warning("Seed service did not exit gracefully, killing...")
                self.seed_process.kill()
        bt.logging.info("Seed service cleanup complete")

    except Exception as e:
        bt.logging.error(f"Error during validator cleanup: {e}")
        bt.logging.error(traceback.format_exc())


def cleanup_executors(self: Validator):
    """
    Shuts down thread and process executors used by the validator.

    Executors cleaned:
        - reward_executor (ProcessPoolExecutor)
        - save_state_executor (ThreadPoolExecutor)
        - maintenance_executor (ThreadPoolExecutor)
        - multiprocessing manager (if present)

    Behavior:
        - Each executor is shut down gracefully with wait=True
        - For ProcessPoolExecutor, attempts graceful shutdown first
        - Falls back to immediate termination if graceful fails
        - Logs success or failure for each executor

    Returns:
        None
    """
    if hasattr(self, 'reward_executor') and self.reward_executor is not None:
        try:
            bt.logging.info("Shutting down reward_executor...")
            self.reward_executor.shutdown(wait=True, cancel_futures=False)
            bt.logging.info("reward_executor shut down successfully")
        except Exception as ex:
            bt.logging.error(f"Error shutting down reward_executor: {ex}")
            try:
                bt.logging.warning("Attempting to terminate reward_executor processes...")
                for process in self.reward_executor._processes.values():
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2.0)
                        if process.is_alive():
                            process.kill()
                bt.logging.info("reward_executor processes terminated")
            except Exception as term_ex:
                bt.logging.error(f"Error terminating reward_executor: {term_ex}")

    if hasattr(self, 'query_ipc_executor') and self.query_ipc_executor is not None:
        try:
            bt.logging.info("Shutting down query_ipc_executor...")
            self.query_ipc_executor.shutdown(wait=True, cancel_futures=False)
            bt.logging.info("query_ipc_executor shut down successfully")
        except Exception as ex:
            bt.logging.error(f"Error shutting down query_ipc_executor: {ex}")

    if hasattr(self, 'reporting_ipc_executor') and self.reporting_ipc_executor is not None:
        try:
            bt.logging.info("Shutting down reporting_ipc_executor...")
            self.reporting_ipc_executor.shutdown(wait=True, cancel_futures=False)
            bt.logging.info("reporting_ipc_executor shut down successfully")
        except Exception as ex:
            bt.logging.error(f"Error shutting down reporting_ipc_executor: {ex}")

    thread_executors = {
        'save_state_executor': getattr(self, 'save_state_executor', None),
        'maintenance_executor': getattr(self, 'maintenance_executor', None),
        '_mvtrx_push_executor': getattr(self, '_mvtrx_push_executor', None),
    }

    for name, executor in thread_executors.items():
        if executor is not None:
            try:
                bt.logging.info(f"Shutting down {name}...")
                executor.shutdown(wait=True, cancel_futures=False)
                bt.logging.info(f"{name} shut down successfully")
            except Exception as ex:
                bt.logging.error(f"Error shutting down {name}: {ex}")

    if hasattr(self, 'manager'):
        try:
            bt.logging.info("Shutting down multiprocessing manager...")
            self.manager.shutdown()
            bt.logging.info("Manager shut down successfully")
        except Exception as ex:
            bt.logging.error(f"Error shutting down manager: {ex}")

    bt.logging.info("Executor cleanup complete")


def cleanup_event_loop(self: Validator):
    """
    Gracefully shuts down the main event loop and any pending tasks.

    Behavior:
        - Cancels all pending tasks in the main loop
        - Waits for task cancellation to complete
        - Stops the event loop if still running
        - Closes the event loop

    Returns:
        None
    """
    try:
        if hasattr(self, 'main_loop') and self.main_loop and not self.main_loop.is_closed():
            bt.logging.info("Shutting down main event loop...")

            pending = asyncio.all_tasks(self.main_loop)
            if pending:
                bt.logging.info(f"Cancelling {len(pending)} pending tasks...")
                for task in pending:
                    task.cancel()

                self.main_loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

            if self.main_loop.is_running():
                self.main_loop.stop()

            self.main_loop.close()
            bt.logging.info("Main event loop shut down successfully")
    except Exception as ex:
        bt.logging.error(f"Error shutting down main event loop: {ex}")
        bt.logging.error(traceback.format_exc())


def cleanup(self: Validator):
    """
    Performs full resource cleanup for the validator during shutdown.
    """
    if self._cleanup_done:
        bt.logging.debug("Cleanup already completed, skipping")
        return

    bt.logging.info("Starting validator cleanup...")
    self._cleanup_done = True

    try:
        bt.logging.info("Waiting for active operations to complete...")
        wait_timeout = 30.0
        wait_start = time.time()

        while (self.shared_state_rewarding or
            self.shared_state_saving or
            self.shared_state_reporting or
            self.maintaining or
            self.compressing or
            self.querying):

            elapsed = time.time() - wait_start
            if elapsed > wait_timeout:
                bt.logging.warning(
                    f"Timeout waiting for operations after {elapsed:.2f}s"
                )
                break
            time.sleep(0.1)

        cleanup_executors(self)
        cleanup_ipc(self)
        cleanup_event_loop(self)

        bt.logging.success("Validator cleanup completed successfully")

    except Exception as ex:
        bt.logging.error(f"Error during cleanup: {ex}")
        bt.logging.error(traceback.format_exc())
