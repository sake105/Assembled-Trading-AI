"""Lightweight cross-platform file lock (Item 66).

Provides advisory file-level locking so concurrent backtests or paper-runner
instances writing to the same output path don't corrupt each other.

Design:
  * Lock file: ``<target_path>.lock`` — created before writing, removed after.
  * In-process: ``threading.Lock`` per lock-file path prevents same-process races.
  * Cross-process: ``os.open(..., O_CREAT | O_EXCL)`` is atomic on POSIX/Windows
    NTFS — only one process can create the lock file, others spin-wait.
  * Timeout: prevents deadlock if a process crashes while holding the lock.

No external dependencies required (no portalocker).

Usage::

    from src.assembled_core.utils.file_lock import FileLock

    with FileLock("output/equity_curve.parquet", timeout=10.0):
        df.to_parquet("output/equity_curve.parquet")
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from types import TracebackType

log = logging.getLogger(__name__)

_IN_PROCESS_LOCKS: dict[str, threading.Lock] = {}
_META_LOCK = threading.Lock()

_POLL_INTERVAL = 0.05  # seconds between lock-acquire retries


def _get_thread_lock(lock_path: str) -> threading.Lock:
    with _META_LOCK:
        if lock_path not in _IN_PROCESS_LOCKS:
            _IN_PROCESS_LOCKS[lock_path] = threading.Lock()
        return _IN_PROCESS_LOCKS[lock_path]


class FileLock:
    """Advisory file lock using a ``.lock`` sentinel file.

    Args:
        path: The file to protect. The lock file will be ``<path>.lock``.
        timeout: Maximum seconds to wait for the lock (default: 30).
        exclusive: Reserved for future use — always True (exclusive lock).
    """

    def __init__(
        self,
        path: str | Path,
        timeout: float = 30.0,
        exclusive: bool = True,
    ) -> None:
        self._path = Path(path)
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._timeout = timeout
        self._thread_lock = _get_thread_lock(str(self._lock_path))
        self._fd: int | None = None

    # ------------------------------------------------------------------ acquire
    def acquire(self) -> None:
        """Acquire the lock. Raises ``TimeoutError`` if *timeout* exceeded."""
        deadline = time.monotonic() + self._timeout
        # Thread lock uses the same timeout so callers from different threads
        # don't block indefinitely when the timeout is short.
        thread_remaining = max(0.0, deadline - time.monotonic())
        if not self._thread_lock.acquire(timeout=thread_remaining):
            raise TimeoutError(
                f"Could not acquire lock on {self._path} within {self._timeout}s"
            )
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                # O_EXCL guarantees atomicity: only one opener succeeds
                self._fd = os.open(
                    str(self._lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.write(self._fd, str(os.getpid()).encode())
                log.debug(
                    "[file_lock] acquired %s (pid=%d)", self._lock_path, os.getpid()
                )
                return
            except FileExistsError:
                if time.monotonic() >= deadline:
                    self._thread_lock.release()
                    raise TimeoutError(
                        f"Could not acquire lock on {self._path} within {self._timeout}s"
                    )
                time.sleep(_POLL_INTERVAL)
            except Exception:
                self._thread_lock.release()
                raise

    # ------------------------------------------------------------------ release
    def release(self) -> None:
        """Release the lock and remove the sentinel file."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            self._lock_path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning(
                "[file_lock] could not remove lock file %s: %s", self._lock_path, exc
            )
        finally:
            try:
                self._thread_lock.release()
            except RuntimeError:
                pass  # already released
        log.debug("[file_lock] released %s", self._lock_path)

    # ------------------------------------------------------------------ context
    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


__all__ = ["FileLock"]
