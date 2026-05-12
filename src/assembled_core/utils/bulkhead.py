# src/assembled_core/utils/bulkhead.py
"""Bulkhead-pattern concurrency isolation (audit C2-013).

Per-integration semaphores + a tiny in-process circuit-breaker so one
misbehaving external dependency (broker, news provider, FRED) cannot
exhaust the local resource budget and take the rest of the system with
it. Pure stdlib (asyncio.Semaphore + threading.RLock counters) — no
pybreaker dependency.

Pattern (Netflix / Hystrix):

    bh = Bulkhead("alpaca", max_concurrent=8, failure_threshold=5,
                  cooldown_seconds=30)
    async with bh.acquire():
        await alpaca_client.list_orders()

If 5 consecutive calls inside a bulkhead raise, the breaker opens and
subsequent acquires raise ``BulkheadOpenError`` for ``cooldown_seconds``.
After cooldown, the next acquire is treated as a probe — success closes
the breaker, failure re-opens it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class BulkheadOpenError(RuntimeError):
    """Raised when an integration's bulkhead breaker is currently open."""


class Bulkhead:
    """Per-integration concurrency limit + circuit breaker.

    Args:
        name: integration name (used in logs).
        max_concurrent: max simultaneous calls allowed. > 0.
        failure_threshold: consecutive failures that trip the breaker.
        cooldown_seconds: time the breaker stays open before allowing a probe.
    """

    def __init__(
        self,
        name: str,
        *,
        max_concurrent: int = 8,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        self.name = name
        self.max_concurrent = max_concurrent
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._sem = asyncio.Semaphore(max_concurrent)
        self._lock = threading.RLock()
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                return False
            return True

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if (
                self._consecutive_failures >= self.failure_threshold
                and self._opened_at is None
            ):
                self._opened_at = time.monotonic()
                logger.warning(
                    "[bulkhead:%s] breaker OPEN after %d consecutive failures",
                    self.name,
                    self._consecutive_failures,
                )

    def record_success(self) -> None:
        with self._lock:
            if self._opened_at is not None:
                logger.info(
                    "[bulkhead:%s] breaker CLOSED after probe success", self.name
                )
            self._consecutive_failures = 0
            self._opened_at = None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Acquire one slot. Raises BulkheadOpenError when the breaker is open."""
        if self.is_open:
            raise BulkheadOpenError(
                f"bulkhead '{self.name}' breaker open — cooldown in progress"
            )
        async with self._sem:
            try:
                yield
            except Exception:
                self.record_failure()
                raise
            else:
                self.record_success()


__all__ = ["Bulkhead", "BulkheadOpenError"]
