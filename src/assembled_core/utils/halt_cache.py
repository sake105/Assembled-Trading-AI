"""60s-refresh halt-symbol cache (audit C5-091, closes _tc_sizing TODO line 1715).

A small in-process cache wrapping a callable that returns the current
universe of halted symbols. Pipeline runners populate
``ctx.halted_symbols`` from ``HaltCache.snapshot()`` once per cycle;
the cache itself decides whether to call the supplier again or hand
back the previously-fetched set.

Design rules:
    - stdlib only (no threading lock for now; pipeline is single-writer)
    - bounded freshness via wall-clock TTL (default 60s)
    - fail-soft: supplier exceptions never propagate — the previous
      snapshot is returned instead, and a warning is logged
    - empty snapshots are valid; ``None`` from supplier ⇒ empty set
    - ``Clock`` injection-point so tests advance time deterministically

This file lives under ``utils/`` (cross-cutting helper) per the
hexagonal migration plan. It does NOT import from ports/adapters/
domain — it's a pure helper any layer can use.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, Protocol

logger = logging.getLogger(__name__)


class _ClockLike(Protocol):
    """Minimal clock interface — anything with ``monotonic()``-style float."""

    def now_monotonic(self) -> float: ...


class _SystemMonotonicClock:
    """Default clock — wraps ``time.monotonic`` so tests can swap it."""

    def now_monotonic(self) -> float:
        return time.monotonic()


DEFAULT_TTL_SECONDS = 60.0


class HaltCache:
    """Wraps a halt-supplier with TTL caching and fail-soft semantics.

    Usage::

        cache = HaltCache(supplier=lambda: broker_client.get_halted_symbols())
        # in the per-cycle pipeline tick:
        ctx.halted_symbols = cache.snapshot()
    """

    def __init__(
        self,
        *,
        supplier: Callable[[], Iterable[str] | None],
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: _ClockLike | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        self._supplier = supplier
        self._ttl = float(ttl_seconds)
        self._clock: _ClockLike = (
            clock if clock is not None else _SystemMonotonicClock()
        )
        self._cached: frozenset[str] = frozenset()
        self._last_refresh: float | None = None
        self._consecutive_failures: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self) -> frozenset[str]:
        """Return the current halt-set, refreshing if TTL has expired."""
        if self._needs_refresh():
            self._refresh()
        return self._cached

    def force_refresh(self) -> frozenset[str]:
        """Bypass TTL — useful right after a paper-engine halt signal."""
        self._refresh()
        return self._cached

    @property
    def last_refresh_age_seconds(self) -> float | None:
        if self._last_refresh is None:
            return None
        return self._clock.now_monotonic() - self._last_refresh

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _needs_refresh(self) -> bool:
        if self._last_refresh is None:
            return True
        return (self._clock.now_monotonic() - self._last_refresh) >= self._ttl

    def _refresh(self) -> None:
        try:
            raw = self._supplier()
        except Exception as exc:  # noqa: BLE001 — fail-soft is the contract
            self._consecutive_failures += 1
            logger.warning(
                "[halt_cache] supplier raised %s on refresh #%d — "
                "keeping previous snapshot of %d symbol(s)",
                exc.__class__.__name__,
                self._consecutive_failures,
                len(self._cached),
            )
            return
        # Success path: normalize to a frozenset[str].
        if raw is None:
            new_set: frozenset[str] = frozenset()
        else:
            new_set = frozenset(str(s) for s in raw if s)
        added = new_set - self._cached
        removed = self._cached - new_set
        if added or removed:
            logger.info(
                "[halt_cache] refresh: added=%d removed=%d total=%d",
                len(added),
                len(removed),
                len(new_set),
            )
        self._cached = new_set
        self._last_refresh = self._clock.now_monotonic()
        self._consecutive_failures = 0


__all__ = ["HaltCache", "DEFAULT_TTL_SECONDS"]
