"""Circuit breaker for flash-crash detection and automatic trading halts.

Monitors market-level or portfolio-level price drops and triggers a
configurable trading halt when thresholds are breached.

Usage::

    from src.assembled_core.risk.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(drop_threshold_pct=3.0, window_minutes=15)
    cb.observe(price=100.0, timestamp=t0)
    cb.observe(price=96.5, timestamp=t1)   # -3.5% in window -> TRIPPED
    assert cb.is_tripped
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    """Flash-crash circuit breaker with configurable threshold and cooldown.

    Attributes:
        drop_threshold_pct: Maximum allowed drop (%) within the observation
            window.  Default 3.0 means a 3% drop trips the breaker.
        window_minutes: Rolling observation window in minutes (default 15).
        cooldown_minutes: How long the breaker stays tripped after the last
            trigger (default 30).
    """

    drop_threshold_pct: float = 3.0
    window_minutes: int = 15
    cooldown_minutes: int = 30

    # Internal state
    _observations: deque = field(default_factory=deque, repr=False)
    _tripped_at: datetime | None = field(default=None, repr=False)
    _trip_count: int = field(default=0, repr=False)

    def observe(self, price: float, timestamp: datetime | None = None) -> bool:
        """Record a price observation and check for circuit-breaker trigger.

        Args:
            price: Current market/portfolio price level.
            timestamp: Observation time (defaults to now UTC).

        Returns:
            True if the circuit breaker just tripped on this observation.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        self._observations.append((timestamp, price))

        # Prune old observations outside the window
        cutoff = timestamp - timedelta(minutes=self.window_minutes)
        while self._observations and self._observations[0][0] < cutoff:
            self._observations.popleft()

        if len(self._observations) < 2:
            return False

        # Check for drop from window high to current price
        window_high = max(p for _, p in self._observations)
        if window_high <= 0:
            return False

        drop_pct = (window_high - price) / window_high * 100.0

        if drop_pct >= self.drop_threshold_pct:
            self._tripped_at = timestamp
            self._trip_count += 1
            logger.critical(
                "[CircuitBreaker] TRIPPED: %.1f%% drop in %d-min window "
                "(high=%.2f, current=%.2f, threshold=%.1f%%). "
                "Trading halted for %d minutes. Trip #%d.",
                drop_pct, self.window_minutes, window_high, price,
                self.drop_threshold_pct, self.cooldown_minutes, self._trip_count,
            )
            return True

        return False

    @property
    def is_tripped(self) -> bool:
        """Check if the circuit breaker is currently tripped (in cooldown)."""
        if self._tripped_at is None:
            return False
        now = datetime.now(timezone.utc)
        cooldown_end = self._tripped_at + timedelta(minutes=self.cooldown_minutes)
        return now < cooldown_end

    @property
    def trip_count(self) -> int:
        """Total number of times the breaker has tripped."""
        return self._trip_count

    def reset(self) -> None:
        """Manually reset the circuit breaker (e.g., after ops review)."""
        self._tripped_at = None
        self._observations.clear()
        logger.info("[CircuitBreaker] Manually reset by operator.")

    def get_state(self) -> dict:
        """Return current circuit breaker state for monitoring."""
        return {
            "is_tripped": self.is_tripped,
            "trip_count": self._trip_count,
            "tripped_at": self._tripped_at.isoformat() if self._tripped_at else None,
            "drop_threshold_pct": self.drop_threshold_pct,
            "window_minutes": self.window_minutes,
            "cooldown_minutes": self.cooldown_minutes,
            "observations_in_window": len(self._observations),
        }


@dataclass
class VolCircuitBreaker:
    """Volatility-spike circuit breaker (Sprint 4 / Plan C28).

    Complements :class:`CircuitBreaker` by tripping when realised short-term
    volatility exceeds a configurable multiple of realised long-term
    volatility. Both volatilities are computed from daily close returns
    passed in by the caller — the breaker itself is stateless beyond the
    trip flag and trip count.

    Default trigger: realised 5d stdev > 2.0 * realised 60d stdev.

    Usage::

        vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.0)
        if vcb.check_returns(recent_returns):
            # breaker just tripped — throttle, alert, etc.
            ...
    """

    short_window: int = 5
    long_window: int = 60
    ratio_threshold: float = 2.0
    cooldown_minutes: int = 30

    _tripped_at: datetime | None = field(default=None, repr=False)
    _trip_count: int = field(default=0, repr=False)
    _last_ratio: float = field(default=0.0, repr=False)

    def check_returns(self, returns: list[float] | "object") -> bool:
        """Evaluate a sequence of recent returns and trip if vol ratio exceeds threshold.

        Args:
            returns: Sequence of daily returns in chronological order.
                Must contain at least ``long_window`` observations, otherwise
                the breaker stays inactive.

        Returns:
            True if the breaker just tripped on this call.
        """
        # Local import to keep the module lightweight and optional-dep-free
        try:
            import statistics
        except Exception:  # pragma: no cover - statistics is stdlib
            return False

        try:
            seq = list(returns)
        except TypeError:
            return False

        if len(seq) < self.long_window or self.long_window < 2 or self.short_window < 2:
            return False

        long_slice = seq[-self.long_window:]
        short_slice = seq[-self.short_window:]

        try:
            long_vol = statistics.pstdev(long_slice)
            short_vol = statistics.pstdev(short_slice)
        except statistics.StatisticsError:
            return False

        if long_vol <= 1e-12:
            return False

        ratio = short_vol / long_vol
        self._last_ratio = ratio

        if ratio >= self.ratio_threshold:
            self._tripped_at = datetime.now(timezone.utc)
            self._trip_count += 1
            logger.critical(
                "[VolCircuitBreaker] TRIPPED: short/long vol ratio=%.3f "
                "(short=%.5f over %d bars, long=%.5f over %d bars, threshold=%.2f). "
                "Trip #%d.",
                ratio, short_vol, self.short_window, long_vol, self.long_window,
                self.ratio_threshold, self._trip_count,
            )
            return True

        return False

    @property
    def is_tripped(self) -> bool:
        if self._tripped_at is None:
            return False
        now = datetime.now(timezone.utc)
        cooldown_end = self._tripped_at + timedelta(minutes=self.cooldown_minutes)
        return now < cooldown_end

    @property
    def trip_count(self) -> int:
        return self._trip_count

    @property
    def last_ratio(self) -> float:
        return self._last_ratio

    def reset(self) -> None:
        self._tripped_at = None
        self._last_ratio = 0.0
        logger.info("[VolCircuitBreaker] Manually reset by operator.")

    def get_state(self) -> dict:
        return {
            "is_tripped": self.is_tripped,
            "trip_count": self._trip_count,
            "last_ratio": self._last_ratio,
            "tripped_at": self._tripped_at.isoformat() if self._tripped_at else None,
            "short_window": self.short_window,
            "long_window": self.long_window,
            "ratio_threshold": self.ratio_threshold,
            "cooldown_minutes": self.cooldown_minutes,
        }
