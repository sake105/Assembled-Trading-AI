"""Per-ticker news velocity tracker.

Analogous to the sector/global VelocityTracker, but at the ticker level —
useful for catching ticker-specific news spikes (e.g. single-stock events
like earnings surprises, CEO changes, M&A rumours).

Usage:
    vt = TickerVelocityTracker(short_window_min=15, long_window_min=60)
    vt.update(events, now=now)
    surging = vt.surging_tickers(now=now)
    for ticker, velocity in surging:
        print(ticker, velocity)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


@dataclass
class TickerSignal:
    ticker: str
    short_count: int
    long_count: int
    velocity: float  # ratio of short-window rate to prior rate
    is_surge: bool


class TickerVelocityTracker:
    """Tracks event counts per ticker and detects velocity surges."""

    def __init__(
        self,
        short_window_min: int = 15,
        long_window_min: int = 60,
        surge_threshold: float = 3.0,
        min_short_events: int = 2,
        max_buffer_per_ticker: int = 500,
    ) -> None:
        self._short_td = timedelta(minutes=short_window_min)
        self._long_td = timedelta(minutes=long_window_min)
        self._surge_threshold = surge_threshold
        self._min_short = min_short_events
        self._max_buffer = max_buffer_per_ticker
        self._buffers: dict[str, deque[datetime]] = {}

    def update(self, events: list, now: datetime | None = None) -> list[TickerSignal]:
        """Append events and return the current ticker signals."""
        if now is None:
            now = datetime.now(tz=timezone.utc)

        for evt in events:
            try:
                ts = (
                    getattr(evt, "published_at", None)
                    or getattr(evt, "ingested_at", None)
                    or now
                )
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                tickers = list(getattr(evt, "tickers", []) or [])
                if not tickers:
                    # also try affected_assets if no ticker list
                    tickers = list(getattr(evt, "affected_assets", []) or [])
                for t in tickers:
                    t = t.upper().strip()
                    if not t:
                        continue
                    buf = self._buffers.setdefault(t, deque(maxlen=self._max_buffer))
                    buf.append(ts)
            except Exception as exc:
                logger.debug("[SKIP] TickerVelocity update: %s", exc)

        # prune & compute
        signals: list[TickerSignal] = []
        cutoff = now - self._long_td
        short_cutoff = now - self._short_td
        for ticker, buf in list(self._buffers.items()):
            while buf and buf[0] < cutoff:
                buf.popleft()
            if not buf:
                del self._buffers[ticker]
                continue
            short_count = sum(1 for ts in buf if ts >= short_cutoff)
            long_count = len(buf)
            prior_count = long_count - short_count
            short_rate = short_count / max(self._short_td.total_seconds() / 60, 1)
            prior_rate = prior_count / max((self._long_td - self._short_td).total_seconds() / 60, 1)
            if prior_rate > 0:
                velocity = round(short_rate / prior_rate, 3)
            elif short_rate > 0:
                velocity = float(self._surge_threshold + 1)
            else:
                velocity = 1.0
            is_surge = velocity >= self._surge_threshold and short_count >= self._min_short
            signals.append(TickerSignal(
                ticker=ticker,
                short_count=short_count,
                long_count=long_count,
                velocity=velocity,
                is_surge=is_surge,
            ))
        return signals

    def surging_tickers(self, now: datetime | None = None) -> list[tuple[str, float]]:
        """Return tickers currently in surge, sorted by velocity desc."""
        signals = self.update([], now=now)
        surging = [(s.ticker, s.velocity) for s in signals if s.is_surge]
        surging.sort(key=lambda x: -x[1])
        return surging

    def clear(self) -> None:
        self._buffers.clear()


__all__ = ["TickerVelocityTracker", "TickerSignal"]
