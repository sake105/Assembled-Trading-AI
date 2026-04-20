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
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# H10: tier-based weights so a T0/T1 wire carries more weight than a blog.
_TIER_WEIGHTS = {
    "T0": 1.5,
    "T1": 1.2,
    "T2": 1.0,
    "T3": 0.7,
    "T4": 0.5,
}


def _resolve_tier_weight(evt: object) -> float:
    tier = getattr(evt, "source_tier", None)
    key = getattr(tier, "value", str(tier) if tier is not None else "T2")
    return float(_TIER_WEIGHTS.get(key, 1.0))


@dataclass
class TickerSignal:
    ticker: str
    short_count: float  # weighted count in short window (tier-weighted)
    long_count: float   # weighted count in long window
    velocity: float     # ratio of short-window rate to prior rate
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
        # H10: buffer entries are (ts, content_hash, weight) so we can dedup
        # the same headline across sources and apply tier weighting.
        self._buffers: dict[str, deque[tuple[datetime, str, float]]] = {}
        # Dedup memory per ticker: content_hash seen within short window.
        self._seen_hashes: dict[str, dict[str, datetime]] = {}
        # H2: guard mutation paths for multi-threaded producers (RSS +
        # GDELT + enricher) that share one tracker instance.
        self._lock = threading.RLock()

    def update(self, events: list, now: datetime | None = None) -> list[TickerSignal]:
        """Append events and return the current ticker signals."""
        if now is None:
            now = datetime.now(tz=timezone.utc)

        with self._lock:
            return self._update_locked(events, now)

    def _update_locked(self, events: list, now: datetime) -> list[TickerSignal]:
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
                content_hash = (
                    getattr(evt, "content_hash", "") or ""
                ).strip().lower() or getattr(evt, "event_id", "")
                weight = _resolve_tier_weight(evt)
                for t in tickers:
                    t = t.upper().strip()
                    if not t:
                        continue
                    buf = self._buffers.setdefault(t, deque(maxlen=self._max_buffer))
                    seen = self._seen_hashes.setdefault(t, {})
                    # H10: dedup same content across sources inside the short
                    # window; keeps cross-wire coverage from inflating velocity.
                    if content_hash:
                        prev = seen.get(content_hash)
                        if prev is not None and (ts - prev) < self._short_td:
                            continue
                        seen[content_hash] = ts
                    buf.append((ts, content_hash, weight))
            except Exception as exc:
                logger.debug("[SKIP] TickerVelocity update: %s", exc)

        # prune & compute
        signals: list[TickerSignal] = []
        cutoff = now - self._long_td
        short_cutoff = now - self._short_td
        for ticker, buf in list(self._buffers.items()):
            while buf and buf[0][0] < cutoff:
                buf.popleft()
            if not buf:
                del self._buffers[ticker]
                self._seen_hashes.pop(ticker, None)
                continue
            # H10: weighted counts instead of naive cardinality.
            short_count = sum(w for ts, _, w in buf if ts >= short_cutoff)
            long_count = sum(w for _, _, w in buf)
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
                short_count=round(short_count, 3),
                long_count=round(long_count, 3),
                velocity=velocity,
                is_surge=is_surge,
            ))
            # H10: prune seen_hashes older than short window to bound memory.
            seen = self._seen_hashes.get(ticker)
            if seen:
                stale = [h for h, t in seen.items() if t < short_cutoff]
                for h in stale:
                    seen.pop(h, None)
        return signals

    def surging_tickers(self, now: datetime | None = None) -> list[tuple[str, float]]:
        """Return tickers currently in surge, sorted by velocity desc."""
        signals = self.update([], now=now)
        surging = [(s.ticker, s.velocity) for s in signals if s.is_surge]
        surging.sort(key=lambda x: -x[1])
        return surging

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._seen_hashes.clear()


__all__ = ["TickerVelocityTracker", "TickerSignal"]
