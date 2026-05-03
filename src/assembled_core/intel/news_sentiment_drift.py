"""News sentiment drift / trajectory tracker.

Aggregates per-ticker and per-sector sentiment over a sliding window and
computes the trajectory (slope) — i.e. is sentiment improving, deteriorating,
or flat? This catches slow-burn regime changes that single-event severity
would miss (e.g. 20 mildly bearish stories in a row on the same name).

Usage:
    tracker = SentimentDriftTracker(window_min=60)
    tracker.update(events, now=now)
    report = tracker.report(now=now)
    for entry in report:
        print(entry.key, entry.mean_sentiment, entry.slope)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftEntry:
    key: str                   # "TICKER:AAPL" or "SECTOR:tech"
    n_events: int
    mean_sentiment: float
    slope: float               # simple slope per minute
    drift_direction: str       # "improving", "deteriorating", "flat"
    latest_sentiment: float


class SentimentDriftTracker:
    """Sliding-window sentiment aggregation with trajectory detection."""

    def __init__(
        self,
        window_min: int = 60,
        min_events: int = 3,
        slope_threshold: float = 0.01,   # sentiment-units per minute
        max_buffer: int = 500,
    ) -> None:
        self._window_td = timedelta(minutes=window_min)
        self._min_events = min_events
        self._slope_th = slope_threshold
        self._max_buffer = max_buffer
        # key -> deque[(ts, sentiment)]
        self._buffers: dict[str, deque[tuple[datetime, float]]] = {}

    def update(self, events: list, now: datetime | None = None) -> None:
        if now is None:
            now = datetime.now(tz=timezone.utc)
        for evt in events:
            try:
                ts = getattr(evt, "published_at", None) or getattr(evt, "ingested_at", None) or now
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                sent = max(-1.0, min(1.0, float(getattr(evt, "sentiment_score", 0.0) or 0.0)))
                tickers = list(getattr(evt, "tickers", []) or [])
                if not tickers:
                    tickers = list(getattr(evt, "affected_assets", []) or [])
                for t in tickers:
                    t = t.upper().strip()
                    if t:
                        self._append(f"TICKER:{t}", ts, sent)
                sectors = list(getattr(evt, "affected_sectors", []) or [])
                for s in sectors:
                    s = s.lower().strip()
                    if s:
                        self._append(f"SECTOR:{s}", ts, sent)
            except Exception as exc:
                logger.debug("[SKIP] SentimentDrift update: %s", exc)

    def _append(self, key: str, ts: datetime, sentiment: float) -> None:
        buf = self._buffers.setdefault(key, deque(maxlen=self._max_buffer))
        buf.append((ts, sentiment))

    def report(self, now: datetime | None = None) -> list[DriftEntry]:
        if now is None:
            now = datetime.now(tz=timezone.utc)
        cutoff = now - self._window_td
        out: list[DriftEntry] = []
        for key, buf in list(self._buffers.items()):
            # prune
            while buf and buf[0][0] < cutoff:
                buf.popleft()
            if not buf:
                del self._buffers[key]
                continue
            if len(buf) < self._min_events:
                continue
            points = list(buf)
            mean_s = sum(p[1] for p in points) / len(points)
            slope = _least_squares_slope(points, anchor=now)
            if slope > self._slope_th:
                direction = "improving"
            elif slope < -self._slope_th:
                direction = "deteriorating"
            else:
                direction = "flat"
            out.append(DriftEntry(
                key=key,
                n_events=len(points),
                mean_sentiment=round(mean_s, 4),
                slope=round(slope, 6),
                drift_direction=direction,
                latest_sentiment=round(points[-1][1], 4),
            ))
        out.sort(key=lambda e: -abs(e.slope))
        return out

    def clear(self) -> None:
        self._buffers.clear()


def _least_squares_slope(
    points: list[tuple[datetime, float]], anchor: datetime
) -> float:
    """Simple OLS slope of sentiment vs. minutes from anchor.

    Returns 0.0 if variance of x is zero (single point or all identical).
    """
    if len(points) < 2:
        return 0.0
    xs = np.array([(p[0] - anchor).total_seconds() / 60.0 for p in points])
    ys = np.array([p[1] for p in points])
    xs_c = xs - xs.mean()
    ys_c = ys - ys.mean()
    den = float(np.dot(xs_c, xs_c))
    if den == 0.0:
        return 0.0
    return float(np.dot(xs_c, ys_c) / den)


__all__ = ["SentimentDriftTracker", "DriftEntry"]
