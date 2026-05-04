"""Breaking news velocity tracker.

Detects surges in news volume that signal an accelerating event —
a key early-warning indicator before a story becomes fully confirmed.

Velocity = (events_in_last_window - events_in_prior_window) / prior_window
Surge is flagged when velocity exceeds a configurable threshold.

Usage:
    tracker = VelocityTracker(short_window_min=15, long_window_min=60)
    result = tracker.update(events)
    if result.is_surge:
        print(f"Breaking surge: {result.surge_sectors} at velocity {result.velocity:.2f}x")
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VelocityResult:
    """Result of a velocity analysis pass."""

    timestamp: datetime
    short_count: int  # events in short window
    long_count: int  # events in long window
    velocity: float  # short_rate / long_rate (>1 = accelerating)
    is_surge: bool  # True when velocity >= surge_threshold
    surge_sectors: list[str] = field(default_factory=list)
    surge_event_types: list[str] = field(default_factory=list)
    avg_severity: float = 0.0


class VelocityTracker:
    """Rolling-window news velocity tracker.

    Maintains a bounded deque of (timestamp, event) pairs. On each update
    compares short-window event rate vs long-window rate to detect acceleration.
    """

    def __init__(
        self,
        short_window_min: float = 15.0,
        long_window_min: float = 60.0,
        surge_threshold: float = 2.5,
        max_buffer: int = 2_000,
    ) -> None:
        self._short_td = timedelta(minutes=short_window_min)
        self._long_td = timedelta(minutes=long_window_min)
        self._surge_threshold = surge_threshold
        self._buffer: deque[tuple[datetime, Any]] = deque(maxlen=max_buffer)

    def update(
        self,
        events: list,
        now: datetime | None = None,
    ) -> VelocityResult:
        """Add new events and compute velocity.

        Args:
            events: List of NewsEvent objects (need published_at / ingested_at).
            now: Reference timestamp. Defaults to utcnow.

        Returns:
            VelocityResult with velocity and surge flag.
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)

        # Add to buffer
        for evt in events:
            ts = getattr(evt, "published_at", None) or getattr(evt, "ingested_at", None)
            if ts is None:
                ts = now
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            self._buffer.append((ts, evt))

        # Prune expired (older than long window)
        cutoff = now - self._long_td
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

        # Count events in each window
        short_cutoff = now - self._short_td
        short_events = [e for ts, e in self._buffer if ts >= short_cutoff]
        long_events = [e for ts, e in self._buffer]

        short_count = len(short_events)
        long_count = len(long_events)

        # Compute velocity
        # Rate = events / minutes in each window
        short_rate = short_count / max(self._short_td.total_seconds() / 60, 1)
        long_window_nonshort = max(
            (self._long_td - self._short_td).total_seconds() / 60, 1
        )
        prior_count = long_count - short_count
        prior_rate = prior_count / long_window_nonshort

        if prior_rate > 0:
            velocity = round(short_rate / prior_rate, 3)
        elif short_rate > 0:
            velocity = float(self._surge_threshold + 1)  # infinite acceleration
        else:
            velocity = 1.0

        is_surge = velocity >= self._surge_threshold and short_count >= 3

        # Always compute sector/type stats and severity for short window
        sector_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        severity_total = 0.0
        for evt in short_events:
            for s in getattr(evt, "affected_sectors", []) or []:
                sector_counts[s] = sector_counts.get(s, 0) + 1
            for et in getattr(evt, "event_types", []) or []:
                type_counts[et] = type_counts.get(et, 0) + 1
            severity_total += float(getattr(evt, "severity", 0.0) or 0.0)

        avg_severity = (
            round(severity_total / max(len(short_events), 1), 3)
            if short_events
            else 0.0
        )

        surge_sectors: list[str] = []
        surge_event_types: list[str] = []
        if is_surge:
            surge_sectors = sorted(sector_counts, key=lambda k: -sector_counts[k])[:3]
            surge_event_types = sorted(type_counts, key=lambda k: -type_counts[k])[:3]
            logger.info(
                "[WARN] News velocity surge: %.1fx — sectors=%s types=%s short_count=%d",
                velocity,
                surge_sectors,
                surge_event_types,
                short_count,
            )

        return VelocityResult(
            timestamp=now,
            short_count=short_count,
            long_count=long_count,
            velocity=velocity,
            is_surge=is_surge,
            surge_sectors=surge_sectors,
            surge_event_types=surge_event_types,
            avg_severity=avg_severity,
        )

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)


__all__ = ["VelocityTracker", "VelocityResult"]
