"""Clock abstractions for deterministic replay.

From 42_EVENT_REPLAY_SYSTEM.md — Clock Protocol.

All time-dependent code should accept a Clock instead of calling
datetime.utcnow() directly. This makes replay deterministic: the
ReplayClock is driven by ClockTick events rather than wall time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    """Minimal clock protocol — returns current datetime."""

    def now(self) -> datetime: ...


class RealClock:
    """Wall-clock implementation for live operation."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class ReplayClock:
    """Deterministic clock driven by ClockTick events.

    The current time is set explicitly via advance_to() and never
    reads from the system clock. Time can only move forward.
    """

    def __init__(self, start_time: datetime) -> None:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        self._current = start_time

    def now(self) -> datetime:
        return self._current

    def advance_to(self, timestamp: datetime) -> None:
        """Advance clock to timestamp. Raises ValueError if timestamp is in the past."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if timestamp < self._current:
            raise ValueError(
                f"Cannot go backwards: {timestamp.isoformat()} < {self._current.isoformat()}"
            )
        self._current = timestamp


__all__ = ["Clock", "RealClock", "ReplayClock"]
