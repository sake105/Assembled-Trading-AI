# src/assembled_core/adapters/outbound/clock_adapter.py
"""Clock port implementations — SystemClock (real) + FrozenClock (tests).

Audit C-001 + C-003. These are the canonical "test-seam" pair: the
domain depends on ``Clock``; production wires ``SystemClock``; tests
wire ``FrozenClock`` (with the ability to advance time deterministically).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.assembled_core.ports.clock import Clock


class SystemClock(Clock):
    """Real wall-clock — always returns ``datetime.now(timezone.utc)``."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FrozenClock(Clock):
    """A clock that returns a controlled point in time.

    Tests can advance the clock with ``.tick(seconds)`` so time-dependent
    branches (cooldowns, rate-limit refills, NTP drift checks) can be
    exercised deterministically.
    """

    def __init__(self, *, initial: datetime | None = None) -> None:
        self._now = (
            initial
            if initial is not None
            else datetime(2026, 1, 1, tzinfo=timezone.utc)
        )

    def now(self) -> datetime:
        return self._now

    def tick(self, *, seconds: float = 0, minutes: float = 0, hours: float = 0) -> None:
        delta = timedelta(seconds=seconds, minutes=minutes, hours=hours)
        self._now = self._now + delta

    def set(self, when: datetime) -> None:
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        self._now = when


__all__ = ["SystemClock", "FrozenClock"]
