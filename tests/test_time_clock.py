"""Tests for assembled_core/time/clock.py (spec 42_EVENT_REPLAY_SYSTEM)."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from assembled_core.time.clock import Clock, RealClock, ReplayClock


class TestRealClock:
    def test_returns_datetime(self):
        clock = RealClock()
        now = clock.now()
        assert isinstance(now, datetime)

    def test_timezone_aware(self):
        clock = RealClock()
        assert clock.now().tzinfo is not None

    def test_advances_with_time(self):
        clock = RealClock()
        t1 = clock.now()
        t2 = clock.now()
        assert t2 >= t1


class TestReplayClock:
    def _start(self):
        return datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    def test_returns_start_time(self):
        clock = ReplayClock(self._start())
        assert clock.now() == self._start()

    def test_advance_moves_time_forward(self):
        clock = ReplayClock(self._start())
        new_time = self._start() + timedelta(hours=1)
        clock.advance_to(new_time)
        assert clock.now() == new_time

    def test_advance_to_same_time_ok(self):
        clock = ReplayClock(self._start())
        clock.advance_to(self._start())  # no-op, should not raise
        assert clock.now() == self._start()

    def test_advance_backwards_raises(self):
        clock = ReplayClock(self._start())
        past = self._start() - timedelta(seconds=1)
        with pytest.raises(ValueError, match="Cannot go backwards"):
            clock.advance_to(past)

    def test_naive_start_gets_utc(self):
        naive = datetime(2024, 1, 15, 10, 0, 0)
        clock = ReplayClock(naive)
        assert clock.now().tzinfo == timezone.utc

    def test_naive_advance_gets_utc(self):
        clock = ReplayClock(self._start())
        naive_future = datetime(2024, 1, 15, 12, 0, 0)
        clock.advance_to(naive_future)
        assert clock.now().tzinfo == timezone.utc

    def test_does_not_read_wall_clock(self):
        past = datetime(2000, 1, 1, tzinfo=timezone.utc)
        clock = ReplayClock(past)
        assert clock.now().year == 2000  # frozen in the past

    def test_sequential_advances(self):
        clock = ReplayClock(self._start())
        t1 = self._start() + timedelta(minutes=5)
        t2 = self._start() + timedelta(minutes=10)
        t3 = self._start() + timedelta(minutes=15)
        clock.advance_to(t1)
        clock.advance_to(t2)
        clock.advance_to(t3)
        assert clock.now() == t3

    def test_satisfies_clock_protocol(self):
        clock: Clock = ReplayClock(self._start())
        assert isinstance(clock.now(), datetime)
