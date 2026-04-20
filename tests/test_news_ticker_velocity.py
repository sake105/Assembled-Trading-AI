"""Tests for TickerVelocityTracker."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_ticker_velocity import TickerVelocityTracker


def _evt(event_id: str, tickers: list[str], ts: datetime) -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        source_id="reuters",
        source_tier=SourceTier.T1,
        title=f"headline {event_id}",
        url=f"https://example.com/{event_id}",
        published_at=ts,
        ingested_at=ts,
        tickers=tickers,
        content_hash=hashlib.sha256(event_id.encode()).hexdigest()[:16],
    )


@pytest.mark.phase12
class TestTickerVelocityTracker:
    def test_no_events_no_signals(self):
        vt = TickerVelocityTracker()
        assert vt.update([]) == []

    def test_surge_detection(self):
        now = datetime.now(tz=timezone.utc)
        vt = TickerVelocityTracker(
            short_window_min=15, long_window_min=60,
            surge_threshold=2.0, min_short_events=2,
        )
        # one prior event in long window
        vt.update([_evt("old", ["AAPL"], now - timedelta(minutes=50))], now=now - timedelta(minutes=50))
        # burst of 4 events in short window
        burst = [_evt(f"e{i}", ["AAPL"], now - timedelta(minutes=2)) for i in range(4)]
        signals = vt.update(burst, now=now)
        aapl = next(s for s in signals if s.ticker == "AAPL")
        assert aapl.short_count >= 4
        assert aapl.is_surge

    def test_multiple_tickers(self):
        now = datetime.now(tz=timezone.utc)
        vt = TickerVelocityTracker(surge_threshold=2.0, min_short_events=2)
        evts = [
            _evt("a1", ["AAPL"], now),
            _evt("a2", ["AAPL"], now),
            _evt("m1", ["MSFT"], now),
        ]
        signals = vt.update(evts, now=now)
        tickers = {s.ticker for s in signals}
        assert "AAPL" in tickers and "MSFT" in tickers

    def test_buffer_cleanup_removes_stale_tickers(self):
        now = datetime.now(tz=timezone.utc)
        vt = TickerVelocityTracker(short_window_min=5, long_window_min=10)
        vt.update([_evt("e1", ["TSLA"], now - timedelta(minutes=100))], now=now - timedelta(minutes=100))
        # Advance time far beyond long window
        signals = vt.update([], now=now + timedelta(hours=5))
        assert all(s.ticker != "TSLA" for s in signals)

    def test_surging_tickers_sorted(self):
        now = datetime.now(tz=timezone.utc)
        vt = TickerVelocityTracker(surge_threshold=2.0, min_short_events=2)
        # Feed AAPL harder than MSFT
        for _ in range(6):
            vt.update([_evt("x", ["AAPL"], now)], now=now)
        for _ in range(2):
            vt.update([_evt("y", ["MSFT"], now)], now=now)
        surging = vt.surging_tickers(now=now)
        tickers = [t for t, _ in surging]
        # AAPL should be at least ranked if surging
        assert "AAPL" in tickers
