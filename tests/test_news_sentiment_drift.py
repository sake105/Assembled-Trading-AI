"""Tests for SentimentDriftTracker."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_sentiment_drift import SentimentDriftTracker


def _evt(
    event_id: str,
    tickers: list[str],
    sent: float,
    ts: datetime,
    sectors: list[str] | None = None,
) -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        source_id="reuters",
        source_tier=SourceTier.T1,
        title=f"headline {event_id}",
        url=f"https://example.com/{event_id}",
        published_at=ts,
        ingested_at=ts,
        tickers=tickers,
        affected_sectors=sectors or [],
        sentiment_score=sent,
        content_hash=hashlib.sha256(event_id.encode()).hexdigest()[:16],
    )


@pytest.mark.fast
class TestSentimentDriftTracker:
    def test_empty_report(self):
        tr = SentimentDriftTracker()
        assert tr.report() == []

    def test_below_min_events_skipped(self):
        tr = SentimentDriftTracker(min_events=3)
        now = datetime.now(tz=timezone.utc)
        tr.update([_evt("a", ["AAPL"], -0.2, now)], now=now)
        assert tr.report(now=now) == []

    def test_deteriorating_sentiment(self):
        tr = SentimentDriftTracker(window_min=60, min_events=3, slope_threshold=0.001)
        now = datetime.now(tz=timezone.utc)
        # sentiment drops over time for AAPL
        evts = [
            _evt("e1", ["AAPL"], 0.5, now - timedelta(minutes=50)),
            _evt("e2", ["AAPL"], 0.0, now - timedelta(minutes=30)),
            _evt("e3", ["AAPL"], -0.4, now - timedelta(minutes=5)),
        ]
        tr.update(evts, now=now)
        rep = tr.report(now=now)
        aapl = next(e for e in rep if e.key == "TICKER:AAPL")
        assert aapl.drift_direction == "deteriorating"
        assert aapl.slope < 0

    def test_improving_sentiment(self):
        tr = SentimentDriftTracker(window_min=60, min_events=3, slope_threshold=0.001)
        now = datetime.now(tz=timezone.utc)
        evts = [
            _evt("e1", ["MSFT"], -0.4, now - timedelta(minutes=50)),
            _evt("e2", ["MSFT"], 0.0, now - timedelta(minutes=30)),
            _evt("e3", ["MSFT"], 0.5, now - timedelta(minutes=5)),
        ]
        tr.update(evts, now=now)
        rep = tr.report(now=now)
        msft = next(e for e in rep if e.key == "TICKER:MSFT")
        assert msft.drift_direction == "improving"
        assert msft.slope > 0

    def test_sector_aggregation(self):
        tr = SentimentDriftTracker(min_events=2, slope_threshold=0.001)
        now = datetime.now(tz=timezone.utc)
        evts = [
            _evt("e1", [], 0.1, now - timedelta(minutes=30), sectors=["tech"]),
            _evt("e2", [], -0.3, now - timedelta(minutes=10), sectors=["tech"]),
            _evt("e3", [], -0.5, now, sectors=["tech"]),
        ]
        tr.update(evts, now=now)
        rep = tr.report(now=now)
        keys = [e.key for e in rep]
        assert "SECTOR:tech" in keys

    def test_window_pruning(self):
        tr = SentimentDriftTracker(window_min=10, min_events=2)
        now = datetime.now(tz=timezone.utc)
        tr.update(
            [
                _evt("old1", ["X"], -0.5, now - timedelta(hours=5)),
                _evt("old2", ["X"], -0.5, now - timedelta(hours=5)),
            ],
            now=now - timedelta(hours=5),
        )
        rep = tr.report(now=now)
        assert all(e.key != "TICKER:X" for e in rep)

    def test_flat_sentiment(self):
        tr = SentimentDriftTracker(min_events=3, slope_threshold=0.05)
        now = datetime.now(tz=timezone.utc)
        evts = [
            _evt("e1", ["CON"], 0.1, now - timedelta(minutes=30)),
            _evt("e2", ["CON"], 0.1, now - timedelta(minutes=20)),
            _evt("e3", ["CON"], 0.1, now - timedelta(minutes=10)),
        ]
        tr.update(evts, now=now)
        rep = tr.report(now=now)
        con = next(e for e in rep if e.key == "TICKER:CON")
        assert con.drift_direction == "flat"
