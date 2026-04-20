"""Tests for CorroborationTracker."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_corroboration import CorroborationTracker


def _make_event(
    event_id: str,
    title: str,
    source_id: str = "reuters",
    tier: SourceTier = SourceTier.T1,
    ts: datetime | None = None,
) -> NewsEvent:
    ts = ts or datetime.now(tz=timezone.utc)
    return NewsEvent(
        event_id=event_id,
        source_id=source_id,
        source_tier=tier,
        title=title,
        url=f"https://example.com/{event_id}",
        published_at=ts,
        ingested_at=ts,
        content_hash=hashlib.sha256((title + event_id).encode()).hexdigest()[:16],
    )


@pytest.mark.phase12
class TestCorroborationTracker:
    def test_single_source_low_score(self):
        tracker = CorroborationTracker()
        evt = _make_event("e1", "Russia launches attack on Ukraine")
        tracker.ingest([evt])
        score = tracker.corroboration_score(evt)
        assert score.n_sources == 1
        assert 0 <= score.score < 0.5

    def test_multiple_sources_boost(self):
        tracker = CorroborationTracker(saturation=4.0)
        title = "Russia launches attack on Ukraine"
        evts = [
            _make_event("e1", title, source_id="reuters", tier=SourceTier.T1),
            _make_event("e2", title, source_id="ap", tier=SourceTier.T1),
            _make_event("e3", title, source_id="bbc", tier=SourceTier.T1),
        ]
        tracker.ingest(evts)
        score = tracker.corroboration_score(evts[0])
        assert score.n_sources == 3
        assert score.score > 0.5

    def test_tier_weighting(self):
        tracker = CorroborationTracker(saturation=4.0)
        title = "Sanctions imposed on Russia"
        # Two T0 sources should give more weight than two T3
        t0_evts = [
            _make_event("a1", title, source_id="reuters", tier=SourceTier.T0),
            _make_event("a2", title, source_id="ap", tier=SourceTier.T0),
        ]
        tracker.ingest(t0_evts)
        s_t0 = tracker.corroboration_score(t0_evts[0])

        tracker2 = CorroborationTracker(saturation=4.0)
        t3_evts = [
            _make_event("b1", title, source_id="blog1", tier=SourceTier.T3),
            _make_event("b2", title, source_id="blog2", tier=SourceTier.T3),
        ]
        tracker2.ingest(t3_evts)
        s_t3 = tracker2.corroboration_score(t3_evts[0])

        assert s_t0.score > s_t3.score

    def test_same_source_twice_counted_once(self):
        tracker = CorroborationTracker()
        title = "Breaking news"
        evts = [
            _make_event("e1", title, source_id="reuters", tier=SourceTier.T1),
            _make_event("e2", title, source_id="reuters", tier=SourceTier.T1),
        ]
        tracker.ingest(evts)
        score = tracker.corroboration_score(evts[0])
        assert score.n_sources == 1

    def test_pruning_removes_old_entries(self):
        tracker = CorroborationTracker(retention_hours=1.0)
        now = datetime.now(tz=timezone.utc)
        old_evt = _make_event(
            "old",
            "old news",
            ts=now - timedelta(hours=5),
        )
        fresh_evt = _make_event("fresh", "fresh news", ts=now)
        tracker.ingest([old_evt, fresh_evt])
        dropped = tracker.prune(now=now)
        assert dropped >= 1
        assert tracker.unique_stories() >= 1

    def test_enricher_populates_corroboration_fields(self):
        from src.assembled_core.intel.news_enricher import NewsEventEnricher

        enricher = NewsEventEnricher()
        title = "Multi-source story"
        evts = [
            _make_event("e1", title, source_id="reuters", tier=SourceTier.T1),
            _make_event("e2", title, source_id="ap", tier=SourceTier.T1),
        ]
        enriched = enricher.enrich(evts)
        assert all(e.corroboration_n_sources >= 2 for e in enriched)
        assert all(e.corroboration_score > 0 for e in enriched)
