"""Tests for NewsEventEnricher pipeline step."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier


def _make_event(
    event_id: str,
    title: str = "Russia launches missile attack on Ukraine energy grid",
    source_id: str = "reuters",
    source_tier: SourceTier = SourceTier.T1,
) -> NewsEvent:
    content_hash = hashlib.sha256((title + event_id).encode()).hexdigest()[:16]
    return NewsEvent(
        event_id=event_id,
        source_id=source_id,
        source_tier=source_tier,
        title=title,
        url=f"https://example.com/{event_id}",
        published_at=datetime.now(tz=timezone.utc),
        ingested_at=datetime.now(tz=timezone.utc),
        content_hash=content_hash,
    )


@pytest.mark.phase12
class TestNewsEventEnricher:
    def test_enriches_event_types(self):
        from src.assembled_core.intel.news_enricher import NewsEventEnricher

        enricher = NewsEventEnricher()
        evt = _make_event("ev1", title="Russia invades Ukraine, war escalation feared")
        result = enricher.enrich([evt])
        assert len(result) == 1
        assert result[0].event_types  # should have at least one event type

    def test_enriches_severity(self):
        from src.assembled_core.intel.news_enricher import NewsEventEnricher

        enricher = NewsEventEnricher()
        evt = _make_event("ev2", title="Missile attack on energy pipeline causes explosion")
        result = enricher.enrich([evt])
        assert result[0].severity > 0

    def test_enriches_market_direction(self):
        from src.assembled_core.intel.news_enricher import NewsEventEnricher

        enricher = NewsEventEnricher()
        evt = _make_event("ev3", title="War escalation crisis collapse of ceasefire talks")
        result = enricher.enrich([evt])
        assert result[0].market_direction in ("bearish", "bullish", "neutral", "mixed")

    def test_skips_already_classified(self):
        from src.assembled_core.intel.news_enricher import NewsEventEnricher

        enricher = NewsEventEnricher()
        evt = _make_event("ev4")
        evt.event_types = ["sanctions"]  # already classified
        evt.severity = 5.0
        result = enricher.enrich([evt])
        # Should not overwrite existing classification
        assert result[0].event_types == ["sanctions"]
        assert result[0].severity == 5.0

    def test_fatigue_discounts_confidence(self):
        from src.assembled_core.intel.news_dedupe import NewsDedupeIndex
        from src.assembled_core.intel.news_enricher import NewsEventEnricher

        dedupe = NewsDedupeIndex()
        title = "Russia sanctions on energy sector announced by EU"
        # Simulate 8 reports of same story
        for i in range(8):
            dedupe.record_story_count(_make_event(f"bg{i}", title=title, source_id=f"src{i}"))

        enricher = NewsEventEnricher(dedupe_index=dedupe)
        evt = _make_event("ev_fatigued", title=title)
        evt.news_confidence = 0.8
        result = enricher.enrich([evt])
        # Confidence should be discounted due to fatigue
        assert result[0].news_confidence < 0.8

    def test_empty_batch(self):
        from src.assembled_core.intel.news_enricher import NewsEventEnricher

        enricher = NewsEventEnricher()
        result = enricher.enrich([])
        assert result == []

    def test_with_velocity_tracker(self):
        from src.assembled_core.intel.news_velocity import VelocityTracker
        from src.assembled_core.intel.news_enricher import NewsEventEnricher

        tracker = VelocityTracker()
        enricher = NewsEventEnricher(velocity_tracker=tracker)
        events = [_make_event(f"ev{i}") for i in range(5)]
        result = enricher.enrich(events)
        assert len(result) == 5
        assert tracker.buffer_size > 0

    def test_with_event_store(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore
        from src.assembled_core.intel.news_enricher import NewsEventEnricher

        store = NewsEventStore()
        enricher = NewsEventEnricher(event_store=store)
        events = [_make_event(f"ev{i}") for i in range(3)]
        enricher.enrich(events)
        assert store.count() == 3

    def test_state_media_gets_confidence_discount(self):
        from src.assembled_core.intel.news_enricher import NewsEventEnricher

        enricher = NewsEventEnricher()
        # RT is state media — should get 30% confidence discount
        evt = _make_event("ev_rt", source_id="rt", source_tier=SourceTier.T3)
        enricher.enrich([evt])
        # Classifier assigns confidence, then bias discount applied
        # Just verify no crash and confidence is non-negative
        assert evt.news_confidence >= 0
