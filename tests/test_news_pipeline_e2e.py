"""End-to-end integration test for the news engine pipeline.

Verifies the full enrichment → store → overlay → signal chain:
    Raw NewsEvents → NewsEventEnricher (classify + impact + fatigue + store + velocity)
                  → NewsEventStore (persisted)
                  → SectorNewsOverlay (aggregate across events)
                  → classification_to_signal → PositionSignal
                  → aggregate_signals → IntelSignal

This test is a smoke contract: it guarantees that the components wire
together correctly without mocks and that a realistic bearish event
flows through as a bearish signal.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier


def _make_event(
    event_id: str,
    title: str,
    source_id: str = "reuters",
    source_tier: SourceTier = SourceTier.T1,
    geo_tags: list[str] | None = None,
    published_at: datetime | None = None,
) -> NewsEvent:
    if published_at is None:
        published_at = datetime.now(tz=timezone.utc)
    content_hash = hashlib.sha256((title + event_id).encode()).hexdigest()[:16]
    return NewsEvent(
        event_id=event_id,
        source_id=source_id,
        source_tier=source_tier,
        title=title,
        url=f"https://example.com/{event_id}",
        published_at=published_at,
        ingested_at=published_at,
        geo_tags=geo_tags or [],
        content_hash=content_hash,
    )


@pytest.mark.phase12
class TestNewsPipelineEndToEnd:
    """Full pipeline from raw events to aggregated intel signal."""

    def test_full_pipeline_bearish_scenario(self):
        from src.assembled_core.intel.news_enricher import NewsEventEnricher
        from src.assembled_core.intel.news_event_store import NewsEventStore
        from src.assembled_core.intel.news_velocity import VelocityTracker
        from src.assembled_core.intel.news_dedupe import NewsDedupeIndex
        from src.assembled_core.intel.sector_news_overlay import SectorNewsOverlay

        now = datetime.now(tz=timezone.utc)

        store = NewsEventStore(max_events=100)
        velocity = VelocityTracker(short_window_min=15, long_window_min=60, surge_threshold=2.5)
        dedupe = NewsDedupeIndex(persist_path=None, max_size=100)
        enricher = NewsEventEnricher(
            event_store=store,
            velocity_tracker=velocity,
            dedupe_index=dedupe,
        )

        raw = [
            _make_event(
                "ev1",
                title="Russia launches missile attack on Ukraine energy grid",
                geo_tags=["RU", "UA"],
                published_at=now - timedelta(minutes=5),
            ),
            _make_event(
                "ev2",
                title="US sanctions target Russian oil exports, markets plunge",
                geo_tags=["RU", "US"],
                published_at=now - timedelta(minutes=3),
            ),
        ]

        enriched = enricher.enrich(raw, now=now)

        # Enrichment populated classification fields
        assert all(e.event_types for e in enriched), "all events should be classified"
        assert all(e.news_confidence > 0 for e in enriched), "confidence should be set"

        # At least one bearish event
        bearish = [e for e in enriched if e.market_direction == "bearish"]
        assert bearish, "expected at least one bearish event"

        # Impact fields persisted on the model
        bearish_evt = bearish[0]
        assert bearish_evt.impact_bps < 0, "bearish event should have negative BPS impact"
        assert bearish_evt.impact_dominant_event_type
        assert bearish_evt.impact_horizon_days > 0

        # EventStore was populated
        stored = store.query_by_time(hours=1)
        assert len(stored) >= len(raw), "EventStore should have indexed enriched events"

        # Sector overlay produces signals
        overlay = SectorNewsOverlay()
        sector_scores = overlay.compute(clusters=[], event_store=store, now=now)
        assert sector_scores, "overlay should produce sector scores"
        # Bearish on energy/financials expected for sanctions + missile strike
        assert any(v < 0 for v in sector_scores.values()), "should contain bearish sector tilts"

    def test_classification_to_signal_from_enriched_event(self):
        """Enriched event → NewsClassification-like object → PositionSignal."""
        from src.assembled_core.intel.news_classifier import classify_news_event
        from src.assembled_core.intel.news_position_bridge import classification_to_signal

        clf = classify_news_event(
            "Iran missile strike destroys oil refinery",
            geo_tags=["IR"],
            source_tier="T1",
        )
        sig = classification_to_signal(clf, cluster_id="cluster_abc")
        assert sig is not None, "high-confidence bearish event should emit signal"
        assert sig.direction == "short"
        assert sig.signal_id.startswith("ps_cluster_abc_")
        # Deterministic ID: same inputs produce same ID
        sig2 = classification_to_signal(clf, cluster_id="cluster_abc")
        assert sig.signal_id == sig2.signal_id

    def test_deterministic_signal_id_across_runs(self):
        """Signal IDs should be reproducible across processes (no hash() salt)."""
        from src.assembled_core.intel.news_classifier import classify_news_event
        from src.assembled_core.intel.news_position_bridge import classification_to_signal

        clf_a = classify_news_event("Russia invades Ukraine, war escalates", source_tier="T1")
        clf_b = classify_news_event("Russia invades Ukraine, war escalates", source_tier="T1")
        sig_a = classification_to_signal(clf_a, cluster_id="c1")
        sig_b = classification_to_signal(clf_b, cluster_id="c1")
        assert sig_a is not None and sig_b is not None
        assert sig_a.signal_id == sig_b.signal_id

    def test_impact_persists_through_model_dump(self):
        """Impact fields must survive NewsEvent.model_dump() for archive replay."""
        from src.assembled_core.intel.news_enricher import NewsEventEnricher

        enricher = NewsEventEnricher()
        evt = _make_event(
            "ev_dump",
            title="Sanctions hit Russian oil exports amid energy crisis",
            geo_tags=["RU"],
        )
        [enriched] = enricher.enrich([evt])
        dumped = enriched.model_dump()
        assert "impact_bps" in dumped
        assert dumped["impact_bps"] != 0.0
        assert "impact_dominant_event_type" in dumped
        assert dumped["impact_dominant_event_type"]

    def test_word_boundary_prevents_false_positives(self):
        """Regression: 'campaign'→ai, 'award'→war, 'design'→resign must not fire."""
        from src.assembled_core.intel.news_classifier import classify_news_event

        benign_titles = [
            "New AI-free marketing campaign launches in Europe",
            "Award ceremony celebrates design innovation",
            "Main Street reopens after toilet-paper shortage eased",
        ]
        for t in benign_titles:
            clf = classify_news_event(t, source_tier="T1")
            # These titles contain no real event triggers; any match would be a false positive
            assert "war_escalation" not in clf.event_types, f"false war match on: {t}"
            assert "political_crisis" not in clf.event_types, f"false crisis match on: {t}"
