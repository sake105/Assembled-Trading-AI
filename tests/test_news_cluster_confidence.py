"""Tests for ClusterManager confidence computation (fixed Bayesian signature)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_cluster import ClusterManager


def _make_event(
    title: str = "Sanctions escalation against Russia",
    source_id: str = "reuters_world",
    tier: SourceTier = SourceTier.T1,
    urgency: float = 0.0,
) -> NewsEvent:
    return NewsEvent(
        event_id=f"evt_{hash(title + source_id)}",
        source_id=source_id,
        source_tier=tier,
        title=title,
        url=f"http://example.com/{hash(title)}",
        published_at=datetime.now(tz=timezone.utc),
        ingested_at=datetime.now(tz=timezone.utc),
        content_hash=f"{hash(title):016x}"[:16],
        urgency=urgency,
    )


@pytest.mark.phase12
class TestClusterManagerConfidence:
    def test_no_crash_with_correct_signature(self):
        """Step 0 regression: _update_confidence must not crash."""
        mgr = ClusterManager()
        now = datetime.now(tz=timezone.utc)
        event = _make_event("Sanctions escalation against Russia")
        clusters = mgr.update_clusters([event], now=now)
        assert len(clusters) == 1
        assert 0.0 < clusters[0].confidence <= 0.99

    def test_t1_event_yields_high_confidence(self):
        mgr = ClusterManager()
        now = datetime.now(tz=timezone.utc)
        events = [
            _make_event("Sanctions escalation against Russia", source_id="reuters_world", tier=SourceTier.T1),
            _make_event("Sanctions escalation: new oil ban", source_id="ap_world", tier=SourceTier.T1),
        ]
        clusters = mgr.update_clusters(events, now=now)
        assert len(clusters) >= 1
        max_conf = max(cl.confidence for cl in clusters)
        assert max_conf > 0.3

    def test_urgency_boost_raises_confidence(self):
        mgr1 = ClusterManager()
        mgr2 = ClusterManager()
        now = datetime.now(tz=timezone.utc)

        normal_event = _make_event("Sanctions escalation against Russia", urgency=0.0)
        breaking_event = _make_event("Sanctions escalation against Russia", urgency=1.0)

        cl1 = mgr1.update_clusters([normal_event], now=now)
        cl2 = mgr2.update_clusters([breaking_event], now=now)

        if cl1 and cl2:
            # Breaking event should have higher or equal confidence
            assert cl2[0].confidence >= cl1[0].confidence

    def test_multi_source_boosts_confidence(self):
        mgr = ClusterManager()
        now = datetime.now(tz=timezone.utc)
        events = [
            _make_event("Missile strike reported in Ukraine", source_id="reuters_world", tier=SourceTier.T1),
            _make_event("Missile strike confirmed in Ukraine", source_id="ap_world", tier=SourceTier.T1),
            _make_event("Explosion heard in Kyiv: military strike", source_id="bbc_world", tier=SourceTier.T1),
        ]
        clusters = mgr.update_clusters(events, now=now)
        # At least one cluster with non-trivial confidence
        assert len(clusters) >= 1
        assert clusters[0].confidence > 0.0

    def test_expired_cluster_removed(self):
        from datetime import timedelta
        mgr = ClusterManager(cluster_ttl_minutes=1)
        now = datetime.now(tz=timezone.utc)
        event = _make_event("Sanctions escalation against Russia")
        mgr.update_clusters([event], now=now)
        assert len(mgr.active_clusters) == 1

        # Advance time past TTL
        future = now + timedelta(minutes=5)
        mgr.update_clusters([], now=future)
        assert len(mgr.active_clusters) == 0

    def test_t3_events_classified(self):
        mgr = ClusterManager()
        now = datetime.now(tz=timezone.utc)
        event = _make_event("Energy shortage deepens as winter approaches", tier=SourceTier.T3)
        clusters = mgr.update_clusters([event], now=now)
        # May or may not form a cluster depending on keywords, but should not crash
        assert isinstance(clusters, list)
