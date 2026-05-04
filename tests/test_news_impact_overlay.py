"""Tests for NewsImpactEstimator and SectorNewsOverlay."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier


def _make_event(
    event_id: str,
    affected_sectors: list[str] | None = None,
    market_direction: str = "bearish",
    severity: float = 6.0,
    news_confidence: float = 0.7,
    event_types: list[str] | None = None,
    hours_ago: float = 1.0,
) -> NewsEvent:
    ts = datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)
    ch = hashlib.sha256(event_id.encode()).hexdigest()[:16]
    return NewsEvent(
        event_id=event_id,
        source_id="reuters",
        source_tier=SourceTier.T1,
        title=f"Test {event_id}",
        url=f"https://example.com/{event_id}",
        published_at=ts,
        ingested_at=ts,
        content_hash=ch,
        affected_sectors=affected_sectors or [],
        event_types=event_types or [],
        market_direction=market_direction,
        severity=severity,
        news_confidence=news_confidence,
    )


class _FakeClassification:
    def __init__(
        self,
        event_types: list[str],
        severity: float,
        market_direction: str,
        time_horizon: str,
        confidence: float,
    ) -> None:
        self.event_types = event_types
        self.severity = severity
        self.market_direction = market_direction
        self.time_horizon = time_horizon
        self.confidence = confidence


# ---------------------------------------------------------------------------
# NewsImpactEstimator
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestNewsImpactEstimator:
    def test_war_escalation_gives_negative_bps(self):
        from src.assembled_core.intel.news_impact_estimator import NewsImpactEstimator

        est = NewsImpactEstimator()
        clf = _FakeClassification(
            event_types=["war_escalation"],
            severity=8.0,
            market_direction="bearish",
            time_horizon="intraday",
            confidence=0.8,
        )
        impact = est.estimate(clf, geo_tags=[], source_tier="T1")
        assert impact.bps < 0
        assert impact.direction == "bearish"
        assert impact.dominant_event_type == "war_escalation"

    def test_geo_premium_applied_for_high_risk_country(self):
        from src.assembled_core.intel.news_impact_estimator import NewsImpactEstimator

        est = NewsImpactEstimator()
        clf = _FakeClassification(
            event_types=["war_escalation"],
            severity=7.0,
            market_direction="bearish",
            time_horizon="intraday",
            confidence=0.7,
        )
        impact_no_geo = est.estimate(clf, geo_tags=[], source_tier="T2")
        impact_with_geo = est.estimate(clf, geo_tags=["RU", "IR"], source_tier="T2")
        # Impact with Russia + Iran should be more negative
        assert impact_with_geo.bps < impact_no_geo.bps
        assert impact_with_geo.geo_premium_bps > 0

    def test_earnings_bullish_gives_positive_bps(self):
        from src.assembled_core.intel.news_impact_estimator import NewsImpactEstimator

        est = NewsImpactEstimator()
        clf = _FakeClassification(
            event_types=["earnings"],
            severity=3.0,
            market_direction="bullish",
            time_horizon="short",
            confidence=0.6,
        )
        impact = est.estimate(clf, source_tier="T1")
        assert impact.bps > 0

    def test_diplomatic_gives_positive_bps(self):
        from src.assembled_core.intel.news_impact_estimator import NewsImpactEstimator

        est = NewsImpactEstimator()
        clf = _FakeClassification(
            event_types=["diplomatic"],
            severity=4.0,
            market_direction="bullish",
            time_horizon="medium",
            confidence=0.55,
        )
        impact = est.estimate(clf, source_tier="T1")
        assert impact.bps > 0

    def test_t3_source_has_lower_impact(self):
        from src.assembled_core.intel.news_impact_estimator import NewsImpactEstimator

        est = NewsImpactEstimator()
        clf = _FakeClassification(
            event_types=["sanctions"],
            severity=7.0,
            market_direction="bearish",
            time_horizon="short",
            confidence=0.7,
        )
        impact_t1 = est.estimate(clf, source_tier="T1")
        impact_t3 = est.estimate(clf, source_tier="T3")
        assert abs(impact_t1.bps) > abs(impact_t3.bps)

    def test_long_horizon_has_lower_impact(self):
        from src.assembled_core.intel.news_impact_estimator import NewsImpactEstimator

        est = NewsImpactEstimator()
        clf_intraday = _FakeClassification(
            event_types=["war_escalation"],
            severity=8.0,
            market_direction="bearish",
            time_horizon="intraday",
            confidence=0.7,
        )
        clf_long = _FakeClassification(
            event_types=["war_escalation"],
            severity=8.0,
            market_direction="bearish",
            time_horizon="long",
            confidence=0.7,
        )
        est_intraday = est.estimate(clf_intraday, source_tier="T1")
        est_long = est.estimate(clf_long, source_tier="T1")
        assert abs(est_intraday.bps) > abs(est_long.bps)

    def test_estimate_batch(self):
        from src.assembled_core.intel.news_impact_estimator import NewsImpactEstimator

        est = NewsImpactEstimator()
        clfs = [
            _FakeClassification(["war_escalation"], 8.0, "bearish", "intraday", 0.8),
            _FakeClassification(["earnings"], 3.0, "bullish", "short", 0.6),
        ]
        results = est.estimate_batch(clfs, source_tier="T1")
        assert len(results) == 2
        assert results[0].bps < 0
        assert results[1].bps > 0

    def test_horizon_days(self):
        from src.assembled_core.intel.news_impact_estimator import NewsImpactEstimator

        est = NewsImpactEstimator()
        clf = _FakeClassification(["sanctions"], 6.0, "bearish", "medium", 0.6)
        impact = est.estimate(clf, source_tier="T2")
        assert impact.horizon_days == 20


# ---------------------------------------------------------------------------
# SectorNewsOverlay
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestSectorNewsOverlay:
    def _make_cluster(self, trigger_val: str, confidence: float):
        class FakeCluster:
            created_at = datetime.now(tz=timezone.utc)

        cl = FakeCluster()
        cl.trigger_type = type("TT", (), {"value": trigger_val})()
        cl.confidence = confidence
        return cl

    def test_returns_dict(self):
        from src.assembled_core.intel.sector_news_overlay import SectorNewsOverlay

        overlay = SectorNewsOverlay()
        clusters = [self._make_cluster("war_escalation", 0.75)]
        result = overlay.compute(clusters=clusters)
        assert isinstance(result, dict)

    def test_war_escalation_boosts_defense(self):
        from src.assembled_core.intel.sector_news_overlay import SectorNewsOverlay

        overlay = SectorNewsOverlay()
        clusters = [self._make_cluster("war_escalation", 0.80)]
        result = overlay.compute(clusters=clusters)
        assert "defense" in result
        assert result["defense"] > 0  # defense is long during war escalation

    def test_war_escalation_hurts_consumer(self):
        from src.assembled_core.intel.sector_news_overlay import SectorNewsOverlay

        overlay = SectorNewsOverlay()
        clusters = [self._make_cluster("war_escalation", 0.80)]
        result = overlay.compute(clusters=clusters)
        assert "consumer" in result
        assert result["consumer"] < 0

    def test_no_clusters_returns_empty(self):
        from src.assembled_core.intel.sector_news_overlay import SectorNewsOverlay

        overlay = SectorNewsOverlay()
        result = overlay.compute(clusters=[])
        assert result == {}

    def test_scores_bounded(self):
        from src.assembled_core.intel.sector_news_overlay import SectorNewsOverlay

        overlay = SectorNewsOverlay()
        clusters = [
            self._make_cluster("war_escalation", 0.90),
            self._make_cluster("energy_disruption", 0.80),
            self._make_cluster("market_stress", 0.70),
        ]
        result = overlay.compute(clusters=clusters)
        for sector, score in result.items():
            assert -1.0 <= score <= 1.0, f"{sector}: {score} out of bounds"

    def test_event_store_contribution(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore
        from src.assembled_core.intel.sector_news_overlay import SectorNewsOverlay

        store = NewsEventStore()
        for i in range(5):
            store.add(
                _make_event(
                    f"ev{i}",
                    affected_sectors=["energy"],
                    market_direction="bearish",
                    severity=7.0,
                    news_confidence=0.7,
                )
            )

        overlay = SectorNewsOverlay()
        result = overlay.compute(
            clusters=[], event_store=store, store_lookback_hours=24.0
        )
        assert "energy" in result
        assert result["energy"] < 0  # bearish energy news

    def test_sanctions_hurts_financials(self):
        from src.assembled_core.intel.sector_news_overlay import SectorNewsOverlay

        overlay = SectorNewsOverlay()
        clusters = [self._make_cluster("sanctions", 0.75)]
        result = overlay.compute(clusters=clusters)
        assert "financials" in result
        assert result["financials"] < 0
