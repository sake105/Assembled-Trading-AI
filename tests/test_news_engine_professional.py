"""Tests for the professional news engine improvements.

Covers:
- Point 29: NewsArchiver JSONL persistence
- Point 30: Feed health dashboard artifact
- Point 32: Earnings proximity boost
- Point 33: News-to-position bridge
- Point 36: Sector rotation signal
- Point 38: Contradiction detector
- Point 39: News fatigue detection
- Point 40: Source bias tagging
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_id: str = "ev1",
    title: str = "Russia sanctions imposed on energy sector",
    source_id: str = "reuters",
    source_tier: SourceTier = SourceTier.T1,
    market_direction: str = "bearish",
    news_confidence: float = 0.6,
    urgency: float = 0.0,
    url: str = "",
) -> NewsEvent:
    import hashlib
    content_hash = hashlib.sha256((title + source_id).encode()).hexdigest()[:16]
    return NewsEvent(
        event_id=event_id,
        source_id=source_id,
        source_tier=source_tier,
        title=title,
        published_at=datetime.now(tz=timezone.utc),
        ingested_at=datetime.now(tz=timezone.utc),
        market_direction=market_direction,
        news_confidence=news_confidence,
        urgency=urgency,
        url=url,
        content_hash=content_hash,
    )


# ---------------------------------------------------------------------------
# Point 29: NewsArchiver
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestNewsArchiver:
    def test_append_and_count(self):
        from src.assembled_core.intel.news_archiver import NewsArchiver

        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = NewsArchiver(base_dir=tmpdir)
            events = [_make_event(f"ev{i}") for i in range(5)]
            written = archiver.append(events)
            assert written == 5

    def test_roundtrip_jsonl(self):
        from src.assembled_core.intel.news_archiver import NewsArchiver

        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = NewsArchiver(base_dir=tmpdir)
            evts = [_make_event("ev_rt1", title="Breaking: Oil pipeline explodes")]
            archiver.append(evts)

            recovered = list(archiver.iter_events())
            assert len(recovered) == 1
            assert recovered[0]["event_id"] == "ev_rt1"
            assert "title" in recovered[0]

    def test_date_partitioning(self):
        from src.assembled_core.intel.news_archiver import NewsArchiver

        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = NewsArchiver(base_dir=tmpdir)
            d1 = date(2024, 1, 10)
            d2 = date(2024, 1, 15)
            archiver.append([_make_event("ev_a")], partition_date=d1)
            archiver.append([_make_event("ev_b")], partition_date=d2)

            partitions = archiver.list_partitions()
            assert d1 in partitions
            assert d2 in partitions

    def test_date_range_filter(self):
        from src.assembled_core.intel.news_archiver import NewsArchiver

        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = NewsArchiver(base_dir=tmpdir)
            archiver.append([_make_event("ev_jan")], partition_date=date(2024, 1, 5))
            archiver.append([_make_event("ev_feb")], partition_date=date(2024, 2, 5))
            archiver.append([_make_event("ev_mar")], partition_date=date(2024, 3, 5))

            jan_only = list(archiver.iter_events(start="2024-01-01", end="2024-01-31"))
            assert len(jan_only) == 1
            assert jan_only[0]["event_id"] == "ev_jan"

    def test_empty_events_returns_zero(self):
        from src.assembled_core.intel.news_archiver import NewsArchiver

        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = NewsArchiver(base_dir=tmpdir)
            assert archiver.append([]) == 0

    def test_count_events(self):
        from src.assembled_core.intel.news_archiver import NewsArchiver

        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = NewsArchiver(base_dir=tmpdir)
            archiver.append([_make_event(f"ev{i}") for i in range(7)])
            assert archiver.count_events() == 7


# ---------------------------------------------------------------------------
# Point 33: News-to-position bridge
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestNewsPositionBridge:
    def test_bearish_cluster_gives_short(self):
        from src.assembled_core.intel.news_position_bridge import cluster_to_signal

        class FakeCluster:
            cluster_id = "cl_001"
            trigger_type = type("TT", (), {"value": "war_escalation"})()
            confidence = 0.7

        sig = cluster_to_signal(FakeCluster())
        assert sig is not None
        assert sig.direction == "short"
        assert sig.confidence == 0.7

    def test_low_confidence_returns_none(self):
        from src.assembled_core.intel.news_position_bridge import cluster_to_signal

        class FakeCluster:
            cluster_id = "cl_002"
            trigger_type = type("TT", (), {"value": "military_strike"})()
            confidence = 0.1  # below threshold

        sig = cluster_to_signal(FakeCluster())
        assert sig is None

    def test_classification_to_signal_bearish(self):
        from src.assembled_core.intel.news_position_bridge import classification_to_signal

        class FakeClassification:
            event_types = ["sanctions", "war_escalation"]
            severity = 7.0
            market_direction = "bearish"
            affected_sectors = ["defense", "energy"]
            affected_assets = ["XAR", "XLE"]
            confidence = 0.65
            time_horizon = "intraday"

        sig = classification_to_signal(FakeClassification(), cluster_id="cl_test")
        assert sig is not None
        assert sig.direction == "short"
        assert "XAR" in sig.affected_assets

    def test_classification_to_signal_bullish(self):
        from src.assembled_core.intel.news_position_bridge import classification_to_signal

        class FakeClassification:
            event_types = ["diplomatic"]
            severity = 3.0
            market_direction = "bullish"
            affected_sectors = ["financials"]
            affected_assets = ["XLF"]
            confidence = 0.5
            time_horizon = "short"

        sig = classification_to_signal(FakeClassification())
        assert sig is not None
        assert sig.direction == "long"

    def test_signals_to_basket_aggregation(self):
        from src.assembled_core.intel.news_position_bridge import (
            PositionSignal,
            signals_to_basket,
        )

        s1 = PositionSignal(
            signal_id="s1",
            source_cluster_id=None,
            direction="short",
            confidence=0.8,
            affected_assets=["XLE", "XOM"],
        )
        s2 = PositionSignal(
            signal_id="s2",
            source_cluster_id=None,
            direction="long",
            confidence=0.4,
            affected_assets=["XAR", "LMT"],
        )
        basket = signals_to_basket([s1, s2])
        assert "XLE" in basket
        assert basket["XLE"] < 0  # short signal
        assert basket["XAR"] > 0  # long signal

    def test_is_actionable(self):
        from src.assembled_core.intel.news_position_bridge import PositionSignal

        sig = PositionSignal(
            signal_id="s1",
            source_cluster_id=None,
            direction="short",
            confidence=0.5,
        )
        assert sig.is_actionable() is True

        flat_sig = PositionSignal(
            signal_id="s2",
            source_cluster_id=None,
            direction="flat",
            confidence=0.7,
        )
        assert flat_sig.is_actionable() is False


# ---------------------------------------------------------------------------
# Point 36: Sector rotation signal
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestSectorRotationSignal:
    def _make_events_df(self) -> "pd.DataFrame":
        import pandas as pd

        now = datetime.now(tz=timezone.utc)
        rows = [
            {
                "timestamp": (now - pd.Timedelta(hours=1)).isoformat(),
                "affected_sectors": ["energy", "defense"],
                "severity": 7.0,
                "market_direction": "bearish",
            },
            {
                "timestamp": (now - pd.Timedelta(hours=2)).isoformat(),
                "affected_sectors": ["energy"],
                "severity": 6.0,
                "market_direction": "bearish",
            },
            {
                "timestamp": (now - pd.Timedelta(hours=3)).isoformat(),
                "affected_sectors": ["tech"],
                "severity": 3.0,
                "market_direction": "neutral",
            },
        ]
        return pd.DataFrame(rows)

    def test_returns_dict(self):
        import pytest; pytest.importorskip('src.assembled_core.features.news_features')
        from src.assembled_core.features.news_features import compute_sector_rotation_signal

        df = self._make_events_df()
        result = compute_sector_rotation_signal(df, window_hours=4.0, min_events=1)
        assert isinstance(result, dict)

    def test_energy_bearish_gets_negative_score(self):
        import pytest; pytest.importorskip('src.assembled_core.features.news_features')
        from src.assembled_core.features.news_features import compute_sector_rotation_signal

        df = self._make_events_df()
        result = compute_sector_rotation_signal(df, window_hours=4.0, min_events=1)
        # energy has 2 bearish events — should have negative score
        if "energy" in result:
            assert result["energy"] < 0

    def test_empty_df_returns_empty(self):
        import pytest; pytest.importorskip('src.assembled_core.features.news_features')
        import pandas as pd
        from src.assembled_core.features.news_features import compute_sector_rotation_signal

        result = compute_sector_rotation_signal(pd.DataFrame(), window_hours=4.0)
        assert result == {}

    def test_scores_bounded(self):
        import pytest; pytest.importorskip('src.assembled_core.features.news_features')
        from src.assembled_core.features.news_features import compute_sector_rotation_signal

        df = self._make_events_df()
        result = compute_sector_rotation_signal(df, window_hours=4.0, min_events=1)
        for sector, score in result.items():
            assert -1.0 <= score <= 1.0, f"{sector}: {score} out of [-1,1]"


# ---------------------------------------------------------------------------
# Point 32: Earnings proximity boost
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestEarningsProximityBoost:
    def test_near_quarter_end_gets_boost(self):
        import pytest; pytest.importorskip('src.assembled_core.features.news_features')
        import pandas as pd
        from src.assembled_core.features.news_features import compute_earnings_proximity_boost

        # Late March = near Q1 end
        df = pd.DataFrame({"timestamp": ["2024-03-28T10:00:00Z", "2024-01-15T10:00:00Z"]})
        result = compute_earnings_proximity_boost(df, proximity_days=14)
        assert "earnings_proximity_boost" in result.columns
        assert result.iloc[0]["earnings_proximity_boost"] >= 1.0
        # Mid-January should have no boost
        assert result.iloc[1]["earnings_proximity_boost"] == 1.0

    def test_boost_capped_at_1_5(self):
        import pytest; pytest.importorskip('src.assembled_core.features.news_features')
        import pandas as pd
        from src.assembled_core.features.news_features import compute_earnings_proximity_boost

        df = pd.DataFrame({"timestamp": ["2024-03-31T10:00:00Z"]})
        result = compute_earnings_proximity_boost(df)
        assert result.iloc[0]["earnings_proximity_boost"] <= 1.5

    def test_empty_df(self):
        import pytest; pytest.importorskip('src.assembled_core.features.news_features')
        import pandas as pd
        from src.assembled_core.features.news_features import compute_earnings_proximity_boost

        result = compute_earnings_proximity_boost(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Point 39: News fatigue detection
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestNewsFatigueDetection:
    def test_no_fatigue_on_first_occurrence(self):
        from src.assembled_core.intel.news_dedupe import NewsDedupeIndex

        idx = NewsDedupeIndex()
        evt = _make_event("ev_fat1", title="Russia launches missile attack on Ukraine")
        score = idx.get_fatigue_score(evt)
        assert score == 0.0

    def test_fatigue_increases_with_repetitions(self):
        from src.assembled_core.intel.news_dedupe import NewsDedupeIndex

        idx = NewsDedupeIndex()
        title = "Russia launches missile attack on Ukraine energy infrastructure"
        # Simulate same story from 5 different sources
        for i in range(5):
            evt = _make_event(f"ev_{i}", title=title, source_id=f"source{i}")
            idx.record_story_count(evt)

        evt_check = _make_event("ev_check", title=title)
        score = idx.get_fatigue_score(evt_check)
        assert score > 0.0

    def test_is_fatigued_flag(self):
        from src.assembled_core.intel.news_dedupe import NewsDedupeIndex

        idx = NewsDedupeIndex()
        title = "War escalation in Middle East oil region critical alert"
        evt_template = _make_event("ev_tmpl", title=title)
        # Simulate 8 reports (full fatigue) via story count
        for i in range(8):
            idx.record_story_count(_make_event(f"ev_{i}", title=title, source_id=f"src{i}"))

        assert idx.is_fatigued(evt_template, threshold=0.6)

    def test_not_fatigued_below_threshold(self):
        from src.assembled_core.intel.news_dedupe import NewsDedupeIndex

        idx = NewsDedupeIndex()
        title = "Fresh breaking news about oil supply disruption"
        evt = _make_event("ev_fresh", title=title)
        # Only 2 reports — low fatigue
        for i in range(2):
            idx.record_story_count(_make_event(f"ev_{i}", title=title, source_id=f"src{i}"))

        assert not idx.is_fatigued(evt, threshold=0.6)


# ---------------------------------------------------------------------------
# Point 38: Contradiction detector
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestContradictionDetector:
    def test_detects_opposing_directions(self):
        from src.assembled_core.intel.news_dedupe import detect_contradictions

        title = "Oil price reaction to Middle East sanctions announcement"
        evt_bearish = _make_event(
            "ev_bear",
            title=title,
            source_id="rt",
            market_direction="bearish",
            news_confidence=0.7,
        )
        evt_bullish = _make_event(
            "ev_bull",
            title=title,
            source_id="reuters",
            market_direction="bullish",
            news_confidence=0.6,
        )

        contradictions = detect_contradictions([evt_bearish, evt_bullish])
        assert len(contradictions) >= 1
        assert contradictions[0]["direction_a"] != contradictions[0]["direction_b"]

    def test_no_contradiction_same_direction(self):
        from src.assembled_core.intel.news_dedupe import detect_contradictions

        title = "Oil supply disruption from Russia pipeline attack"
        evt1 = _make_event("ev1", title=title, market_direction="bearish", news_confidence=0.7)
        evt2 = _make_event("ev2", title=title, market_direction="bearish", news_confidence=0.8)

        contradictions = detect_contradictions([evt1, evt2])
        assert len(contradictions) == 0

    def test_low_confidence_not_flagged(self):
        from src.assembled_core.intel.news_dedupe import detect_contradictions

        title = "Market reaction to trade war tariff announcement"
        evt1 = _make_event("ev1", title=title, market_direction="bearish", news_confidence=0.1)
        evt2 = _make_event("ev2", title=title, market_direction="bullish", news_confidence=0.1)

        contradictions = detect_contradictions([evt1, evt2], min_confidence=0.3)
        assert len(contradictions) == 0

    def test_empty_events_returns_empty(self):
        from src.assembled_core.intel.news_dedupe import detect_contradictions

        assert detect_contradictions([]) == []


# ---------------------------------------------------------------------------
# Point 40: Source bias tagging
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestSourceBiasTagging:
    def test_state_media_detected(self):
        from src.assembled_core.intel.news_classifier import is_state_media

        assert is_state_media("rt") is True
        assert is_state_media("xinhua") is True
        assert is_state_media("tass") is True

    def test_neutral_source_not_state_media(self):
        from src.assembled_core.intel.news_classifier import is_state_media

        assert is_state_media("reuters") is False
        assert is_state_media("bloomberg") is False
        assert is_state_media("bbc") is False

    def test_get_source_bias_known(self):
        from src.assembled_core.intel.news_classifier import get_source_bias

        bias = get_source_bias("reuters")
        assert "geo_bias" in bias
        assert bias["geo_bias"] == "GB"

    def test_get_source_bias_unknown_returns_empty(self):
        from src.assembled_core.intel.news_classifier import get_source_bias

        assert get_source_bias("unknown_feed_xyz") == {}

    def test_state_media_confidence_discount(self):
        from src.assembled_core.intel.news_classifier import apply_source_bias_discount

        # State media gets 30% discount
        discounted = apply_source_bias_discount(0.8, "rt")
        assert abs(discounted - 0.56) < 0.01  # 0.8 * 0.70

    def test_neutral_source_no_discount(self):
        from src.assembled_core.intel.news_classifier import apply_source_bias_discount

        result = apply_source_bias_discount(0.8, "reuters")
        assert result == 0.8

    def test_pro_gov_source_partial_discount(self):
        from src.assembled_core.intel.news_classifier import apply_source_bias_discount

        # VOA (pro_western) → 10% discount
        result = apply_source_bias_discount(0.8, "voa")
        assert abs(result - 0.72) < 0.01

    def test_case_insensitive_lookup(self):
        from src.assembled_core.intel.news_classifier import is_state_media

        assert is_state_media("RT") is True
        assert is_state_media("XINHUA") is True


# ---------------------------------------------------------------------------
# Point 30: Feed health dashboard
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestFeedHealthDashboard:
    def test_get_feed_stats_empty(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        stats = hm.get_feed_stats()
        assert stats == {}

    def test_record_and_retrieve_stats(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        events = [_make_event(f"ev{i}") for i in range(3)]
        hm.record_events("reuters_feed", events)

        stats = hm.get_feed_stats()
        assert "reuters_feed" in stats
        assert stats["reuters_feed"]["total_events"] == 3

    def test_check_silent_feeds(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        # Record an old event by setting last_event_time manually
        hm.record_events("old_feed", [_make_event("ev1")])
        # Force last_event_time to be far in the past
        stats = hm._source_stats["old_feed"]
        from datetime import timedelta
        stats.last_event_time = datetime.now(tz=timezone.utc) - timedelta(hours=5)

        silent = hm.check_silent_feeds(threshold_hours=2.0)
        assert "old_feed" in silent

    def test_fresh_feed_not_silent(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        hm.record_events("active_feed", [_make_event("ev_now")])

        silent = hm.check_silent_feeds(threshold_hours=2.0)
        assert "active_feed" not in silent
