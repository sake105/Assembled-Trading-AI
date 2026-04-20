"""Tests for NewsEventStore and NewsSignalAggregator."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier


def _make_event(
    event_id: str,
    title: str = "Test news event",
    source_id: str = "reuters",
    source_tier: SourceTier = SourceTier.T1,
    tickers: list[str] | None = None,
    affected_sectors: list[str] | None = None,
    event_types: list[str] | None = None,
    geo_tags: list[str] | None = None,
    severity: float = 3.0,
    news_confidence: float = 0.6,
    hours_ago: float = 0.5,
) -> NewsEvent:
    content_hash = hashlib.sha256((title + event_id).encode()).hexdigest()[:16]
    ts = datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)
    return NewsEvent(
        event_id=event_id,
        source_id=source_id,
        source_tier=source_tier,
        title=title,
        url=f"https://example.com/{event_id}",
        published_at=ts,
        ingested_at=ts,
        content_hash=content_hash,
        tickers=tickers or [],
        affected_sectors=affected_sectors or [],
        event_types=event_types or [],
        geo_tags=geo_tags or [],
        severity=severity,
        news_confidence=news_confidence,
    )


# ---------------------------------------------------------------------------
# NewsEventStore
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestNewsEventStore:
    def test_add_and_count(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        events = [_make_event(f"ev{i}") for i in range(10)]
        store.add_many(events)
        assert store.count() == 10

    def test_query_by_ticker(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        store.add(_make_event("ev_nvda", tickers=["NVDA"], title="Nvidia AI chip news"))
        store.add(_make_event("ev_aapl", tickers=["AAPL"], title="Apple earnings beat"))
        store.add(_make_event("ev_both", tickers=["NVDA", "AMD"], title="Chip sector news"))

        nvda_events = store.query_by_ticker("NVDA")
        assert len(nvda_events) == 2
        assert all("ev_nvda" == e.event_id or "ev_both" == e.event_id for e in nvda_events)

    def test_query_by_sector(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        store.add(_make_event("ev_en1", affected_sectors=["energy"], title="Oil pipeline attack"))
        store.add(_make_event("ev_en2", affected_sectors=["energy", "defense"], title="Russia energy cuts"))
        store.add(_make_event("ev_fin", affected_sectors=["financials"], title="Fed rate hike"))

        energy = store.query_by_sector("energy")
        assert len(energy) == 2
        financials = store.query_by_sector("financials")
        assert len(financials) == 1

    def test_query_by_event_type(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        store.add(_make_event("ev_war", event_types=["war_escalation"], title="War escalation update"))
        store.add(_make_event("ev_san", event_types=["sanctions"], title="New sanctions announced"))
        store.add(_make_event("ev_earn", event_types=["earnings"], title="Q1 earnings beat"))

        war_events = store.query_by_event_type("war_escalation")
        assert len(war_events) == 1
        assert war_events[0].event_id == "ev_war"

    def test_query_by_geo(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        store.add(_make_event("ev_ru", geo_tags=["RU"], title="Russia sanctions"))
        store.add(_make_event("ev_cn", geo_tags=["CN"], title="China trade war"))
        store.add(_make_event("ev_ru2", geo_tags=["RU", "UA"], title="Russia-Ukraine conflict"))

        ru_events = store.query_by_geo("RU")
        assert len(ru_events) == 2

    def test_query_by_time_excludes_old(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        store.add(_make_event("ev_recent", hours_ago=0.5))
        store.add(_make_event("ev_old", hours_ago=25.0))

        recent = store.query_by_time(hours=1.0)
        assert len(recent) == 1
        assert recent[0].event_id == "ev_recent"

    def test_query_by_severity(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        store.add(_make_event("ev_high", severity=8.0))
        store.add(_make_event("ev_low", severity=2.0))

        high_sev = store.query_by_severity(min_severity=6.0)
        assert len(high_sev) == 1
        assert high_sev[0].event_id == "ev_high"

    def test_top_sectors(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        for i in range(5):
            store.add(_make_event(f"en{i}", affected_sectors=["energy"]))
        for i in range(2):
            store.add(_make_event(f"fin{i}", affected_sectors=["financials"]))

        top = store.top_sectors(hours=24.0, n=3)
        assert top[0][0] == "energy"
        assert top[0][1] == 5

    def test_top_tickers(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        for i in range(4):
            store.add(_make_event(f"ev{i}", tickers=["NVDA"]))
        store.add(_make_event("ev_aapl", tickers=["AAPL"]))

        top = store.top_tickers(hours=24.0, n=5)
        assert top[0][0] == "NVDA"
        assert top[0][1] == 4

    def test_avg_severity(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        store.add(_make_event("ev1", severity=4.0))
        store.add(_make_event("ev2", severity=6.0))

        avg = store.avg_severity(hours=24.0)
        assert abs(avg - 5.0) < 0.01

    def test_eviction_on_overflow(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore(max_events=10)
        for i in range(15):
            store.add(_make_event(f"ev{i}"))
        # After eviction should be ~7-8 events
        assert store.count() <= 10

    def test_clear(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        store.add_many([_make_event(f"ev{i}") for i in range(5)])
        store.clear()
        assert store.count() == 0
        assert store.query_by_ticker("NVDA") == []

    def test_query_by_source(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        store.add(_make_event("ev_r1", source_id="reuters"))
        store.add(_make_event("ev_r2", source_id="reuters"))
        store.add(_make_event("ev_b1", source_id="bloomberg"))

        reuters = store.query_by_source("reuters")
        assert len(reuters) == 2

    def test_query_custom_predicate(self):
        from src.assembled_core.intel.news_event_store import NewsEventStore

        store = NewsEventStore()
        store.add(_make_event("ev_hi", news_confidence=0.9))
        store.add(_make_event("ev_lo", news_confidence=0.2))

        high_conf = store.query(lambda e: getattr(e, "news_confidence", 0) >= 0.8)
        assert len(high_conf) == 1
        assert high_conf[0].event_id == "ev_hi"


# ---------------------------------------------------------------------------
# IntelSignalAggregator
# ---------------------------------------------------------------------------


@pytest.mark.phase12
class TestIntelSignalAggregator:
    def _make_cluster(
        self,
        cluster_id: str,
        trigger_type_val: str,
        confidence: float,
    ):
        class FakeCluster:
            pass
        cl = FakeCluster()
        cl.cluster_id = cluster_id
        cl.trigger_type = type("TT", (), {"value": trigger_type_val})()
        cl.confidence = confidence
        return cl

    def test_aggregate_bearish_clusters(self):
        from src.assembled_core.intel.news_signal_aggregator import aggregate_signals

        clusters = [
            self._make_cluster("cl1", "war_escalation", 0.75),
            self._make_cluster("cl2", "sanctions", 0.65),
        ]
        # Legacy FakeCluster has no supporting_events; disable the
        # corroboration gate for direction-only assertions.
        sig = aggregate_signals(clusters, require_corroboration_gate=False)
        assert sig.net_direction == "bearish"
        assert sig.aggregate_confidence > 0
        assert sig.n_clusters == 2

    def test_empty_clusters_gives_neutral(self):
        from src.assembled_core.intel.news_signal_aggregator import aggregate_signals

        sig = aggregate_signals([])
        assert sig.net_direction == "neutral"
        assert sig.aggregate_confidence == 0.0
        assert sig.risk_level == "LOW"

    def test_risk_level_high(self):
        from src.assembled_core.intel.news_signal_aggregator import aggregate_signals

        clusters = [self._make_cluster("cl1", "war_escalation", 0.90)]
        sig = aggregate_signals(clusters, require_corroboration_gate=False)
        assert sig.risk_level in ("HIGH", "CRITICAL")

    def test_risk_level_low_for_weak_signal(self):
        from src.assembled_core.intel.news_signal_aggregator import aggregate_signals

        clusters = [self._make_cluster("cl1", "war_escalation", 0.25)]
        sig = aggregate_signals(clusters, min_confidence=0.3)
        # 0.25 < min_confidence → filtered out → neutral
        assert sig.net_direction == "neutral"
        assert sig.risk_level == "LOW"

    def test_is_actionable(self):
        from src.assembled_core.intel.news_signal_aggregator import aggregate_signals

        clusters = [self._make_cluster("cl1", "military_strike", 0.85)]
        sig = aggregate_signals(clusters, require_corroboration_gate=False)
        assert sig.is_actionable()

    def test_asset_basket_populated(self):
        from src.assembled_core.intel.news_signal_aggregator import aggregate_signals
        from src.assembled_core.intel.news_position_bridge import PositionSignal

        pos_signals = [
            PositionSignal(
                signal_id="s1",
                source_cluster_id=None,
                direction="short",
                confidence=0.7,
                affected_assets=["XLE", "XOM"],
                affected_sectors=["energy"],
                event_types=["energy_disruption"],
            )
        ]
        # pre-built PositionSignals skip the gate (it only runs when
        # position_signals is None)
        sig = aggregate_signals([], position_signals=pos_signals)
        assert "XLE" in sig.asset_basket
        assert sig.asset_basket["XLE"] < 0  # short position

    def test_corroboration_gate_drops_uncorroborated_signal(self):
        """K1: signals without ≥2 distinct T0/T1 sources must be dropped."""
        from src.assembled_core.intel.models import NewsEvent, SourceTier
        from src.assembled_core.intel.news_signal_aggregator import (
            aggregate_signals,
            get_corroboration_drop_count,
            reset_corroboration_drop_count,
        )
        from datetime import datetime, timezone

        reset_corroboration_drop_count()
        drop_before = get_corroboration_drop_count()

        # Single T3 source → gate should drop it.
        cl = self._make_cluster("cl_uncorr", "war_escalation", 0.80)
        cl.supporting_events = [NewsEvent(
            event_id="e1",
            source_id="rt",
            source_tier=SourceTier.T3,
            title="something",
            url="https://x/e1",
            published_at=datetime.now(tz=timezone.utc),
            ingested_at=datetime.now(tz=timezone.utc),
            content_hash="h1",
        )]
        sig = aggregate_signals([cl], require_corroboration_gate=True)
        assert sig.net_direction == "neutral"
        assert get_corroboration_drop_count() > drop_before

    def test_corroboration_gate_passes_multi_tier1(self):
        """K1: gate must pass when ≥2 distinct T0/T1 sources corroborate."""
        from src.assembled_core.intel.models import NewsEvent, SourceTier
        from src.assembled_core.intel.news_signal_aggregator import aggregate_signals
        from datetime import datetime, timezone

        cl = self._make_cluster("cl_corr", "war_escalation", 0.80)
        now = datetime.now(tz=timezone.utc)
        cl.supporting_events = [
            NewsEvent(event_id=f"e{i}", source_id=src, source_tier=SourceTier.T1,
                      title="t", url=f"https://x/e{i}", published_at=now,
                      ingested_at=now, content_hash=f"h{i}")
            for i, src in enumerate(["reuters", "ap", "bbc"])
        ]
        sig = aggregate_signals([cl], require_corroboration_gate=True)
        assert sig.net_direction == "bearish"

    def test_sector_exposure_populated(self):
        from src.assembled_core.intel.news_signal_aggregator import aggregate_signals
        from src.assembled_core.intel.news_position_bridge import PositionSignal

        pos_signals = [
            PositionSignal(
                signal_id="s1",
                source_cluster_id=None,
                direction="short",
                confidence=0.8,
                affected_sectors=["defense", "energy"],
                event_types=["war_escalation"],
            )
        ]
        sig = aggregate_signals([], position_signals=pos_signals)
        assert "energy" in sig.sector_exposure or "defense" in sig.sector_exposure
