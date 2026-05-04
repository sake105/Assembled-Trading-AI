"""End-to-end integration test wiring all F1–F16 components together.

This test exercises the full chain on synthetic events:

    raw events
      -> language detection
      -> classifier
      -> impact estimator
      -> corroboration tracker
      -> ticker velocity
      -> sentiment drift
      -> contradiction detector
      -> source voting
      -> entity graph
      -> macro-calendar gating
      -> alert engine
      -> archive write/read roundtrip
      -> position bridge with corroboration gate

Goal: catch regressions where one module breaks the contract another relies on.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_alerts import AlertEngine
from src.assembled_core.intel.news_archive import (
    NewsArchiveReader,
    NewsArchiveWriter,
)
from src.assembled_core.intel.news_contradiction import ContradictionDetector
from src.assembled_core.intel.news_corroboration import CorroborationTracker
from src.assembled_core.intel.news_decay import NewsDecay
from src.assembled_core.intel.news_entity_graph import EntityCoGraph
from src.assembled_core.intel.news_enricher import NewsEventEnricher
from src.assembled_core.intel.news_language import detect_language
from src.assembled_core.intel.news_macro_calendar import MacroCalendar, MacroEvent
from src.assembled_core.intel.news_position_bridge import (
    PositionSignal,
    require_corroboration,
)
from src.assembled_core.intel.news_sentiment_drift import SentimentDriftTracker
from src.assembled_core.intel.news_source_voting import vote_direction
from src.assembled_core.intel.news_ticker_velocity import TickerVelocityTracker


def _evt(
    eid: str,
    title: str,
    src: str,
    tier: SourceTier,
    ts: datetime,
    tickers: list[str] | None = None,
    sectors: list[str] | None = None,
    direction: str = "bearish",
    severity: float = 7.0,
    sentiment: float = -0.4,
) -> NewsEvent:
    return NewsEvent(
        event_id=eid,
        source_id=src,
        source_tier=tier,
        title=title,
        url=f"https://example.com/{eid}",
        published_at=ts,
        ingested_at=ts,
        tickers=tickers or [],
        affected_sectors=sectors or [],
        market_direction=direction,
        severity=severity,
        sentiment_score=sentiment,
        content_hash=hashlib.sha256((title + eid).encode()).hexdigest()[:16],
    )


@pytest.mark.phase12
class TestFullNewsPipeline:
    def test_full_chain(self, tmp_path):
        now = datetime.now(tz=timezone.utc)
        title = "Russia escalates sanctions enforcement against energy sector"
        events = [
            _evt(
                "e1",
                title,
                "reuters",
                SourceTier.T1,
                now - timedelta(minutes=20),
                tickers=["XOM", "CVX"],
                sectors=["energy"],
            ),
            _evt(
                "e2",
                title,
                "ap",
                SourceTier.T1,
                now - timedelta(minutes=10),
                tickers=["XOM"],
                sectors=["energy"],
            ),
            _evt(
                "e3",
                title,
                "bbc",
                SourceTier.T1,
                now - timedelta(minutes=5),
                tickers=["XOM", "CVX"],
                sectors=["energy"],
            ),
            # one outlier with opposite framing → contradiction signal
            _evt(
                "e4",
                title,
                "rt",
                SourceTier.T3,
                now,
                tickers=["XOM"],
                sectors=["energy"],
                direction="bullish",
                severity=4.0,
                sentiment=0.2,
            ),
        ]

        # 1) language detection
        for e in events:
            assert detect_language(e.title) == "en"

        # Snapshot the raw direction labels before the enricher rewrites them.
        # ContradictionDetector relies on the originally-set market_direction.
        raw_for_contradiction = [e.model_copy(deep=True) for e in events]

        # 2) enricher (language + classify + impact + corroboration + velocity)
        enricher = NewsEventEnricher()
        enriched = enricher.enrich(events)
        # all events should have non-empty language and impact attrs
        for e in enriched:
            assert e.language == "en"
            assert e.impact_horizon_days >= 0
            # impact_bps may be 0 if classifier finds no matching event_type
            assert isinstance(e.impact_bps, float)

        # 3) corroboration directly: high-tier events agree → high score
        corr = CorroborationTracker(saturation=4.0)
        corr.ingest(enriched[:3])
        score = corr.corroboration_score(enriched[0])
        assert score.n_sources == 3
        assert score.score > 0.4

        # 4) contradiction: T3 outlier disagrees with T1 majority
        # Run on the pre-enrichment snapshot — the classifier normalises
        # market_direction across the group and would mask the outlier.
        contra = ContradictionDetector()
        rep = contra.analyse(raw_for_contradiction)
        flagged = [e for e in rep.values() if e.contradicts]
        assert flagged, "expected at least one contradiction entry"

        # 5) source voting: T1 majority overpowers T3
        vote = vote_direction(enriched)
        assert vote.winner == "bearish"

        # 6) ticker velocity: 4 events in <30 min → likely surge
        vt = TickerVelocityTracker(
            short_window_min=30,
            long_window_min=120,
            surge_threshold=2.0,
            min_short_events=3,
        )
        signals = vt.update(enriched, now=now)
        assert any(s.ticker == "XOM" for s in signals)

        # 7) sentiment drift on the energy sector — direction should be coherent
        drift = SentimentDriftTracker(
            window_min=60, min_events=3, slope_threshold=0.0001
        )
        drift.update(enriched, now=now)
        rep2 = drift.report(now=now)
        assert any(e.key == "SECTOR:energy" for e in rep2)

        # 8) entity graph
        eg = EntityCoGraph()
        eg.ingest(enriched)
        # XOM and CVX co-occur on multiple events → an edge
        assert eg.edge_weight("xom", "cvx") >= 1

        # 9) macro calendar gating: no FOMC scheduled near `now` → not a blackout
        cal = MacroCalendar()
        cal.add(MacroEvent("FOMC1", "fomc", now + timedelta(days=10)))
        assert cal.is_blackout("fomc", now=now) is False

        # 10) alert engine fires on the high-severity T1 event
        ae = AlertEngine(min_severity=6.0, include_default_log_handler=False)
        captured: list = []
        ae.add_handler(captured.append)
        alerts = ae.evaluate(enriched)
        assert len(alerts) >= 1
        assert captured  # handler was called

        # 11) decay: compute at one half-life to assert the curve shape
        # rather than pinning a specific table value (tuning drifts).
        d = NewsDecay()
        prof = d.profile("sanctions")
        frac = d.impact_remaining("sanctions", prof.parameter_min)
        assert 0.49 < frac < 0.51

        # 12) archive roundtrip
        path = tmp_path / "events.jsonl"
        with NewsArchiveWriter(path) as w:
            w.append(enriched)
        replayed = list(NewsArchiveReader(path).iter_events())
        assert len(replayed) == len(enriched)

        # 13) position bridge corroboration gate
        sig = PositionSignal(
            signal_id="ps_test",
            source_cluster_id=None,
            direction="short",
            confidence=0.7,
            affected_assets=["XOM"],
        )
        gated = require_corroboration(sig, enriched[:3])
        assert gated is sig
        gated2 = require_corroboration(sig, enriched[:1])
        assert gated2 is None
