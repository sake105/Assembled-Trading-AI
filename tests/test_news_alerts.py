"""Tests for AlertEngine."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_alerts import AlertEngine, NewsAlert


def _evt(eid: str, severity: float = 0.0, event_types: list[str] | None = None,
         corr_score: float = 0.0, corr_n: int = 0) -> NewsEvent:
    ts = datetime.now(tz=timezone.utc)
    return NewsEvent(
        event_id=eid,
        source_id="reuters",
        source_tier=SourceTier.T1,
        title=f"hl {eid}",
        url=f"https://example.com/{eid}",
        published_at=ts,
        ingested_at=ts,
        content_hash=hashlib.sha256(eid.encode()).hexdigest()[:16],
        severity=severity,
        event_types=event_types or [],
        corroboration_score=corr_score,
        corroboration_n_sources=corr_n,
    )


@pytest.mark.phase12
class TestAlertEngine:
    def test_no_events_no_alerts(self):
        eng = AlertEngine()
        assert eng.evaluate([]) == []

    def test_below_threshold_no_alert(self):
        eng = AlertEngine(min_severity=8.0, include_default_log_handler=False)
        out = eng.evaluate([_evt("e1", severity=5.0)])
        assert out == []

    def test_critical_alert(self):
        eng = AlertEngine(min_severity=8.0, include_default_log_handler=False)
        out = eng.evaluate([_evt("e1", severity=9.0, event_types=["war_escalation"])])
        assert len(out) == 1
        assert out[0].kind == "critical"

    def test_corroborated_alert(self):
        eng = AlertEngine(
            min_severity=99.0,    # disable critical channel
            min_corroboration_score=0.7,
            min_corroboration_sources=3,
            include_default_log_handler=False,
        )
        out = eng.evaluate([_evt("e1", corr_score=0.9, corr_n=4)])
        assert len(out) == 1
        assert out[0].kind == "corroborated"

    def test_handler_invoked(self):
        eng = AlertEngine(min_severity=1.0, include_default_log_handler=False)
        captured: list[NewsAlert] = []
        eng.add_handler(captured.append)
        eng.evaluate([_evt("e1", severity=5.0)])
        assert len(captured) == 1
        assert captured[0].event_id == "e1"

    def test_failing_handler_does_not_break_others(self):
        eng = AlertEngine(min_severity=1.0, include_default_log_handler=False)
        captured: list[NewsAlert] = []
        eng.add_handler(lambda a: (_ for _ in ()).throw(RuntimeError("boom")))
        eng.add_handler(captured.append)
        eng.evaluate([_evt("e1", severity=5.0)])
        assert len(captured) == 1

    def test_surge_alert(self):
        eng = AlertEngine(include_default_log_handler=False)
        captured: list[NewsAlert] = []
        eng.add_handler(captured.append)
        a = eng.surge_alert("AAPL", velocity=4.5)
        assert a.kind == "surge"
        assert captured[-1].extra["ticker"] == "AAPL"

    def test_contradiction_alert(self):
        eng = AlertEngine(include_default_log_handler=False)
        captured: list[NewsAlert] = []
        eng.add_handler(captured.append)
        a = eng.contradiction_alert("story:abc", "bearish_vs_bullish")
        assert a.kind == "contradiction"
        assert captured[-1].extra["split"] == "bearish_vs_bullish"

    def test_clear_handlers(self):
        eng = AlertEngine(min_severity=1.0)
        eng.clear_handlers()
        out = eng.evaluate([_evt("e1", severity=9.0)])
        assert len(out) == 1  # alert built but no dispatch crash
