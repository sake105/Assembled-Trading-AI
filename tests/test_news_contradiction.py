"""Tests for ContradictionDetector."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_contradiction import (
    ContradictionDetector,
    _source_camp,
)


def _make_event(
    event_id: str,
    title: str,
    source_id: str,
    tier: SourceTier = SourceTier.T1,
    market_direction: str = "neutral",
    severity: float = 5.0,
) -> NewsEvent:
    ts = datetime.now(tz=timezone.utc)
    return NewsEvent(
        event_id=event_id,
        source_id=source_id,
        source_tier=tier,
        title=title,
        url=f"https://example.com/{event_id}",
        published_at=ts,
        ingested_at=ts,
        content_hash=hashlib.sha256((title + event_id).encode()).hexdigest()[:16],
        market_direction=market_direction,
        severity=severity,
    )


@pytest.mark.phase12
class TestSourceCamp:
    def test_state_media(self):
        assert _source_camp("rt") == "state"
        assert _source_camp("xinhua") == "state"

    def test_western_mainstream(self):
        assert _source_camp("reuters") == "western"
        assert _source_camp("bbc") == "western"

    def test_other(self):
        assert _source_camp("some_blog") == "other"


@pytest.mark.phase12
class TestContradictionDetector:
    def test_agreement_does_not_flag(self):
        det = ContradictionDetector()
        title = "Peace talks advance between Russia and Ukraine"
        evts = [
            _make_event("e1", title, "reuters", market_direction="bullish"),
            _make_event("e2", title, "rt", SourceTier.T3, market_direction="bullish"),
        ]
        report = det.analyse(evts)
        entries = list(report.values())
        assert any(not e.contradicts for e in entries)

    def test_direction_disagreement_flags(self):
        det = ContradictionDetector()
        title = "Russian strikes hit Ukrainian energy infrastructure"
        evts = [
            _make_event("w1", title, "reuters", market_direction="bearish", severity=7.0),
            _make_event("w2", title, "bbc", market_direction="bearish", severity=7.0),
            _make_event("s1", title, "rt", SourceTier.T3, market_direction="bullish", severity=5.0),
            _make_event("s2", title, "tass", SourceTier.T3, market_direction="bullish", severity=5.0),
        ]
        report = det.analyse(evts)
        entries = list(report.values())
        contradicting = [e for e in entries if e.contradicts]
        assert contradicting, "expected direction contradiction"
        entry = contradicting[0]
        assert entry.western_direction == "bearish"
        assert entry.state_direction == "bullish"
        assert "bearish_vs_bullish" == entry.direction_split

    def test_no_overlap_returns_no_contradiction(self):
        det = ContradictionDetector()
        title = "Only western outlets cover this"
        evts = [
            _make_event("w1", title, "reuters", market_direction="bearish"),
            _make_event("w2", title, "bbc", market_direction="bearish"),
        ]
        report = det.analyse(evts)
        for entry in report.values():
            assert entry.contradicts is False

    def test_severity_gap_flags(self):
        det = ContradictionDetector()
        title = "Casualty reporting on ongoing conflict"
        evts = [
            _make_event("w1", title, "reuters", market_direction="bearish", severity=8.0),
            _make_event("s1", title, "xinhua", SourceTier.T3, market_direction="bearish", severity=2.0),
        ]
        report = det.analyse(evts)
        entries = list(report.values())
        assert any(e.contradicts and e.direction_split == "severity_gap" for e in entries)
