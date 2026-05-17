"""Tests for source voting helpers."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_source_voting import (
    vote_direction,
    vote_event_type,
)


def _evt(
    eid: str,
    src: str,
    tier: SourceTier,
    direction: str = "neutral",
    event_types: list[str] | None = None,
) -> NewsEvent:
    ts = datetime.now(tz=timezone.utc)
    return NewsEvent(
        event_id=eid,
        source_id=src,
        source_tier=tier,
        title=f"hl {eid}",
        url=f"https://example.com/{eid}",
        published_at=ts,
        ingested_at=ts,
        content_hash=hashlib.sha256(eid.encode()).hexdigest()[:16],
        market_direction=direction,
        event_types=event_types or [],
    )


@pytest.mark.fast
class TestVoteDirection:
    def test_empty(self):
        r = vote_direction([])
        assert r.winner == "" and r.total_weight == 0.0

    def test_simple_majority(self):
        evs = [
            _evt("e1", "reuters", SourceTier.T1, "bearish"),
            _evt("e2", "ap", SourceTier.T1, "bearish"),
            _evt("e3", "blogX", SourceTier.T2, "bullish"),
        ]
        r = vote_direction(evs)
        assert r.winner == "bearish"

    def test_tier_weight_overrides_count(self):
        # 1 OFAC (T0=3.0) vs 2 T2 (each 1.0) → OFAC wins
        evs = [
            _evt("e1", "ofac", SourceTier.T0, "bearish"),
            _evt("e2", "blogA", SourceTier.T2, "bullish"),
            _evt("e3", "blogB", SourceTier.T2, "bullish"),
        ]
        r = vote_direction(evs)
        assert r.winner == "bearish"

    def test_state_media_discount(self):
        # state-media gets 0.5x discount
        evs = [
            _evt("e1", "rt", SourceTier.T3, "bullish"),  # tier T3=0.4 * 0.5 = 0.2
            _evt("e2", "reuters", SourceTier.T1, "bearish"),  # 2.0
        ]
        r = vote_direction(evs)
        assert r.winner == "bearish"


@pytest.mark.fast
class TestVoteEventType:
    def test_empty_v2(self):
        assert vote_event_type([]).winner == ""

    def test_majority_event_type(self):
        evs = [
            _evt("e1", "reuters", SourceTier.T1, event_types=["sanctions"]),
            _evt("e2", "ap", SourceTier.T1, event_types=["sanctions"]),
            _evt("e3", "blog", SourceTier.T2, event_types=["earnings"]),
        ]
        r = vote_event_type(evs)
        assert r.winner == "sanctions"

    def test_no_event_types_yields_empty(self):
        evs = [_evt("e1", "reuters", SourceTier.T1)]
        assert vote_event_type(evs).winner == ""
