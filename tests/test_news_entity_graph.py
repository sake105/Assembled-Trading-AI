"""Tests for EntityCoGraph."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_entity_graph import EntityCoGraph


def _evt(eid: str, entities: list[str], ts: datetime | None = None,
         tickers: list[str] | None = None) -> NewsEvent:
    ts = ts or datetime.now(tz=timezone.utc)
    return NewsEvent(
        event_id=eid,
        source_id="reuters",
        source_tier=SourceTier.T1,
        title=f"hl {eid}",
        url=f"https://example.com/{eid}",
        published_at=ts,
        ingested_at=ts,
        content_hash=hashlib.sha256(eid.encode()).hexdigest()[:16],
        entities=entities,
        tickers=tickers or [],
    )


@pytest.mark.phase12
class TestEntityCoGraph:
    def test_empty_graph(self):
        g = EntityCoGraph()
        assert g.size == 0
        assert g.neighbours("anything") == []
        assert g.top_entities() == []

    def test_single_entity_no_edge(self):
        g = EntityCoGraph()
        g.ingest([_evt("e1", ["openai"])])
        assert g.size == 0  # no edges from a single entity

    def test_pair_creates_edge(self):
        g = EntityCoGraph()
        g.ingest([_evt("e1", ["openai", "nvidia"])])
        assert g.has_edge("openai", "nvidia")
        assert g.has_edge("nvidia", "openai")
        assert g.edge_weight("openai", "nvidia") == 1

    def test_multiple_co_occurrences_increment(self):
        g = EntityCoGraph()
        g.ingest([
            _evt("e1", ["openai", "nvidia"]),
            _evt("e2", ["openai", "nvidia"]),
            _evt("e3", ["openai", "nvidia"]),
        ])
        assert g.edge_weight("openai", "nvidia") == 3

    def test_neighbours_sorted_and_filtered(self):
        g = EntityCoGraph()
        g.ingest([
            _evt("e1", ["openai", "nvidia"]),
            _evt("e2", ["openai", "msft"]),
            _evt("e3", ["openai", "nvidia"]),
        ])
        nbrs = g.neighbours("openai")
        assert nbrs[0] == ("nvidia", 2)
        filt = g.neighbours("openai", min_weight=2)
        assert filt == [("nvidia", 2)]

    def test_tickers_and_entities_merged(self):
        g = EntityCoGraph()
        g.ingest([_evt("e1", ["acme corp"], tickers=["AAPL"])])
        # both entities should connect
        assert g.has_edge("acme corp", "aapl")

    def test_top_entities_ordering(self):
        g = EntityCoGraph()
        g.ingest([
            _evt("e1", ["openai", "nvidia"]),
            _evt("e2", ["openai", "msft"]),
            _evt("e3", ["openai", "google"]),
        ])
        top = g.top_entities(n=4)
        assert top[0].entity == "openai"
        assert top[0].degree == 3

    def test_window_pruning(self):
        g = EntityCoGraph(retention_hours=1.0)
        old = datetime.now(tz=timezone.utc) - timedelta(hours=5)
        g.ingest([_evt("old", ["a", "b"], ts=old)], now=old)
        # advance time well past retention
        future = datetime.now(tz=timezone.utc)
        g.prune(now=future)
        assert g.has_edge("a", "b") is False
        assert g.size == 0
