"""Tests for wave-81 module wiring into trading_cycle.py.

Covers:
  Step 8.96 — intel.news_entity_graph (EntityCoGraph / EntityStat)
  Step 8.97 — intel.news_event_store (NewsEventStore)
  Step 8.98 — intel.news_ingest (records_to_news_events / GdeltFetcher)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_entity_graph import EntityCoGraph, EntityStat
from src.assembled_core.intel.news_event_store import NewsEventStore
from src.assembled_core.intel.news_ingest import records_to_news_events, GdeltFetcher


# ---------------------------------------------------------------------------
# news_entity_graph (Step 8.96)
# ---------------------------------------------------------------------------

def test_entity_co_graph_creates():
    ecg = EntityCoGraph()
    assert isinstance(ecg, EntityCoGraph)


def test_entity_co_graph_empty_counts():
    ecg = EntityCoGraph()
    assert len(ecg._counts) == 0


def test_entity_co_graph_ingest_empty():
    ecg = EntityCoGraph()
    ecg.ingest([])
    assert len(ecg._counts) == 0


def test_entity_co_graph_adjacency_starts_empty():
    ecg = EntityCoGraph()
    assert len(ecg._adj) == 0


def test_entity_stat_importable():
    assert EntityStat is not None


# ---------------------------------------------------------------------------
# news_event_store (Step 8.97)
# ---------------------------------------------------------------------------

def test_news_event_store_creates():
    nes = NewsEventStore()
    assert isinstance(nes, NewsEventStore)


def test_news_event_store_empty():
    nes = NewsEventStore()
    assert len(nes._events) == 0


def test_news_event_store_indices_empty():
    nes = NewsEventStore()
    assert len(nes._idx_ticker) == 0
    assert len(nes._idx_sector) == 0


def test_news_event_store_max_events_param():
    nes = NewsEventStore(max_events=100)
    assert nes._max_events == 100


# ---------------------------------------------------------------------------
# news_ingest (Step 8.98)
# ---------------------------------------------------------------------------

def test_records_to_news_events_empty():
    result = records_to_news_events([])
    assert isinstance(result, list)
    assert len(result) == 0


def test_gdelt_fetcher_creates(tmp_path):
    fetcher = GdeltFetcher(state_path=tmp_path / "state.json")
    assert isinstance(fetcher, GdeltFetcher)


def test_gdelt_fetcher_load_state_fresh(tmp_path):
    fetcher = GdeltFetcher(state_path=tmp_path / "state.json")
    state = fetcher.load_state()
    assert state is not None
