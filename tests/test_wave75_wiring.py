"""Tests for wave-75 module wiring into trading_cycle.py.

Covers:
  Step 8.78 — intel.news_classifier (classify_news_event / NewsClassification)
  Step 8.79 — intel.news_cluster (ClusterManager)
  Step 8.80 — intel.news_corroboration (CorroborationTracker / CorroborationScore)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_classifier import (
    classify_news_event,
    NewsClassification,
    classify_batch,
)
from src.assembled_core.intel.news_cluster import ClusterManager
from src.assembled_core.intel.news_corroboration import (
    CorroborationTracker,
    CorroborationScore,
)


# ---------------------------------------------------------------------------
# news_classifier (Step 8.78)
# ---------------------------------------------------------------------------

def test_classify_news_event_returns_classification():
    result = classify_news_event("Federal Reserve raises rates by 25bps")
    assert isinstance(result, NewsClassification)


def test_classify_news_event_has_direction():
    result = classify_news_event("Market crash: stocks plummet on recession fears")
    assert result.market_direction in ("bearish", "bullish", "neutral", "mixed")


def test_classify_news_event_severity_range():
    result = classify_news_event("Central bank announcement")
    assert 0.0 <= result.severity <= 10.0


def test_classify_news_event_empty_headline():
    result = classify_news_event("")
    assert isinstance(result, NewsClassification)
    assert result.event_types == []


def test_classify_batch_returns_list():
    titles = ["Fed raises rates", "Earnings beat expectations", "Trade war escalates"]
    results = classify_batch(titles)
    assert isinstance(results, list)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# news_cluster (Step 8.79)
# ---------------------------------------------------------------------------

def test_cluster_manager_creates():
    cm = ClusterManager()
    assert isinstance(cm, ClusterManager)


def test_cluster_manager_empty_update():
    cm = ClusterManager()
    clusters = cm.update_clusters([])
    assert isinstance(clusters, list)
    assert len(clusters) == 0


def test_cluster_manager_custom_ttl():
    cm = ClusterManager(cluster_ttl_minutes=120)
    assert cm._ttl_minutes == 120


# ---------------------------------------------------------------------------
# news_corroboration (Step 8.80)
# ---------------------------------------------------------------------------

def test_corroboration_tracker_creates():
    ct = CorroborationTracker()
    assert isinstance(ct, CorroborationTracker)


def test_corroboration_tracker_ingest_empty():
    ct = CorroborationTracker()
    ct.ingest([])
    assert len(ct._entries) == 0


def test_corroboration_tracker_custom_params():
    ct = CorroborationTracker(retention_hours=12.0, saturation=3.0)
    assert ct._saturation == 3.0


def test_corroboration_score_importable():
    assert CorroborationScore is not None
