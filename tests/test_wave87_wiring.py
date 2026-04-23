"""Tests for wave-87 module wiring into trading_cycle.py.

Covers:
  Step 8.114 — intel.weaponized_interdependence (get_known_wi_pairs / WIScore)
  Step 8.115 — intel.wild_card_detector (detect_volume_anomaly / detect_cross_domain_spike)
  Step 8.116 — intel.dependency_graph (DependencyGraph)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.intel.weaponized_interdependence import (
    get_known_wi_pairs,
    WIScore,
    compute_wi_score,
)
from src.assembled_core.intel.wild_card_detector import (
    detect_volume_anomaly,
    detect_cross_domain_spike,
)
from src.assembled_core.intel.dependency_graph import DependencyGraph


# ---------------------------------------------------------------------------
# weaponized_interdependence (Step 8.114)
# ---------------------------------------------------------------------------

def test_get_known_wi_pairs_returns_list():
    pairs = get_known_wi_pairs()
    assert isinstance(pairs, list)


def test_get_known_wi_pairs_not_empty():
    pairs = get_known_wi_pairs()
    assert len(pairs) > 0


def test_get_known_wi_pairs_items_are_dicts():
    pairs = get_known_wi_pairs()
    for p in pairs[:3]:
        assert isinstance(p, dict)


def test_wi_score_importable():
    assert WIScore is not None


def test_compute_wi_score_importable():
    assert compute_wi_score is not None


# ---------------------------------------------------------------------------
# wild_card_detector (Step 8.115)
# ---------------------------------------------------------------------------

def test_detect_volume_anomaly_empty_series():
    result = detect_volume_anomaly(pd.Series([], dtype=float))
    assert isinstance(result, dict)
    assert "is_anomaly" in result


def test_detect_volume_anomaly_no_anomaly_on_constant():
    series = pd.Series([100.0] * 40)
    result = detect_volume_anomaly(series)
    assert result["is_anomaly"] is False


def test_detect_volume_anomaly_returns_zscore():
    series = pd.Series([100.0] * 40)
    result = detect_volume_anomaly(series)
    assert "zscore" in result


def test_detect_cross_domain_spike_importable():
    assert detect_cross_domain_spike is not None


# ---------------------------------------------------------------------------
# dependency_graph (Step 8.116)
# ---------------------------------------------------------------------------

def test_dependency_graph_creates():
    dg = DependencyGraph()
    assert isinstance(dg, DependencyGraph)


def test_dependency_graph_empty_nodes():
    dg = DependencyGraph()
    assert len(dg._nodes) == 0


def test_dependency_graph_empty_adj():
    dg = DependencyGraph()
    assert len(dg._adj_out) == 0
    assert len(dg._adj_in) == 0
