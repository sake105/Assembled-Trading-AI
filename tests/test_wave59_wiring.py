"""Tests for wave-59 module wiring into trading_cycle.py.

Covers:
  Step 2.39 — features.satellite_proxy_features (compute_copper_gold_ratio / compute_bdi_features)
  Step 2.40 — features.supply_chain_features (build_supply_chain_features)
  Step 3.94 — strategies.signal_decay_gate (compute_multipliers / apply_multipliers)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.satellite_proxy_features import (
    compute_copper_gold_ratio,
    compute_oil_gold_ratio,
    compute_bdi_features,
)
from src.assembled_core.features.supply_chain_features import (
    build_supply_chain_features,
    compute_supply_chain_depth,
    compute_single_source_dependency,
)
from src.assembled_core.strategies.signal_decay_gate import (
    compute_multipliers,
    apply_multipliers,
)


# ---------------------------------------------------------------------------
# satellite_proxy_features (Step 2.39)
# ---------------------------------------------------------------------------

def test_compute_copper_gold_ratio_returns_series():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    copper = pd.Series(rng.uniform(3.5, 5.0, 30), index=idx)
    gold = pd.Series(rng.uniform(1800, 2000, 30), index=idx)
    result = compute_copper_gold_ratio(copper, gold)
    assert isinstance(result, pd.Series)
    assert len(result) == 30


def test_compute_copper_gold_ratio_name():
    rng = np.random.default_rng(0)
    copper = pd.Series(rng.uniform(3.5, 5.0, 30))
    gold = pd.Series(rng.uniform(1800, 2000, 30))
    result = compute_copper_gold_ratio(copper, gold)
    assert result.name == "copper_gold_ratio"


def test_compute_oil_gold_ratio_returns_series():
    rng = np.random.default_rng(0)
    oil = pd.Series(rng.uniform(60, 100, 30))
    gold = pd.Series(rng.uniform(1800, 2000, 30))
    result = compute_oil_gold_ratio(oil, gold)
    assert isinstance(result, pd.Series)


def test_compute_bdi_features_returns_df():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    bdi = pd.Series(rng.uniform(1000, 3000, 100), index=idx)
    result = compute_bdi_features(bdi)
    assert isinstance(result, pd.DataFrame)
    assert "bdi_level" in result.columns
    assert "bdi_zscore" in result.columns


def test_compute_bdi_features_length():
    rng = np.random.default_rng(0)
    bdi = pd.Series(rng.uniform(1000, 3000, 100))
    result = compute_bdi_features(bdi)
    assert len(result) == 100


# ---------------------------------------------------------------------------
# supply_chain_features (Step 2.40)
# ---------------------------------------------------------------------------

def test_build_supply_chain_features_empty_symbols():
    result = build_supply_chain_features([])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_build_supply_chain_features_no_edges():
    symbols = ["AAPL", "MSFT", "GOOG"]
    result = build_supply_chain_features(symbols)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(symbols)


def test_build_supply_chain_features_has_cols():
    symbols = ["AAPL", "MSFT"]
    result = build_supply_chain_features(symbols)
    assert "supply_chain_depth" in result.columns
    assert "single_source_dep" in result.columns


def test_compute_supply_chain_depth_empty_edges():
    result = compute_supply_chain_depth([], ["AAPL"])
    assert isinstance(result, dict)
    assert result.get("AAPL", 0) == 0


def test_compute_single_source_dependency_empty():
    result = compute_single_source_dependency([], ["AAPL"])
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# signal_decay_gate (Step 3.94)
# ---------------------------------------------------------------------------

def test_compute_multipliers_empty_factors():
    result = compute_multipliers([])
    assert isinstance(result, dict)
    assert len(result) == 0


def test_compute_multipliers_unknown_factors():
    result = compute_multipliers(["momentum", "value", "quality"])
    assert isinstance(result, dict)
    assert len(result) == 3


def test_compute_multipliers_default_healthy():
    result = compute_multipliers(["momentum"])
    assert result["momentum"] == 1.0


def test_apply_multipliers_returns_tuple():
    weights = {"momentum": 0.5, "value": 0.3, "quality": 0.2}
    effective, multipliers = apply_multipliers(weights)
    assert isinstance(effective, dict)
    assert isinstance(multipliers, dict)
    # disabled mode → weights unchanged
    for k, v in weights.items():
        assert abs(effective[k] - v) < 1e-9
