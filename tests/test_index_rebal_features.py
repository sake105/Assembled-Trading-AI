"""Tests for index rebalancing front-running features."""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.features.index_rebal_features import (
    compute_predicted_demand,
    build_index_rebal_features,
    get_index_rebal_feature_names,
)


@pytest.mark.phase12
class TestComputePredictedDemand:
    def test_basic(self):
        demand = compute_predicted_demand(
            market_cap=50e9,
            index_weight=0.01,
            index_aum=7_000e9,
            shares_float=500e6,
            current_price=100.0,
        )
        assert demand > 0

    def test_zero_float(self):
        assert compute_predicted_demand(50e9, 0.01, 7_000e9, 0, 100.0) == 0.0

    def test_zero_price(self):
        assert compute_predicted_demand(50e9, 0.01, 7_000e9, 500e6, 0) == 0.0


@pytest.mark.phase12
class TestBuildIndexRebalFeatures:
    def test_basic_v2(self):
        changes = pd.DataFrame(
            {
                "symbol": ["AAPL", "XYZ"],
                "effective_date": ["2024-06-21", "2024-06-21"],
                "action": ["add", "delete"],
                "index_name": ["SP500", "SP500"],
            }
        )
        result = build_index_rebal_features(changes)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "index_addition_flag" in result.columns
        assert "index_demand_score" in result.columns

    def test_addition_positive(self):
        changes = pd.DataFrame(
            {
                "symbol": ["NEW"],
                "effective_date": ["2024-06-21"],
                "action": ["add"],
                "index_name": ["SP500"],
            }
        )
        result = build_index_rebal_features(changes)
        # Addition should have positive flag
        assert (result["index_addition_flag"] == 1.0).all()

    def test_deletion_negative(self):
        changes = pd.DataFrame(
            {
                "symbol": ["OLD"],
                "effective_date": ["2024-06-21"],
                "action": ["delete"],
                "index_name": ["SP500"],
            }
        )
        result = build_index_rebal_features(changes)
        assert (result["index_addition_flag"] == -1.0).all()

    def test_empty(self):
        result = build_index_rebal_features(pd.DataFrame())
        assert len(result) == 0

    def test_window_size(self):
        changes = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "effective_date": ["2024-06-21"],
                "action": ["add"],
                "index_name": ["SP500"],
            }
        )
        result = build_index_rebal_features(changes)
        # Should create 6 rows (T-5 to T)
        assert len(result) == 6

    def test_feature_names(self):
        names = get_index_rebal_feature_names()
        assert len(names) == 4
        assert "index_addition_flag" in names
