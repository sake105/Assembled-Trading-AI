"""Tests for intermarket cross-asset factors module."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.features.intermarket_factors import (
    align_intermarket_factors_to_panel,
    get_intermarket_factor_names,
)


def _synthetic_intermarket_factors(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create synthetic intermarket factors (avoids yfinance/FRED calls)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "bond_equity_ratio_20d": 0.5 + rng.normal(0, 0.05, n),
            "dollar_trend_20d": rng.normal(0, 0.02, n),
            "dollar_trend_60d": rng.normal(0, 0.03, n),
            "credit_spread_change_5d": rng.normal(0, 0.01, n),
            "credit_spread_change_20d": rng.normal(0, 0.02, n),
            "gold_equity_divergence": rng.normal(0, 0.03, n),
            "yield_curve_slope": 1.5 + rng.normal(0, 0.5, n),
            "yield_10y": 4.0 + rng.normal(0, 0.3, n),
            "yield_2y": 3.5 + rng.normal(0, 0.2, n),
            "hy_ig_ratio": 0.85 + rng.normal(0, 0.02, n),
            "bond_equity_divergence_flag": rng.choice(
                [-1.0, 0.0, 1.0], n, p=[0.2, 0.6, 0.2]
            ),
        }
    )


@pytest.mark.phase12
class TestGetIntermarketFactorNames:
    def test_returns_list(self):
        names = get_intermarket_factor_names()
        assert isinstance(names, list)
        assert len(names) == 11

    def test_expected_names(self):
        names = get_intermarket_factor_names()
        assert "bond_equity_ratio_20d" in names
        assert "yield_curve_slope" in names
        assert "gold_equity_divergence" in names
        assert "hy_ig_ratio" in names
        assert "bond_equity_divergence_flag" in names


@pytest.mark.phase12
class TestAlignIntermarketFactors:
    def test_basic_merge(self):
        factors = _synthetic_intermarket_factors(n=50)
        dates = factors["timestamp"].tolist()
        panel = pd.DataFrame(
            {
                "timestamp": dates * 3,
                "symbol": ["AAPL"] * 50 + ["MSFT"] * 50 + ["GOOG"] * 50,
                "close": np.random.default_rng(1).normal(150, 20, 150),
            }
        )
        merged = align_intermarket_factors_to_panel(panel, factors)
        assert "bond_equity_ratio_20d" in merged.columns
        assert "yield_curve_slope" in merged.columns
        assert len(merged) == 150

    def test_empty_factors(self):
        panel = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01")],
                "symbol": ["A"],
                "close": [100.0],
            }
        )
        result = align_intermarket_factors_to_panel(panel, pd.DataFrame())
        assert len(result) == 1
        assert "bond_equity_ratio_20d" not in result.columns

    def test_empty_panel(self):
        factors = _synthetic_intermarket_factors(n=10)
        result = align_intermarket_factors_to_panel(pd.DataFrame(), factors)
        assert result.empty

    def test_pit_safety(self):
        """Factors from future dates should not leak into past rows."""
        factors = _synthetic_intermarket_factors(n=20)
        # Panel has dates earlier than factors
        early_dates = pd.bdate_range("2022-01-01", periods=5)
        panel = pd.DataFrame(
            {
                "timestamp": list(early_dates),
                "symbol": ["AAPL"] * 5,
                "close": [100.0] * 5,
            }
        )
        merged = align_intermarket_factors_to_panel(panel, factors)
        # Factor columns should be NaN for dates before factor data starts
        assert merged["bond_equity_ratio_20d"].isna().all()


@pytest.mark.phase12
class TestSyntheticFactorStructure:
    def test_columns_match_names(self):
        factors = _synthetic_intermarket_factors()
        expected_names = get_intermarket_factor_names()
        for name in expected_names:
            assert name in factors.columns

    def test_divergence_flag_values(self):
        factors = _synthetic_intermarket_factors(n=200)
        unique_vals = set(factors["bond_equity_divergence_flag"].unique())
        assert unique_vals <= {-1.0, 0.0, 1.0}

    def test_yield_curve_slope_reasonable(self):
        factors = _synthetic_intermarket_factors()
        # Slope should be around 1.5 +/- noise
        mean_slope = factors["yield_curve_slope"].mean()
        assert 0.0 < mean_slope < 3.0
