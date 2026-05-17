"""Tests for options-derived regime signals module."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.features.options_derived_signals import (
    build_options_regime_factors,
    align_options_factors_to_panel,
    get_options_factor_names,
    compute_vix_term_structure,
    compute_implied_vs_realized_spread,
    compute_skew_vix_divergence,
)


def _synthetic_cboe(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "vix": 15.0 + rng.normal(0, 5, n).clip(8, 80),
            "vix3m": 18.0 + rng.normal(0, 4, n).clip(10, 70),
            "put_call_ratio": 0.85 + rng.normal(0, 0.2, n).clip(0.3, 2.0),
        }
    )


@pytest.mark.fast
class TestBuildOptionsRegimeFactors:
    def test_basic_output(self):
        cboe = _synthetic_cboe()
        result = build_options_regime_factors(cboe)
        assert "timestamp" in result.columns
        assert "vix_level" in result.columns
        assert "vix_change_5d" in result.columns
        assert "vix_term_slope" in result.columns
        assert len(result) == len(cboe)

    def test_vix_regime_categories(self):
        cboe = _synthetic_cboe()
        result = build_options_regime_factors(cboe)
        regimes = set(result["vix_regime"].unique())
        assert regimes <= {"low", "normal", "high", "extreme"}

    def test_pcr_extreme_values(self):
        cboe = _synthetic_cboe()
        result = build_options_regime_factors(cboe)
        extremes = result["equity_put_call_extreme"].unique()
        assert set(extremes) <= {-1.0, 0.0, 1.0}

    def test_vix_zscore(self):
        cboe = _synthetic_cboe(n=300)
        result = build_options_regime_factors(cboe)
        # z-score should be present after enough data
        valid = result["vix_zscore_252d"].dropna()
        assert len(valid) > 0
        assert all(np.isfinite(valid))

    def test_empty_input(self):
        result = build_options_regime_factors(pd.DataFrame())
        assert "timestamp" in result.columns
        assert len(result) == 0

    def test_missing_vix_column(self):
        cboe = pd.DataFrame(
            {
                "timestamp": pd.bdate_range("2024-01-01", periods=10),
                "put_call_ratio": [0.9] * 10,
            }
        )
        result = build_options_regime_factors(cboe)
        assert "put_call_ratio_raw" in result.columns
        assert "vix_level" not in result.columns

    def test_missing_pcr_column(self):
        cboe = pd.DataFrame(
            {
                "timestamp": pd.bdate_range("2024-01-01", periods=10),
                "vix": [20.0] * 10,
                "vix3m": [22.0] * 10,
            }
        )
        result = build_options_regime_factors(cboe)
        assert "vix_level" in result.columns
        assert "put_call_ratio_raw" not in result.columns


@pytest.mark.fast
class TestAlignOptionsToPanel:
    def test_merge(self):
        cboe = _synthetic_cboe(n=50)
        factors = build_options_regime_factors(cboe)
        dates = cboe["timestamp"].tolist()
        panel = pd.DataFrame(
            {
                "timestamp": dates * 2,
                "symbol": ["AAPL"] * 50 + ["MSFT"] * 50,
                "close": np.random.default_rng(1).normal(100, 10, 100),
            }
        )
        merged = align_options_factors_to_panel(panel, factors)
        assert "vix_level" in merged.columns
        assert len(merged) == 100

    def test_empty_factors(self):
        panel = pd.DataFrame(
            {"timestamp": [pd.Timestamp("2024-01-01")], "symbol": ["A"], "close": [100]}
        )
        result = align_options_factors_to_panel(panel, pd.DataFrame())
        assert len(result) == 1


@pytest.mark.fast
class TestGetFactorNames:
    def test_returns_list(self):
        names = get_options_factor_names()
        assert isinstance(names, list)
        assert "vix_level" in names
        assert "equity_put_call_extreme" in names
        assert len(names) == 9


@pytest.mark.fast
class TestHelperFunctions:
    def test_vix_term_structure_contango(self):
        result = compute_vix_term_structure(vix=15.0, vix3m=20.0)
        assert result > 1.0  # contango

    def test_vix_term_structure_backwardation(self):
        result = compute_vix_term_structure(vix=30.0, vix3m=25.0)
        assert result < 1.0  # backwardation

    def test_vix_term_structure_zero_vix(self):
        result = compute_vix_term_structure(vix=0.0, vix3m=20.0)
        assert result == 1.0  # safe fallback

    def test_implied_vs_realized_spread(self):
        spread = compute_implied_vs_realized_spread(vix=20.0, realized_vol_20d=0.15)
        assert spread == pytest.approx(0.05, abs=0.001)

    def test_skew_vix_divergence(self):
        result = compute_skew_vix_divergence(skew=130.0, vix=20.0)
        assert result == pytest.approx(6.5, abs=0.01)

    def test_skew_vix_divergence_low_vix(self):
        result = compute_skew_vix_divergence(skew=130.0, vix=0.5)
        assert result == 0.0
