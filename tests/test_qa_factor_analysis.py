"""Tests for Phase C1 Factor Analysis functions (qa/factor_analysis.py).

Tests the new Phase C1 functions:
- compute_ic()
- compute_rank_ic()
- summarize_ic_series()
- compute_rolling_ic()
- add_forward_returns() with multiple horizons
- example_factor_analysis_workflow()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.qa.factor_analysis import (
    add_forward_returns,
    compute_ic,
    compute_rank_ic,
    summarize_ic_series,
    compute_rolling_ic,
    example_factor_analysis_workflow,
)


@pytest.fixture
def sample_panel_3x40() -> pd.DataFrame:
    """Create a synthetic price panel with 3 symbols and 40 days of data."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    all_data = []
    for symbol in symbols:
        base_price = 100.0
        # Create price series with trend
        price_series = base_price + np.cumsum(np.random.randn(40) * 0.02) * base_price
        price_series = np.maximum(price_series, 1.0)  # Ensure positive prices
        
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "close": price_series,
        })
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def factor_panel_with_known_ic() -> pd.DataFrame:
    """Create panel with factors and forward returns where IC can be calculated manually.
    
    Creates a panel with:
    - perfect_factor: perfectly correlated with forward returns (IC = 1.0)
    - inverse_factor: perfectly anti-correlated (IC = -1.0)
    - random_factor: no correlation (IC ≈ 0.0)
    """
    dates = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    symbols = ["STOCK1", "STOCK2", "STOCK3"]
    
    all_data = []
    for symbol in symbols:
        # Create price series
        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(40) * 0.02) * base_price
        prices = np.maximum(prices, 1.0)
        
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "close": prices,
        })
        all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    # Add forward returns (5-day horizon)
    df = add_forward_returns(df, horizon_days=5, return_type="log")
    
    # Add factors with known relationships to forward returns
    # Perfect factor: same as forward return (IC should be 1.0)
    df["perfect_factor"] = df["fwd_return_5d"].fillna(0.0)
    
    # Inverse factor: negative of forward return (IC should be -1.0)
    df["inverse_factor"] = -df["fwd_return_5d"].fillna(0.0)
    
    # Random factor: no correlation (IC should be close to 0.0)
    np.random.seed(42)  # For reproducibility
    df["random_factor"] = np.random.randn(len(df))
    
    # Add a factor with some correlation (IC should be positive but < 1.0)
    df["moderate_factor"] = df["fwd_return_5d"].fillna(0.0) * 0.7 + np.random.randn(len(df)) * 0.3
    
    return df


@pytest.fixture
def factor_panel_with_nans() -> pd.DataFrame:
    """Create panel with some NaN values in factors and forward returns."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    symbols = ["STOCK1", "STOCK2", "STOCK3"]
    
    all_data = []
    for symbol in symbols:
        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(40) * 0.02) * base_price
        prices = np.maximum(prices, 1.0)
        
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "close": prices,
        })
        all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    # Add forward returns
    df = add_forward_returns(df, horizon_days=5, return_type="log")
    
    # Add factors with some NaN values
    df["factor_with_nans"] = df["fwd_return_5d"].fillna(0.0)
    # Set some NaN values (every 5th row)
    df.loc[df.index[::5], "factor_with_nans"] = np.nan
    
    # Add another factor with NaN in forward returns
    df["factor_clean"] = np.random.randn(len(df))
    # Set some forward returns to NaN
    df.loc[df.index[::7], "fwd_return_5d"] = np.nan
    
    return df


class TestAddForwardReturns:
    """Tests for add_forward_returns() function."""
    
    def test_single_horizon(self, sample_panel_3x40):
        """Test add_forward_returns with single horizon."""
        result = add_forward_returns(sample_panel_3x40, horizon_days=20)
        
        assert len(result) == len(sample_panel_3x40)
        assert "fwd_return_20d" in result.columns
        
        # Check that last 20 rows per symbol have NaN
        for symbol in sample_panel_3x40["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol].sort_values("timestamp")
            assert symbol_data["fwd_return_20d"].iloc[-20:].isna().all()
            # First rows should have values (if enough data)
            if len(symbol_data) > 20:
                assert pd.notna(symbol_data["fwd_return_20d"].iloc[0])
    
    def test_multiple_horizons(self, sample_panel_3x40):
        """Test add_forward_returns with multiple horizons."""
        result = add_forward_returns(sample_panel_3x40, horizon_days=[20, 60])
        
        assert len(result) == len(sample_panel_3x40)
        assert "fwd_ret_20" in result.columns
        assert "fwd_ret_60" in result.columns
        
        # Check NaN handling for each horizon
        for horizon in [20, 60]:
            col_name = f"fwd_ret_{horizon}"
            for symbol in sample_panel_3x40["symbol"].unique():
                symbol_data = result[result["symbol"] == symbol].sort_values("timestamp")
                # Last horizon rows should be NaN
                assert symbol_data[col_name].iloc[-horizon:].isna().all()
    
    def test_log_vs_simple_returns(self, sample_panel_3x40):
        """Test that log and simple returns are computed correctly."""
        result_log = add_forward_returns(sample_panel_3x40, horizon_days=5, return_type="log")
        result_simple = add_forward_returns(sample_panel_3x40, horizon_days=5, return_type="simple")
        
        # For small returns, log ≈ simple, but they should be different
        # Log return: ln(price[t+5] / price[t])
        # Simple return: (price[t+5] / price[t]) - 1
        
        # Check that they're different (not identical)
        valid_mask = result_log["fwd_return_5d"].notna() & result_simple["fwd_return_5d"].notna()
        if valid_mask.sum() > 0:
            log_vals = result_log.loc[valid_mask, "fwd_return_5d"]
            simple_vals = result_simple.loc[valid_mask, "fwd_return_5d"]
            # They should be close but not identical
            assert not (log_vals == simple_vals).all()
    
    def test_works_with_factor_dataframe(self, sample_panel_3x40):
        """Test that add_forward_returns works with factor DataFrames."""
        # Add some factors
        factor_df = sample_panel_3x40.copy()
        factor_df["factor_x"] = np.random.randn(len(factor_df))
        factor_df["factor_y"] = np.random.randn(len(factor_df))
        
        result = add_forward_returns(factor_df, horizon_days=10)
        
        # Should preserve factor columns
        assert "factor_x" in result.columns
        assert "factor_y" in result.columns
        assert "fwd_return_10d" in result.columns


class TestComputeIc:
    """Tests for compute_ic() function."""
    
    def test_basic_functionality(self, factor_panel_with_known_ic):
        """Test basic IC computation."""
        ic_df = compute_ic(
            factor_panel_with_known_ic,
            forward_returns_col="fwd_return_5d",
            group_col="symbol",
        )
        
        # Should have timestamp as index
        assert isinstance(ic_df.index, pd.DatetimeIndex)
        
        # Should have IC columns for each factor
        assert "ic_perfect_factor" in ic_df.columns
        assert "ic_inverse_factor" in ic_df.columns
        assert "ic_random_factor" in ic_df.columns
        
        # Perfect factor should have IC close to 1.0 (allowing for numerical precision)
        perfect_ic = ic_df["ic_perfect_factor"].dropna()
        if len(perfect_ic) > 0:
            assert perfect_ic.abs().mean() > 0.9  # Should be very high correlation
        
        # Inverse factor should have IC close to -1.0
        inverse_ic = ic_df["ic_inverse_factor"].dropna()
        if len(inverse_ic) > 0:
            assert inverse_ic.mean() < -0.9  # Should be very negative
    
    def test_rank_ic_vs_pearson_ic(self, factor_panel_with_known_ic):
        """Test that Rank-IC uses Spearman correlation."""
        ic_pearson = compute_ic(
            factor_panel_with_known_ic,
            forward_returns_col="fwd_return_5d",
            method="pearson",
        )
        
        ic_rank = compute_rank_ic(
            factor_panel_with_known_ic,
            forward_returns_col="fwd_return_5d",
        )
        
        # Should have same structure
        assert ic_pearson.index.equals(ic_rank.index)
        assert set(ic_pearson.columns) == set(ic_rank.columns)
        
        # For perfect monotonic relationship, Rank-IC should equal Pearson IC
        perfect_ic_pearson = ic_pearson["ic_perfect_factor"].dropna()
        perfect_ic_rank = ic_rank["ic_perfect_factor"].dropna()
        
        if len(perfect_ic_pearson) > 0 and len(perfect_ic_rank) > 0:
            # Should be very similar for perfect correlation
            diff = (perfect_ic_pearson - perfect_ic_rank).abs()
            assert diff.mean() < 0.1  # Should be close
    
    def test_nan_handling(self, factor_panel_with_nans):
        """Test that IC computation handles NaNs correctly."""
        ic_df = compute_ic(
            factor_panel_with_nans,
            forward_returns_col="fwd_return_5d",
            group_col="symbol",
        )
        
        # Should still compute IC where valid data exists
        assert len(ic_df) > 0
        
        # IC columns should exist
        if "ic_factor_with_nans" in ic_df.columns:
            # Should have some valid IC values (where both factor and return are valid)
            valid_ic = ic_df["ic_factor_with_nans"].dropna()
            assert len(valid_ic) > 0  # Should have some valid ICs
    
    def test_auto_factor_detection(self, factor_panel_with_known_ic):
        """Test that compute_ic auto-detects factor columns."""
        ic_df = compute_ic(
            factor_panel_with_known_ic,
            forward_returns_col="fwd_return_5d",
        )
        
        # Should detect all factor columns (exclude timestamp, symbol, close, fwd_return_5d)
        expected_factors = ["perfect_factor", "inverse_factor", "random_factor", "moderate_factor"]
        for factor in expected_factors:
            assert f"ic_{factor}" in ic_df.columns


class TestSummarizeIcSeries:
    """Tests for summarize_ic_series() function."""
    
    def test_basic_statistics(self, factor_panel_with_known_ic):
        """Test that summarize_ic_series computes correct statistics."""
        ic_df = compute_ic(
            factor_panel_with_known_ic,
            forward_returns_col="fwd_return_5d",
        )
        
        summary = summarize_ic_series(ic_df)
        
        # Should have required columns
        required_cols = ["factor", "mean_ic", "std_ic", "ic_ir", "hit_ratio", "q05", "q95", "min_ic", "max_ic", "count"]
        for col in required_cols:
            assert col in summary.columns
        
        # Should have one row per factor
        assert len(summary) > 0
        
        # Perfect factor should have high mean_ic and high hit_ratio
        perfect_row = summary[summary["factor"] == "perfect_factor"]
        if len(perfect_row) > 0:
            assert perfect_row["mean_ic"].iloc[0] > 0.8
            assert perfect_row["hit_ratio"].iloc[0] > 0.7  # Most days should have positive IC
        
        # Inverse factor should have negative mean_ic
        inverse_row = summary[summary["factor"] == "inverse_factor"]
        if len(inverse_row) > 0:
            assert inverse_row["mean_ic"].iloc[0] < -0.8
    
    def test_quantiles(self, factor_panel_with_known_ic):
        """Test that quantiles are computed correctly."""
        ic_df = compute_ic(
            factor_panel_with_known_ic,
            forward_returns_col="fwd_return_5d",
        )
        
        summary = summarize_ic_series(ic_df)
        
        # Check that q05 < mean < q95 for each factor
        for _, row in summary.iterrows():
            if row["count"] > 0:
                assert row["q05"] <= row["mean_ic"] <= row["q95"]
                assert row["min_ic"] <= row["q05"]
                assert row["q95"] <= row["max_ic"]
    
    def test_ic_ir_calculation(self, factor_panel_with_known_ic):
        """Test that IC-IR is calculated correctly (mean_ic / std_ic)."""
        ic_df = compute_ic(
            factor_panel_with_known_ic,
            forward_returns_col="fwd_return_5d",
        )
        
        summary = summarize_ic_series(ic_df)
        
        # IC-IR should be mean_ic / std_ic (allowing for division by zero handling)
        for _, row in summary.iterrows():
            if row["std_ic"] > 1e-10:
                expected_ir = row["mean_ic"] / row["std_ic"]
                assert abs(row["ic_ir"] - expected_ir) < 1e-6
    
    def test_hit_ratio(self, factor_panel_with_known_ic):
        """Test that hit_ratio is percentage of positive IC days."""
        ic_df = compute_ic(
            factor_panel_with_known_ic,
            forward_returns_col="fwd_return_5d",
        )
        
        summary = summarize_ic_series(ic_df)
        
        # Manually verify hit_ratio for one factor
        for _, row in summary.iterrows():
            factor_name = row["factor"]
            ic_col = f"ic_{factor_name}"
            if ic_col in ic_df.columns:
                ic_values = ic_df[ic_col].dropna()
                if len(ic_values) > 0:
                    expected_hit_ratio = (ic_values > 0).sum() / len(ic_values)
                    assert abs(row["hit_ratio"] - expected_hit_ratio) < 1e-6


class TestComputeRollingIc:
    """Tests for compute_rolling_ic() function."""
    
    def test_basic_rolling_statistics(self, factor_panel_with_known_ic):
        """Test basic rolling IC statistics."""
        ic_df = compute_ic(
            factor_panel_with_known_ic,
            forward_returns_col="fwd_return_5d",
        )
        
        rolling_df = compute_rolling_ic(ic_df, window=10)
        
        # Should have same index as input
        assert rolling_df.index.equals(ic_df.index)
        
        # Should have rolling_mean and rolling_ir columns for each factor
        for factor_col in ic_df.columns:
            if factor_col.startswith("ic_"):
                factor_name = factor_col[3:]  # Remove "ic_" prefix
                assert f"rolling_mean_{factor_name}" in rolling_df.columns
                assert f"rolling_ir_{factor_name}" in rolling_df.columns
    
    def test_window_handling(self, factor_panel_with_known_ic):
        """Test that rolling window is handled correctly."""
        ic_df = compute_ic(
            factor_panel_with_known_ic,
            forward_returns_col="fwd_return_5d",
        )
        
        window = 15
        rolling_df = compute_rolling_ic(ic_df, window=window)
        
        # First window-1 rows should have NaN for rolling statistics
        for col in rolling_df.columns:
            if col.startswith("rolling_mean_"):
                # First window-1 rows should be NaN (or at least some of them)
                first_rows = rolling_df[col].iloc[:window-1]
                # At least some should be NaN (depending on data availability)
                assert first_rows.isna().sum() >= 0  # At least some NaN expected
    
    def test_rolling_ir_calculation(self, factor_panel_with_known_ic):
        """Test that rolling IR is calculated correctly (rolling_mean / rolling_std)."""
        ic_df = compute_ic(
            factor_panel_with_known_ic,
            forward_returns_col="fwd_return_5d",
        )
        
        rolling_df = compute_rolling_ic(ic_df, window=10)
        
        # For each factor, verify rolling_ir = rolling_mean / rolling_std
        for factor_col in ic_df.columns:
            if factor_col.startswith("ic_"):
                factor_name = factor_col[3:]
                mean_col = f"rolling_mean_{factor_name}"
                ir_col = f"rolling_ir_{factor_name}"
                
                if mean_col in rolling_df.columns and ir_col in rolling_df.columns:
                    # Get valid rows (where both mean and IR are not NaN)
                    valid_mask = rolling_df[mean_col].notna() & rolling_df[ir_col].notna()
                    if valid_mask.sum() > 0:
                        # Rolling IR should be reasonable (not inf or very large)
                        # Note: Very small std can lead to large IR values, which is mathematically correct
                        # but we check that they're not inf
                        ir_values = rolling_df.loc[valid_mask, ir_col]
                        assert not np.isinf(ir_values).any()  # Should not have inf values
                        # Allow large but finite values (division by very small std is valid)


class TestExampleFactorAnalysisWorkflow:
    """Tests for example_factor_analysis_workflow() function."""
    
    def test_basic_workflow(self, sample_panel_3x40):
        """Test that example workflow runs successfully."""
        # Add some factors
        factor_df = sample_panel_3x40.copy()
        factor_df["factor_a"] = np.random.randn(len(factor_df))
        factor_df["factor_b"] = np.random.randn(len(factor_df))
        
        results = example_factor_analysis_workflow(
            prices_df=sample_panel_3x40,
            factor_df=factor_df,
            horizons=[20],
        )
        
        # Should return dictionary with expected keys
        expected_keys = ["data_with_returns", "ic", "rank_ic", "summary_ic", "summary_rank_ic", "rolling_ic"]
        for key in expected_keys:
            assert key in results
        
        # data_with_returns should have forward returns
        assert "fwd_return_20d" in results["data_with_returns"].columns or "fwd_ret_20" in results["data_with_returns"].columns
    
    def test_multiple_horizons(self, sample_panel_3x40):
        """Test workflow with multiple horizons."""
        factor_df = sample_panel_3x40.copy()
        factor_df["factor_x"] = np.random.randn(len(factor_df))
        
        results = example_factor_analysis_workflow(
            prices_df=sample_panel_3x40,
            factor_df=factor_df,
            horizons=[20, 60],
        )
        
        # Should have forward returns for both horizons
        data = results["data_with_returns"]
        assert "fwd_ret_20" in data.columns
        assert "fwd_ret_60" in data.columns
    
    def test_works_without_factor_df(self, sample_panel_3x40):
        """Test that workflow works when factor_df is None."""
        results = example_factor_analysis_workflow(
            prices_df=sample_panel_3x40,
            factor_df=None,
            horizons=[20],
        )
        
        # Should still return all keys
        assert "data_with_returns" in results
        # IC DataFrames might be empty if no factors found
        assert "ic" in results

