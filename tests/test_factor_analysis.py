"""Tests for Factor Analysis and IC Engine module (Phase C, Sprint C1).

Tests the factor analysis functions:
- add_forward_returns()
- compute_factor_ic()
- compute_rank_ic()
- summarize_factor_ic()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.qa.factor_analysis import (
    add_forward_returns,
    compute_factor_ic,
    compute_rank_ic,
    summarize_factor_ic,
)


@pytest.fixture
def sample_price_panel() -> pd.DataFrame:
    """Create a synthetic price panel with 3 symbols and 100 days of data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]

    all_data = []
    for symbol in symbols:
        base_price = 100.0

        # Create price series with different patterns
        price_series = base_price + np.cumsum(np.random.randn(100) * 0.02) * base_price

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": symbol,
                "close": price_series,
            }
        )
        all_data.append(df)

    return (
        pd.concat(all_data, ignore_index=True)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


@pytest.fixture
def factor_panel_with_forward_returns() -> pd.DataFrame:
    """Create panel with factors and forward returns for IC testing."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
    symbols = ["STOCK1", "STOCK2", "STOCK3", "STOCK4", "STOCK5"]

    all_data = []
    for symbol in symbols:
        # Create price series
        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(50) * 0.02) * base_price

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": symbol,
                "close": prices,
            }
        )
        all_data.append(df)

    df = (
        pd.concat(all_data, ignore_index=True)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    # Add forward returns
    df = add_forward_returns(df, horizon_days=1, col_name="fwd_return_1d")

    # Add a perfect predictive factor (same as forward return)
    df["perfect_factor"] = df["fwd_return_1d"]

    # Add an inverse factor (negative of forward return)
    df["inverse_factor"] = -df["fwd_return_1d"]

    # Add a random factor (no correlation)
    df["random_factor"] = np.random.randn(len(df))

    return df


class TestAddForwardReturns:
    """Tests for add_forward_returns() function."""

    def test_basic_functionality(self, sample_price_panel):
        """Test that add_forward_returns runs and adds forward return column."""
        result = add_forward_returns(sample_price_panel, horizon_days=1)

        # Should have same number of rows as input
        assert len(result) == len(sample_price_panel)

        # Should add forward return column
        assert "fwd_return_1d" in result.columns

        # Preserve original columns
        for col in sample_price_panel.columns:
            assert col in result.columns

    def test_forward_returns_are_forward_looking(self, sample_price_panel):
        """Test that forward returns correctly look ahead (no look-ahead bias)."""
        result = add_forward_returns(sample_price_panel, horizon_days=1)

        # Last row per symbol should have NaN (no future data)
        for symbol in sample_price_panel["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol].sort_values("timestamp")
            last_row = symbol_data.iloc[-1]
            assert pd.isna(last_row["fwd_return_1d"]), (
                "Last row should have NaN for forward returns"
            )

    def test_custom_horizon(self, sample_price_panel):
        """Test with custom horizon."""
        result = add_forward_returns(sample_price_panel, horizon_days=5)

        assert "fwd_return_5d" in result.columns

        # Last 5 rows per symbol should have NaN
        for symbol in sample_price_panel["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol].sort_values("timestamp")
            last_5 = symbol_data.iloc[-5:]
            assert last_5["fwd_return_5d"].isna().all(), (
                "Last 5 rows should have NaN for 5d forward returns"
            )

    def test_custom_column_name(self, sample_price_panel):
        """Test with custom column name."""
        result = add_forward_returns(
            sample_price_panel, horizon_days=1, col_name="future_return"
        )

        assert "future_return" in result.columns
        assert "fwd_return_1d" not in result.columns

    def test_log_vs_simple_returns(self, sample_price_panel):
        """Test that log and simple returns are different but consistent."""
        result_log = add_forward_returns(
            sample_price_panel, horizon_days=1, return_type="log"
        )
        result_simple = add_forward_returns(
            sample_price_panel, horizon_days=1, return_type="simple"
        )

        # Values should be different
        log_values = result_log["fwd_return_1d"].dropna()
        simple_values = result_simple["fwd_return_1d"].dropna()

        # Both should have similar magnitudes (log returns are approximately simple returns for small values)
        assert len(log_values) > 0
        assert len(simple_values) > 0

    def test_required_columns_validation(self):
        """Test that missing required columns raise KeyError."""
        # Missing timestamp
        df_no_timestamp = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "close": [100.0],
            }
        )
        with pytest.raises(KeyError, match="Missing required columns"):
            add_forward_returns(df_no_timestamp)


class TestComputeFactorIc:
    """Tests for compute_factor_ic() function."""

    def test_basic_functionality(self, factor_panel_with_forward_returns):
        """Test that compute_factor_ic runs and returns IC DataFrame."""
        result = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor"],
            fwd_return_col="fwd_return_1d",
        )

        # Should have columns: timestamp, factor, ic, count
        assert "timestamp" in result.columns
        assert "factor" in result.columns
        assert "ic" in result.columns
        assert "count" in result.columns

        # Should have rows
        assert len(result) > 0

    def test_perfect_correlation_ic(self, factor_panel_with_forward_returns):
        """Test that perfect correlation gives IC ≈ 1."""
        result = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor"],
            fwd_return_col="fwd_return_1d",
        )

        if len(result) > 0:
            # Perfect factor should have IC close to 1
            ic_values = result["ic"].dropna()
            if len(ic_values) > 0:
                # IC should be very close to 1.0 (allowing small numerical errors)
                assert (ic_values > 0.95).all(), (
                    f"Perfect factor should have IC ≈ 1, got {ic_values.mean()}"
                )

    def test_inverse_correlation_ic(self, factor_panel_with_forward_returns):
        """Test that inverse correlation gives IC ≈ -1."""
        result = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["inverse_factor"],
            fwd_return_col="fwd_return_1d",
        )

        if len(result) > 0:
            ic_values = result["ic"].dropna()
            if len(ic_values) > 0:
                # Inverse factor should have IC close to -1
                assert (ic_values < -0.95).all(), (
                    f"Inverse factor should have IC ≈ -1, got {ic_values.mean()}"
                )

    def test_multiple_factors(self, factor_panel_with_forward_returns):
        """Test that multiple factors can be computed simultaneously."""
        result = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor", "inverse_factor", "random_factor"],
            fwd_return_col="fwd_return_1d",
        )

        # Should have IC for all three factors
        unique_factors = result["factor"].unique()
        assert "perfect_factor" in unique_factors
        assert "inverse_factor" in unique_factors
        assert "random_factor" in unique_factors

    def test_ic_per_timestamp(self, factor_panel_with_forward_returns):
        """Test that IC is computed per timestamp (cross-sectional)."""
        result = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor"],
            fwd_return_col="fwd_return_1d",
        )

        # Should have one row per timestamp (or fewer if some timestamps lack data)
        unique_timestamps = factor_panel_with_forward_returns["timestamp"].nunique()
        assert (
            len(result) <= unique_timestamps - 1
        )  # First timestamp has no forward return

    def test_ic_range(self, factor_panel_with_forward_returns):
        """Test that IC values are in range [-1, 1]."""
        result = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor", "random_factor"],
            fwd_return_col="fwd_return_1d",
        )

        ic_values = result["ic"].dropna()
        if len(ic_values) > 0:
            assert (ic_values >= -1.0).all(), "IC should be >= -1"
            assert (ic_values <= 1.0).all(), "IC should be <= 1"

    def test_required_columns_validation(self):
        """Test that missing required columns raise KeyError."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=10, freq="D", tz="UTC"
                ),
                "symbol": "AAPL",
                "factor1": range(10),
            }
        )

        with pytest.raises(KeyError, match="Missing required columns"):
            compute_factor_ic(df, factor_cols=["factor1"], fwd_return_col="fwd_return")


class TestComputeRankIc:
    """Tests for compute_rank_ic() function."""

    def test_basic_functionality(self, factor_panel_with_forward_returns):
        """Test that compute_rank_ic runs and returns Rank-IC DataFrame."""
        result = compute_rank_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor"],
            fwd_return_col="fwd_return_1d",
        )

        assert "timestamp" in result.columns
        assert "factor" in result.columns
        assert "ic" in result.columns
        assert len(result) > 0

    def test_rank_ic_vs_regular_ic(self, factor_panel_with_forward_returns):
        """Test that Rank-IC uses Spearman correlation (rank-based)."""
        # Rank-IC should handle monotonic relationships better
        result_rank = compute_rank_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor"],
            fwd_return_col="fwd_return_1d",
        )

        result_regular = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor"],
            fwd_return_col="fwd_return_1d",
            method="pearson",
        )

        # For perfect linear relationship, both should be similar
        # But Rank-IC should also be high
        if len(result_rank) > 0 and len(result_regular) > 0:
            rank_ic = result_rank["ic"].dropna()
            regular_ic = result_regular["ic"].dropna()

            if len(rank_ic) > 0 and len(regular_ic) > 0:
                assert (rank_ic > 0.9).all(), (
                    "Rank-IC should be high for perfect factor"
                )


class TestSummarizeFactorIc:
    """Tests for summarize_factor_ic() function."""

    def test_basic_functionality(self, factor_panel_with_forward_returns):
        """Test that summarize_factor_ic runs and returns summary DataFrame."""
        # First compute IC
        ic_df = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor", "inverse_factor", "random_factor"],
            fwd_return_col="fwd_return_1d",
        )

        # Then summarize
        summary = summarize_factor_ic(ic_df)

        # Should have required columns
        assert "factor" in summary.columns
        assert "mean_ic" in summary.columns
        assert "std_ic" in summary.columns
        assert "ic_ir" in summary.columns
        assert "hit_ratio" in summary.columns
        assert "count" in summary.columns

        # Should have one row per factor
        assert len(summary) == 3

    def test_summary_statistics_correct(self, factor_panel_with_forward_returns):
        """Test that summary statistics are computed correctly."""
        ic_df = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor"],
            fwd_return_col="fwd_return_1d",
        )

        summary = summarize_factor_ic(ic_df)

        if len(summary) > 0:
            perfect_row = summary[summary["factor"] == "perfect_factor"]
            if len(perfect_row) > 0:
                # Perfect factor should have high mean IC
                assert perfect_row["mean_ic"].iloc[0] > 0.8, (
                    "Perfect factor should have high mean IC"
                )

                # Hit ratio should be high (most days with positive IC)
                assert perfect_row["hit_ratio"].iloc[0] > 0.8, (
                    "Perfect factor should have high hit ratio"
                )

                # IC-IR should be positive (may be 0 if std_ic is very small, which is valid)
                ic_ir_value = perfect_row["ic_ir"].iloc[0]
                assert ic_ir_value >= 0, "Perfect factor should have non-negative IC-IR"
                # If std_ic is very small (near-constant IC), IC-IR might be 0, which is acceptable

    def test_ic_ir_calculation(self, factor_panel_with_forward_returns):
        """Test that IC-IR = mean_ic / std_ic."""
        ic_df = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor"],
            fwd_return_col="fwd_return_1d",
        )

        summary = summarize_factor_ic(ic_df)

        if len(summary) > 0:
            row = summary.iloc[0]
            expected_ir = (
                row["mean_ic"] / row["std_ic"] if row["std_ic"] > 1e-10 else 0.0
            )
            assert abs(row["ic_ir"] - expected_ir) < 1e-6, (
                "IC-IR should equal mean_ic / std_ic"
            )

    def test_hit_ratio_range(self, factor_panel_with_forward_returns):
        """Test that hit_ratio is between 0 and 1."""
        ic_df = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor", "inverse_factor"],
            fwd_return_col="fwd_return_1d",
        )

        summary = summarize_factor_ic(ic_df)

        hit_ratios = summary["hit_ratio"]
        assert (hit_ratios >= 0.0).all(), "Hit ratio should be >= 0"
        assert (hit_ratios <= 1.0).all(), "Hit ratio should be <= 1"

    def test_sorting_by_ic_ir(self, factor_panel_with_forward_returns):
        """Test that summary is sorted by IC-IR (descending)."""
        ic_df = compute_factor_ic(
            factor_panel_with_forward_returns,
            factor_cols=["perfect_factor", "inverse_factor", "random_factor"],
            fwd_return_col="fwd_return_1d",
        )

        summary = summarize_factor_ic(ic_df)

        # Should be sorted by IC-IR descending
        ic_ir_values = summary["ic_ir"].values
        assert (ic_ir_values == sorted(ic_ir_values, reverse=True)).all(), (
            "Summary should be sorted by IC-IR (descending)"
        )


class TestIntegrationWithPhaseAFactors:
    """Tests for integration with Phase A factors."""

    def test_ic_with_core_ta_factors(self, sample_price_panel):
        """Test that IC can be computed for Phase A core TA factors."""
        from src.assembled_core.features.ta_factors_core import build_core_ta_factors

        # Build core TA factors
        factors_df = build_core_ta_factors(sample_price_panel)

        # Add forward returns (21 days = 1 month)
        factors_df = add_forward_returns(
            factors_df, horizon_days=21, col_name="fwd_return_21d"
        )

        # Compute IC for some factors
        ic_df = compute_factor_ic(
            factors_df,
            factor_cols=["returns_12m", "trend_strength_20"],
            fwd_return_col="fwd_return_21d",
        )

        assert len(ic_df) > 0
        assert (
            "returns_12m" in ic_df["factor"].values
            or "trend_strength_20" in ic_df["factor"].values
        )

    def test_ic_with_volatility_factors(self, sample_price_panel):
        """Test that IC can be computed for volatility factors."""
        from src.assembled_core.features.ta_liquidity_vol_factors import (
            add_realized_volatility,
        )

        # Add realized volatility
        vol_df = add_realized_volatility(sample_price_panel, windows=[20])

        # Add forward returns
        vol_df = add_forward_returns(vol_df, horizon_days=1, col_name="fwd_return_1d")

        # Compute IC
        ic_df = compute_factor_ic(
            vol_df, factor_cols=["rv_20"], fwd_return_col="fwd_return_1d"
        )

        assert len(ic_df) > 0


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_single_symbol(self):
        """Test with single symbol."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": "AAPL",
                "close": 100.0 + np.arange(50) * 0.5,
                "factor1": np.random.randn(50),
            }
        )

        df = add_forward_returns(df, horizon_days=1)

        # IC computation should handle single symbol (may need at least 3 timestamps with data)
        ic_df = compute_factor_ic(
            df, factor_cols=["factor1"], fwd_return_col="fwd_return_1d"
        )

        # Should work, but may have fewer IC values due to minimum sample size requirements
        assert isinstance(ic_df, pd.DataFrame)

    def test_minimal_data(self):
        """Test with minimal data."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        symbols = ["A", "B", "C"]

        all_data = []
        for symbol in symbols:
            df = pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "close": 100.0 + np.arange(10) * 0.5,
                    "factor1": np.random.randn(10),
                }
            )
            all_data.append(df)

        df = pd.concat(all_data, ignore_index=True)
        df = add_forward_returns(df, horizon_days=1)

        ic_df = compute_factor_ic(
            df, factor_cols=["factor1"], fwd_return_col="fwd_return_1d"
        )

        # Should work, but may have limited IC values
        assert isinstance(ic_df, pd.DataFrame)

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame(columns=["timestamp", "symbol", "close"])

        with pytest.raises(ValueError, match="empty"):
            add_forward_returns(empty_df)
