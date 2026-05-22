"""Tests for Factor Report Workflow (Phase C, Sprint C2).

Tests the high-level factor report workflow:
- run_factor_report()
- Integration with Phase A factors
- CLI integration (optional)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.qa.factor_analysis import run_factor_report


@pytest.fixture
def sample_price_panel() -> pd.DataFrame:
    """Create a synthetic price panel with 3 symbols and 200 days of data."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]

    all_data = []
    for symbol in symbols:
        base_price = 100.0 + hash(symbol) % 100

        # Create price series with trend
        price_series = base_price + np.cumsum(np.random.randn(200) * 0.02) * base_price

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": symbol,
                "close": price_series,
                "open": price_series * 0.99,
                "high": price_series * 1.01,
                "low": price_series * 0.98,
                "volume": np.random.randint(1000000, 10000000, size=200),
            }
        )
        all_data.append(df)

    return (
        pd.concat(all_data, ignore_index=True)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


class TestRunFactorReport:
    """Tests for run_factor_report() function."""

    def test_basic_functionality_core(self, sample_price_panel):
        """Test that run_factor_report runs with core factor set."""
        results = run_factor_report(
            prices=sample_price_panel,
            factor_set="core",
            fwd_horizon_days=5,
        )

        # Should return dictionary with expected keys
        assert "factors" in results
        assert "ic" in results
        assert "rank_ic" in results
        assert "summary_ic" in results
        assert "summary_rank_ic" in results

    def test_factors_dataframe_structure(self, sample_price_panel):
        """Test that factors DataFrame has expected structure."""
        results = run_factor_report(
            prices=sample_price_panel,
            factor_set="core",
            fwd_horizon_days=5,
        )

        factors_df = results["factors"]

        # Should preserve original columns
        assert "timestamp" in factors_df.columns
        assert "symbol" in factors_df.columns
        assert "close" in factors_df.columns

        # Should have forward returns
        assert "fwd_return_5d" in factors_df.columns

        # Should have factor columns (at least some)
        factor_cols = [
            col
            for col in factors_df.columns
            if any(
                pattern in col
                for pattern in ["returns_", "momentum_", "trend_strength_", "reversal_"]
            )
        ]
        assert len(factor_cols) > 0, "Should have at least some factor columns"

    def test_ic_not_empty(self, sample_price_panel):
        """Test that IC DataFrame is not empty."""
        results = run_factor_report(
            prices=sample_price_panel,
            factor_set="core",
            fwd_horizon_days=5,
        )

        ic_df = results["ic"]

        # Should have columns
        assert "timestamp" in ic_df.columns
        assert "factor" in ic_df.columns
        assert "ic" in ic_df.columns

        # Should have some rows (may be fewer than timestamps due to forward return requirements)
        assert len(ic_df) > 0

    def test_summary_ic_not_empty(self, sample_price_panel):
        """Test that summary IC DataFrame is not empty."""
        results = run_factor_report(
            prices=sample_price_panel,
            factor_set="core",
            fwd_horizon_days=5,
        )

        summary_ic = results["summary_ic"]

        # Should have expected columns
        assert "factor" in summary_ic.columns
        assert "mean_ic" in summary_ic.columns
        assert "std_ic" in summary_ic.columns
        assert "ic_ir" in summary_ic.columns
        assert "hit_ratio" in summary_ic.columns

        # Should have at least one factor
        assert len(summary_ic) > 0

    def test_vol_liquidity_factor_set(self, sample_price_panel):
        """Test that vol_liquidity factor set works."""
        results = run_factor_report(
            prices=sample_price_panel,
            factor_set="vol_liquidity",
            fwd_horizon_days=5,
        )

        factors_df = results["factors"]

        # Should have volatility factors
        vol_cols = [col for col in factors_df.columns if col.startswith("rv_")]
        assert len(vol_cols) > 0, "Should have realized volatility columns"

        # Should have vol-of-vol if RV columns exist (may be empty if not enough data)
        [col for col in factors_df.columns if col.startswith("vov_")]
        # Structure should be there, but columns may be empty if not enough data

        # Should have summary (may be empty if no IC computed)
        assert "summary_ic" in results

    def test_all_factor_set(self, sample_price_panel):
        """Test that 'all' factor set combines core and vol_liquidity."""
        results = run_factor_report(
            prices=sample_price_panel,
            factor_set="all",
            fwd_horizon_days=5,
        )

        factors_df = results["factors"]

        # Should have core factors
        core_factor_cols = [
            col
            for col in factors_df.columns
            if any(
                pattern in col
                for pattern in ["returns_", "momentum_", "trend_strength_"]
            )
        ]
        assert len(core_factor_cols) > 0, "Should have core factor columns"

        # Should have vol factors (if volume data available)
        [col for col in factors_df.columns if col.startswith("rv_")]
        # May be empty if no volume, but should attempt to compute

        # Should have summary
        assert len(results["summary_ic"]) > 0

    def test_custom_horizon(self, sample_price_panel):
        """Test with custom forward horizon."""
        results = run_factor_report(
            prices=sample_price_panel,
            factor_set="core",
            fwd_horizon_days=21,
        )

        factors_df = results["factors"]

        # Should have forward returns with correct horizon
        assert "fwd_return_21d" in factors_df.columns

        # IC should be computed
        assert len(results["summary_ic"]) > 0

    def test_summary_ic_statistics_valid(self, sample_price_panel):
        """Test that summary IC statistics have valid ranges."""
        results = run_factor_report(
            prices=sample_price_panel,
            factor_set="core",
            fwd_horizon_days=5,
        )

        summary_ic = results["summary_ic"]

        if len(summary_ic) > 0:
            # Mean IC should be in reasonable range
            mean_ics = summary_ic["mean_ic"].dropna()
            if len(mean_ics) > 0:
                assert (mean_ics >= -1.0).all(), "Mean IC should be >= -1"
                assert (mean_ics <= 1.0).all(), "Mean IC should be <= 1"

            # Hit ratio should be between 0 and 1
            hit_ratios = summary_ic["hit_ratio"].dropna()
            if len(hit_ratios) > 0:
                assert (hit_ratios >= 0.0).all(), "Hit ratio should be >= 0"
                assert (hit_ratios <= 1.0).all(), "Hit ratio should be <= 1"

            # Count should be positive
            counts = summary_ic["count"].dropna()
            if len(counts) > 0:
                assert (counts > 0).all(), "Count should be > 0"

    def test_rank_ic_summary(self, sample_price_panel):
        """Test that Rank-IC summary is generated."""
        results = run_factor_report(
            prices=sample_price_panel,
            factor_set="core",
            fwd_horizon_days=5,
        )

        summary_rank_ic = results["summary_rank_ic"]

        # Should have same structure as IC summary
        assert "factor" in summary_rank_ic.columns
        assert "mean_ic" in summary_rank_ic.columns
        assert "ic_ir" in summary_rank_ic.columns

        # Should have at least one factor
        assert len(summary_rank_ic) > 0

    def test_invalid_factor_set_raises_error(self, sample_price_panel):
        """Test that invalid factor_set raises ValueError."""
        with pytest.raises(ValueError, match="Invalid factor_set"):
            run_factor_report(
                prices=sample_price_panel,
                factor_set="invalid",
                fwd_horizon_days=5,
            )

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame(columns=["timestamp", "symbol", "close"])

        with pytest.raises(ValueError, match="empty"):
            run_factor_report(
                prices=empty_df,
                factor_set="core",
                fwd_horizon_days=5,
            )

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises KeyError."""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "close": [100.0],
                # Missing timestamp
            }
        )

        with pytest.raises(KeyError, match="Missing required columns"):
            run_factor_report(
                prices=df,
                factor_set="core",
                fwd_horizon_days=5,
            )


class TestFactorReportIntegration:
    """Tests for integration with Phase A factors."""

    def test_core_factors_included(self, sample_price_panel):
        """Test that core factors from Phase A are included."""
        results = run_factor_report(
            prices=sample_price_panel,
            factor_set="core",
            fwd_horizon_days=5,
        )

        factors_df = results["factors"]

        # Check for expected core factor patterns
        expected_patterns = ["returns_", "momentum_", "trend_strength_", "reversal_"]
        found_patterns = []

        for pattern in expected_patterns:
            matching_cols = [col for col in factors_df.columns if pattern in col]
            if matching_cols:
                found_patterns.append(pattern)

        # Should have at least some expected patterns
        assert len(found_patterns) > 0, (
            f"Should have factors matching patterns: {expected_patterns}"
        )

    def test_summary_sorted_by_ic_ir(self, sample_price_panel):
        """Test that summary is sorted by IC-IR (descending)."""
        results = run_factor_report(
            prices=sample_price_panel,
            factor_set="core",
            fwd_horizon_days=5,
        )

        summary_ic = results["summary_ic"]

        if len(summary_ic) > 1:
            # Should be sorted by IC-IR descending
            ic_ir_values = summary_ic["ic_ir"].values
            is_sorted = (ic_ir_values == sorted(ic_ir_values, reverse=True)).all()
            assert is_sorted, "Summary should be sorted by IC-IR (descending)"


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_minimal_data(self):
        """Test with minimal data (few symbols, few days)."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
        symbols = ["STOCK1", "STOCK2"]

        all_data = []
        for symbol in symbols:
            df = pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "close": 100.0 + np.arange(50) * 0.5,
                    "volume": np.random.randint(1000000, 10000000, size=50),
                }
            )
            all_data.append(df)

        df = pd.concat(all_data, ignore_index=True)

        # Should work, but may have limited factors/IC values (or empty IC if not enough data)
        results = run_factor_report(
            prices=df,
            factor_set="core",
            fwd_horizon_days=5,
        )

        assert isinstance(results, dict)
        assert "summary_ic" in results
        assert "summary_rank_ic" in results
        # Summary may be empty if not enough data for IC calculation
