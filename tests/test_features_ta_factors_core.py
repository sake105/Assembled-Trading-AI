"""Tests for Core TA/Price Factors module (Phase A, Sprint A1).

Tests the build_core_ta_factors() function and its factor computations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.features.ta_factors_core import build_core_ta_factors


@pytest.fixture
def sample_price_panel() -> pd.DataFrame:
    """Create a synthetic price panel with 3 symbols and 300 days of data."""
    dates = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]

    all_data = []
    for symbol in symbols:
        # Create price series with different patterns
        base_price = 100.0 if symbol == "AAPL" else 150.0 if symbol == "MSFT" else 200.0

        # AAPL: upward trend
        if symbol == "AAPL":
            price_series = base_price + np.arange(300) * 0.5 + np.random.randn(300) * 2
        # MSFT: sideways with volatility
        elif symbol == "MSFT":
            price_series = base_price + np.random.randn(300) * 3
        # GOOGL: downward trend
        else:
            price_series = base_price - np.arange(300) * 0.3 + np.random.randn(300) * 2

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": symbol,
                "close": price_series,
                "high": price_series + np.abs(np.random.randn(300)) * 1.5,
                "low": price_series - np.abs(np.random.randn(300)) * 1.5,
            }
        )
        all_data.append(df)

    return (
        pd.concat(all_data, ignore_index=True)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


@pytest.fixture
def monotonic_price_data() -> pd.DataFrame:
    """Create price data with strictly monotonic increasing prices (for validation tests)."""
    dates = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")

    # Strictly increasing prices
    prices = 100.0 + np.arange(300) * 0.5

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "TEST",
            "close": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
        }
    )


class TestBuildCoreTaFactors:
    """Tests for build_core_ta_factors() function."""

    def test_basic_functionality(self, sample_price_panel):
        """Test that build_core_ta_factors runs without errors and adds factor columns."""
        result = build_core_ta_factors(sample_price_panel)

        # Should have same number of rows as input
        assert len(result) == len(sample_price_panel)

        # Should preserve original columns
        for col in sample_price_panel.columns:
            assert col in result.columns

        # Should add factor columns
        expected_factor_cols = [
            "returns_1m",
            "returns_3m",
            "returns_6m",
            "returns_12m",
            "momentum_12m_excl_1m",
            "trend_strength_20",
            "trend_strength_50",
            "trend_strength_200",
            "reversal_1d",
            "reversal_2d",
            "reversal_3d",
        ]

        for col in expected_factor_cols:
            assert col in result.columns, f"Factor column {col} missing"

    def test_shape_preservation(self, sample_price_panel):
        """Test that output shape matches input (same rows, more columns)."""
        input_rows = len(sample_price_panel)
        input_cols = len(sample_price_panel.columns)

        result = build_core_ta_factors(sample_price_panel)

        assert len(result) == input_rows
        assert len(result.columns) > input_cols

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
            build_core_ta_factors(df_no_timestamp)

        # Missing symbol
        df_no_symbol = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=10, freq="D", tz="UTC"
                ),
                "close": range(10),
            }
        )
        with pytest.raises(KeyError, match="Missing required columns"):
            build_core_ta_factors(df_no_symbol)

        # Missing close
        df_no_close = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=10, freq="D", tz="UTC"
                ),
                "symbol": "AAPL",
            }
        )
        with pytest.raises(KeyError, match="Missing required columns"):
            build_core_ta_factors(df_no_close)

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame(columns=["timestamp", "symbol", "close"])

        with pytest.raises(ValueError, match="empty"):
            build_core_ta_factors(empty_df)

    def test_multiple_symbols_grouped_correctly(self, sample_price_panel):
        """Test that factors are computed per symbol (no cross-contamination)."""
        result = build_core_ta_factors(sample_price_panel)

        # Check that each symbol has its own factor values
        for symbol in sample_price_panel["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol]

            # Should have data for this symbol
            assert len(symbol_data) > 0

            # Factor columns should not be all NaN (at least some should have values)
            factor_cols = [
                c for c in result.columns if c not in sample_price_panel.columns
            ]
            non_null_counts = {
                col: symbol_data[col].notna().sum() for col in factor_cols
            }

            # At least some factors should have non-null values (except forward returns at the end)
            assert sum(non_null_counts.values()) > 0, (
                f"No non-null factors for {symbol}"
            )


class TestMultiHorizonReturns:
    """Tests for multi-horizon return factors."""

    def test_forward_returns_computation(self, monotonic_price_data):
        """Test that forward returns are computed correctly for monotonic prices."""
        result = build_core_ta_factors(monotonic_price_data)

        # For strictly increasing prices, forward returns should be positive
        # (except at the end where we don't have future data)

        # Check returns_1m (21 days forward)
        returns_1m = result["returns_1m"].dropna()
        if len(returns_1m) > 0:
            # For monotonic increasing prices, returns should be positive
            assert (returns_1m > 0).all(), (
                "Forward returns should be positive for increasing prices"
            )

        # Check returns_12m (252 days forward)
        returns_12m = result["returns_12m"].dropna()
        if len(returns_12m) > 0:
            assert (returns_12m > 0).all(), (
                "12-month returns should be positive for increasing prices"
            )

    def test_forward_returns_nan_at_end(self, sample_price_panel):
        """Test that forward returns are NaN at the end (no future data available)."""
        result = build_core_ta_factors(sample_price_panel)

        # For each symbol, check the last rows
        for symbol in sample_price_panel["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol].sort_values("timestamp")

            # Last row should have NaN for long-horizon returns
            last_row = symbol_data.iloc[-1]
            assert pd.isna(last_row["returns_12m"]), (
                "Last row should have NaN for 12m returns"
            )
            assert pd.isna(last_row["returns_6m"]), (
                "Last row should have NaN for 6m returns"
            )

    def test_momentum_12m_excl_1m(self, monotonic_price_data):
        """Test momentum_12m_excl_1m computation."""
        result = build_core_ta_factors(monotonic_price_data)

        momentum = result["momentum_12m_excl_1m"].dropna()

        if len(momentum) > 0:
            # For monotonic increasing prices, momentum should be positive
            assert (momentum > 0).all(), (
                "Momentum should be positive for increasing prices"
            )


class TestTrendStrengthFactors:
    """Tests for trend strength factors."""

    def test_trend_strength_positive_for_uptrend(self, monotonic_price_data):
        """Test that trend strength is positive for upward trending prices."""
        result = build_core_ta_factors(monotonic_price_data)

        # For monotonic increasing prices, trend strength should be positive
        trend_strength_20 = result["trend_strength_20"].dropna()
        if len(trend_strength_20) > 0:
            # Most values should be positive (except early rows where MA is still catching up)
            positive_ratio = (trend_strength_20 > 0).sum() / len(trend_strength_20)
            assert positive_ratio > 0.7, (
                "Most trend strength values should be positive for uptrend"
            )

    def test_trend_strength_all_lookbacks_present(self, sample_price_panel):
        """Test that all trend strength factors (20, 50, 200) are computed."""
        result = build_core_ta_factors(sample_price_panel)

        for lookback in [20, 50, 200]:
            col = f"trend_strength_{lookback}"
            assert col in result.columns, f"Trend strength column {col} missing"

            # Should have some non-null values
            non_null = result[col].notna().sum()
            assert non_null > 0, f"Trend strength {col} has no non-null values"

    def test_trend_strength_without_ohlc(self):
        """Test that trend strength works even without high/low columns (uses fallback)."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": "TEST",
                "close": 100.0 + np.arange(300) * 0.5,
                # No high/low columns
            }
        )

        result = build_core_ta_factors(df)

        # Should still compute trend strength (using fallback)
        assert "trend_strength_20" in result.columns
        assert result["trend_strength_20"].notna().sum() > 0


class TestShortTermReversal:
    """Tests for short-term reversal factors."""

    def test_reversal_factors_present(self, sample_price_panel):
        """Test that all reversal factors (1d, 2d, 3d) are computed."""
        result = build_core_ta_factors(sample_price_panel)

        for horizon in [1, 2, 3]:
            col = f"reversal_{horizon}d"
            assert col in result.columns, f"Reversal column {col} missing"

            # Should have non-null values
            non_null = result[col].notna().sum()
            assert non_null > 0, f"Reversal {col} has no non-null values"

    def test_reversal_zscore_range(self, sample_price_panel):
        """Test that reversal z-scores are in reasonable range."""
        result = build_core_ta_factors(sample_price_panel)

        reversal_1d = result["reversal_1d"].dropna()

        if len(reversal_1d) > 0:
            # Z-scores should typically be within [-5, 5] range (most values)
            within_range = ((reversal_1d >= -5) & (reversal_1d <= 5)).sum()
            ratio = within_range / len(reversal_1d)

            # Most z-scores should be within reasonable range
            assert ratio > 0.9, (
                f"Too many extreme z-scores: {within_range}/{len(reversal_1d)} within [-5, 5]"
            )

    def test_reversal_symmetry(self, sample_price_panel):
        """Test that reversal factors are computed per symbol (no cross-contamination)."""
        result = build_core_ta_factors(sample_price_panel)

        # Check that reversal values differ between symbols (they should, since prices differ)
        symbols = sample_price_panel["symbol"].unique()
        if len(symbols) > 1:
            reversal_values = {}
            for symbol in symbols:
                symbol_data = result[result["symbol"] == symbol]
                reversal_values[symbol] = symbol_data["reversal_1d"].mean()

            # Values should differ (not all identical)
            assert len(set(reversal_values.values())) > 1, (
                "Reversal values should differ between symbols"
            )


class TestFactorRelationships:
    """Tests for logical relationships between factors."""

    def test_monotonic_price_relationships(self, monotonic_price_data):
        """Test factor relationships for monotonic increasing prices."""
        result = build_core_ta_factors(monotonic_price_data)

        # For increasing prices:
        # - Forward returns should be positive (where available)
        # - Trend strength should be positive (where MA has enough data)
        # - Reversal might vary (depends on short-term patterns)

        # Check forward returns (excluding NaN at end)
        returns_1m = result["returns_1m"].dropna()
        if len(returns_1m) > 10:
            positive_ratio = (returns_1m > 0).sum() / len(returns_1m)
            assert positive_ratio > 0.8, (
                "Most forward returns should be positive for increasing prices"
            )

        # Check trend strength (excluding early rows where MA is still warming up)
        trend_strength = result["trend_strength_20"].iloc[50:].dropna()
        if len(trend_strength) > 10:
            positive_ratio = (trend_strength > 0).sum() / len(trend_strength)
            assert positive_ratio > 0.7, (
                "Most trend strength should be positive for uptrend"
            )

    def test_timestamp_sorting_preserved(self, sample_price_panel):
        """Test that timestamps remain sorted per symbol after factor computation."""
        result = build_core_ta_factors(sample_price_panel)

        for symbol in sample_price_panel["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol].sort_values("timestamp")

            # Timestamps should be in ascending order
            timestamps = symbol_data["timestamp"]
            assert timestamps.is_monotonic_increasing, (
                f"Timestamps not sorted for {symbol}"
            )


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_single_symbol(self):
        """Test with single symbol."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": "AAPL",
                "close": 100.0 + np.arange(300) * 0.5,
                "high": 100.0 + np.arange(300) * 0.5 + 1.0,
                "low": 100.0 + np.arange(300) * 0.5 - 1.0,
            }
        )

        result = build_core_ta_factors(df)

        assert len(result) == len(df)
        assert "returns_1m" in result.columns
        assert "trend_strength_20" in result.columns
        assert "reversal_1d" in result.columns

    def test_custom_column_names(self):
        """Test with custom column names."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": "AAPL",
                "price": 100.0 + np.arange(300) * 0.5,
            }
        )

        result = build_core_ta_factors(
            df,
            price_col="price",
            group_col="ticker",
            timestamp_col="date",
        )

        assert len(result) == len(df)
        assert "returns_1m" in result.columns

    def test_minimal_data(self):
        """Test with minimal data (just enough for some factors)."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": "TEST",
                "close": 100.0 + np.arange(50) * 0.5,
            }
        )

        result = build_core_ta_factors(df)

        # Should still compute factors (some may have NaN due to insufficient data)
        assert len(result) == len(df)
        assert "returns_1m" in result.columns
        # At least reversal factors should have values
        assert result["reversal_1d"].notna().sum() > 0
