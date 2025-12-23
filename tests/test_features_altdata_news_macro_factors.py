"""Tests for Alt-Data News & Macro Factors module (Phase B2).

Tests the build_news_sentiment_factors() and build_macro_regime_factors() functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.features.altdata_news_macro_factors import (
    build_macro_regime_factors,
    build_news_sentiment_factors,
)


@pytest.fixture
def sample_price_panel() -> pd.DataFrame:
    """Create a synthetic price panel with 2 symbols and 100 days of data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT"]

    all_data = []
    for symbol in symbols:
        base_price = 100.0 if symbol == "AAPL" else 150.0
        price_series = base_price + np.arange(100) * 0.1 + np.random.randn(100) * 1.0

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": symbol,
                "close": price_series,
                "high": price_series + np.abs(np.random.randn(100)) * 0.5,
                "low": price_series - np.abs(np.random.randn(100)) * 0.5,
                "volume": np.random.randint(1000000, 10000000, size=100),
            }
        )
        all_data.append(df)

    return (
        pd.concat(all_data, ignore_index=True)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


@pytest.fixture
def sample_news_sentiment_daily() -> pd.DataFrame:
    """Create synthetic news sentiment daily data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")

    # Per-symbol sentiment
    sentiment_data = []
    for symbol in ["AAPL", "MSFT"]:
        for i, date in enumerate(dates[::2]):  # Every other day
            sentiment_data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "sentiment_score": 0.5 + np.sin(i * 0.1) * 0.3,  # Range: 0.2 to 0.8
                    "sentiment_volume": np.random.randint(5, 20),
                }
            )

    # Market-wide sentiment
    for i, date in enumerate(dates[::3]):  # Every third day
        sentiment_data.append(
            {
                "timestamp": date,
                "symbol": "__MARKET__",
                "sentiment_score": -0.1 + np.sin(i * 0.15) * 0.2,  # Range: -0.3 to 0.1
                "sentiment_volume": np.random.randint(20, 50),
            }
        )

    return (
        pd.DataFrame(sentiment_data)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


@pytest.fixture
def sample_macro_series() -> pd.DataFrame:
    """Create synthetic macro economic series."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")

    macro_data = []
    macro_codes = ["GDP", "CPI", "UNEMPLOYMENT", "FED_RATE", "VIX"]

    for code in macro_codes:
        for i, date in enumerate(dates[::10]):  # Every 10th day
            if code == "GDP":
                value = 2.0 + np.sin(i * 0.2) * 1.0  # Range: 1.0 to 3.0
            elif code == "CPI":
                value = 2.5 + np.sin(i * 0.15) * 1.5  # Range: 1.0 to 4.0
            elif code == "UNEMPLOYMENT":
                value = 5.0 + np.sin(i * 0.1) * 2.0  # Range: 3.0 to 7.0
            elif code == "FED_RATE":
                value = 3.0 + np.sin(i * 0.25) * 2.0  # Range: 1.0 to 5.0
            elif code == "VIX":
                value = 15.0 + np.sin(i * 0.3) * 10.0  # Range: 5.0 to 25.0
            else:
                value = 1.0

            macro_data.append(
                {
                    "timestamp": date,
                    "macro_code": code,
                    "value": value,
                    "country": "US",
                    "release_time": date,
                }
            )

    return (
        pd.DataFrame(macro_data)
        .sort_values(["timestamp", "macro_code"])
        .reset_index(drop=True)
    )


class TestBuildNewsSentimentFactors:
    """Tests for build_news_sentiment_factors()."""

    def test_basic_functionality(self, sample_price_panel, sample_news_sentiment_daily):
        """Test basic news sentiment factor computation."""
        result = build_news_sentiment_factors(
            sample_news_sentiment_daily,
            sample_price_panel,
            lookback_days=20,
        )

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_price_panel)
        assert "timestamp" in result.columns
        assert "symbol" in result.columns
        assert "close" in result.columns

        # Check factor columns
        assert "news_sentiment_mean_20d" in result.columns
        assert "news_sentiment_trend_20d" in result.columns
        assert "news_sentiment_shock_flag" in result.columns
        assert "news_sentiment_volume_20d" in result.columns

    def test_sentiment_mean_aggregation(
        self, sample_price_panel, sample_news_sentiment_daily
    ):
        """Test that sentiment mean is correctly aggregated over lookback window."""
        result = build_news_sentiment_factors(
            sample_news_sentiment_daily,
            sample_price_panel,
            lookback_days=20,
        )

        # Check that sentiment mean values are within reasonable range
        mean_values = result["news_sentiment_mean_20d"].dropna()
        assert len(mean_values) > 0

        # Sentiment scores should be between -1 and 1 (or reasonable range)
        assert (mean_values >= -2.0).all()  # Allow some margin for aggregation
        assert (mean_values <= 2.0).all()

    def test_sentiment_trend_calculation(
        self, sample_price_panel, sample_news_sentiment_daily
    ):
        """Test that sentiment trend is correctly calculated."""
        result = build_news_sentiment_factors(
            sample_news_sentiment_daily,
            sample_price_panel,
            lookback_days=20,
        )

        # Check that trend values exist
        trend_values = result["news_sentiment_trend_20d"].dropna()
        assert len(trend_values) > 0

        # Trend should be finite
        assert np.isfinite(trend_values).all()

    def test_sentiment_shock_flag(
        self, sample_price_panel, sample_news_sentiment_daily
    ):
        """Test that sentiment shock flag is binary (0 or 1)."""
        result = build_news_sentiment_factors(
            sample_news_sentiment_daily,
            sample_price_panel,
            lookback_days=20,
        )

        # Check that shock flag is binary
        shock_values = result["news_sentiment_shock_flag"].dropna()
        if len(shock_values) > 0:
            assert shock_values.isin([0.0, 1.0]).all()

    def test_market_wide_sentiment(self, sample_price_panel):
        """Test that market-wide sentiment is applied to all symbols."""
        # Create market-wide sentiment only
        dates = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
        market_sentiment = pd.DataFrame(
            {
                "timestamp": dates,
                "sentiment_score": 0.5,
                "sentiment_volume": 10,
            }
        )

        result = build_news_sentiment_factors(
            market_sentiment,
            sample_price_panel,
            lookback_days=20,
        )

        # All symbols should have sentiment factors
        for symbol in sample_price_panel["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol]
            assert "news_sentiment_mean_20d" in symbol_data.columns

    def test_empty_sentiment_data(self, sample_price_panel):
        """Test behavior with empty sentiment data."""
        empty_sentiment = pd.DataFrame(
            columns=["timestamp", "symbol", "sentiment_score", "sentiment_volume"]
        )

        result = build_news_sentiment_factors(
            empty_sentiment,
            sample_price_panel,
            lookback_days=20,
        )

        # Should still return price panel with NaN factors
        assert len(result) == len(sample_price_panel)
        assert "news_sentiment_mean_20d" in result.columns
        assert result["news_sentiment_mean_20d"].isna().all()

    def test_custom_lookback_days(
        self, sample_price_panel, sample_news_sentiment_daily
    ):
        """Test with custom lookback window."""
        result = build_news_sentiment_factors(
            sample_news_sentiment_daily,
            sample_price_panel,
            lookback_days=10,
        )

        # Check that column names reflect custom lookback
        assert "news_sentiment_mean_10d" in result.columns
        assert "news_sentiment_trend_10d" in result.columns
        assert "news_sentiment_volume_10d" in result.columns


class TestBuildMacroRegimeFactors:
    """Tests for build_macro_regime_factors()."""

    def test_basic_functionality(self, sample_price_panel, sample_macro_series):
        """Test basic macro regime factor computation."""
        result = build_macro_regime_factors(
            sample_macro_series,
            sample_price_panel,
        )

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_price_panel)
        assert "timestamp" in result.columns
        assert "symbol" in result.columns
        assert "close" in result.columns

        # Check factor columns
        assert "macro_growth_regime" in result.columns
        assert "macro_inflation_regime" in result.columns
        assert "macro_risk_aversion_proxy" in result.columns

    def test_regime_values_range(self, sample_price_panel, sample_macro_series):
        """Test that regime values are in expected range (-1, 0, +1)."""
        result = build_macro_regime_factors(
            sample_macro_series,
            sample_price_panel,
        )

        # Check that regime values are in expected set
        growth_values = result["macro_growth_regime"].dropna()
        if len(growth_values) > 0:
            assert growth_values.isin([-1.0, 0.0, 1.0]).all()

        inflation_values = result["macro_inflation_regime"].dropna()
        if len(inflation_values) > 0:
            assert inflation_values.isin([-1.0, 0.0, 1.0]).all()

        risk_values = result["macro_risk_aversion_proxy"].dropna()
        if len(risk_values) > 0:
            assert risk_values.isin([-1.0, 0.0, 1.0]).all()

    def test_same_regime_per_date(self, sample_price_panel, sample_macro_series):
        """Test that all symbols get the same regime value on the same date."""
        result = build_macro_regime_factors(
            sample_macro_series,
            sample_price_panel,
        )

        # Group by timestamp and check that all symbols have same regime values
        for timestamp in result["timestamp"].unique():
            timestamp_data = result[result["timestamp"] == timestamp]

            growth_values = timestamp_data["macro_growth_regime"].dropna()
            if len(growth_values) > 0:
                # All non-NaN values should be the same
                assert growth_values.nunique() <= 1

            inflation_values = timestamp_data["macro_inflation_regime"].dropna()
            if len(inflation_values) > 0:
                assert inflation_values.nunique() <= 1

            risk_values = timestamp_data["macro_risk_aversion_proxy"].dropna()
            if len(risk_values) > 0:
                assert risk_values.nunique() <= 1

    def test_growth_regime_calculation(self, sample_price_panel):
        """Test that growth regime is correctly calculated from GDP/unemployment."""
        # Create macro series with high GDP (expansion)
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        macro_high_gdp = pd.DataFrame(
            {
                "timestamp": dates,
                "macro_code": "GDP",
                "value": 3.0,  # High GDP growth
                "country": "US",
            }
        )

        result = build_macro_regime_factors(
            macro_high_gdp,
            sample_price_panel,
        )

        # Should have positive growth regime
        growth_values = result["macro_growth_regime"].dropna()
        if len(growth_values) > 0:
            # At least some should be positive (expansion)
            assert (growth_values >= 0).any() or len(growth_values) == 0

    def test_inflation_regime_calculation(self, sample_price_panel):
        """Test that inflation regime is correctly calculated from CPI."""
        # Create macro series with high CPI (high inflation)
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        macro_high_cpi = pd.DataFrame(
            {
                "timestamp": dates,
                "macro_code": "CPI",
                "value": 4.0,  # High inflation
                "country": "US",
            }
        )

        result = build_macro_regime_factors(
            macro_high_cpi,
            sample_price_panel,
        )

        # Should have positive inflation regime
        inflation_values = result["macro_inflation_regime"].dropna()
        if len(inflation_values) > 0:
            # At least some should be positive (high inflation)
            assert (inflation_values >= 0).any() or len(inflation_values) == 0

    def test_empty_macro_data(self, sample_price_panel):
        """Test behavior with empty macro data."""
        empty_macro = pd.DataFrame(
            columns=["timestamp", "macro_code", "value", "country"]
        )

        result = build_macro_regime_factors(
            empty_macro,
            sample_price_panel,
        )

        # Should still return price panel with NaN factors
        assert len(result) == len(sample_price_panel)
        assert "macro_growth_regime" in result.columns
        assert result["macro_growth_regime"].isna().all()

    def test_missing_days_handling(self, sample_price_panel):
        """Test that missing days in macro series are handled gracefully."""
        # Create macro series with gaps
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        macro_sparse = pd.DataFrame(
            {
                "timestamp": dates[::3],  # Every third day only
                "macro_code": "GDP",
                "value": 2.0,
                "country": "US",
            }
        )

        result = build_macro_regime_factors(
            macro_sparse,
            sample_price_panel,
        )

        # Should still return results (with NaN for missing days)
        assert len(result) == len(sample_price_panel)
        assert "macro_growth_regime" in result.columns

    def test_country_filter(self, sample_price_panel):
        """Test that country filter works correctly."""
        # Create macro series with multiple countries
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        macro_multi_country = pd.DataFrame(
            {
                "timestamp": dates,
                "macro_code": "GDP",
                "value": 2.0,
                "country": ["US", "EU", "US", "EU", "US", "EU", "US", "EU", "US", "EU"],
            }
        )

        # Filter to US only
        result_us = build_macro_regime_factors(
            macro_multi_country,
            sample_price_panel,
            country_filter="US",
        )

        # Should still return results
        assert len(result_us) == len(sample_price_panel)
        assert "macro_growth_regime" in result_us.columns


class TestIntegration:
    """Integration tests for news/macro factors."""

    def test_combined_news_and_macro(
        self, sample_price_panel, sample_news_sentiment_daily, sample_macro_series
    ):
        """Test combining news sentiment and macro regime factors."""
        # Build news sentiment factors
        news_factors = build_news_sentiment_factors(
            sample_news_sentiment_daily,
            sample_price_panel,
            lookback_days=20,
        )

        # Build macro regime factors
        macro_factors = build_macro_regime_factors(
            sample_macro_series,
            sample_price_panel,
        )

        # Merge both
        combined = news_factors.merge(
            macro_factors[
                [
                    "timestamp",
                    "symbol",
                    "macro_growth_regime",
                    "macro_inflation_regime",
                    "macro_risk_aversion_proxy",
                ]
            ],
            on=["timestamp", "symbol"],
            how="left",
        )

        # Check that both factor types are present
        assert "news_sentiment_mean_20d" in combined.columns
        assert "macro_growth_regime" in combined.columns
        assert len(combined) == len(sample_price_panel)
