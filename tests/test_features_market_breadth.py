"""Tests for Market Breadth and Risk-On/Risk-Off Indicators module (Phase A, Sprint A3).

Tests the market breadth functions:
- compute_market_breadth_ma()
- compute_advance_decline_line()
- compute_risk_on_off_indicator()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.features.market_breadth import (
    compute_advance_decline_line,
    compute_market_breadth_ma,
    compute_risk_on_off_indicator,
)


@pytest.fixture
def sample_price_panel() -> pd.DataFrame:
    """Create a synthetic price panel with 5 symbols and 300 days of data."""
    dates = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    all_data = []
    for symbol in symbols:
        base_price = 100.0 + hash(symbol) % 100  # Different base prices
        
        # Create price series with different patterns
        price_series = base_price + np.cumsum(np.random.randn(300) * 0.02) * base_price
        
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "close": price_series,
        })
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def trending_price_panel() -> pd.DataFrame:
    """Create price panel with clear uptrend (for monotonicity tests)."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    symbols = ["STOCK1", "STOCK2", "STOCK3"]
    
    all_data = []
    for symbol in symbols:
        # Strictly increasing prices
        prices = 100.0 + np.arange(100) * 0.5
        
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "close": prices,
        })
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


class TestComputeMarketBreadthMa:
    """Tests for compute_market_breadth_ma() function."""
    
    def test_basic_functionality(self, sample_price_panel):
        """Test that compute_market_breadth_ma runs and returns correct format."""
        result = compute_market_breadth_ma(sample_price_panel, ma_window=50)
        
        # Should have one row per unique timestamp
        unique_timestamps = sample_price_panel["timestamp"].nunique()
        assert len(result) <= unique_timestamps
        
        # Should have required columns
        assert "timestamp" in result.columns
        assert "fraction_above_ma_50" in result.columns
        assert "count_above_ma" in result.columns
        assert "count_total" in result.columns
    
    def test_fraction_range(self, sample_price_panel):
        """Test that fraction_above_ma is between 0 and 1."""
        result = compute_market_breadth_ma(sample_price_panel, ma_window=50)
        
        fraction_col = "fraction_above_ma_50"
        fractions = result[fraction_col].dropna()
        
        if len(fractions) > 0:
            assert (fractions >= 0.0).all(), "Fraction should be >= 0"
            assert (fractions <= 1.0).all(), "Fraction should be <= 1"
    
    def test_count_consistency(self, sample_price_panel):
        """Test that count_above_ma <= count_total."""
        result = compute_market_breadth_ma(sample_price_panel, ma_window=50)
        
        mask = result["count_total"].notna()
        if mask.sum() > 0:
            assert (
                result.loc[mask, "count_above_ma"] <= result.loc[mask, "count_total"]
            ).all(), "count_above_ma should not exceed count_total"
    
    def test_custom_ma_window(self, sample_price_panel):
        """Test with custom MA window."""
        result = compute_market_breadth_ma(sample_price_panel, ma_window=20)
        
        assert "fraction_above_ma_20" in result.columns
        assert len(result) > 0
    
    def test_aggregation_over_symbols(self, sample_price_panel):
        """Test that breadth is correctly aggregated across symbols."""
        result = compute_market_breadth_ma(sample_price_panel, ma_window=50)
        
        # Each row should represent one timestamp
        assert result["timestamp"].nunique() == len(result)
        
        # Count total should match number of symbols (for timestamps with full data)
        max_count = result["count_total"].max()
        expected_symbols = sample_price_panel["symbol"].nunique()
        assert max_count <= expected_symbols
    
    def test_uptrend_breadth(self, trending_price_panel):
        """Test that uptrending market has high breadth after MA warms up."""
        result = compute_market_breadth_ma(trending_price_panel, ma_window=20)
        
        # After MA has enough data (e.g., after row 30), fraction should be high
        if len(result) > 30:
            later_fractions = result.iloc[30:]["fraction_above_ma_20"].dropna()
            if len(later_fractions) > 0:
                avg_fraction = later_fractions.mean()
                assert avg_fraction > 0.7, f"For uptrend, fraction should be high, got {avg_fraction}"
    
    def test_required_columns_validation(self):
        """Test that missing required columns raise KeyError."""
        # Missing timestamp
        df_no_timestamp = pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [100.0],
        })
        with pytest.raises(KeyError, match="Missing required columns"):
            compute_market_breadth_ma(df_no_timestamp)
    
    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame(columns=["timestamp", "symbol", "close"])
        
        with pytest.raises(ValueError, match="empty"):
            compute_market_breadth_ma(empty_df)


class TestComputeAdvanceDeclineLine:
    """Tests for compute_advance_decline_line() function."""
    
    def test_basic_functionality(self, sample_price_panel):
        """Test that compute_advance_decline_line runs and returns correct format."""
        result = compute_advance_decline_line(sample_price_panel)
        
        # Should have one row per unique timestamp (minus first, which has no returns)
        unique_timestamps = sample_price_panel["timestamp"].nunique()
        assert len(result) <= unique_timestamps - 1  # First timestamp has no returns
        
        # Should have required columns
        assert "timestamp" in result.columns
        assert "advances" in result.columns
        assert "declines" in result.columns
        assert "net_advances" in result.columns
        assert "ad_line" in result.columns
        assert "ad_line_normalized" in result.columns
    
    def test_net_advances_calculation(self, sample_price_panel):
        """Test that net_advances = advances - declines."""
        result = compute_advance_decline_line(sample_price_panel)
        
        if len(result) > 0:
            computed_net = result["advances"] - result["declines"]
            assert (result["net_advances"] == computed_net).all(), "net_advances should equal advances - declines"
    
    def test_ad_line_cumulative(self, sample_price_panel):
        """Test that A/D Line is cumulative sum of net_advances."""
        result = compute_advance_decline_line(sample_price_panel)
        
        if len(result) > 0:
            expected_cumsum = result["net_advances"].cumsum()
            assert (result["ad_line"] == expected_cumsum).all(), "ad_line should be cumulative sum of net_advances"
    
    def test_ad_line_normalized_starts_at_zero(self, sample_price_panel):
        """Test that normalized A/D Line starts at 0."""
        result = compute_advance_decline_line(sample_price_panel)
        
        if len(result) > 0:
            first_normalized = result["ad_line_normalized"].iloc[0]
            assert abs(first_normalized) < 1e-10, f"Normalized A/D Line should start at 0, got {first_normalized}"
    
    def test_ad_line_monotonicity_property(self, trending_price_panel):
        """Test that A/D Line for uptrending market is generally increasing."""
        result = compute_advance_decline_line(trending_price_panel)
        
        if len(result) > 10:
            # For uptrending market, net_advances should be mostly positive
            positive_days = (result["net_advances"] > 0).sum()
            total_days = len(result)
            positive_ratio = positive_days / total_days
            
            # In an uptrend, most days should have positive net advances
            assert positive_ratio > 0.6, f"For uptrend, positive net advances ratio should be >0.6, got {positive_ratio}"
            
            # A/D Line should be generally increasing
            ad_line_diffs = result["ad_line"].diff().dropna()
            increasing_days = (ad_line_diffs > 0).sum()
            increasing_ratio = increasing_days / len(ad_line_diffs)
            assert increasing_ratio > 0.5, f"For uptrend, A/D Line should be increasing more often, got {increasing_ratio}"
    
    def test_required_columns_validation(self):
        """Test that missing required columns raise KeyError."""
        # Missing timestamp
        df_no_timestamp = pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [100.0],
        })
        with pytest.raises(KeyError, match="Missing required columns"):
            compute_advance_decline_line(df_no_timestamp)
    
    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame(columns=["timestamp", "symbol", "close"])
        
        with pytest.raises(ValueError, match="empty"):
            compute_advance_decline_line(empty_df)


class TestComputeRiskOnOffIndicator:
    """Tests for compute_risk_on_off_indicator() function."""
    
    def test_basic_functionality(self, sample_price_panel):
        """Test that compute_risk_on_off_indicator runs and returns correct format."""
        result = compute_risk_on_off_indicator(sample_price_panel)
        
        # Should have one row per unique timestamp (minus first, which has no returns)
        unique_timestamps = sample_price_panel["timestamp"].nunique()
        assert len(result) <= unique_timestamps - 1
        
        # Should have required columns
        assert "timestamp" in result.columns
        assert "risk_on_ratio" in result.columns
        assert "risk_off_ratio" in result.columns
        assert "risk_on_off_score" in result.columns
    
    def test_ratio_range(self, sample_price_panel):
        """Test that ratios are between 0 and 1."""
        result = compute_risk_on_off_indicator(sample_price_panel)
        
        if len(result) > 0:
            assert (result["risk_on_ratio"] >= 0.0).all(), "risk_on_ratio should be >= 0"
            assert (result["risk_on_ratio"] <= 1.0).all(), "risk_on_ratio should be <= 1"
            assert (result["risk_off_ratio"] >= 0.0).all(), "risk_off_ratio should be >= 0"
            assert (result["risk_off_ratio"] <= 1.0).all(), "risk_off_ratio should be <= 1"
    
    def test_score_range(self, sample_price_panel):
        """Test that risk_on_off_score is between -1 and 1."""
        result = compute_risk_on_off_indicator(sample_price_panel)
        
        if len(result) > 0:
            assert (result["risk_on_off_score"] >= -1.0).all(), "risk_on_off_score should be >= -1"
            assert (result["risk_on_off_score"] <= 1.0).all(), "risk_on_off_score should be <= 1"
    
    def test_ratio_sum(self, sample_price_panel):
        """Test that risk_on_ratio + risk_off_ratio ≈ 1 (excluding flat days)."""
        result = compute_risk_on_off_indicator(sample_price_panel)
        
        if len(result) > 0:
            ratio_sum = result["risk_on_ratio"] + result["risk_off_ratio"]
            # Should be close to 1 (within small tolerance)
            assert (ratio_sum <= 1.01).all(), "risk_on_ratio + risk_off_ratio should be <= 1"
            assert (ratio_sum >= 0.99).all(), "risk_on_ratio + risk_off_ratio should be >= 0.99"
    
    def test_required_columns_validation(self):
        """Test that missing required columns raise KeyError."""
        # Missing timestamp
        df_no_timestamp = pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [100.0],
        })
        with pytest.raises(KeyError, match="Missing required columns"):
            compute_risk_on_off_indicator(df_no_timestamp)
    
    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame(columns=["timestamp", "symbol", "close"])
        
        with pytest.raises(ValueError, match="empty"):
            compute_risk_on_off_indicator(empty_df)


class TestIntegration:
    """Tests for integration between market breadth functions."""
    
    def test_functions_work_together(self, sample_price_panel):
        """Test that all market breadth functions can be used together."""
        # Compute market breadth
        breadth = compute_market_breadth_ma(sample_price_panel, ma_window=50)
        
        # Compute A/D Line
        ad_line = compute_advance_decline_line(sample_price_panel)
        
        # Compute Risk-On/Risk-Off
        risk_indicator = compute_risk_on_off_indicator(sample_price_panel)
        
        # Should all have timestamp column for joining
        assert "timestamp" in breadth.columns
        assert "timestamp" in ad_line.columns
        assert "timestamp" in risk_indicator.columns
        
        # Should be able to merge them
        combined = breadth.merge(ad_line, on="timestamp", how="outer").merge(
            risk_indicator, on="timestamp", how="outer"
        )
        
        assert len(combined) > 0
        assert "fraction_above_ma_50" in combined.columns
        assert "ad_line" in combined.columns
        assert "risk_on_off_score" in combined.columns
    
    def test_universe_level_output(self, sample_price_panel):
        """Test that all functions return universe-level (one row per timestamp) outputs."""
        breadth = compute_market_breadth_ma(sample_price_panel)
        ad_line = compute_advance_decline_line(sample_price_panel)
        risk_indicator = compute_risk_on_off_indicator(sample_price_panel)
        
        # Each should have one row per timestamp (or fewer if some timestamps lack data)
        unique_timestamps = sample_price_panel["timestamp"].nunique()
        
        assert len(breadth) <= unique_timestamps
        assert len(ad_line) <= unique_timestamps - 1  # First timestamp has no returns
        assert len(risk_indicator) <= unique_timestamps - 1
        
        # Timestamps should be unique in each output
        assert breadth["timestamp"].nunique() == len(breadth)
        assert ad_line["timestamp"].nunique() == len(ad_line)
        assert risk_indicator["timestamp"].nunique() == len(risk_indicator)


class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_symbol(self):
        """Test with single symbol."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": "AAPL",
            "close": 100.0 + np.arange(100) * 0.5,
        })
        
        breadth = compute_market_breadth_ma(df, ma_window=20)
        ad_line = compute_advance_decline_line(df)
        
        assert len(breadth) > 0
        assert len(ad_line) > 0
        
        # With single symbol, fraction should be either 0 or 1
        fractions = breadth["fraction_above_ma_20"].dropna()
        if len(fractions) > 0:
            assert ((fractions == 0.0) | (fractions == 1.0)).all(), "With single symbol, fraction should be 0 or 1"
    
    def test_minimal_data(self):
        """Test with minimal data (just enough for some indicators)."""
        dates = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
        symbols = ["A", "B"]
        
        all_data = []
        for symbol in symbols:
            df = pd.DataFrame({
                "timestamp": dates,
                "symbol": symbol,
                "close": 100.0 + np.arange(60) * 0.5,
            })
            all_data.append(df)
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Market breadth should work with MA window < data length
        breadth = compute_market_breadth_ma(df, ma_window=20)
        assert len(breadth) > 0
        
        # A/D Line should work
        ad_line = compute_advance_decline_line(df)
        assert len(ad_line) > 0

