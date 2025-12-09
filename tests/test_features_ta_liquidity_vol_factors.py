"""Tests for Liquidity and Volatility Factors module (Phase A, Sprint A2).

Tests the liquidity and volatility factor functions:
- add_realized_volatility()
- add_vol_of_vol()
- add_turnover_and_liquidity_proxies()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.features.ta_liquidity_vol_factors import (
    add_realized_volatility,
    add_turnover_and_liquidity_proxies,
    add_vol_of_vol,
)


@pytest.fixture
def sample_price_panel() -> pd.DataFrame:
    """Create a synthetic price panel with 2 symbols and 300 days of data."""
    dates = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT"]
    
    all_data = []
    for symbol in symbols:
        base_price = 100.0 if symbol == "AAPL" else 150.0
        
        # Create price series with volatility
        price_series = base_price + np.cumsum(np.random.randn(300) * 0.02) * base_price
        
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "close": price_series,
            "high": price_series * 1.02,
            "low": price_series * 0.98,
            "volume": np.random.lognormal(10, 0.5, 300),
        })
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def price_panel_with_freefloat() -> pd.DataFrame:
    """Create price panel with freefloat column for turnover calculation."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "TEST",
        "close": 100.0 + np.arange(100) * 0.5,
        "high": 100.0 + np.arange(100) * 0.5 + 1.0,
        "low": 100.0 + np.arange(100) * 0.5 - 1.0,
        "volume": np.random.lognormal(8, 0.3, 100),
        "freefloat": 1_000_000.0,  # Constant freefloat
    })
    
    return df


class TestAddRealizedVolatility:
    """Tests for add_realized_volatility() function."""
    
    def test_basic_functionality(self, sample_price_panel):
        """Test that add_realized_volatility runs without errors and adds RV columns."""
        result = add_realized_volatility(sample_price_panel)
        
        # Should have same number of rows as input
        assert len(result) == len(sample_price_panel)
        
        # Should preserve original columns
        for col in sample_price_panel.columns:
            assert col in result.columns
        
        # Should add RV columns (default windows: [20, 60])
        assert "rv_20" in result.columns
        assert "rv_60" in result.columns
    
    def test_custom_windows(self, sample_price_panel):
        """Test with custom window sizes."""
        windows = [10, 30, 90]
        result = add_realized_volatility(sample_price_panel, windows=windows)
        
        for window in windows:
            assert f"rv_{window}" in result.columns
    
    def test_rv_values_positive(self, sample_price_panel):
        """Test that realized volatility values are positive (or NaN at start)."""
        result = add_realized_volatility(sample_price_panel)
        
        rv_20 = result["rv_20"].dropna()
        
        if len(rv_20) > 0:
            assert (rv_20 >= 0).all(), "Realized volatility should be non-negative"
    
    def test_rv_scaling(self, sample_price_panel):
        """Test that RV values are annualized (reasonable range)."""
        result = add_realized_volatility(sample_price_panel)
        
        rv_20 = result["rv_20"].dropna()
        
        if len(rv_20) > 0:
            # Annualized volatility should be in reasonable range (e.g., 0-200%)
            # Very high volatility stocks might exceed, but most should be in range
            within_range = ((rv_20 >= 0) & (rv_20 <= 2.0)).sum()
            ratio = within_range / len(rv_20)
            
            # Most values should be in reasonable range
            assert ratio > 0.8, f"Too many extreme RV values: {within_range}/{len(rv_20)} in [0, 2.0]"
    
    def test_rv_per_symbol(self, sample_price_panel):
        """Test that RV is computed per symbol (no cross-contamination)."""
        result = add_realized_volatility(sample_price_panel)
        
        # Check that each symbol has its own RV values
        for symbol in sample_price_panel["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol]
            
            # Should have data for this symbol
            assert len(symbol_data) > 0
            
            # RV columns should have some non-null values
            assert symbol_data["rv_20"].notna().sum() > 0
    
    def test_required_columns_validation(self):
        """Test that missing required columns raise KeyError."""
        # Missing timestamp
        df_no_timestamp = pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [100.0],
        })
        with pytest.raises(KeyError, match="Missing required columns"):
            add_realized_volatility(df_no_timestamp)
        
        # Missing close
        df_no_close = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": "AAPL",
        })
        with pytest.raises(KeyError, match="Missing required columns"):
            add_realized_volatility(df_no_close)
    
    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame(columns=["timestamp", "symbol", "close"])
        
        with pytest.raises(ValueError, match="empty"):
            add_realized_volatility(empty_df)


class TestAddVolOfVol:
    """Tests for add_vol_of_vol() function."""
    
    def test_basic_functionality(self, sample_price_panel):
        """Test that add_vol_of_vol runs after adding realized volatility."""
        # First add realized volatility
        df_with_rv = add_realized_volatility(sample_price_panel)
        
        # Then add Vol-of-Vol
        result = add_vol_of_vol(df_with_rv)
        
        # Should have same number of rows
        assert len(result) == len(sample_price_panel)
        
        # Should add Vol-of-Vol columns
        assert "vov_20_60" in result.columns
        assert "vov_60_60" in result.columns
    
    def test_vol_of_vol_auto_detect_rv(self, sample_price_panel):
        """Test that Vol-of-Vol auto-detects RV columns."""
        df_with_rv = add_realized_volatility(sample_price_panel, windows=[20, 60])
        
        result = add_vol_of_vol(df_with_rv)
        
        # Should detect and process both RV columns
        assert "vov_20_60" in result.columns
        assert "vov_60_60" in result.columns
    
    def test_vol_of_vol_custom_window(self, sample_price_panel):
        """Test with custom Vol-of-Vol window."""
        df_with_rv = add_realized_volatility(sample_price_panel, windows=[20])
        
        result = add_vol_of_vol(df_with_rv, vol_window=30)
        
        assert "vov_20_30" in result.columns
    
    def test_vol_of_vol_no_rv_columns(self, sample_price_panel):
        """Test that function handles missing RV columns gracefully."""
        # DataFrame without RV columns
        result = add_vol_of_vol(sample_price_panel)
        
        # Should return original DataFrame unchanged (or with warning)
        assert len(result) == len(sample_price_panel)
        # Should not have Vol-of-Vol columns
        assert not any(col.startswith("vov_") for col in result.columns)
    
    def test_vol_of_vol_values_positive(self, sample_price_panel):
        """Test that Vol-of-Vol values are positive (or NaN)."""
        df_with_rv = add_realized_volatility(sample_price_panel)
        result = add_vol_of_vol(df_with_rv)
        
        vov_20_60 = result["vov_20_60"].dropna()
        
        if len(vov_20_60) > 0:
            assert (vov_20_60 >= 0).all(), "Vol-of-Vol should be non-negative"


class TestAddTurnoverAndLiquidityProxies:
    """Tests for add_turnover_and_liquidity_proxies() function."""
    
    def test_basic_functionality(self, sample_price_panel):
        """Test that function runs and adds liquidity proxies."""
        result = add_turnover_and_liquidity_proxies(sample_price_panel)
        
        # Should have same number of rows
        assert len(result) == len(sample_price_panel)
        
        # Should add volume_zscore (if volume available)
        if "volume" in sample_price_panel.columns:
            assert "volume_zscore" in result.columns
        
        # Should add spread_proxy (if high/low/close available)
        if all(col in sample_price_panel.columns for col in ["high", "low", "close"]):
            assert "spread_proxy" in result.columns
    
    def test_turnover_with_freefloat(self, price_panel_with_freefloat):
        """Test turnover calculation when freefloat is available."""
        result = add_turnover_and_liquidity_proxies(
            price_panel_with_freefloat,
            freefloat_col="freefloat"
        )
        
        assert "turnover" in result.columns
        
        # Turnover should be positive (volume / freefloat)
        turnover = result["turnover"].dropna()
        if len(turnover) > 0:
            assert (turnover >= 0).all(), "Turnover should be non-negative"
    
    def test_turnover_without_freefloat(self, sample_price_panel):
        """Test that function works without freefloat (no turnover column)."""
        result = add_turnover_and_liquidity_proxies(sample_price_panel)
        
        # Should not have turnover column if freefloat not provided
        assert "turnover" not in result.columns
    
    def test_volume_zscore(self, sample_price_panel):
        """Test volume z-score calculation."""
        result = add_turnover_and_liquidity_proxies(sample_price_panel)
        
        if "volume" in sample_price_panel.columns:
            assert "volume_zscore" in result.columns
            
            # Volume z-score should have some non-null values
            assert result["volume_zscore"].notna().sum() > 0
    
    def test_spread_proxy(self, sample_price_panel):
        """Test spread proxy calculation ((high - low) / close)."""
        result = add_turnover_and_liquidity_proxies(sample_price_panel)
        
        if all(col in sample_price_panel.columns for col in ["high", "low", "close"]):
            assert "spread_proxy" in result.columns
            
            spread = result["spread_proxy"].dropna()
            
            if len(spread) > 0:
                # Spread should be positive (high > low)
                assert (spread >= 0).all(), "Spread proxy should be non-negative"
    
    def test_without_volume(self):
        """Test function behavior when volume column is missing."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": "TEST",
            "close": 100.0 + np.arange(100) * 0.5,
        })
        
        result = add_turnover_and_liquidity_proxies(df)
        
        # Should not have volume_zscore
        assert "volume_zscore" not in result.columns
    
    def test_required_columns_validation(self):
        """Test that missing required columns raise KeyError."""
        # Missing timestamp
        df_no_timestamp = pd.DataFrame({
            "symbol": ["AAPL"],
            "volume": [1000.0],
        })
        with pytest.raises(KeyError, match="Missing required columns"):
            add_turnover_and_liquidity_proxies(df_no_timestamp)
    
    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame(columns=["timestamp", "symbol"])
        
        with pytest.raises(ValueError, match="empty"):
            add_turnover_and_liquidity_proxies(empty_df)


class TestIntegrationWithCoreFactors:
    """Tests for integration with build_core_ta_factors."""
    
    def test_factors_work_together(self, sample_price_panel):
        """Test that liquidity/vol factors can be used with core TA factors."""
        from src.assembled_core.features.ta_factors_core import build_core_ta_factors
        
        # Build core TA factors
        df_with_core = build_core_ta_factors(sample_price_panel)
        
        # Add realized volatility
        df_with_rv = add_realized_volatility(df_with_core)
        
        # Add Vol-of-Vol
        df_with_vov = add_vol_of_vol(df_with_rv)
        
        # Add liquidity proxies
        result = add_turnover_and_liquidity_proxies(df_with_vov)
        
        # Should have all factor columns
        assert "returns_1m" in result.columns  # From core factors
        assert "rv_20" in result.columns  # From RV
        assert "vov_20_60" in result.columns  # From Vol-of-Vol
        assert "volume_zscore" in result.columns  # From liquidity proxies
        
        # Should have same number of rows
        assert len(result) == len(sample_price_panel)


class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_symbol(self):
        """Test with single symbol."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": "AAPL",
            "close": 100.0 + np.cumsum(np.random.randn(300) * 0.02) * 100.0,
            "high": 100.0 + np.cumsum(np.random.randn(300) * 0.02) * 100.0 * 1.02,
            "low": 100.0 + np.cumsum(np.random.randn(300) * 0.02) * 100.0 * 0.98,
            "volume": np.random.lognormal(10, 0.5, 300),
        })
        
        result_rv = add_realized_volatility(df)
        assert "rv_20" in result_rv.columns
        
        result_liq = add_turnover_and_liquidity_proxies(df)
        assert "volume_zscore" in result_liq.columns
    
    def test_minimal_data(self):
        """Test with minimal data (just enough for some factors)."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": "TEST",
            "close": 100.0 + np.arange(50) * 0.5,
        })
        
        result = add_realized_volatility(df, windows=[20])
        
        # Should still compute factors (some may have NaN due to insufficient data)
        assert len(result) == len(df)
        assert "rv_20" in result.columns

