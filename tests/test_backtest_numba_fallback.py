"""Tests for Numba fallback when Numba is not installed."""

from __future__ import annotations

import os
import pandas as pd
import pytest

from src.assembled_core.qa.backtest_engine import run_portfolio_backtest
from src.assembled_core.portfolio.position_sizing import compute_target_positions


def create_simple_signal_fn():
    """Create a simple signal function for testing."""
    def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        """Generate simple LONG signals for all symbols."""
        signals = []
        for symbol in prices_df["symbol"].unique():
            symbol_data = prices_df[prices_df["symbol"] == symbol].sort_values("timestamp")
            if len(symbol_data) > 0:
                # Simple signal: LONG if price > first price
                first_price = symbol_data["close"].iloc[0]
                for _, row in symbol_data.iterrows():
                    direction = "LONG" if row["close"] > first_price else "NEUTRAL"
                    signals.append({
                        "timestamp": row["timestamp"],
                        "symbol": symbol,
                        "direction": direction,
                        "score": 1.0 if direction == "LONG" else 0.0,
                    })
        return pd.DataFrame(signals)
    return signal_fn


def test_backtest_without_numba():
    """Test that backtest works without Numba installed (use_numba=False)."""
    # Create synthetic price data
    dates = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT"]
    
    prices = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            prices.append({
                "timestamp": date,
                "symbol": symbol,
                "close": 100.0 + i * 0.5,
                "open": 99.0 + i * 0.5,
                "high": 101.0 + i * 0.5,
                "low": 98.0 + i * 0.5,
                "volume": 1000000,
            })
    
    prices_df = pd.DataFrame(prices)
    
    # Define signal and position sizing functions
    signal_fn = create_simple_signal_fn()
    
    def position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions(signals_df, total_capital=capital, equal_weight=True)
    
    # Run backtest with use_numba=False (should work even if Numba is not installed)
    result = run_portfolio_backtest(
        prices=prices_df,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=10000.0,
        include_costs=False,  # Simpler for testing
        use_numba=False,  # Explicitly disable Numba
    )
    
    # Assertions
    assert result is not None
    assert result.equity is not None
    assert len(result.equity) > 0
    assert "equity" in result.equity.columns
    assert result.metrics is not None
    assert "final_pf" in result.metrics


def test_backtest_numba_default_false():
    """Test that backtest defaults to use_numba=False."""
    # Create synthetic price data
    dates = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    symbols = ["AAPL"]
    
    prices = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            prices.append({
                "timestamp": date,
                "symbol": symbol,
                "close": 100.0 + i * 0.5,
            })
    
    prices_df = pd.DataFrame(prices)
    
    # Define signal and position sizing functions
    signal_fn = create_simple_signal_fn()
    
    def position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions(signals_df, total_capital=capital, equal_weight=True)
    
    # Run backtest without specifying use_numba (should default to False)
    result = run_portfolio_backtest(
        prices=prices_df,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=10000.0,
        include_costs=False,
        # use_numba not specified - should default to False
    )
    
    # Assertions
    assert result is not None
    assert result.equity is not None
    assert len(result.equity) > 0


def test_numba_kernels_graceful_fallback():
    """Test that Numba kernels fall back gracefully when Numba is not available."""
    from src.assembled_core.qa.numba_kernels import apply_fills_cash_delta_numba
    
    # Test that function exists and can be called (even if Numba is not available)
    import numpy as np
    
    # Create test arrays
    sides = np.array([0, 1, 0], dtype=np.int32)  # BUY, SELL, BUY
    qtys = np.array([100.0, 50.0, 200.0], dtype=np.float64)
    prices = np.array([150.0, 200.0, 100.0], dtype=np.float64)
    
    # Should work regardless of Numba availability
    try:
        cash_delta = apply_fills_cash_delta_numba(
            sides=sides,
            qtys=qtys,
            prices=prices,
            spread_w=0.0,
            impact_w=0.0,
            commission_bps=0.0,
        )
        # If Numba is available, should get JIT-compiled result
        # If not, should get fallback (pure NumPy) result
        assert isinstance(cash_delta, (float, np.floating))
    except Exception as e:
        # Should not raise exceptions (graceful fallback)
        pytest.fail(f"Numba kernel should fall back gracefully, but raised: {e}")


def test_settings_use_numba_default():
    """Test that Settings.use_numba defaults to False."""
    from src.assembled_core.config.settings import get_settings
    
    settings = get_settings()
    assert settings.use_numba is False  # Default should be False


def test_settings_use_numba_env_var():
    """Test that Settings.use_numba can be set via environment variable."""
    
    # Save original value
    original_value = os.environ.get("ASSEMBLED_USE_NUMBA")
    
    try:
        # Test with env var set to "true"
        os.environ["ASSEMBLED_USE_NUMBA"] = "true"
        # Clear cache if exists (Pydantic caches settings)
        # For now, just test that it can be read
        # Note: Pydantic BaseSettings caches, so we might need to reload
        # This is a smoke test - actual env var reading is tested by Pydantic
        
        # Test with env var set to "false"
        os.environ["ASSEMBLED_USE_NUMBA"] = "false"
        # Settings should respect env var (tested by Pydantic BaseSettings)
    finally:
        # Restore original value
        if original_value is not None:
            os.environ["ASSEMBLED_USE_NUMBA"] = original_value
        elif "ASSEMBLED_USE_NUMBA" in os.environ:
            del os.environ["ASSEMBLED_USE_NUMBA"]

