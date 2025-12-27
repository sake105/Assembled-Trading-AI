"""Tests for Numba equivalence (Numba path vs. fallback path).

This test ensures that the Numba-accelerated path produces identical results
to the pure NumPy/pandas fallback path. If Numba is not available, the fallback
should still produce deterministic results.
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.qa.backtest_engine import BacktestResult, run_portfolio_backtest


def create_synthetic_prices(
    symbols: list[str],
    start_date: str,
    num_days: int,
) -> pd.DataFrame:
    """Create synthetic price data for testing.
    
    Args:
        symbols: List of symbols
        start_date: Start date (YYYY-MM-DD)
        num_days: Number of days of data
    
    Returns:
        DataFrame with columns: timestamp, symbol, close (and OHLCV if needed)
    """
    dates = pd.date_range(start=start_date, periods=num_days, freq="D", tz="UTC")
    
    rows = []
    for date in dates:
        for symbol in symbols:
            # Simple random walk for prices (deterministic with seed)
            # Base price per symbol: AAPL=150, MSFT=300, etc.
            base_price = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0}.get(symbol, 100.0)
            # Add small daily variation (deterministic based on date)
            daily_offset = hash(f"{symbol}{date.date()}") % 100 / 100.0  # -0.5 to 0.5
            price = base_price * (1.0 + (daily_offset - 0.5) * 0.02)  # ±1% variation
            
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "close": price,
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "volume": 1000000.0,
            })
    
    df = pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def create_simple_signal_fn(ma_fast: int = 5, ma_slow: int = 10):
    """Create a simple trend signal function for testing."""
    def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        """Generate simple EMA crossover signals."""
        signals_list = []
        
        for symbol in prices_df["symbol"].unique():
            symbol_prices = prices_df[prices_df["symbol"] == symbol].sort_values("timestamp")
            if len(symbol_prices) < ma_slow:
                continue
            
            # Simple EMA calculation
            close_prices = symbol_prices["close"].values
            ema_fast = pd.Series(close_prices).ewm(span=ma_fast, adjust=False).mean()
            ema_slow = pd.Series(close_prices).ewm(span=ma_slow, adjust=False).mean()
            
            # Generate signal on last row
            last_fast = ema_fast.iloc[-1]
            last_slow = ema_slow.iloc[-1]
            
            if last_fast > last_slow:
                direction = "LONG"
                score = (last_fast - last_slow) / last_slow
            else:
                direction = "NEUTRAL"
                score = 0.0
            
            signals_list.append({
                "timestamp": symbol_prices["timestamp"].iloc[-1],
                "symbol": symbol,
                "direction": direction,
                "score": score,
            })
        
        result = pd.DataFrame(signals_list)
        # Ensure required columns are present
        if result.empty:
            result = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
        return result
    
    return signal_fn


def create_simple_position_sizing_fn():
    """Create a simple equal-weight position sizing function."""
    from src.assembled_core.portfolio.position_sizing import compute_target_positions
    
    def position_sizing_fn(signals_df: pd.DataFrame, total_capital: float) -> pd.DataFrame:
        """Equal-weight position sizing for LONG signals."""
        # Use existing compute_target_positions function
        return compute_target_positions(
            signals_df,
            total_capital=total_capital,
            equal_weight=True,
        )
    
    return position_sizing_fn


def test_numba_equivalence_mini_backtest():
    """Test that Numba path produces same results as fallback path."""
    # Create synthetic prices
    symbols = ["AAPL", "MSFT"]
    prices = create_synthetic_prices(symbols=symbols, start_date="2020-01-01", num_days=20)
    
    # Create simple signal and position sizing functions
    signal_fn = create_simple_signal_fn(ma_fast=5, ma_slow=10)
    position_sizing_fn = create_simple_position_sizing_fn()
    
    start_capital = 10000.0
    
    # Run backtest WITHOUT Numba (pure NumPy/pandas path)
    result_no_numba = run_portfolio_backtest(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=start_capital,
        commission_bps=1.0,  # 1 bps commission
        spread_w=0.5,  # 0.5 bps spread
        impact_w=0.25,  # 0.25 bps impact
        use_numba=False,
        include_trades=True,
    )
    
    # Run backtest WITH Numba (should use Numba if available, fallback otherwise)
    result_with_numba = run_portfolio_backtest(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=start_capital,
        commission_bps=1.0,
        spread_w=0.5,
        impact_w=0.25,
        use_numba=True,
        include_trades=True,
    )
    
    # Assertions: Equity should be nearly identical
    assert isinstance(result_no_numba, BacktestResult)
    assert isinstance(result_with_numba, BacktestResult)
    
    # Compare equity curves (should be nearly identical)
    equity_no_numba = result_no_numba.equity["equity"].values
    equity_with_numba = result_with_numba.equity["equity"].values
    
    assert len(equity_no_numba) == len(equity_with_numba), "Equity curves should have same length"
    
    # Check that equity values are nearly identical (within small tolerance)
    import numpy as np
    np.testing.assert_allclose(
        equity_no_numba,
        equity_with_numba,
        rtol=1e-6,  # Relative tolerance (very small)
        atol=1e-6,  # Absolute tolerance (very small)
        err_msg="Equity curves should be nearly identical between Numba and non-Numba paths",
    )
    
    # Compare final equity (more lenient tolerance for final value)
    final_equity_no_numba = equity_no_numba[-1]
    final_equity_with_numba = equity_with_numba[-1]
    
    assert abs(final_equity_no_numba - final_equity_with_numba) < 1e-3, (
        f"Final equity should be nearly identical: {final_equity_no_numba} vs {final_equity_with_numba}"
    )
    
    # Compare metrics (final_pf, sharpe should be nearly identical)
    metrics_no_numba = result_no_numba.metrics
    metrics_with_numba = result_with_numba.metrics
    
    # Final PF should be identical (within tolerance)
    final_pf_no_numba = metrics_no_numba.get("final_pf", float('nan'))
    final_pf_with_numba = metrics_with_numba.get("final_pf", float('nan'))
    
    if not (pd.isna(final_pf_no_numba) or pd.isna(final_pf_with_numba)):
        assert abs(final_pf_no_numba - final_pf_with_numba) < 1e-6, (
            f"Final PF should be nearly identical: {final_pf_no_numba} vs {final_pf_with_numba}"
        )
    
    # Compare trade counts (should be identical)
    trades_no_numba = metrics_no_numba.get("trades", 0)
    trades_with_numba = metrics_with_numba.get("trades", 0)
    
    assert trades_no_numba == trades_with_numba, (
        f"Trade counts should be identical: {trades_no_numba} vs {trades_with_numba}"
    )
    
    # Compare actual trades DataFrames (if available)
    if result_no_numba.trades is not None and result_with_numba.trades is not None:
        trades_df_no_numba = result_no_numba.trades.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        trades_df_with_numba = result_with_numba.trades.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        
        assert len(trades_df_no_numba) == len(trades_df_with_numba), (
            f"Trade DataFrames should have same length: {len(trades_df_no_numba)} vs {len(trades_df_with_numba)}"
        )
        
        # Compare trade quantities and sides (should be nearly identical)
        if not trades_df_no_numba.empty:
            qty_no_numba = trades_df_no_numba["qty"].values
            qty_with_numba = trades_df_with_numba["qty"].values
            
            np.testing.assert_allclose(
                qty_no_numba,
                qty_with_numba,
                rtol=1e-6,
                atol=1e-6,
                err_msg="Trade quantities should be nearly identical",
            )
            
            # Sides should be identical (exact match)
            sides_no_numba = trades_df_no_numba["side"].values
            sides_with_numba = trades_df_with_numba["side"].values
            
            assert np.array_equal(sides_no_numba, sides_with_numba), (
                "Trade sides should be identical between Numba and non-Numba paths"
            )


def test_numba_equivalence_deterministic_order_counts():
    """Test that order/trade counts are deterministic regardless of Numba usage."""
    # Create synthetic prices
    symbols = ["AAPL", "MSFT"]
    prices = create_synthetic_prices(symbols=symbols, start_date="2020-01-01", num_days=20)
    
    # Create simple signal and position sizing functions
    signal_fn = create_simple_signal_fn(ma_fast=5, ma_slow=10)
    position_sizing_fn = create_simple_position_sizing_fn()
    
    start_capital = 10000.0
    
    # Run backtest multiple times with same seed/conditions
    results_no_numba = []
    results_with_numba = []
    
    for _ in range(3):  # Run 3 times to check determinism
        result_no_numba = run_portfolio_backtest(
            prices=prices.copy(),  # Copy to avoid mutations
            signal_fn=signal_fn,
            position_sizing_fn=position_sizing_fn,
            start_capital=start_capital,
            commission_bps=1.0,
            spread_w=0.5,
            impact_w=0.25,
            use_numba=False,
            include_trades=True,
        )
        results_no_numba.append(result_no_numba.metrics.get("trades", 0))
        
        result_with_numba = run_portfolio_backtest(
            prices=prices.copy(),
            signal_fn=signal_fn,
            position_sizing_fn=position_sizing_fn,
            start_capital=start_capital,
            commission_bps=1.0,
            spread_w=0.5,
            impact_w=0.25,
            use_numba=True,
            include_trades=True,
        )
        results_with_numba.append(result_with_numba.metrics.get("trades", 0))
    
    # All runs should produce same trade count (deterministic)
    assert len(set(results_no_numba)) == 1, "Non-Numba path should be deterministic"
    assert len(set(results_with_numba)) == 1, "Numba path should be deterministic"
    assert results_no_numba[0] == results_with_numba[0], (
        "Numba and non-Numba paths should produce same trade counts"
    )

