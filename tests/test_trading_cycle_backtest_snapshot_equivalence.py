"""Test that snapshot mode (backtest_use_snapshot=True) produces identical results to history-slice mode (backtest_use_snapshot=False).

This test ensures that the performance optimization (snapshot mode) does not change
the behavior for trend_baseline strategy.
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.features.ta_features import add_all_features
from src.assembled_core.pipeline.trading_cycle import TradingContext, run_trading_cycle
from src.assembled_core.portfolio.position_sizing import (
    compute_target_positions_from_trend_signals,
)
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices


def create_synthetic_prices_30_days() -> pd.DataFrame:
    """Create synthetic price data: 30 days, 3 symbols.

    Returns:
        DataFrame with columns: timestamp, symbol, close, high, low, open, volume
        30 days per symbol, 3 symbols = 90 rows total
    """
    dates = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    data = []
    for symbol in symbols:
        base_price = 100.0 if symbol == "AAPL" else (200.0 if symbol == "MSFT" else 150.0)
        for i, date in enumerate(dates):
            # Simple upward trend with noise
            close = base_price + i * 0.5 + (i % 5) * 0.1
            high = close * 1.02
            low = close * 0.98
            open_price = close * 0.99
            volume = 1000000.0 + i * 10000.0
            
            data.append({
                "timestamp": date,
                "symbol": symbol,
                "close": close,
                "high": high,
                "low": low,
                "open": open_price,
                "volume": volume,
            })
    
    df = pd.DataFrame(data)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def test_backtest_snapshot_vs_history_slice_equivalence():
    """Test that snapshot mode produces identical orders to history-slice mode.

    This test:
    1. Creates synthetic prices (30 days, 3 symbols)
    2. Precomputes features for full panel
    3. Runs both modes (snapshot=True vs False) for multiple as_of timestamps
    4. Asserts orders are identical (sorted, no NaNs, same qty/side/symbol)
    """
    # Create synthetic prices
    prices = create_synthetic_prices_30_days()
    
    # Precompute features for full panel
    prices_with_features_full = add_all_features(prices.copy())
    
    # Define signal and sizing functions (trend_baseline strategy)
    ma_fast = 5
    ma_slow = 10
    
    def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(prices_df, ma_fast=ma_fast, ma_slow=ma_slow)
    
    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals,
            total_capital=capital,
            top_n=None,
            min_score=0.0,
        )
    
    # Test multiple as_of timestamps (avoid first few days where MAs might be NaN)
    dates = prices["timestamp"].unique()
    test_as_ofs = dates[10:25]  # Days 11-25 (enough history for MAs)
    
    for as_of in test_as_ofs:
        # Run with snapshot mode (backtest_use_snapshot=True)
        ctx_snapshot = TradingContext(
            prices=prices.copy(),
            as_of=as_of,
            mode="backtest",
            precomputed_prices_with_features=prices_with_features_full.copy(),
            backtest_use_snapshot=True,
            signal_fn=signal_fn,
            position_sizing_fn=sizing_fn,
            capital=10000.0,
            enable_risk_controls=False,  # Disable risk controls for deterministic comparison
            write_outputs=False,
        )
        
        result_snapshot = run_trading_cycle(ctx_snapshot)
        
        # Run with history-slice mode (backtest_use_snapshot=False)
        ctx_history = TradingContext(
            prices=prices.copy(),
            as_of=as_of,
            mode="backtest",
            precomputed_prices_with_features=prices_with_features_full.copy(),
            backtest_use_snapshot=False,
            signal_fn=signal_fn,
            position_sizing_fn=sizing_fn,
            capital=10000.0,
            enable_risk_controls=False,
            write_outputs=False,
        )
        
        result_history = run_trading_cycle(ctx_history)
        
        # Verify both runs succeeded
        assert result_snapshot.status == "success", f"Snapshot mode failed at as_of={as_of}"
        assert result_history.status == "success", f"History-slice mode failed at as_of={as_of}"
        
        # Extract orders
        orders_snapshot = result_snapshot.orders_filtered if not result_snapshot.orders_filtered.empty else result_snapshot.orders
        orders_history = result_history.orders_filtered if not result_history.orders_filtered.empty else result_history.orders
        
        # Assert orders are not empty (or both empty)
        if orders_snapshot.empty and orders_history.empty:
            # Both empty is fine (no signals generated)
            continue
        
        # Assert both have orders or both are empty
        assert not orders_snapshot.empty, f"Snapshot mode produced empty orders at as_of={as_of}"
        assert not orders_history.empty, f"History-slice mode produced empty orders at as_of={as_of}"
        
        # Sort orders for comparison (by symbol, then by side)
        orders_snapshot_sorted = orders_snapshot.sort_values(["symbol", "side", "qty"]).reset_index(drop=True)
        orders_history_sorted = orders_history.sort_values(["symbol", "side", "qty"]).reset_index(drop=True)
        
        # Assert no NaNs in critical columns
        assert not orders_snapshot_sorted[["symbol", "side", "qty", "price"]].isna().any().any(), \
            f"Snapshot mode produced NaNs at as_of={as_of}"
        assert not orders_history_sorted[["symbol", "side", "qty", "price"]].isna().any().any(), \
            f"History-slice mode produced NaNs at as_of={as_of}"
        
        # Assert same number of orders
        assert len(orders_snapshot_sorted) == len(orders_history_sorted), \
            f"Different number of orders at as_of={as_of}: snapshot={len(orders_snapshot_sorted)}, history={len(orders_history_sorted)}"
        
        # Assert same symbols
        symbols_snapshot = set(orders_snapshot_sorted["symbol"].unique())
        symbols_history = set(orders_history_sorted["symbol"].unique())
        assert symbols_snapshot == symbols_history, \
            f"Different symbols at as_of={as_of}: snapshot={symbols_snapshot}, history={symbols_history}"
        
        # Compare order details (qty, side, symbol, price)
        # Use approximate comparison for price (floating point tolerance)
        for col in ["symbol", "side", "qty"]:
            pd.testing.assert_series_equal(
                orders_snapshot_sorted[col],
                orders_history_sorted[col],
                check_names=False,
                check_dtype=False,
                obj=f"Column {col} differs at as_of={as_of}",
            )
        
        # Price comparison with tolerance (1e-6)
        pd.testing.assert_series_equal(
            orders_snapshot_sorted["price"],
            orders_history_sorted["price"],
            check_names=False,
            check_dtype=False,
            rtol=1e-6,
            atol=1e-6,
            obj=f"Price differs at as_of={as_of}",
        )

