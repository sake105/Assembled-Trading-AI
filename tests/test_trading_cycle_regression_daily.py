"""Regression tests for trading_cycle vs. legacy run_daily path.

This test ensures that run_trading_cycle produces identical orders
compared to the legacy manual step-by-step path in run_daily.py.
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.features.ta_features import add_all_features
from src.assembled_core.pipeline.trading_cycle import (
    TradingContext,
    run_trading_cycle,
)
from src.assembled_core.portfolio.position_sizing import (
    compute_target_positions_from_trend_signals,
)
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices


def _legacy_generate_orders_path(
    prices: pd.DataFrame,
    target_date: pd.Timestamp,
    ma_fast: int = 20,
    ma_slow: int = 50,
    total_capital: float = 10000.0,
    top_n: int | None = None,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """Legacy path: manual step-by-step order generation (like old run_daily).
    
    This function replicates the old logic from run_daily.py before B1 refactoring:
    1. Filter prices to target date (last available <= target_date)
    2. Compute features (add_all_features)
    3. Generate signals (generate_trend_signals_from_prices)
    4. Compute target positions (compute_target_positions_from_trend_signals)
    5. Generate orders (generate_orders_from_signals)
    
    Args:
        prices: Full price DataFrame (columns: timestamp, symbol, close, ...)
        target_date: Target date (pd.Timestamp, UTC)
        ma_fast: Fast moving average window
        ma_slow: Slow moving average window
        total_capital: Total capital for position sizing
        top_n: Optional maximum number of positions
        min_score: Minimum signal score threshold
        
    Returns:
        Orders DataFrame (columns: timestamp, symbol, side, qty, price)
    """
    # Step 1: Filter prices to target date (last available per symbol)
    prices_filtered = prices[prices["timestamp"] <= target_date].copy()
    if prices_filtered.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    
    # Get last available timestamp per symbol
    prices_filtered = (
        prices_filtered.groupby("symbol", group_keys=False, dropna=False)
        .last()
        .reset_index()
    )
    
    # Step 2: Compute features
    prices_with_features = add_all_features(
        prices_filtered,
        ma_windows=(ma_fast, ma_slow),
        atr_window=14,
        rsi_window=14,
        include_rsi=True,
    )
    
    # Step 3: Generate signals
    signals = generate_trend_signals_from_prices(
        prices_with_features, ma_fast=ma_fast, ma_slow=ma_slow
    )
    
    # Step 4: Compute target positions
    target_positions = compute_target_positions_from_trend_signals(
        signals, total_capital=total_capital, top_n=top_n, min_score=min_score
    )
    
    # Step 5: Generate orders (legacy: generate_orders_from_targets, same as trading_cycle uses)
    from src.assembled_core.execution.order_generation import generate_orders_from_targets
    
    orders = generate_orders_from_targets(
        target_positions,
        current_positions=None,  # Legacy path doesn't have current positions
        timestamp=target_date,
        prices=prices_with_features,
    )
    
    return orders


def _create_synthetic_prices(
    symbols: list[str] = None, n_days: int = 10, start_date: str = "2025-01-01"
) -> pd.DataFrame:
    """Create synthetic price data for testing.
    
    Args:
        symbols: List of symbols (default: ["AAPL", "MSFT", "GOOGL"])
        n_days: Number of days (default: 10)
        start_date: Start date string (default: "2025-01-01")
        
    Returns:
        DataFrame with columns: timestamp, symbol, close, high, low, open, volume
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL"]
    
    dates = pd.date_range(start_date, periods=n_days, freq="D", tz="UTC")
    
    rows = []
    for symbol in symbols:
        base_price = 100.0 if symbol == "AAPL" else 200.0 if symbol == "MSFT" else 150.0
        for i, date in enumerate(dates):
            # Simple trend: prices increase over time with small random variation
            price = base_price + (i * 0.5) + (i % 3) * 0.2
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "close": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "open": price * 0.995,
                "volume": 1000000.0,
            })
    
    return pd.DataFrame(rows)


def test_trading_cycle_vs_legacy_deterministic() -> None:
    """Test that trading_cycle produces deterministic orders (same inputs -> same outputs)."""
    prices = _create_synthetic_prices(symbols=["AAPL", "MSFT"], n_days=10)
    target_date = pd.Timestamp("2025-01-10", tz="UTC")
    
    # Define functions for trading_cycle
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(df, ma_fast=5, ma_slow=10)
    
    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals, total_capital=capital, top_n=None, min_score=0.0
        )
    
    ctx = TradingContext(
        prices=prices,
        as_of=target_date,
        freq="1d",
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=10000.0,
        enable_risk_controls=False,
        write_outputs=False,
    )
    
    # Run twice - should produce identical results
    result1 = run_trading_cycle(ctx)
    result2 = run_trading_cycle(ctx)
    
    assert result1.status == "success"
    assert result2.status == "success"
    
    # Orders should be identical
    pd.testing.assert_frame_equal(
        result1.orders.sort_values(["symbol", "side"]).reset_index(drop=True),
        result2.orders.sort_values(["symbol", "side"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_trading_cycle_vs_legacy_orders_identical() -> None:
    """Test that trading_cycle produces identical orders compared to legacy path."""
    prices = _create_synthetic_prices(symbols=["AAPL", "MSFT"], n_days=10)
    target_date = pd.Timestamp("2025-01-10", tz="UTC")
    
    ma_fast = 5
    ma_slow = 10
    total_capital = 10000.0
    
    # Legacy path
    legacy_orders = _legacy_generate_orders_path(
        prices=prices,
        target_date=target_date,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        total_capital=total_capital,
        top_n=None,
        min_score=0.0,
    )
    
    # Trading cycle path
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(df, ma_fast=ma_fast, ma_slow=ma_slow)
    
    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals, total_capital=capital, top_n=None, min_score=0.0
        )
    
    ctx = TradingContext(
        prices=prices,
        as_of=target_date,
        freq="1d",
        feature_config={
            "ma_windows": (ma_fast, ma_slow),
            "atr_window": 14,
            "rsi_window": 14,
            "include_rsi": True,
        },
        signal_fn=signal_fn,
        signal_config={"ma_fast": ma_fast, "ma_slow": ma_slow},
        position_sizing_fn=sizing_fn,
        capital=total_capital,
        enable_risk_controls=False,
        write_outputs=False,
    )
    
    cycle_result = run_trading_cycle(ctx)
    assert cycle_result.status == "success"
    
    cycle_orders = cycle_result.orders
    
    # Normalize: sort by symbol, side, qty for comparison
    legacy_sorted = legacy_orders.sort_values(["symbol", "side", "qty"]).reset_index(drop=True)
    cycle_sorted = cycle_orders.sort_values(["symbol", "side", "qty"]).reset_index(drop=True)
    
    # Check structure
    assert set(legacy_sorted.columns) == set(cycle_sorted.columns)
    assert {"timestamp", "symbol", "side", "qty", "price"}.issubset(set(legacy_sorted.columns))
    
    # Check no NaNs
    assert not legacy_sorted[["symbol", "side", "qty"]].isna().any().any()
    assert not cycle_sorted[["symbol", "side", "qty"]].isna().any().any()
    
    # Check order counts match
    assert len(legacy_sorted) == len(cycle_sorted), (
        f"Order count mismatch: legacy={len(legacy_sorted)}, cycle={len(cycle_sorted)}"
    )
    
    # If both have orders, check symbol/side combinations match
    if not legacy_sorted.empty and not cycle_sorted.empty:
        legacy_keys = set(zip(legacy_sorted["symbol"], legacy_sorted["side"]))
        cycle_keys = set(zip(cycle_sorted["symbol"], cycle_sorted["side"]))
        
        # Keys should match (same symbols and sides)
        assert legacy_keys == cycle_keys, (
            f"Order keys mismatch: legacy={legacy_keys}, cycle={cycle_keys}"
        )
        
        # Check quantities and prices are close (within tolerance for float comparison)
        for key in legacy_keys:
            legacy_row = legacy_sorted[
                (legacy_sorted["symbol"] == key[0]) & (legacy_sorted["side"] == key[1])
            ].iloc[0]
            cycle_row = cycle_sorted[
                (cycle_sorted["symbol"] == key[0]) & (cycle_sorted["side"] == key[1])
            ].iloc[0]
            
            # Quantities should be identical (exact)
            assert legacy_row["qty"] == cycle_row["qty"], (
                f"Qty mismatch for {key}: legacy={legacy_row['qty']}, cycle={cycle_row['qty']}"
            )
            
            # Prices should be close (within 0.01 tolerance)
            assert abs(legacy_row["price"] - cycle_row["price"]) < 0.01, (
                f"Price mismatch for {key}: legacy={legacy_row['price']}, cycle={cycle_row['price']}"
            )


def test_trading_cycle_orders_structure() -> None:
    """Test that trading_cycle produces orders with expected structure."""
    prices = _create_synthetic_prices(symbols=["AAPL"], n_days=10)
    target_date = pd.Timestamp("2025-01-10", tz="UTC")
    
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(df, ma_fast=5, ma_slow=10)
    
    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals, total_capital=capital, top_n=None, min_score=0.0
        )
    
    ctx = TradingContext(
        prices=prices,
        as_of=target_date,
        freq="1d",
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=10000.0,
        enable_risk_controls=False,
        write_outputs=False,
    )
    
    result = run_trading_cycle(ctx)
    assert result.status == "success"
    
    orders = result.orders
    
    # Check required columns
    required_cols = {"timestamp", "symbol", "side", "qty", "price"}
    assert required_cols.issubset(set(orders.columns)), (
        f"Missing required columns. Expected: {required_cols}, Got: {set(orders.columns)}"
    )
    
    # Check no NaNs in critical columns
    if not orders.empty:
        assert not orders[["symbol", "side", "qty"]].isna().any().any(), (
            "Orders contain NaNs in symbol, side, or qty columns"
        )
        
        # Check side values are valid
        valid_sides = {"BUY", "SELL"}
        invalid_sides = set(orders["side"].unique()) - valid_sides
        assert len(invalid_sides) == 0, (
            f"Invalid side values: {invalid_sides}"
        )
        
        # Check qty is positive
        assert (orders["qty"] > 0).all(), "Orders contain non-positive quantities"
        
        # Check price is positive (if not empty)
        if "price" in orders.columns:
            prices_valid = orders["price"].dropna()
            if not prices_valid.empty:
                assert (prices_valid > 0).all(), "Orders contain non-positive prices"


def test_trading_cycle_with_three_symbols() -> None:
    """Test trading_cycle with 3 symbols to ensure multi-symbol handling."""
    prices = _create_synthetic_prices(symbols=["AAPL", "MSFT", "GOOGL"], n_days=10)
    target_date = pd.Timestamp("2025-01-10", tz="UTC")
    
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(df, ma_fast=5, ma_slow=10)
    
    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals, total_capital=capital, top_n=None, min_score=0.0
        )
    
    ctx = TradingContext(
        prices=prices,
        as_of=target_date,
        freq="1d",
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=10000.0,
        enable_risk_controls=False,
        write_outputs=False,
    )
    
    result = run_trading_cycle(ctx)
    assert result.status == "success"
    
    # Should have filtered prices for 3 symbols
    assert result.prices_filtered["symbol"].nunique() == 3
    
    # Should have features built
    assert not result.prices_with_features.empty
    assert len(result.prices_with_features.columns) > len(result.prices_filtered.columns)
    
    # Should have signals for potentially all symbols
    assert not result.signals.empty
    
    # Orders should be well-formed
    if not result.orders.empty:
        assert result.orders["symbol"].nunique() <= 3
        assert (result.orders["qty"] > 0).all()


def test_trading_cycle_empty_signals() -> None:
    """Test trading_cycle handles empty signals gracefully."""
    prices = _create_synthetic_prices(symbols=["AAPL"], n_days=2)  # Too few days for MA signals
    target_date = pd.Timestamp("2025-01-02", tz="UTC")
    
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        # Return empty signals (no signals generated)
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    
    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        # Return empty target positions
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    
    ctx = TradingContext(
        prices=prices,
        as_of=target_date,
        freq="1d",
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=10000.0,
        enable_risk_controls=False,
        write_outputs=False,
    )
    
    result = run_trading_cycle(ctx)
    assert result.status == "success"
    
    # Should have empty orders
    assert result.orders.empty or len(result.orders) == 0

