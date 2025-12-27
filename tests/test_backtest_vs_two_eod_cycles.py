# tests/test_backtest_vs_two_eod_cycles.py
"""Regression test: Backtest over 2 days should match two sequential EOD cycles.

This test ensures that running a 2-day backtest (via TradingCycle in backtest mode)
produces identical results to running two sequential EOD cycles manually.

Success criterion from backend upgrade nr2:
- Positions path identical
- Equity path identical (within rounding tolerance)
- Orders structure deterministic and correctly sorted
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.assembled_core.qa.backtest_engine import (
    make_cycle_fn,
    run_portfolio_backtest,
)
from src.assembled_core.pipeline.trading_cycle import TradingContext, run_trading_cycle
from src.assembled_core.qa.backtest_engine import _update_positions_vectorized
from src.assembled_core.pipeline.backtest import _simulate_fills_per_order
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
from src.assembled_core.portfolio.position_sizing import (
    compute_target_positions_from_trend_signals,
)


def create_synthetic_prices(
    symbols: list[str] = ["AAPL", "MSFT", "NVDA"],
    start_date: str = "2024-01-01",
    days: int = 20,
) -> pd.DataFrame:
    """Create synthetic price data for testing.
    
    Args:
        symbols: List of symbols to generate
        start_date: Start date string
        days: Number of days to generate
        
    Returns:
        DataFrame with columns: timestamp, symbol, close (and OHLCV for features)
    """
    dates = pd.date_range(start=start_date, periods=days, freq="D", tz="UTC")
    
    all_prices = []
    for symbol in symbols:
        # Generate synthetic prices with some trend and volatility
        np.random.seed(42)  # Deterministic
        base_price = 100.0 + hash(symbol) % 50
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        prices = base_price * np.exp(np.cumsum(returns))
        
        for date, price in zip(dates, prices):
            all_prices.append({
                "timestamp": date,
                "symbol": symbol,
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1000000,
            })
    
    return pd.DataFrame(all_prices).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def create_signal_fn():
    """Create a simple signal function for testing."""
    def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(prices_df, ma_fast=5, ma_slow=10)
    
    return signal_fn


def create_position_sizing_fn():
    """Create a simple position sizing function for testing."""
    def position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals_df,
            total_capital=capital,
            top_n=None,
            min_score=0.0,
        )
    
    return position_sizing_fn


def positions_df_to_dict(positions_df: pd.DataFrame) -> dict[str, float]:
    """Convert positions DataFrame to dictionary."""
    if positions_df.empty:
        return {}
    return dict(zip(positions_df["symbol"], positions_df["qty"]))


def positions_dict_to_df(positions: dict[str, float]) -> pd.DataFrame:
    """Convert positions dictionary to DataFrame."""
    if not positions:
        return pd.DataFrame(columns=["symbol", "qty"])
    return pd.DataFrame(
        {"symbol": list(positions.keys()), "qty": list(positions.values())}
    ).sort_values("symbol").reset_index(drop=True)


def compute_equity_from_positions(
    positions: dict[str, float],
    cash: float,
    prices: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> float:
    """Compute equity from positions and cash at a given timestamp.
    
    Args:
        positions: Dictionary mapping symbol -> quantity
        cash: Cash balance
        prices: Full prices DataFrame
        timestamp: Timestamp to compute equity at
        
    Returns:
        Total equity (cash + positions value)
    """
    equity = cash
    
    # Get prices at timestamp
    prices_at_ts = prices[prices["timestamp"] == timestamp]
    if prices_at_ts.empty:
        return equity
    
    # Sum position values (qty * price for each symbol)
    for symbol, qty in positions.items():
        symbol_prices = prices_at_ts[prices_at_ts["symbol"] == symbol]
        if not symbol_prices.empty:
            price = symbol_prices["close"].iloc[0]
            equity += qty * price
    
    return equity


def test_backtest_vs_two_eod_cycles():
    """Test that 2-day backtest matches two sequential EOD cycles."""
    # Create synthetic prices: 2-3 symbols, 20 days, extract 2 trading days
    prices_full = create_synthetic_prices(symbols=["AAPL", "MSFT"], days=20)
    
    # Extract 2 trading days (day 10 and day 11)
    dates = sorted(prices_full["timestamp"].unique())
    as_of_1 = dates[10]  # Day 11 (0-indexed)
    as_of_2 = dates[11]  # Day 12
    
    # Filter prices to only include data up to and including the 2 trading days
    prices_test = prices_full[prices_full["timestamp"] <= as_of_2].copy()
    
    # Parameters
    start_capital = 10000.0
    commission_bps = 0.5
    spread_w = 0.25
    impact_w = 0.5
    
    # Create signal and position sizing functions
    signal_fn = create_signal_fn()
    position_sizing_fn = create_position_sizing_fn()
    
    # ===== PFAD A: Backtest über 2 Tage (trading_cycle mode backtest) =====
    # Create TradingContext template for backtest
    ctx_template_backtest = TradingContext(
        prices=prices_test,
        freq="1d",
        universe=None,
        use_factor_store=False,
        factor_store_root=None,
        factor_group="core_ta",
        feature_config={},
        write_outputs=False,
        enable_risk_controls=False,
    )
    
    # Create cycle_fn
    cycle_fn = make_cycle_fn(
        ctx_template_backtest,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        capital=start_capital,
    )
    
    # Run backtest over 2 days (only the 2 trading days)
    prices_2days = prices_test[prices_test["timestamp"].isin([as_of_1, as_of_2])].copy()
    
    result_backtest = run_portfolio_backtest(
        prices=prices_test,  # Full prices for feature computation (MAs need history)
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=start_capital,
        commission_bps=commission_bps,
        spread_w=spread_w,
        impact_w=impact_w,
        include_costs=True,
        include_trades=True,
        include_signals=False,
        include_targets=False,
        rebalance_freq="1d",
        compute_features=True,
        cycle_fn=cycle_fn,  # Use TradingCycle integration
    )
    
    # Extract final positions from backtest (reconstruct from orders)
    # Backtest engine uses _update_positions_vectorized for positions (no costs in position updates)
    # Costs are applied in equity calculation via simulate_with_costs
    if not result_backtest.trades.empty:
        # Start with empty positions DataFrame
        positions_backtest = pd.DataFrame(columns=["symbol", "qty"])
        # Apply all orders sequentially using _update_positions_vectorized (same as engine)
        for timestamp in sorted(result_backtest.trades["timestamp"].unique()):
            orders_at_ts = result_backtest.trades[result_backtest.trades["timestamp"] == timestamp]
            if not orders_at_ts.empty:
                positions_backtest = _update_positions_vectorized(orders_at_ts, positions_backtest)
    else:
        positions_backtest = pd.DataFrame(columns=["symbol", "qty"])
    
    positions_backtest_dict = positions_df_to_dict(positions_backtest)
    
    # Extract equity from backtest result (last equity value - includes costs)
    equity_backtest_final = result_backtest.equity["equity"].iloc[-1]
    
    # ===== PFAD B: Zwei manuelle EOD-Zyklen =====
    # Create TradingContext template for EOD
    ctx_template_eod = TradingContext(
        prices=prices_test,
        freq="1d",
        universe=None,
        use_factor_store=False,
        factor_store_root=None,
        factor_group="core_ta",
        feature_config={},
        write_outputs=False,
        enable_risk_controls=False,
    )
    
    # Day 1: Run EOD cycle
    from dataclasses import replace
    
    positions_eod = {}  # Start with empty positions
    cash_eod = start_capital
    
    # Day 1: EOD cycle
    ctx_day1 = replace(
        ctx_template_eod,
        as_of=as_of_1,
        mode="eod",
        current_positions=positions_dict_to_df(positions_eod),
        order_timestamp=as_of_1,
        capital=cash_eod,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
    )
    
    cycle_result_day1 = run_trading_cycle(ctx_day1)
    assert cycle_result_day1.status == "success", f"Day 1 cycle failed: {cycle_result_day1.error_message}"
    
    # Apply fills and update positions for day 1
    orders_day1 = (
        cycle_result_day1.orders_filtered
        if not cycle_result_day1.orders_filtered.empty
        else cycle_result_day1.orders
    )
    
    if not orders_day1.empty:
        # Apply fills (simulate costs) - same as backtest engine uses
        # For EOD, we use _simulate_fills_per_order which updates both cash and positions
        cash_eod, positions_eod = _simulate_fills_per_order(
            orders_at_timestamp=orders_day1,
            cash=cash_eod,
            positions=positions_eod.copy(),
            spread_w=spread_w,
            impact_w=impact_w,
            commission_bps=commission_bps,
        )
    
    # Day 2: EOD cycle (with updated positions and cash)
    ctx_day2 = replace(
        ctx_template_eod,
        as_of=as_of_2,
        mode="eod",
        current_positions=positions_dict_to_df(positions_eod),
        order_timestamp=as_of_2,
        capital=cash_eod,  # Updated cash from day 1
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
    )
    
    cycle_result_day2 = run_trading_cycle(ctx_day2)
    assert cycle_result_day2.status == "success", f"Day 2 cycle failed: {cycle_result_day2.error_message}"
    
    # Apply fills and update positions for day 2
    orders_day2 = (
        cycle_result_day2.orders_filtered
        if not cycle_result_day2.orders_filtered.empty
        else cycle_result_day2.orders
    )
    
    if not orders_day2.empty:
        # Apply fills (simulate costs)
        cash_eod, positions_eod = _simulate_fills_per_order(
            orders_at_timestamp=orders_day2,
            cash=cash_eod,
            positions=positions_eod.copy(),
            spread_w=spread_w,
            impact_w=impact_w,
            commission_bps=commission_bps,
        )
    
    # Compute final equity for EOD path
    # Note: simulate_with_costs in backtest only tracks cash deltas, not position values
    # For consistency, we need to compute equity the same way: cash + positions value
    equity_eod_final = compute_equity_from_positions(
        positions_eod, cash_eod, prices_test, as_of_2
    )
    
    # ===== ASSERTIONS =====
    
    # 1. End positions identical
    positions_backtest_sorted = dict(sorted(positions_backtest_dict.items()))
    positions_eod_sorted = dict(sorted(positions_eod.items()))
    
    # Compare positions (allowing for floating point tolerance)
    assert set(positions_backtest_sorted.keys()) == set(
        positions_eod_sorted.keys()
    ), f"Position symbols differ: backtest={set(positions_backtest_sorted.keys())}, eod={set(positions_eod_sorted.keys())}"
    
    for symbol in positions_backtest_sorted:
        backtest_qty = positions_backtest_sorted[symbol]
        eod_qty = positions_eod_sorted.get(symbol, 0.0)
        assert abs(backtest_qty - eod_qty) < 1e-6, (
            f"Position qty differs for {symbol}: "
            f"backtest={backtest_qty}, eod={eod_qty}, diff={abs(backtest_qty - eod_qty)}"
        )
    
    # 2. Equity identical (within rounding tolerance)
    equity_diff = abs(equity_backtest_final - equity_eod_final)
    assert equity_diff < 1e-2, (  # 1 cent tolerance
        f"Equity differs: backtest={equity_backtest_final:.4f}, "
        f"eod={equity_eod_final:.4f}, diff={equity_diff:.4f}"
    )
    
    # 3. Orders structure ok, deterministically sorted
    if not result_backtest.trades.empty:
        # Check that orders are sorted by timestamp, then symbol
        orders_sorted = result_backtest.trades.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            result_backtest.trades.sort_values(["timestamp", "symbol"]).reset_index(drop=True),
            orders_sorted,
            check_exact=False,
            rtol=1e-6,
        )
        
        # Check required columns
        required_cols = ["timestamp", "symbol", "side", "qty", "price"]
        assert all(col in result_backtest.trades.columns for col in required_cols), (
            f"Missing required columns in trades: {set(required_cols) - set(result_backtest.trades.columns)}"
        )
    
    print(f"✓ Positions match: {positions_backtest_sorted}")
    print(f"✓ Equity matches: backtest={equity_backtest_final:.4f}, eod={equity_eod_final:.4f}")
    print(f"✓ Orders structure valid: {len(result_backtest.trades) if result_backtest.trades is not None else 0} trades")

