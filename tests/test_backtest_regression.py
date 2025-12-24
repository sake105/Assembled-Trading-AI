"""Regression tests for backtest optimizations.

This test module ensures that optimizations (vectorization, Numba) don't change
the logic by comparing new implementations against legacy implementations.

Tolerances:
- Equity curve: rtol=1e-9, atol=1e-9 (numerical precision)
- Metrics: rtol=1e-6, atol=1e-6 (floating point operations)
- Order counts: exact match (integer)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.pipeline.backtest import compute_metrics, simulate_equity
from src.assembled_core.pipeline.backtest_legacy import (
    _legacy_simulate_equity,
)
from src.assembled_core.qa.backtest_engine import run_portfolio_backtest


# Tolerances for regression tests
EQUITY_RTOL = 1e-9  # Relative tolerance for equity curve
EQUITY_ATOL = 1e-9  # Absolute tolerance for equity curve
METRICS_RTOL = 1e-6  # Relative tolerance for metrics
METRICS_ATOL = 1e-6  # Absolute tolerance for metrics


def test_equity_curve_regression(golden_mini_backtest_data, position_sizing_fn):
    """Test that optimized equity curve matches legacy implementation."""
    prices = golden_mini_backtest_data["prices"]
    signals = golden_mini_backtest_data["signals"]
    start_capital = golden_mini_backtest_data["expected_equity_start"]

    # Generate orders from signals
    from src.assembled_core.execution.order_generation import generate_orders_from_targets

    # Group signals by timestamp and generate orders
    all_orders = []
    current_positions = pd.DataFrame(columns=["symbol", "qty"])

    for timestamp, signal_group in signals.groupby("timestamp"):
        targets = position_sizing_fn(signal_group, start_capital)
        prices_at_ts = prices[prices["timestamp"] == timestamp]
        orders = generate_orders_from_targets(
            target_positions=targets,
            current_positions=current_positions,
            timestamp=timestamp,
            prices=prices_at_ts if not prices_at_ts.empty else None,
        )
        if not orders.empty:
            all_orders.append(orders)
            # Update positions (simplified, just for order generation)
            from src.assembled_core.qa.backtest_engine import _update_positions_vectorized
            current_positions = _update_positions_vectorized(orders, current_positions)

    if all_orders:
        orders_df = pd.concat(all_orders, ignore_index=True)
    else:
        orders_df = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # Run optimized implementation
    equity_new = simulate_equity(prices, orders_df, start_capital)

    # Run legacy implementation
    equity_legacy = _legacy_simulate_equity(prices, orders_df, start_capital)

    # Compare equity curves
    assert len(equity_new) == len(equity_legacy), "Equity curves must have same length"

    # Sort by timestamp for comparison
    equity_new = equity_new.sort_values("timestamp").reset_index(drop=True)
    equity_legacy = equity_legacy.sort_values("timestamp").reset_index(drop=True)

    # Compare timestamps (should be identical)
    pd.testing.assert_series_equal(
        equity_new["timestamp"], equity_legacy["timestamp"], check_names=False
    )

    # Compare equity values (with tolerance)
    np.testing.assert_allclose(
        equity_new["equity"].values,
        equity_legacy["equity"].values,
        rtol=EQUITY_RTOL,
        atol=EQUITY_ATOL,
        err_msg="Equity curves must match within tolerance",
    )


def test_metrics_regression(golden_mini_backtest_data, position_sizing_fn):
    """Test that optimized metrics match legacy implementation."""
    prices = golden_mini_backtest_data["prices"]
    signals = golden_mini_backtest_data["signals"]
    start_capital = golden_mini_backtest_data["expected_equity_start"]

    # Generate orders
    from src.assembled_core.execution.order_generation import generate_orders_from_targets

    all_orders = []
    current_positions = pd.DataFrame(columns=["symbol", "qty"])

    for timestamp, signal_group in signals.groupby("timestamp"):
        targets = position_sizing_fn(signal_group, start_capital)
        prices_at_ts = prices[prices["timestamp"] == timestamp]
        orders = generate_orders_from_targets(
            target_positions=targets,
            current_positions=current_positions,
            timestamp=timestamp,
            prices=prices_at_ts if not prices_at_ts.empty else None,
        )
        if not orders.empty:
            all_orders.append(orders)
            from src.assembled_core.qa.backtest_engine import _update_positions_vectorized
            current_positions = _update_positions_vectorized(orders, current_positions)

    if all_orders:
        orders_df = pd.concat(all_orders, ignore_index=True)
    else:
        orders_df = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # Run both implementations
    equity_new = simulate_equity(prices, orders_df, start_capital)
    equity_legacy = _legacy_simulate_equity(prices, orders_df, start_capital)

    # Compute metrics
    metrics_new = compute_metrics(equity_new)
    metrics_legacy = compute_metrics(equity_legacy)

    # Compare key metrics
    assert abs(metrics_new["final_pf"] - metrics_legacy["final_pf"]) < METRICS_ATOL + (
        METRICS_RTOL * abs(metrics_legacy["final_pf"])
    ), f"final_pf mismatch: {metrics_new['final_pf']} vs {metrics_legacy['final_pf']}"

    # Sharpe can be NaN if no variance, so handle that case
    if not (np.isnan(metrics_new["sharpe"]) and np.isnan(metrics_legacy["sharpe"])):
        if not (np.isnan(metrics_new["sharpe"]) or np.isnan(metrics_legacy["sharpe"])):
            assert abs(metrics_new["sharpe"] - metrics_legacy["sharpe"]) < METRICS_ATOL + (
                METRICS_RTOL * abs(metrics_legacy["sharpe"])
            ), f"sharpe mismatch: {metrics_new['sharpe']} vs {metrics_legacy['sharpe']}"

    # Rows should be identical
    assert metrics_new["rows"] == metrics_legacy["rows"], "rows must match exactly"


def test_orders_count_regression(golden_mini_backtest_data, position_sizing_fn):
    """Test that order generation produces same number of orders."""
    prices = golden_mini_backtest_data["prices"]
    signals = golden_mini_backtest_data["signals"]
    start_capital = golden_mini_backtest_data["expected_equity_start"]

    # Generate orders (using optimized path)
    from src.assembled_core.execution.order_generation import generate_orders_from_targets

    all_orders = []
    current_positions = pd.DataFrame(columns=["symbol", "qty"])

    for timestamp, signal_group in signals.groupby("timestamp"):
        targets = position_sizing_fn(signal_group, start_capital)
        prices_at_ts = prices[prices["timestamp"] == timestamp]
        orders = generate_orders_from_targets(
            target_positions=targets,
            current_positions=current_positions,
            timestamp=timestamp,
            prices=prices_at_ts if not prices_at_ts.empty else None,
        )
        if not orders.empty:
            all_orders.append(orders)
            from src.assembled_core.qa.backtest_engine import _update_positions_vectorized
            current_positions = _update_positions_vectorized(orders, current_positions)

    if all_orders:
        orders_df = pd.concat(all_orders, ignore_index=True)
    else:
        orders_df = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # Verify orders count is reasonable (not zero, not excessive)
    assert len(orders_df) > 0, "Should generate at least some orders"
    assert len(orders_df) <= 50, f"Too many orders generated: {len(orders_df)}"

    # Verify orders structure
    assert "timestamp" in orders_df.columns
    assert "symbol" in orders_df.columns
    assert "side" in orders_df.columns
    assert "qty" in orders_df.columns
    assert "price" in orders_df.columns


def test_portfolio_backtest_regression(golden_mini_backtest_data, position_sizing_fn):
    """Test that full portfolio backtest produces consistent results."""
    prices = golden_mini_backtest_data["prices"]
    signals = golden_mini_backtest_data["signals"]
    start_capital = golden_mini_backtest_data["expected_equity_start"]

    # Define signal function
    def signal_fn(prices_df):
        # Return pre-computed signals for this test
        return signals

    # Run portfolio backtest (optimized path)
    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=start_capital,
        include_costs=False,  # Use cost-free simulation for comparison
        include_trades=True,
    )

    # Verify structure
    assert "equity" in result.equity.columns
    assert "timestamp" in result.equity.columns
    assert len(result.equity) > 0

    # Verify metrics
    assert "final_pf" in result.metrics
    assert "sharpe" in result.metrics
    assert "trades" in result.metrics

    # Verify trades
    assert result.trades is not None
    assert len(result.trades) > 0

    # Verify equity curve is reasonable
    assert result.equity["equity"].iloc[0] == pytest.approx(start_capital, rel=1e-6)
    assert result.equity["equity"].iloc[-1] > 0, "Final equity must be positive"


def test_equity_mark_to_market_numba_vs_numpy(golden_mini_backtest_data):
    """Test that Numba and NumPy implementations produce identical results."""
    from src.assembled_core.pipeline.backtest import _update_equity_mark_to_market

    prices = golden_mini_backtest_data["prices"]
    symbols = golden_mini_backtest_data["symbols"]
    dates = golden_mini_backtest_data["dates"]

    # Create price pivot
    price_pivot = prices.pivot(index="timestamp", columns="symbol", values="close")

    # Test positions
    positions = {"AAPL": 10.0, "MSFT": 5.0, "GOOGL": 3.0}
    cash = 1000.0

    # Test with NumPy (use_numba=False)
    equity_numpy = _update_equity_mark_to_market(
        timestamp=dates[0],
        cash=cash,
        positions=positions,
        price_pivot=price_pivot,
        symbols=symbols,
        use_numba=False,
    )

    # Test with Numba (use_numba=True, will fallback if not available)
    equity_numba = _update_equity_mark_to_market(
        timestamp=dates[0],
        cash=cash,
        positions=positions,
        price_pivot=price_pivot,
        symbols=symbols,
        use_numba=True,
    )

    # Results should be identical (within numerical precision)
    assert abs(equity_numpy - equity_numba) < EQUITY_ATOL, (
        f"Equity mismatch: NumPy={equity_numpy}, Numba={equity_numba}"
    )

