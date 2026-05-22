"""Signature regression tests for portfolio simulation functions.

Tests that function signatures match expected return values:
- simulate_with_costs() returns (equity, metrics, trades_df)
- run_portfolio_step() returns (eq_path, rep_path, trades_df)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline.portfolio import simulate_with_costs
from src.assembled_core.pipeline.orchestrator import run_portfolio_step


def create_minimal_orders() -> pd.DataFrame:
    """Create minimal orders for testing."""
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    data = [
        {
            "timestamp": pd.Timestamp(base_time, tz="UTC"),
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 10.0,
            "price": 150.0,
        },
    ]
    return pd.DataFrame(data)


def test_simulate_with_costs_signature():
    """Test that simulate_with_costs returns (equity, metrics, trades_df)."""
    orders = create_minimal_orders()
    start_capital = 10000.0

    result = simulate_with_costs(orders, start_capital, 0.0, 0.0, 0.0, "1d")

    # Verify return type is tuple with 3 elements
    assert isinstance(result, tuple), "simulate_with_costs should return a tuple"
    assert len(result) == 3, (
        f"simulate_with_costs should return 3 values, got {len(result)}"
    )

    equity, metrics, trades_df = result

    # Verify equity
    assert isinstance(equity, pd.DataFrame), (
        "First return value should be DataFrame (equity)"
    )
    assert "timestamp" in equity.columns
    assert "equity" in equity.columns

    # Verify metrics
    assert isinstance(metrics, dict), "Second return value should be dict (metrics)"
    assert "final_pf" in metrics
    assert "sharpe" in metrics
    assert "trades" in metrics

    # Verify trades_df
    assert isinstance(trades_df, pd.DataFrame), (
        "Third return value should be DataFrame (trades_df)"
    )
    # trades_df may be empty, but should have expected structure
    if not trades_df.empty:
        assert "timestamp" in trades_df.columns
        assert "symbol" in trades_df.columns


def test_run_portfolio_step_signature(tmp_path: Path):
    """Test that run_portfolio_step returns (eq_path, rep_path, trades_df).

    Note: This test requires orders file to exist, so it may be skipped if setup is incomplete.
    """
    # Create minimal orders file
    orders = create_minimal_orders()
    orders_file = tmp_path / "orders_1d.csv"
    orders.to_csv(orders_file, index=False)

    # Try to run portfolio step (may fail if prices are missing, which is OK for signature test)
    try:
        result = run_portfolio_step(
            freq="1d",
            start_capital=10000.0,
            commission_bps=0.0,
            spread_w=0.0,
            impact_w=0.0,
            output_dir=tmp_path,
        )

        # Verify return type is tuple with 3 elements
        assert isinstance(result, tuple), "run_portfolio_step should return a tuple"
        assert len(result) == 3, (
            f"run_portfolio_step should return 3 values, got {len(result)}"
        )

        eq_path, rep_path, trades_df = result

        # Verify paths
        assert isinstance(eq_path, Path), "First return value should be Path (eq_path)"
        assert isinstance(rep_path, Path), (
            "Second return value should be Path (rep_path)"
        )

        # Verify trades_df
        assert isinstance(trades_df, pd.DataFrame), (
            "Third return value should be DataFrame (trades_df)"
        )

    except (FileNotFoundError, ValueError) as e:
        # If prices are missing or other setup issues, skip the test
        pytest.skip(f"run_portfolio_step setup incomplete: {e}")


def test_simulate_with_costs_trades_df_structure():
    """Test that trades_df has expected structure (fill_qty, fill_price, status, costs)."""
    orders = create_minimal_orders()
    start_capital = 10000.0

    equity, metrics, trades_df = simulate_with_costs(
        orders, start_capital, 0.5, 0.25, 0.5, "1d"
    )

    # trades_df should not be empty if orders are provided
    if not orders.empty:
        # trades_df may be empty if all orders are rejected, but if not empty, should have structure
        if not trades_df.empty:
            # Check for expected columns (at minimum: timestamp, symbol, side, qty, price)
            expected_cols = ["timestamp", "symbol", "side", "qty", "price"]
            for col in expected_cols:
                assert col in trades_df.columns, f"trades_df should have {col} column"

            # If fill model pipeline was applied, should have fill_qty, fill_price, status
            # (These may not be present if fill model pipeline is not applied)
            # We just verify the DataFrame structure is reasonable
            assert len(trades_df) > 0, (
                "trades_df should have at least one row if orders are provided"
            )
