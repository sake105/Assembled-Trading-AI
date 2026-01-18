# tests/test_risk_turnover_cap.py
"""Tests for turnover cap risk limit (Sprint 8 R2).

Tests verify:
1. Orders are proportionally reduced when turnover > cap
2. Deterministic rounding (no flakiness)
3. Orders unchanged when turnover <= cap
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.pre_trade_checks import (
    PreTradeConfig,
    run_pre_trade_checks,
)


def test_two_orders_proportionally_reduced_when_turnover_exceeds_cap() -> None:
    """Test that 2 orders are proportionally reduced when turnover > cap."""
    # Orders: BUY 100 AAPL @ 150, BUY 50 MSFT @ 200
    # Total notional = 100*150 + 50*200 = 15000 + 10000 = 25000
    # Equity = 10000
    # Turnover = 25000 / 10000 = 2.5 (250%)
    # Cap = 0.5 (50%)
    # Scale factor = 0.5 / 2.5 = 0.2
    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "BUY"],
        "qty": [100.0, 50.0],
        "price": [150.0, 200.0],
    })

    equity = 10000.0
    config = PreTradeConfig(turnover_cap=0.5)  # 50% cap

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        equity=equity,
        config=config,
    )

    # Both orders should be reduced by scale_factor = 0.2
    # AAPL: 100 * 0.2 = 20 (rounded: 20)
    # MSFT: 50 * 0.2 = 10 (rounded: 10)
    assert len(filtered) == 2, "Both orders should remain (reduced, not dropped)"
    
    aapl_row = filtered[filtered["symbol"] == "AAPL"].iloc[0]
    msft_row = filtered[filtered["symbol"] == "MSFT"].iloc[0]

    assert abs(aapl_row["qty"] - 20.0) < 1e-10, "AAPL qty should be reduced to 20"
    assert abs(msft_row["qty"] - 10.0) < 1e-10, "MSFT qty should be reduced to 10"

    # Check reduction reasons
    assert len(result.reduced_orders) == 2, "Should have 2 reduction reasons"
    assert all(
        r["reason"] == "RISK_REDUCE_TURNOVER_CAP" for r in result.reduced_orders
    ), "Should have RISK_REDUCE_TURNOVER_CAP reason"

    # Verify explain dict
    for r in result.reduced_orders:
        assert "total_turnover" in r["explain"], "explain should have total_turnover"
        assert "cap" in r["explain"], "explain should have cap"
        assert "scale_factor" in r["explain"], "explain should have scale_factor"
        assert abs(r["explain"]["total_turnover"] - 2.5) < 1e-10, "total_turnover should be 2.5"
        assert abs(r["explain"]["cap"] - 0.5) < 1e-10, "cap should be 0.5"
        assert abs(r["explain"]["scale_factor"] - 0.2) < 1e-10, "scale_factor should be 0.2"


def test_deterministic_rounding() -> None:
    """Test that rounding is deterministic (same inputs → same outputs)."""
    # Orders with fractional qty after scaling
    # BUY 33 AAPL @ 150 → notional = 4950
    # Equity = 10000, turnover = 0.495, cap = 0.3
    # Scale factor = 0.3 / 0.495 ≈ 0.606
    # New qty = 33 * 0.606 ≈ 20.0 (should round to 20)
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [33.0],
        "price": [150.0],
    })

    equity = 10000.0
    config = PreTradeConfig(turnover_cap=0.3)  # 30% cap

    # Run twice
    result1, filtered1 = run_pre_trade_checks(
        orders,
        portfolio=None,
        equity=equity,
        config=config,
    )

    result2, filtered2 = run_pre_trade_checks(
        orders,
        portfolio=None,
        equity=equity,
        config=config,
    )

    # Results should be identical
    pd.testing.assert_frame_equal(
        filtered1.sort_values("symbol").reset_index(drop=True),
        filtered2.sort_values("symbol").reset_index(drop=True),
        check_dtype=False,
    )
    assert (
        filtered1.sort_values("symbol").reset_index(drop=True).equals(
            filtered2.sort_values("symbol").reset_index(drop=True)
        )
    ), "Reduction should be deterministic"

    # Reduction reasons should be identical
    assert result1.reduced_orders == result2.reduced_orders, "Reduction reasons should be identical"

    # Verify rounding: 33 * (0.3 / 0.495) ≈ 20.0
    expected_qty = int(33.0 * (0.3 / 0.495))  # Floor rounding
    actual_qty = filtered1["qty"].iloc[0]
    assert abs(actual_qty - expected_qty) < 1e-10, f"Qty should be rounded to {expected_qty}, got {actual_qty}"


def test_turnover_below_cap_unchanged() -> None:
    """Test that orders are unchanged when turnover <= cap."""
    # Orders: BUY 10 AAPL @ 150
    # Total notional = 10 * 150 = 1500
    # Equity = 10000
    # Turnover = 1500 / 10000 = 0.15 (15%)
    # Cap = 0.5 (50%)
    # Turnover < cap, so orders should be unchanged
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [10.0],
        "price": [150.0],
    })

    equity = 10000.0
    config = PreTradeConfig(turnover_cap=0.5)  # 50% cap

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        equity=equity,
        config=config,
    )

    # Order should be unchanged
    assert len(filtered) == 1, "Order should remain"
    assert abs(filtered["qty"].iloc[0] - 10.0) < 1e-10, "Qty should be unchanged (10.0)"
    assert len(result.reduced_orders) == 0, "Should have no reduction reasons"


def test_turnover_exactly_at_cap_unchanged() -> None:
    """Test that orders are unchanged when turnover exactly equals cap."""
    # Orders: BUY 50 AAPL @ 150
    # Total notional = 50 * 150 = 7500
    # Equity = 10000
    # Turnover = 7500 / 10000 = 0.75 (75%)
    # Cap = 0.75 (75%)
    # Turnover == cap, so orders should be unchanged
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [50.0],
        "price": [150.0],
    })

    equity = 10000.0
    config = PreTradeConfig(turnover_cap=0.75)  # 75% cap

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        equity=equity,
        config=config,
    )

    # Order should be unchanged
    assert len(filtered) == 1, "Order should remain"
    assert abs(filtered["qty"].iloc[0] - 50.0) < 1e-10, "Qty should be unchanged (50.0)"
    assert len(result.reduced_orders) == 0, "Should have no reduction reasons"


def test_small_order_dropped_after_reduction() -> None:
    """Test that orders with qty <= 0 after reduction are dropped."""
    # Orders: BUY 1 AAPL @ 150
    # Total notional = 1 * 150 = 150
    # Equity = 10000
    # Turnover = 150 / 10000 = 0.015 (1.5%)
    # Cap = 0.001 (0.1%)
    # Scale factor = 0.001 / 0.015 ≈ 0.0667
    # New qty = 1 * 0.0667 ≈ 0.067 → rounded to 0 → dropped
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [1.0],
        "price": [150.0],
    })

    equity = 10000.0
    config = PreTradeConfig(turnover_cap=0.001)  # 0.1% cap

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        equity=equity,
        config=config,
    )

    # Order should be dropped (qty becomes 0 after rounding)
    assert len(filtered) == 0, "Order should be dropped"
    assert len(result.reduced_orders) == 1, "Should have 1 reduction reason (dropped)"
    assert result.reduced_orders[0]["new_qty"] == 0.0, "new_qty should be 0.0 (dropped)"


def test_negative_qty_handled_correctly() -> None:
    """Test that SELL orders (negative qty) are handled correctly."""
    # Orders: SELL 100 AAPL @ 150
    # Total notional = abs(-100) * 150 = 15000
    # Equity = 10000
    # Turnover = 15000 / 10000 = 1.5 (150%)
    # Cap = 0.5 (50%)
    # Scale factor = 0.5 / 1.5 ≈ 0.333
    # New qty = -100 * 0.333 ≈ -33.33 → rounded to -33
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["SELL"],
        "qty": [100.0],
        "price": [150.0],
    })

    equity = 10000.0
    config = PreTradeConfig(turnover_cap=0.5)  # 50% cap

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        equity=equity,
        config=config,
    )

    # Order should be reduced (negative qty preserved)
    assert len(filtered) == 1, "Order should remain"
    # Scale factor = 0.5 / 1.5 = 0.333...
    # New qty = 100 * 0.333... = 33.33...
    # int(33.33) = 33, then negate for SELL → -33
    expected_qty = -int(100.0 * (0.5 / 1.5))  # Should be -33
    actual_qty = filtered["qty"].iloc[0]
    assert actual_qty < 0.0, "Qty should be negative (SELL)"
    assert abs(actual_qty - expected_qty) < 1e-10, f"Qty should be {expected_qty}, got {actual_qty}"
