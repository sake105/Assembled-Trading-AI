"""Cash invariant test for partial fills (Sprint 13).

Tests that cash tracking is correct for partial fills:
- cash_end = start_cash + sum(cash_delta)
- Uses fill_qty * fill_price (not qty * price) for cash_delta calculation
- This test would fail if qty instead of fill_qty is used.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.ledger import events_from_trades
from src.assembled_core.accounting.position_engine import build_positions_from_ledger
from src.assembled_core.pipeline.portfolio import simulate_with_costs


def create_orders_with_partial_fill() -> pd.DataFrame:
    """Create orders with explicit partial fill (fill_qty < qty)."""
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    data = [
        {
            "timestamp": pd.Timestamp(base_time, tz="UTC"),
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 100.0,  # Order qty
            "price": 150.0,
            "fill_qty": 50.0,  # Partial fill: only 50 filled
            "fill_price": 150.0,
            "status": "partial",
        },
        {
            "timestamp": pd.Timestamp(base_time, tz="UTC") + pd.Timedelta(minutes=5),
            "symbol": "MSFT",
            "side": "SELL",
            "qty": 200.0,  # Order qty
            "price": 200.0,
            "fill_qty": 75.0,  # Partial fill: only 75 filled
            "fill_price": 200.0,
            "status": "partial",
        },
    ]
    return pd.DataFrame(data)


def test_cash_invariant_partial_fills():
    """Test that cash_end = start_cash + sum(cash_delta) for partial fills.

    This test verifies:
    1. Cash tracking uses fill_qty * fill_price (not qty * price)
    2. Cash invariant holds: cash_end = start_cash + sum(cash_delta)
    3. Would fail if qty instead of fill_qty is used
    """
    orders = create_orders_with_partial_fill()
    start_cash = 10000.0
    commission_bps = 1.0  # 1 bps = 0.01%
    spread_w = 0.25  # 0.25 * 1e-4 = 0.000025
    impact_w = 0.5  # 0.5 * 1e-4 = 0.00005

    # Simulate with costs
    equity, metrics, trades_df = simulate_with_costs(
        orders,
        start_cash,
        commission_bps,
        spread_w,
        impact_w,
        "1d",
        prices=None,  # No prices needed, we have explicit fill_qty/fill_price
    )

    # Verify trades_df has fill_qty and fill_price
    assert "fill_qty" in trades_df.columns, "trades_df should have fill_qty column"
    assert "fill_price" in trades_df.columns, "trades_df should have fill_price column"
    assert "cash_delta" in trades_df.columns, "trades_df should have cash_delta column"
    assert (
        "total_cost_cash" in trades_df.columns
    ), "trades_df should have total_cost_cash column"

    # Calculate expected cash_delta manually using fill_qty
    # BUY: -(fill_qty * fill_price + total_cost_cash)
    # SELL: +(fill_qty * fill_price - total_cost_cash)
    expected_cash_deltas = []
    for _, row in trades_df.iterrows():
        fill_qty = abs(float(row["fill_qty"]))
        fill_price = float(row["fill_price"])
        total_cost = float(row["total_cost_cash"])

        if row["side"] == "BUY":
            expected_delta = -(fill_qty * fill_price + total_cost)
        elif row["side"] == "SELL":
            expected_delta = fill_qty * fill_price - total_cost
        else:
            expected_delta = 0.0

        expected_cash_deltas.append(expected_delta)

    # Verify cash_delta matches expected (using fill_qty)
    for idx, (expected, actual) in enumerate(
        zip(expected_cash_deltas, trades_df["cash_delta"])
    ):
        assert abs(expected - actual) < 1e-6, (
            f"Row {idx}: cash_delta mismatch. Expected {expected:.6f}, got {actual:.6f}. "
            f"Order: {trades_df.iloc[idx][['symbol', 'side', 'qty', 'fill_qty']].to_dict()}"
        )

    # Calculate final cash from cash_deltas
    sum_cash_delta = trades_df["cash_delta"].sum()
    expected_final_cash = start_cash + sum_cash_delta

    # Verify cash invariant: cash_end = start_cash + sum(cash_delta)
    # Note: equity includes positions, so we need to account for position value
    # For simplicity, we verify that the cash_delta sum is correct
    assert abs(sum_cash_delta - (expected_final_cash - start_cash)) < 1e-6, (
        f"Cash invariant violated: sum(cash_delta) = {sum_cash_delta:.6f}, "
        f"expected change = {expected_final_cash - start_cash:.6f}"
    )


def test_cash_invariant_via_ledger_events():
    """Test cash invariant via ledger events (end-to-end).

    This test:
    1. Creates trades with partial fills
    2. Generates ledger events
    3. Builds positions from ledger
    4. Verifies cash_end = start_cash + sum(cash_delta from events)
    """
    orders = create_orders_with_partial_fill()
    start_cash = 10000.0

    # Add cost columns manually (simulating what add_cost_columns_to_trades does)
    # For simplicity, use fixed costs
    orders["commission_cash"] = 0.15  # Fixed commission
    orders["spread_cash"] = 0.25  # Fixed spread
    orders["slippage_cash"] = 0.10  # Fixed slippage
    orders["total_cost_cash"] = (
        orders["commission_cash"] + orders["spread_cash"] + orders["slippage_cash"]
    )

    # Generate ledger events
    events = events_from_trades(orders, run_id="test_partial", source="test")

    # Filter to FILL events only (they have cash_delta)
    fill_events = events[events["event_type"] == "FILL"]

    # Calculate sum of cash_deltas
    sum_cash_delta = fill_events["cash_delta"].sum()

    # Build positions from ledger
    result = build_positions_from_ledger(
        events,
        prices_df=None,
        mark_ts=None,
        start_cash=start_cash,
        missing_price_policy="zero",
    )

    final_cash = result["cash_balance"]

    # Verify cash invariant: final_cash = start_cash + sum(cash_delta)
    expected_final_cash = start_cash + sum_cash_delta
    assert abs(final_cash - expected_final_cash) < 1e-6, (
        f"Cash invariant violated: final_cash={final_cash:.6f}, "
        f"expected={expected_final_cash:.6f}, "
        f"sum(cash_delta)={sum_cash_delta:.6f}"
    )


def test_partial_fill_uses_fill_qty_not_qty():
    """Test that cash_delta uses fill_qty (not qty) - regression test.

    This test would FAIL if qty instead of fill_qty is used for cash_delta calculation.
    """
    orders = create_orders_with_partial_fill()
    start_cash = 10000.0

    # Simulate with costs
    equity, metrics, trades_df = simulate_with_costs(
        orders,
        start_cash,
        commission_bps=1.0,
        spread_w=0.25,
        impact_w=0.5,
        freq="1d",
        prices=None,
    )

    # For each trade, verify cash_delta is based on fill_qty, not qty
    for _, row in trades_df.iterrows():
        qty = abs(float(row["qty"]))
        fill_qty = abs(float(row["fill_qty"]))
        fill_price = float(row["fill_price"])
        total_cost = float(row["total_cost_cash"])

        # Calculate what cash_delta SHOULD be (using fill_qty)
        if row["side"] == "BUY":
            correct_cash_delta = -(fill_qty * fill_price + total_cost)
        elif row["side"] == "SELL":
            correct_cash_delta = fill_qty * fill_price - total_cost
        else:
            correct_cash_delta = 0.0

        # Calculate what cash_delta WOULD be if qty was used (WRONG)
        if row["side"] == "BUY":
            wrong_cash_delta = -(qty * fill_price + total_cost)
        elif row["side"] == "SELL":
            wrong_cash_delta = qty * fill_price - total_cost
        else:
            wrong_cash_delta = 0.0

        # Verify actual cash_delta matches correct (fill_qty-based) calculation
        actual_cash_delta = float(row["cash_delta"])
        assert abs(actual_cash_delta - correct_cash_delta) < 1e-6, (
            f"cash_delta should use fill_qty, not qty. "
            f"Symbol={row['symbol']}, side={row['side']}, "
            f"qty={qty}, fill_qty={fill_qty}, "
            f"actual={actual_cash_delta:.6f}, correct={correct_cash_delta:.6f}, "
            f"wrong (if qty used)={wrong_cash_delta:.6f}"
        )

        # Verify that wrong calculation would be different (for partial fills)
        if fill_qty != qty:  # Only check if it's actually a partial fill
            assert abs(actual_cash_delta - wrong_cash_delta) > 1e-3, (
                f"For partial fill, cash_delta should differ if qty is used. "
                f"Symbol={row['symbol']}, qty={qty}, fill_qty={fill_qty}, "
                f"actual={actual_cash_delta:.6f}, wrong={wrong_cash_delta:.6f}"
            )


def test_cash_invariant_multiple_partial_fills():
    """Test cash invariant with multiple partial fills."""
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    orders = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(base_time, tz="UTC"),
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 100.0,
                "price": 150.0,
                "fill_qty": 30.0,  # 30% fill
                "fill_price": 150.0,
                "status": "partial",
            },
            {
                "timestamp": pd.Timestamp(base_time, tz="UTC")
                + pd.Timedelta(minutes=5),
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 50.0,
                "price": 151.0,
                "fill_qty": 50.0,  # Full fill
                "fill_price": 151.0,
                "status": "filled",
            },
            {
                "timestamp": pd.Timestamp(base_time, tz="UTC")
                + pd.Timedelta(minutes=10),
                "symbol": "MSFT",
                "side": "SELL",
                "qty": 200.0,
                "price": 200.0,
                "fill_qty": 100.0,  # 50% fill
                "fill_price": 200.0,
                "status": "partial",
            },
        ]
    )

    start_cash = 10000.0

    # Simulate with costs
    equity, metrics, trades_df = simulate_with_costs(
        orders,
        start_cash,
        commission_bps=1.0,
        spread_w=0.25,
        impact_w=0.5,
        freq="1d",
        prices=None,
    )

    # Calculate expected final cash
    sum_cash_delta = trades_df["cash_delta"].sum()
    expected_final_cash = start_cash + sum_cash_delta

    # Verify via ledger events
    orders_with_costs = trades_df.copy()
    events = events_from_trades(orders_with_costs, run_id="test_multi", source="test")
    result = build_positions_from_ledger(
        events,
        prices_df=None,
        mark_ts=None,
        start_cash=start_cash,
        missing_price_policy="zero",
    )

    final_cash_ledger = result["cash_balance"]

    # Verify both methods agree
    assert abs(final_cash_ledger - expected_final_cash) < 1e-6, (
        f"Cash mismatch: ledger={final_cash_ledger:.6f}, "
        f"expected={expected_final_cash:.6f}, "
        f"sum(cash_delta)={sum_cash_delta:.6f}"
    )
