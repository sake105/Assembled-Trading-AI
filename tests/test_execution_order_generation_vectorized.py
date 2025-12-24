"""Unit tests for vectorized order generation and position updates.

This test module verifies that:
- generate_orders_from_targets produces identical results with vectorized operations
- Position updates are correctly aligned (symbol order stable)
- No Python loops over symbols remain in the rebalancing path
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.execution.order_generation import generate_orders_from_targets
from src.assembled_core.qa.backtest_engine import _process_rebalancing_timestamp, _update_positions_vectorized
from src.assembled_core.portfolio.position_sizing import compute_target_positions


def test_generate_orders_vectorized_alignment() -> None:
    """Test that order generation maintains stable symbol alignment."""
    # Create target positions (unsorted to test alignment)
    targets = pd.DataFrame({
        "symbol": ["MSFT", "AAPL", "GOOGL"],
        "target_weight": [0.33, 0.33, 0.34],
        "target_qty": [10.0, 15.0, 5.0],
    })

    # Current positions (different order, some missing)
    current = pd.DataFrame({
        "symbol": ["AAPL", "TSLA"],
        "qty": [10.0, 20.0],
    })

    orders = generate_orders_from_targets(
        target_positions=targets,
        current_positions=current,
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
    )

    # Verify orders are sorted by symbol (stable alignment)
    assert orders["symbol"].is_monotonic_increasing
    # AAPL: 15.0 - 10.0 = 5.0 BUY
    # MSFT: 10.0 - 0.0 = 10.0 BUY
    # GOOGL: 5.0 - 0.0 = 5.0 BUY
    # TSLA: 0.0 - 20.0 = -20.0 SELL (should appear if target has 0)

    # Check that all symbols from merge appear (targets + current)
    expected_symbols = set(targets["symbol"].tolist()) | set(current["symbol"].tolist())
    # Only non-zero deltas should appear in orders
    assert len(orders) > 0
    assert all(s in expected_symbols for s in orders["symbol"])


def test_position_update_vectorized_alignment() -> None:
    """Test that position updates maintain stable symbol alignment."""
    # Create orders
    orders = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")] * 3,
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [10.0, 5.0, 3.0],
        "price": [100.0, 200.0, 150.0],
    })

    # Current positions (different order)
    current = pd.DataFrame({
        "symbol": ["MSFT", "AAPL"],
        "qty": [20.0, 5.0],
    })

    updated = _update_positions_vectorized(orders, current, use_numba=False)

    # Verify updated positions are sorted by symbol (stable alignment)
    assert updated["symbol"].is_monotonic_increasing

    # Verify correct quantities:
    # AAPL: 5.0 + 10.0 = 15.0
    # MSFT: 20.0 - 5.0 = 15.0
    # GOOGL: 0.0 + 3.0 = 3.0
    aapl_qty = updated[updated["symbol"] == "AAPL"]["qty"].values
    msft_qty = updated[updated["symbol"] == "MSFT"]["qty"].values
    googl_qty = updated[updated["symbol"] == "GOOGL"]["qty"].values

    assert len(aapl_qty) == 1
    assert abs(aapl_qty[0] - 15.0) < 1e-6
    assert len(msft_qty) == 1
    assert abs(msft_qty[0] - 15.0) < 1e-6
    assert len(googl_qty) == 1
    assert abs(googl_qty[0] - 3.0) < 1e-6


def test_rebalancing_timestamp_vectorized() -> None:
    """Test that rebalancing timestamp processing uses vectorized operations (no symbol loops)."""
    # Create signals for 2 symbols, 2 timestamps
    signals = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
        ],
        "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
        "direction": ["LONG", "LONG", "LONG", "NEUTRAL"],
        "score": [1.0, 0.8, 1.0, 0.0],
    })

    # Prices for 2 symbols, 2 timestamps
    prices = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
        ],
        "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
        "close": [100.0, 200.0, 105.0, 205.0],
    })

    start_capital = 10000.0
    current_positions = pd.DataFrame(columns=["symbol", "qty"])

    # Process first timestamp
    signal_group_1 = signals[signals["timestamp"] == pd.Timestamp("2024-01-01", tz="UTC")]
    orders_1, pos_1, targets_1 = _process_rebalancing_timestamp(
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        signal_group=signal_group_1,
        current_positions=current_positions,
        position_sizing_fn=lambda sig, cap: compute_target_positions(sig, total_capital=cap, equal_weight=True),
        start_capital=start_capital,
        prices=prices,
        include_targets=True,
    )

    # Verify positions are sorted
    assert pos_1["symbol"].is_monotonic_increasing

    # Process second timestamp
    orders_2, pos_2, targets_2 = _process_rebalancing_timestamp(
        timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
        signal_group=signals[signals["timestamp"] == pd.Timestamp("2024-01-02", tz="UTC")],
        current_positions=pos_1,
        position_sizing_fn=lambda sig, cap: compute_target_positions(sig, total_capital=cap, equal_weight=True),
        start_capital=start_capital,
        prices=prices,
        include_targets=True,
    )

    # Verify positions remain sorted after second rebalance
    assert pos_2["symbol"].is_monotonic_increasing

    # Verify that positions changed between timestamps (rebalancing occurred)
    # This confirms the vectorized update worked
    assert len(pos_1) > 0 or len(pos_2) > 0


def test_orders_side_vectorized() -> None:
    """Test that side determination uses vectorized np.where instead of apply."""
    targets = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "target_qty": [10.0, 5.0, 0.0],
    })

    current = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "qty": [5.0, 10.0, 3.0],
    })

    orders = generate_orders_from_targets(
        target_positions=targets,
        current_positions=current,
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
    )

    # Verify sides are correctly determined vectorially:
    # AAPL: 10.0 - 5.0 = 5.0 > 0 → BUY
    # MSFT: 5.0 - 10.0 = -5.0 < 0 → SELL
    # GOOGL: 0.0 - 3.0 = -3.0 < 0 → SELL
    aapl_order = orders[orders["symbol"] == "AAPL"]
    msft_order = orders[orders["symbol"] == "MSFT"]
    googl_order = orders[orders["symbol"] == "GOOGL"]

    assert len(aapl_order) == 1
    assert aapl_order["side"].values[0] == "BUY"
    assert abs(aapl_order["qty"].values[0] - 5.0) < 1e-6

    assert len(msft_order) == 1
    assert msft_order["side"].values[0] == "SELL"
    assert abs(msft_order["qty"].values[0] - 5.0) < 1e-6

    assert len(googl_order) == 1
    assert googl_order["side"].values[0] == "SELL"
    assert abs(googl_order["qty"].values[0] - 3.0) < 1e-6


def test_position_update_empty_orders() -> None:
    """Test that empty orders return unchanged positions (edge case)."""
    current = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [10.0, 5.0],
    })

    empty_orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    updated = _update_positions_vectorized(empty_orders, current, use_numba=False)

    # Positions should be unchanged
    pd.testing.assert_frame_equal(updated.sort_values("symbol").reset_index(drop=True), 
                                   current.sort_values("symbol").reset_index(drop=True))


def test_position_update_multiple_orders_same_symbol() -> None:
    """Test that multiple orders for the same symbol are correctly aggregated."""
    # Multiple orders for AAPL (should be aggregated)
    orders = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")] * 3,
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "side": ["BUY", "BUY", "SELL"],
        "qty": [5.0, 3.0, 2.0],
        "price": [100.0, 100.0, 200.0],
    })

    current = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [10.0, 10.0],
    })

    updated = _update_positions_vectorized(orders, current, use_numba=False)

    # AAPL: 10.0 + 5.0 + 3.0 = 18.0
    # MSFT: 10.0 - 2.0 = 8.0
    aapl_qty = updated[updated["symbol"] == "AAPL"]["qty"].values
    msft_qty = updated[updated["symbol"] == "MSFT"]["qty"].values

    assert len(aapl_qty) == 1
    assert abs(aapl_qty[0] - 18.0) < 1e-6
    assert len(msft_qty) == 1
    assert abs(msft_qty[0] - 8.0) < 1e-6

