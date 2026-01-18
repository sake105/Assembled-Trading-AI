# tests/test_risk_drawdown_derisk.py
"""Tests for drawdown de-risking policy (Sprint 8 R3).

Tests verify:
1. Orders unchanged when drawdown < threshold
2. Orders scaled when drawdown >= threshold
3. Deterministic rounding
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


def test_drawdown_below_threshold_unchanged() -> None:
    """Test that orders are unchanged when drawdown < threshold."""
    # Current equity = 9000, peak equity = 10000
    # Drawdown = 1 - 9000/10000 = 0.1 (10%)
    # Threshold = 0.2 (20%)
    # Drawdown < threshold, so orders should be unchanged
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
    })

    current_equity = 9000.0
    peak_equity = 10000.0
    config = PreTradeConfig(drawdown_threshold=0.2, de_risk_scale=0.25)  # 20% threshold, 25% scale

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_equity=current_equity,
        peak_equity=peak_equity,
        config=config,
    )

    # Order should be unchanged
    assert len(filtered) == 1, "Order should remain"
    assert abs(filtered["qty"].iloc[0] - 100.0) < 1e-10, "Qty should be unchanged (100.0)"
    assert len(result.reduced_orders) == 0, "Should have no reduction reasons"


def test_drawdown_at_threshold_scaled() -> None:
    """Test that orders are scaled when drawdown exactly equals threshold."""
    # Current equity = 7999, peak equity = 10000
    # Drawdown = 1 - 7999/10000 = 0.2001 (20.01%)
    # Threshold = 0.2 (20%)
    # Drawdown > threshold, so scale should be applied
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
    })

    current_equity = 7999.0
    peak_equity = 10000.0
    config = PreTradeConfig(drawdown_threshold=0.2, de_risk_scale=0.25)  # 20% threshold, 25% scale

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_equity=current_equity,
        peak_equity=peak_equity,
        config=config,
    )

    # Order should be scaled (drawdown > threshold, so >= is true)
    assert len(filtered) == 1, "Order should remain"
    # 100 * 0.25 = 25, rounded to 25
    assert abs(filtered["qty"].iloc[0] - 25.0) < 1e-10, "Qty should be scaled to 25.0"
    assert len(result.reduced_orders) == 1, "Should have 1 reduction reason"
    assert result.reduced_orders[0]["reason"] == "RISK_DERISK_DRAWDOWN", "Should have RISK_DERISK_DRAWDOWN reason"


def test_drawdown_above_threshold_scaled() -> None:
    """Test that orders are scaled when drawdown > threshold."""
    # Current equity = 7000, peak equity = 10000
    # Drawdown = 1 - 7000/10000 = 0.3 (30%)
    # Threshold = 0.2 (20%)
    # Drawdown > threshold, so orders should be scaled by de_risk_scale = 0.25
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
    })

    current_equity = 7000.0
    peak_equity = 10000.0
    config = PreTradeConfig(drawdown_threshold=0.2, de_risk_scale=0.25)  # 20% threshold, 25% scale

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_equity=current_equity,
        peak_equity=peak_equity,
        config=config,
    )

    # Order should be scaled: 100 * 0.25 = 25
    assert len(filtered) == 1, "Order should remain"
    assert abs(filtered["qty"].iloc[0] - 25.0) < 1e-10, "Qty should be scaled to 25.0"

    # Check reduction reason
    assert len(result.reduced_orders) == 1, "Should have 1 reduction reason"
    assert result.reduced_orders[0]["reason"] == "RISK_DERISK_DRAWDOWN", "Should have RISK_DERISK_DRAWDOWN reason"
    assert abs(result.reduced_orders[0]["explain"]["drawdown"] - 0.3) < 1e-10, "Drawdown should be 0.3"
    assert abs(result.reduced_orders[0]["explain"]["threshold"] - 0.2) < 1e-10, "Threshold should be 0.2"
    assert abs(result.reduced_orders[0]["explain"]["de_risk_scale"] - 0.25) < 1e-10, "De-risk scale should be 0.25"


def test_drawdown_full_block() -> None:
    """Test that orders are fully blocked when de_risk_scale = 0.0."""
    # Current equity = 7000, peak equity = 10000
    # Drawdown = 1 - 7000/10000 = 0.3 (30%)
    # Threshold = 0.2 (20%)
    # Drawdown > threshold, de_risk_scale = 0.0, so all orders should be blocked
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
    })

    current_equity = 7000.0
    peak_equity = 10000.0
    config = PreTradeConfig(drawdown_threshold=0.2, de_risk_scale=0.0)  # 20% threshold, full block

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_equity=current_equity,
        peak_equity=peak_equity,
        config=config,
    )

    # Order should be dropped (qty becomes 0 after scaling)
    assert len(filtered) == 0, "Order should be dropped"
    assert len(result.reduced_orders) == 1, "Should have 1 reduction reason (dropped)"
    assert result.reduced_orders[0]["new_qty"] == 0.0, "new_qty should be 0.0 (dropped)"


def test_deterministic_rounding() -> None:
    """Test that rounding is deterministic (same inputs → same outputs)."""
    # Current equity = 7000, peak equity = 10000
    # Drawdown = 0.3, threshold = 0.2, scale = 0.25
    # Order: 33 shares → 33 * 0.25 = 8.25 → rounded to 8
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [33.0],
        "price": [150.0],
    })

    current_equity = 7000.0
    peak_equity = 10000.0
    config = PreTradeConfig(drawdown_threshold=0.2, de_risk_scale=0.25)

    # Run twice
    result1, filtered1 = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_equity=current_equity,
        peak_equity=peak_equity,
        config=config,
    )

    result2, filtered2 = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_equity=current_equity,
        peak_equity=peak_equity,
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

    # Verify rounding: 33 * 0.25 = 8.25 → 8
    expected_qty = int(33.0 * 0.25)
    actual_qty = filtered1["qty"].iloc[0]
    assert abs(actual_qty - expected_qty) < 1e-10, f"Qty should be rounded to {expected_qty}, got {actual_qty}"


def test_negative_qty_handled_correctly() -> None:
    """Test that SELL orders (negative qty) are handled correctly."""
    # Current equity = 7000, peak equity = 10000
    # Drawdown = 0.3, threshold = 0.2, scale = 0.25
    # SELL order: 100 shares → 100 * 0.25 = 25 → rounded to 25, then negate → -25
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["SELL"],
        "qty": [100.0],
        "price": [150.0],
    })

    current_equity = 7000.0
    peak_equity = 10000.0
    config = PreTradeConfig(drawdown_threshold=0.2, de_risk_scale=0.25)

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_equity=current_equity,
        peak_equity=peak_equity,
        config=config,
    )

    # Order should be scaled (negative qty preserved)
    assert len(filtered) == 1, "Order should remain"
    # 100 * 0.25 = 25, then negate for SELL → -25
    expected_qty = -int(100.0 * 0.25)  # Should be -25
    actual_qty = filtered["qty"].iloc[0]
    assert actual_qty < 0.0, "Qty should be negative (SELL)"
    assert abs(actual_qty - expected_qty) < 1e-10, f"Qty should be {expected_qty}, got {actual_qty}"


def test_missing_equity_skips_check() -> None:
    """Test that check is skipped when equity data is missing."""
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
    })

    config = PreTradeConfig(drawdown_threshold=0.2, de_risk_scale=0.25)

    # Missing current_equity
    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_equity=None,
        peak_equity=10000.0,
        config=config,
    )

    # Order should be unchanged (check skipped)
    assert len(filtered) == 1, "Order should remain unchanged"
    assert "drawdown_derisk_check" in result.summary, "Should have skip reason in summary"
    assert result.summary["drawdown_derisk_check"] == "skipped_no_current_equity", "Should skip due to missing current_equity"

    # Missing peak_equity
    result2, filtered2 = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_equity=9000.0,
        peak_equity=None,
        config=config,
    )

    # Order should be unchanged (check skipped)
    assert len(filtered2) == 1, "Order should remain unchanged"
    assert result2.summary["drawdown_derisk_check"] == "skipped_no_peak_equity", "Should skip due to missing peak_equity"
