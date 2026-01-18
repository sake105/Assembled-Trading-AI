# tests/test_risk_max_weight_per_symbol.py
"""Tests for max_weight_per_symbol risk limit (Sprint 8 R1).

Tests verify:
1. BUY order is reduced when it would exceed limit
2. SELL order is not blocked when it reduces exposure
3. Deterministic reduction (same inputs → same outputs)
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


def test_buy_order_reduced_when_exceeds_limit() -> None:
    """Test that BUY order is reduced when it would exceed max_weight_per_symbol."""
    # Current: AAPL=50 shares @ 150.0 = 7500 (75% of 10000 equity)
    current_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [50.0],
    })

    # Order: BUY 50 more AAPL → would be 100 shares @ 150.0 = 15000 (150% of equity)
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [50.0],
        "price": [150.0],
    })

    prices_latest = pd.DataFrame({
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    equity = 10000.0
    config = PreTradeConfig(max_weight_per_symbol=0.10)  # 10% limit

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        config=config,
    )

    # Order should be reduced so that target weight = 10%
    # max_target_notional = 0.10 * 10000 = 1000
    # max_target_qty = 1000 / 150 = 6.67
    # current_qty = 50, so order_delta_needed = 6.67 - 50 = -43.33
    # But we're BUYing, so we need to reduce the BUY order
    # Actually: target_qty = 50 + 50 = 100, but limit is 6.67
    # So we need to reduce order so that target_qty = 6.67
    # order_delta_needed = 6.67 - 50 = -43.33
    # Since we're BUYing 50, we need to reduce to: 50 + (-43.33) = 6.67
    # Wait, that's wrong. Let me recalculate:
    # target_qty = current_qty + order_delta
    # We want: target_qty = 6.67, current_qty = 50
    # So: order_delta = 6.67 - 50 = -43.33
    # But order_delta for BUY is +qty, so we need qty = -43.33, which doesn't make sense.
    # Actually, the limit is on weight, not qty. Let me think:
    # We want: abs(target_qty * price / equity) <= max_weight_per_symbol
    # So: abs(target_qty * 150 / 10000) <= 0.10
    # abs(target_qty) <= 0.10 * 10000 / 150 = 6.67
    # So target_qty should be in [-6.67, 6.67]
    # Current qty = 50, so we need to reduce to 6.67
    # order_delta = 6.67 - 50 = -43.33
    # Since we're BUYing, order_delta = +qty, so we need qty = -43.33, which is invalid.
    # Actually, we should SELL instead. But the order is BUY, so we reduce it to 0 or block it.
    # Let me reconsider: if current is 50 and limit is 6.67, we're already over limit.
    # A BUY order would make it worse, so we should block it or reduce to 0.
    # But the requirement says "reduce", so let's reduce to the point where we're at the limit.
    # Since we're already over, we need to SELL to get to the limit.
    # But the order is BUY, so we can't execute it. We should reduce it to 0.
    # Actually, let me check the logic again: if current weight > limit, and order is BUY,
    # then we should block or reduce to 0. But if order is SELL, we should allow it.
    # Let me test with a case where current is below limit, and BUY would exceed it.

    # Revised test: Current: AAPL=5 shares @ 150.0 = 750 (7.5% of equity)
    current_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [5.0],
    })

    # Order: BUY 10 more → would be 15 shares @ 150.0 = 2250 (22.5% of equity, exceeds 10% limit)
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [10.0],
        "price": [150.0],
    })

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        config=config,
    )

    # Order should be reduced so that target weight = 10%
    # max_target_notional = 0.10 * 10000 = 1000
    # max_target_qty = 1000 / 150 = 6.67
    # current_qty = 5, so order_delta_needed = 6.67 - 5 = 1.67
    # Since we're BUYing, we reduce qty from 10 to 1.67
    assert len(filtered) == 1, "Order should not be blocked"
    assert filtered["qty"].iloc[0] < 10.0, "Order qty should be reduced"
    assert abs(filtered["qty"].iloc[0] - 1.67) < 0.1, "Order qty should be reduced to ~1.67"

    # Check that reduction reason is recorded
    assert len(result.reduced_orders) > 0, "Should have reduction reason"
    assert any(
        r["reason"] == "RISK_REDUCE_MAX_WEIGHT_PER_SYMBOL" for r in result.reduced_orders
    ), "Should have RISK_REDUCE_MAX_WEIGHT_PER_SYMBOL reason"


def test_sell_order_not_blocked_when_reduces_exposure() -> None:
    """Test that SELL order is not blocked when it reduces exposure (even if overweight)."""
    # Current: AAPL=100 shares @ 150.0 = 15000 (150% of equity, over limit)
    current_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })

    # Order: SELL 50 AAPL → would be 50 shares @ 150.0 = 7500 (75% of equity, still over but reduced)
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["SELL"],
        "qty": [50.0],
        "price": [150.0],
    })

    prices_latest = pd.DataFrame({
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    equity = 10000.0
    config = PreTradeConfig(max_weight_per_symbol=0.10)  # 10% limit

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        config=config,
    )

    # SELL order should not be blocked (it reduces exposure)
    # Even though result is still over limit, the order reduces it, so allow it
    assert len(filtered) == 1, "SELL order should not be blocked"
    assert filtered["qty"].iloc[0] == 50.0, "SELL order qty should be unchanged"


def test_deterministic_reduction() -> None:
    """Test that reduction is deterministic (same inputs → same outputs)."""
    current_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [5.0],
    })

    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [10.0],
        "price": [150.0],
    })

    prices_latest = pd.DataFrame({
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    equity = 10000.0
    config = PreTradeConfig(max_weight_per_symbol=0.10)

    # Run twice
    result1, filtered1 = run_pre_trade_checks(
        orders,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        config=config,
    )

    result2, filtered2 = run_pre_trade_checks(
        orders,
        current_positions=current_positions,
        prices_latest=prices_latest,
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


def test_equity_zero_raises_error() -> None:
    """Test that equity <= 0 raises clear error message."""
    current_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [5.0],
    })

    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [10.0],
        "price": [150.0],
    })

    prices_latest = pd.DataFrame({
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    config = PreTradeConfig(max_weight_per_symbol=0.10)

    # Equity = 0 should raise error
    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=0.0,
        config=config,
    )

    # Should have blocked reason
    assert not result.is_ok, "Should fail with equity = 0"
    assert any("equity" in reason.lower() for reason in result.blocked_reasons), "Should mention equity in error"


def test_missing_price_raises_error() -> None:
    """Test that missing price raises clear error message."""
    current_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [5.0],
    })

    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [10.0],
        "price": [150.0],
    })

    # Missing price for AAPL
    prices_latest = pd.DataFrame({
        "symbol": ["MSFT"],
        "close": [200.0],
    })

    equity = 10000.0
    config = PreTradeConfig(max_weight_per_symbol=0.10)

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        config=config,
    )

    # Should have blocked reason
    assert not result.is_ok, "Should fail with missing price"
    assert any("price" in reason.lower() for reason in result.blocked_reasons), "Should mention price in error"
