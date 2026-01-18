# tests/test_fill_model_costs_consistency.py
"""Tests for cost consistency with fill model (Sprint 7 / C5).

Tests verify:
1. total_cost_cash == commission+spread+slippage always
2. rejected fills have zero costs
3. partial fills costs scale with filled notional
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.fill_model import PartialFillModel
from src.assembled_core.execution.fill_model_pipeline import apply_fill_model_pipeline
from src.assembled_core.execution.transaction_costs import (
    CommissionModel,
    SpreadModel,
    SlippageModel,
    add_cost_columns_to_trades,
)


def test_total_cost_equals_sum_of_components() -> None:
    """Test that total_cost_cash always equals sum of components."""
    # Create trades with full fills
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "side": ["BUY", "SELL", "BUY", "SELL", "BUY"],
        "qty": [100.0, 50.0, 75.0, 25.0, 200.0],
        "price": [150.0, 152.0, 148.0, 150.0, 155.0],
    })

    # Create prices
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "close": [150.0, 152.0, 148.0, 150.0, 155.0],
        "volume": [1e6] * 5,
    })

    # Apply fill model pipeline (skip session gate if exchange_calendars missing)
    fills = apply_fill_model_pipeline(
        trades,
        prices=prices,
        freq="1d",
        partial_fill_model=None,
        strict_session_gate=False,  # Permissive fallback if exchange_calendars missing
    )

    # Add cost columns
    commission_model = CommissionModel(mode="bps", commission_bps=1.0, fixed_per_trade=0.0)
    spread_model = SpreadModel(adv_window=20, buckets=None, fallback_spread_bps=5.0)
    slippage_model = SlippageModel(vol_window=20, k=1.0, min_bps=0.0, max_bps=50.0, fallback_slippage_bps=10.0)
    
    fills_with_costs = add_cost_columns_to_trades(
        fills,
        commission_model=commission_model,
        spread_model=spread_model,
        slippage_model=slippage_model,
        prices=prices,
    )

    # Check: total_cost_cash == commission_cash + spread_cash + slippage_cash
    computed_total = (
        fills_with_costs["commission_cash"]
        + fills_with_costs["spread_cash"]
        + fills_with_costs["slippage_cash"]
    )
    pd.testing.assert_series_equal(
        fills_with_costs["total_cost_cash"],
        computed_total,
        check_names=False,
        rtol=1e-10,
    )
    # Verify that total_cost_cash equals sum of components
    assert (fills_with_costs["total_cost_cash"] == computed_total).all(), "total_cost_cash should equal sum of components"


def test_rejected_fills_have_zero_costs() -> None:
    """Test that rejected fills have zero costs."""
    # Create trades on weekend (will be rejected by session gate)
    weekend_timestamp = pd.Timestamp("2024-01-06", tz="UTC")  # Saturday
    trades = pd.DataFrame({
        "timestamp": [weekend_timestamp],
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
    })

    # Create prices
    prices = pd.DataFrame({
        "timestamp": [weekend_timestamp],
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    # Apply fill model pipeline (session gate will reject weekend orders)
    # Use strict=False to handle missing exchange_calendars gracefully
    fills = apply_fill_model_pipeline(
        trades,
        prices=prices,
        freq="1d",
        partial_fill_model=None,
        strict_session_gate=False,  # Permissive fallback if exchange_calendars missing
    )

    # Add cost columns
    commission_model = CommissionModel(mode="bps", commission_bps=1.0, fixed_per_trade=0.0)
    spread_model = SpreadModel(adv_window=20, buckets=None, fallback_spread_bps=5.0)
    slippage_model = SlippageModel(vol_window=20, k=1.0, min_bps=0.0, max_bps=50.0, fallback_slippage_bps=10.0)
    
    fills_with_costs = add_cost_columns_to_trades(
        fills,
        commission_model=commission_model,
        spread_model=spread_model,
        slippage_model=slippage_model,
        prices=prices,
    )

    # Check: rejected fills have zero costs
    rejected_mask = fills_with_costs["status"] == "rejected"
    if rejected_mask.any():
        assert (fills_with_costs.loc[rejected_mask, "commission_cash"] == 0.0).all(), "Rejected fills should have zero commission"
        assert (fills_with_costs.loc[rejected_mask, "spread_cash"] == 0.0).all(), "Rejected fills should have zero spread"
        assert (fills_with_costs.loc[rejected_mask, "slippage_cash"] == 0.0).all(), "Rejected fills should have zero slippage"
        assert (fills_with_costs.loc[rejected_mask, "total_cost_cash"] == 0.0).all(), "Rejected fills should have zero total cost"


def test_partial_fills_costs_scale_with_filled_notional() -> None:
    """Test that partial fills costs scale with filled notional (not original qty)."""
    # Create large order that will be partially filled
    trades = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-25", tz="UTC")],  # After 20+ days for ADV
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100000.0],  # Large order
        "price": [150.0],
    })

    # Create prices with sufficient history for ADV
    timestamps = pd.date_range("2024-01-01", periods=30, freq="1d", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": ["AAPL"] * 30,
        "close": [150.0] * 30,
        "volume": [1e6] * 30,  # 1M shares per day
    })

    # Apply fill model pipeline with partial fill model (participation_cap = 0.05)
    partial_model = PartialFillModel(adv_window=20, participation_cap=0.05, min_fill_qty=0.0)
    
    fills = apply_fill_model_pipeline(
        trades,
        prices=prices,
        freq="1d",
        partial_fill_model=partial_model,
        strict_session_gate=False,  # Permissive fallback if exchange_calendars missing
    )

    # Add cost columns
    commission_model = CommissionModel(mode="bps", commission_bps=1.0, fixed_per_trade=0.0)
    spread_model = SpreadModel(adv_window=20, buckets=None, fallback_spread_bps=5.0)
    slippage_model = SlippageModel(vol_window=20, k=1.0, min_bps=0.0, max_bps=50.0, fallback_slippage_bps=10.0)
    
    fills_with_costs = add_cost_columns_to_trades(
        fills,
        commission_model=commission_model,
        spread_model=spread_model,
        slippage_model=slippage_model,
        prices=prices,
    )

    # Check: fill_qty < qty (partial fill)
    assert fills_with_costs["fill_qty"].iloc[0] < fills_with_costs["qty"].iloc[0], "Should be partial fill"
    
    # Check: costs are based on fill_qty * fill_price (not original qty * price)
    filled_notional = fills_with_costs["fill_qty"].iloc[0] * fills_with_costs["fill_price"].iloc[0]
    original_notional = fills_with_costs["qty"].iloc[0] * fills_with_costs["price"].iloc[0]
    
    # Commission should be proportional to filled notional
    # commission_bps = 1.0, so commission = filled_notional * 0.0001
    expected_commission = filled_notional * 0.0001
    actual_commission = fills_with_costs["commission_cash"].iloc[0]
    
    # Allow small tolerance for rounding
    assert abs(actual_commission - expected_commission) < 0.01, "Commission should scale with filled notional"
    
    # Total cost should be less than if based on original notional
    # (since fill_qty < qty)
    assert actual_commission < (original_notional * 0.0001), "Costs should be less for partial fills"
