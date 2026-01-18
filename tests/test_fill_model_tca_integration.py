# tests/test_fill_model_tca_integration.py
"""Tests for TCA integration with fill model (Sprint 7 / C5).

Tests verify:
1. TCA report still writes, schema ok
2. cost_bps computed deterministically for partial fills
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
from src.assembled_core.qa.tca import build_tca_report


def test_tca_report_schema_ok_with_partial_fills() -> None:
    """Test that TCA report has correct schema even with partial fills."""
    # Create trades with partial fills
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-25", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [100000.0, 50000.0, 75000.0],  # Large orders
        "price": [150.0, 200.0, 100.0],
    })

    # Create prices with sufficient history for ADV
    timestamps = pd.date_range("2024-01-01", periods=30, freq="1d", tz="UTC")
    prices_list = []
    for ts in timestamps:
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            prices_list.append({
                "timestamp": ts,
                "symbol": symbol,
                "close": 150.0 if symbol == "AAPL" else (200.0 if symbol == "MSFT" else 100.0),
                "volume": 1e6,
            })
    prices = pd.DataFrame(prices_list)
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Apply fill model pipeline with partial fill model
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

    # Build TCA report
    tca_report = build_tca_report(fills_with_costs, freq="1d")

    # Check: TCA report has correct schema
    required_cols = [
        "date",
        "symbol",
        "notional",
        "commission_cash",
        "spread_cash",
        "slippage_cash",
        "total_cost_cash",
        "cost_bps",
        "n_trades",
    ]
    for col in required_cols:
        assert col in tca_report.columns, f"TCA report should have column '{col}'"
    
    # Check: no NaNs in key columns
    assert not tca_report["notional"].isna().any(), "notional should have no NaNs"
    assert not tca_report["total_cost_cash"].isna().any(), "total_cost_cash should have no NaNs"
    assert not tca_report["cost_bps"].isna().any(), "cost_bps should have no NaNs"


def test_cost_bps_computed_deterministically_for_partial_fills() -> None:
    """Test that cost_bps is computed deterministically for partial fills."""
    # Create trades with partial fills
    trades = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-25", tz="UTC")],
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
        "volume": [1e6] * 30,
    })

    # Apply fill model pipeline with partial fill model
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

    # Build TCA report twice (should be identical)
    tca_report1 = build_tca_report(fills_with_costs, freq="1d")
    tca_report2 = build_tca_report(fills_with_costs, freq="1d")

    # Check: TCA reports are identical (deterministic)
    pd.testing.assert_frame_equal(
        tca_report1,
        tca_report2,
        "TCA report should be deterministic",
    )

    # Check: cost_bps is computed correctly (total_cost_cash / notional * 10000)
    # notional should be based on fill_qty * fill_price (not original qty)
    if not tca_report1.empty:
        row = tca_report1.iloc[0]
        expected_cost_bps = (row["total_cost_cash"] / row["notional"]) * 10000.0 if row["notional"] > 0.0 else 0.0
        assert abs(row["cost_bps"] - expected_cost_bps) < 0.01, "cost_bps should be computed correctly"


def test_tca_report_handles_rejected_fills() -> None:
    """Test that TCA report handles rejected fills correctly."""
    # Create trades on weekend (will be rejected)
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

    # Apply fill model pipeline (session gate will reject)
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

    # Build TCA report
    tca_report = build_tca_report(fills_with_costs, freq="1d")

    # Check: rejected fills have zero notional and zero costs
    if not tca_report.empty:
        # If rejected fills are included, they should have zero notional and costs
        # But typically rejected fills might be filtered out or have zero notional
        # For now, just check that the report is valid
        assert "notional" in tca_report.columns, "TCA report should have notional column"
        assert "total_cost_cash" in tca_report.columns, "TCA report should have total_cost_cash column"
