"""Tests for broker snapshot policy 'require' (Sprint 13).

Tests that 'require' policy fails fast when snapshot is missing.
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

from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades


def test_broker_snapshot_require_fails_when_missing(tmp_path: Path):
    """Test that 'require' policy raises ValueError when snapshot is missing."""
    # Create minimal trades
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    trades = pd.DataFrame([
        {
            "timestamp": pd.Timestamp(base_time, tz="UTC"),
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 5.0,
            "price": 150.0,
            "fill_qty": 5.0,
            "fill_price": 150.0,
            "status": "filled",
            "total_cost_cash": 0.0,
        },
    ])
    
    orders = trades.copy()
    run_id = "test_require"
    as_of_date = pd.Timestamp(base_time, tz="UTC")
    
    # Build ledger with policy "require" but no snapshot exists
    with pytest.raises(ValueError, match="Broker snapshot required but not found"):
        build_ledger_from_trades(
            orders_df=orders,
            trades_df=trades,
            run_id=run_id,
            output_dir=tmp_path,
            as_of_date=as_of_date,
            prices_df=None,
            start_cash=10000.0,
            broker_snapshot_policy="require",
            write_paper_broker_snapshot=False,
        )


def test_broker_snapshot_require_succeeds_when_present(tmp_path: Path):
    """Test that 'require' policy succeeds when snapshot exists."""
    from src.assembled_core.accounting.broker_snapshot_store import (
        store_broker_snapshot_json,
    )
    
    # Create minimal trades
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    trades = pd.DataFrame([
        {
            "timestamp": pd.Timestamp(base_time, tz="UTC"),
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 5.0,
            "price": 150.0,
            "fill_qty": 5.0,
            "fill_price": 150.0,
            "status": "filled",
            "total_cost_cash": 0.0,
        },
    ])
    
    orders = trades.copy()
    run_id = "test_require_success"
    as_of_date = pd.Timestamp(base_time, tz="UTC")
    
    # Store broker snapshot first
    broker_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [5.0],
    })
    store_broker_snapshot_json(
        cash=10000.0,
        positions_df=broker_positions,
        output_dir=tmp_path,
        run_id=run_id,
        as_of_date=as_of_date,
    )
    
    # Build ledger with policy "require" - should succeed
    result = build_ledger_from_trades(
        orders_df=orders,
        trades_df=trades,
        run_id=run_id,
        output_dir=tmp_path,
        as_of_date=as_of_date,
        prices_df=None,
        start_cash=10000.0,
        broker_snapshot_policy="require",
        write_paper_broker_snapshot=False,
    )
    
    # Should succeed without error
    assert result["reconciliation_result"] is not None
    assert result["broker_snapshot_path"] is not None
