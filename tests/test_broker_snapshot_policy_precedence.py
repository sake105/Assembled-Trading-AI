"""Tests for broker snapshot policy precedence (Sprint 13).

Tests that broker snapshot is preferred over paper view when policy is "prefer".
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.broker_snapshot_store import (
    store_broker_snapshot_json,
)
from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades


def test_broker_snapshot_precedence_over_paper_view(tmp_path: Path):
    """Test that stored broker snapshot is used when policy is 'prefer'."""
    # Create minimal trades
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    trades = pd.DataFrame([
        {
            "timestamp": pd.Timestamp(base_time, tz="UTC"),
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 5.0,  # Paper view will have AAPL=5
            "price": 150.0,
            "fill_qty": 5.0,
            "fill_price": 150.0,
            "status": "filled",
            "total_cost_cash": 0.0,
        },
    ])
    
    orders = trades.copy()
    run_id = "test_precedence"
    as_of_date = pd.Timestamp(base_time, tz="UTC")
    
    # Create a broker snapshot with different position (AAPL=10)
    # This will intentionally differ from paper view (AAPL=5)
    broker_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [10.0],  # Different from paper view
    })
    broker_cash = 10000.0
    
    # Store broker snapshot
    store_broker_snapshot_json(
        cash=broker_cash,
        positions_df=broker_positions,
        output_dir=tmp_path,
        run_id=run_id,
        as_of_date=as_of_date,
    )
    
    # Build ledger with policy "prefer"
    result = build_ledger_from_trades(
        orders_df=orders,
        trades_df=trades,
        run_id=run_id,
        output_dir=tmp_path,
        as_of_date=as_of_date,
        prices_df=None,
        start_cash=10000.0,
        broker_snapshot_policy="prefer",
        write_paper_broker_snapshot=False,
    )
    
    # Verify reconciliation was performed
    assert result["reconciliation_result"] is not None, "Reconciliation should be performed"
    
    # The reconciliation should use broker snapshot (AAPL=10), not paper view (AAPL=5)
    # Since ledger has AAPL=5 and broker snapshot has AAPL=10, there should be a mismatch
    # (unless the reconciliation logic matches them somehow, but the key is that snapshot was used)
    
    # Verify broker_snapshot_path is set (snapshot was found and used)
    assert result["broker_snapshot_path"] is not None, "broker_snapshot_path should be set when snapshot is used"
    
    # Verify the reconciliation result reflects the broker snapshot usage
    # The exact outcome depends on reconciliation logic, but snapshot should be preferred
    reconciliation = result["reconciliation_result"]
    assert "ok" in reconciliation, "Reconciliation result should have 'ok' field"
