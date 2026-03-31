"""Tests for EOD pipeline broker snapshot policy 'require' (Sprint 13).

Tests that 'require' policy fails fast when snapshot is missing.
This test directly tests build_ledger_from_trades() which is called by the EOD pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades


def test_eod_policy_require_raises_when_snapshot_missing(tmp_path: Path) -> None:
    """Test that build_ledger_from_trades() with policy='require' raises ValueError when snapshot is missing."""
    # Minimal synthetic inputs: orders/trades such that ledger step runs,
    # but no broker snapshot exists in output/broker_snapshot_<run_id>/...
    run_id = "test_run_require_snapshot"
    output_dir = tmp_path

    # Minimal orders and trades
    orders_df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-02T00:00:00Z", tz="UTC"),
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 10.0,
                "price": 100.0,
            }
        ]
    )

    trades_df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-02T00:00:00Z", tz="UTC"),
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 10.0,
                "price": 100.0,  # Required column
                "fill_qty": 10.0,
                "fill_price": 100.0,
                "status": "filled",
                "commission_cash": 0.0,
                "spread_cash": 0.0,
                "slippage_cash": 0.0,
                "total_cost_cash": 0.0,
            }
        ]
    )

    # Run build_ledger_from_trades with policy="require" - should raise ValueError
    with pytest.raises(ValueError, match="Broker snapshot required but not found"):
        build_ledger_from_trades(
            orders_df=orders_df,
            trades_df=trades_df,
            run_id=run_id,
            output_dir=output_dir,
            as_of_date=pd.Timestamp("2024-01-02", tz="UTC"),
            prices_df=None,
            start_cash=100000.0,
            broker_snapshot_policy="require",
            write_paper_broker_snapshot=False,
            broker_snapshot_run_id=None,  # Will use run_id
        )
