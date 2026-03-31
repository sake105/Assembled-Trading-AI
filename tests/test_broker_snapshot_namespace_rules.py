"""Namespace rules for broker snapshot usage (Sprint 13).

Goal:
- Prevent using snapshots from the wrong run_id/namespace by mistake.
- Make require/prefer behavior around namespaces explicit and testable.
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
    broker_snapshot_base_path,
    store_broker_snapshot_json,
)
from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades


def _minimal_trades(as_of: datetime) -> pd.DataFrame:
    """Create a minimal trades DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(as_of, tz="UTC"),
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 5.0,
                "price": 100.0,
                "fill_qty": 5.0,
                "fill_price": 100.0,
                "status": "filled",
                "total_cost_cash": 0.0,
            },
        ]
    )


def test_require_uses_correct_namespace(tmp_path: Path) -> None:
    """Policy=require should look only in the requested namespace (run_id)."""
    output_dir = tmp_path
    as_of = datetime(2025, 1, 15, 10, 0, 0)
    as_of_ts = pd.Timestamp(as_of, tz="UTC")

    # Ledger run_id (for events)
    ledger_run_id = "ledger_run"
    # Snapshot namespace to use
    snapshot_ns = "snapshot_ns_ok"

    # Create trades/orders
    trades = _minimal_trades(as_of)
    orders = trades.copy()

    # Store a broker snapshot under the snapshot namespace
    positions = pd.DataFrame({"symbol": ["AAPL"], "qty": [5.0]})
    store_broker_snapshot_json(
        cash=10000.0,
        positions_df=positions,
        output_dir=output_dir,
        run_id=snapshot_ns,
        as_of_date=as_of_ts,
    )

    # Build ledger with policy=require and explicit snapshot namespace
    result = build_ledger_from_trades(
        orders_df=orders,
        trades_df=trades,
        run_id=ledger_run_id,
        output_dir=output_dir,
        as_of_date=as_of_ts,
        prices_df=None,
        start_cash=10000.0,
        broker_snapshot_policy="require",
        write_paper_broker_snapshot=False,
        broker_snapshot_run_id=snapshot_ns,
    )

    broker_meta = result.get("broker_meta") or {}
    assert broker_meta.get("broker_view_source") == "stored_snapshot"
    assert broker_meta.get("broker_snapshot_run_id") == snapshot_ns


def test_require_with_wrong_namespace_raises(tmp_path: Path) -> None:
    """Policy=require should fail if the requested namespace has no snapshot."""
    output_dir = tmp_path
    as_of = datetime(2025, 1, 15, 10, 0, 0)
    as_of_ts = pd.Timestamp(as_of, tz="UTC")

    trades = _minimal_trades(as_of)
    orders = trades.copy()

    # Store snapshot in a different namespace
    actual_ns = "actual_ns"
    positions = pd.DataFrame({"symbol": ["AAPL"], "qty": [5.0]})
    store_broker_snapshot_json(
        cash=10000.0,
        positions_df=positions,
        output_dir=output_dir,
        run_id=actual_ns,
        as_of_date=as_of_ts,
    )

    wrong_ns = "wrong_ns"

    # Expected base path for error message
    expected_base = broker_snapshot_base_path(output_dir, wrong_ns)

    try:
        _ = build_ledger_from_trades(
            orders_df=orders,
            trades_df=trades,
            run_id="ledger_run",
            output_dir=output_dir,
            as_of_date=as_of_ts,
            prices_df=None,
            start_cash=10000.0,
            broker_snapshot_policy="require",
            write_paper_broker_snapshot=False,
            broker_snapshot_run_id=wrong_ns,
        )
    except ValueError as exc:
        msg = str(exc)
        assert "Broker snapshot required but not found" in msg
        assert f"run_id={wrong_ns}" in msg
        assert "as_of_date=2025-01-15" in msg
        assert str(expected_base) in msg
    else:
        raise AssertionError(
            "Expected ValueError for missing snapshot in wrong namespace"
        )


def test_prefer_with_wrong_namespace_falls_back_to_paper_view(tmp_path: Path) -> None:
    """Policy=prefer should fall back to paper view if requested namespace is empty."""
    output_dir = tmp_path
    as_of = datetime(2025, 1, 15, 10, 0, 0)
    as_of_ts = pd.Timestamp(as_of, tz="UTC")

    trades = _minimal_trades(as_of)
    orders = trades.copy()

    wrong_ns = "non_existing_ns"

    result = build_ledger_from_trades(
        orders_df=orders,
        trades_df=trades,
        run_id="ledger_run",
        output_dir=output_dir,
        as_of_date=as_of_ts,
        prices_df=None,
        start_cash=10000.0,
        broker_snapshot_policy="prefer",
        write_paper_broker_snapshot=False,
        broker_snapshot_run_id=wrong_ns,
    )

    broker_meta = result.get("broker_meta") or {}
    assert broker_meta.get("broker_view_source") == "paper_view"
    # broker_snapshot_run_id should reflect the namespace that was actually used (requested)
    assert broker_meta.get("broker_snapshot_run_id") == wrong_ns


def test_default_namespace_uses_ledger_run_id_when_none(tmp_path: Path) -> None:
    """If broker_snapshot_run_id is None, default to ledger run_id for namespace."""
    output_dir = tmp_path
    as_of = datetime(2025, 1, 15, 10, 0, 0)
    as_of_ts = pd.Timestamp(as_of, tz="UTC")

    ledger_run_id = "ledger_run_default_ns"
    trades = _minimal_trades(as_of)
    orders = trades.copy()

    # Store snapshot under the ledger run_id namespace
    positions = pd.DataFrame({"symbol": ["AAPL"], "qty": [5.0]})
    store_broker_snapshot_json(
        cash=10000.0,
        positions_df=positions,
        output_dir=output_dir,
        run_id=ledger_run_id,
        as_of_date=as_of_ts,
    )

    # Call without explicit broker_snapshot_run_id
    result = build_ledger_from_trades(
        orders_df=orders,
        trades_df=trades,
        run_id=ledger_run_id,
        output_dir=output_dir,
        as_of_date=as_of_ts,
        prices_df=None,
        start_cash=10000.0,
        broker_snapshot_policy="prefer",
        write_paper_broker_snapshot=False,
        broker_snapshot_run_id=None,
    )

    broker_meta = result.get("broker_meta") or {}
    assert broker_meta.get("broker_view_source") == "stored_snapshot"
    # Finalized namespace should be the ledger run_id
    assert broker_meta.get("broker_snapshot_run_id") == ledger_run_id
