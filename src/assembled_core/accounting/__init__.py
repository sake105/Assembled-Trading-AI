"""Accounting and ledger system for paper trading (Sprint 13).

This package provides:
- Ledger event generation from orders/trades
- Position engine with average cost basis tracking
- Ledger storage (parquet-based, deterministic)
- Reconciliation engine (ledger vs broker snapshots)
"""

from src.assembled_core.accounting.ledger import (
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    events_from_orders,
    events_from_trades,
    generate_event_id,
)
from src.assembled_core.accounting.ledger_store import (
    ledger_base_path,
    list_ledger_runs,
    load_ledger_events_parquet,
    store_daily_snapshot_parquet,
    store_ledger_events_parquet,
)
from src.assembled_core.accounting.position_engine import build_positions_from_ledger
from src.assembled_core.accounting.reconciliation import reconcile_ledger_vs_broker
from src.assembled_core.accounting.reconciliation_report import (
    write_reconcile_report_csv,
    write_reconcile_report_json,
    write_reconcile_report_md,
)
from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades

__all__ = [
    # Contract
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    # Event generation
    "events_from_orders",
    "events_from_trades",
    "generate_event_id",
    # Storage
    "ledger_base_path",
    "store_ledger_events_parquet",
    "load_ledger_events_parquet",
    "store_daily_snapshot_parquet",
    "list_ledger_runs",
    # Position engine
    "build_positions_from_ledger",
    # Reconciliation
    "reconcile_ledger_vs_broker",
    # Reconciliation reports
    "write_reconcile_report_csv",
    "write_reconcile_report_json",
    "write_reconcile_report_md",
    # Integration
    "build_ledger_from_trades",
]
