"""Accounting report writer (Sprint 13).

This module provides functions to write daily accounting reports
in CSV, JSON, and optional Markdown formats with deterministic output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _json_serialize_nan(obj: Any) -> Any:
    """JSON serializer that converts NaN/Inf to None (for deterministic JSON output)."""
    if isinstance(obj, dict):
        return {k: _json_serialize_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_serialize_nan(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj


def write_accounting_report_csv(
    positions_result: dict,
    output_dir: Path | str,
    run_id: str,
    as_of: pd.Timestamp | str,
    *,
    start_cash: float,
    reconciliation_result: dict | None = None,
    ledger_pack_path: str | None = None,
    reconcile_report_path: str | None = None,
    costs_breakdown: dict[str, float] | None = None,
    broker_meta: dict[str, Any] | None = None,
) -> Path:
    """Write accounting report to CSV file.

    Args:
        positions_result: Result dict from build_positions_from_ledger() with keys:
            - positions_df: DataFrame with columns: symbol, qty, avg_price, realized_pnl, unrealized_pnl, ...
            - cash_balance: float (final cash balance)
            - summary: dict with total_realized_pnl, total_unrealized_pnl, etc.
        output_dir: Base output directory
        run_id: Run identifier
        as_of: Report date (UTC, tz-aware)
        start_cash: Starting cash balance
        reconciliation_result: Optional reconciliation result dict
        ledger_pack_path: Optional path to ledger pack (relative to output_dir)
        reconcile_report_path: Optional path to reconciliation report (relative to output_dir)
        costs_breakdown: Optional dict with keys: commission_cash, spread_cash, slippage_cash, total_cost_cash

    Returns:
        Path to written CSV file
    """
    # Normalize as_of
    if isinstance(as_of, str):
        as_of = pd.to_datetime(as_of, utc=True)
    if as_of.tz is None:
        as_of = as_of.tz_localize("UTC")

    # Format date as YYYY-MM-DD
    date_str = as_of.strftime("%Y-%m-%d")

    # Create output directory
    output_path = Path(output_dir)
    accounting_dir = output_path / f"accounting_report_{run_id}"
    accounting_dir.mkdir(parents=True, exist_ok=True)

    # Build CSV file path
    csv_path = accounting_dir / f"accounting_{date_str}.csv"

    # Extract data
    positions_df = positions_result.get("positions_df", pd.DataFrame())
    cash_balance = positions_result.get("cash_balance", start_cash)
    summary = positions_result.get("summary", {})

    total_realized_pnl = summary.get("total_realized_pnl", 0.0)
    total_unrealized_pnl = summary.get("total_unrealized_pnl", 0.0)
    total_pnl = summary.get("total_pnl", total_realized_pnl + total_unrealized_pnl)

    # Fixed schema columns (including optional reconciliation / broker fields)
    fixed_columns = [
        "section",
        "symbol",
        "cash_start",
        "cash_end",
        "cash_change",
        "realized_pnl",
        "unrealized_pnl",
        "total_pnl",
        "commission_cash",
        "spread_cash",
        "slippage_cash",
        "total_cost_cash",
        "reconciliation_ok",
        "cash_end_matches_reconcile_cash",
        "reconcile_report_path",
        "broker_view_source",
        "broker_snapshot_run_id",
        "broker_snapshot_date",
        "broker_snapshot_path",
        "schema_version",
    ]

    # Build report rows
    report_rows = []

    # Consistency flag: does accounting cash_end match broker cash in reconciliation?
    # We approximate this via reconciliation_result["cash_match"] if reconciliation_result is provided.
    cash_end_matches_reconcile_cash: bool | None = None
    if reconciliation_result is not None:
        cash_end_matches_reconcile_cash = bool(reconciliation_result.get("cash_match"))

    # Broker meta fields (optional, fixed schema)
    broker_view_source = ""
    broker_snapshot_run_id = ""
    broker_snapshot_date = ""
    broker_snapshot_path = ""
    if broker_meta is not None:
        broker_view_source = str(broker_meta.get("broker_view_source", "") or "")
        broker_snapshot_run_id = str(
            broker_meta.get("broker_snapshot_run_id", "") or ""
        )
        broker_snapshot_date = str(broker_meta.get("broker_snapshot_date", "") or "")
        broker_snapshot_path = str(broker_meta.get("broker_snapshot_path", "") or "")

    # Summary row
    report_rows.append(
        {
            "section": "SUMMARY",
            "symbol": "",
            "cash_start": start_cash,
            "cash_end": cash_balance,
            "cash_change": cash_balance - start_cash,
            "realized_pnl": total_realized_pnl,
            "unrealized_pnl": total_unrealized_pnl,
            "total_pnl": total_pnl,
            "commission_cash": (
                costs_breakdown.get("commission_cash", 0.0) if costs_breakdown else 0.0
            ),
            "spread_cash": (
                costs_breakdown.get("spread_cash", 0.0) if costs_breakdown else 0.0
            ),
            "slippage_cash": (
                costs_breakdown.get("slippage_cash", 0.0) if costs_breakdown else 0.0
            ),
            "total_cost_cash": (
                costs_breakdown.get("total_cost_cash", 0.0) if costs_breakdown else 0.0
            ),
            "reconciliation_ok": (
                reconciliation_result.get("ok") if reconciliation_result else None
            ),
            "cash_end_matches_reconcile_cash": cash_end_matches_reconcile_cash,
            "reconcile_report_path": reconcile_report_path or "",
            "broker_view_source": broker_view_source,
            "broker_snapshot_run_id": broker_snapshot_run_id,
            "broker_snapshot_date": broker_snapshot_date,
            "broker_snapshot_path": broker_snapshot_path,
            "schema_version": 1,
        }
    )

    # Per-symbol rows (sorted by symbol)
    if not positions_df.empty:
        for row in positions_df.itertuples(index=False):
            _rpnl = row.realized_pnl if hasattr(row, "realized_pnl") else float("nan")
            _upnl = row.unrealized_pnl if hasattr(row, "unrealized_pnl") else float("nan")
            report_rows.append(
                {
                    "section": "POSITION",
                    "symbol": str(row.symbol),
                    "cash_start": None,
                    "cash_end": None,
                    "cash_change": None,
                    "realized_pnl": float(_rpnl) if pd.notna(_rpnl) else 0.0,
                    "unrealized_pnl": float(_upnl) if pd.notna(_upnl) else 0.0,
                    "total_pnl": (
                        float(_rpnl + _upnl)
                        if pd.notna(_rpnl) and pd.notna(_upnl)
                        else None
                    ),
                    "commission_cash": None,
                    "spread_cash": None,
                    "slippage_cash": None,
                    "total_cost_cash": None,
                    "reconciliation_ok": None,
                    "cash_end_matches_reconcile_cash": None,
                    "reconcile_report_path": "",
                    "broker_view_source": broker_view_source,
                    "broker_snapshot_run_id": broker_snapshot_run_id,
                    "broker_snapshot_date": broker_snapshot_date,
                    "broker_snapshot_path": broker_snapshot_path,
                    "schema_version": 1,
                }
            )

    # Build DataFrame with fixed schema
    report_df = pd.DataFrame(report_rows)
    # Ensure all fixed columns exist (for schema stability)
    for col in fixed_columns:
        if col not in report_df.columns:
            report_df[col] = ""

    # Replace NaN with None for CSV (pandas will write as empty)
    report_df = report_df.fillna("")

    # Write CSV (deterministic: sorted by section, then symbol)
    report_df = report_df[fixed_columns].sort_values(
        ["section", "symbol"], kind="mergesort"
    )
    report_df.to_csv(csv_path, index=False)

    logger.info(f"Accounting report CSV written: {csv_path}")
    return csv_path


def write_accounting_report_json(
    positions_result: dict,
    output_dir: Path | str,
    run_id: str,
    as_of: pd.Timestamp | str,
    *,
    start_cash: float,
    reconciliation_result: dict | None = None,
    ledger_pack_path: str | None = None,
    reconcile_report_path: str | None = None,
    costs_breakdown: dict[str, float] | None = None,
    broker_meta: dict[str, Any] | None = None,
) -> Path:
    """Write accounting report to JSON file.

    Args:
        positions_result: Result dict from build_positions_from_ledger()
        output_dir: Base output directory
        run_id: Run identifier
        as_of: Report date (UTC, tz-aware)
        start_cash: Starting cash balance
        reconciliation_result: Optional reconciliation result dict
        ledger_pack_path: Optional path to ledger pack (relative to output_dir)
        reconcile_report_path: Optional path to reconciliation report (relative to output_dir)
        costs_breakdown: Optional dict with cost breakdown

    Returns:
        Path to written JSON file
    """
    # Normalize as_of
    if isinstance(as_of, str):
        as_of = pd.to_datetime(as_of, utc=True)
    if as_of.tz is None:
        as_of = as_of.tz_localize("UTC")

    # Format date as YYYY-MM-DD
    date_str = as_of.strftime("%Y-%m-%d")

    # Create output directory
    output_path = Path(output_dir)
    accounting_dir = output_path / f"accounting_report_{run_id}"
    accounting_dir.mkdir(parents=True, exist_ok=True)

    # Build JSON file path
    json_path = accounting_dir / f"accounting_{date_str}.json"

    # Extract data
    positions_df = positions_result.get("positions_df", pd.DataFrame())
    cash_balance = positions_result.get("cash_balance", start_cash)
    summary = positions_result.get("summary", {})

    # Build positions list (sorted by symbol, deterministic)
    positions_list = []
    if not positions_df.empty:
        positions_sorted = positions_df.sort_values("symbol", kind="mergesort")
        for row in positions_sorted.itertuples(index=False):
            _qty = row.qty if hasattr(row, "qty") else float("nan")
            _avg = row.avg_price if hasattr(row, "avg_price") else float("nan")
            _rpnl = row.realized_pnl if hasattr(row, "realized_pnl") else float("nan")
            _upnl = row.unrealized_pnl if hasattr(row, "unrealized_pnl") else float("nan")
            _notional = row.notional if hasattr(row, "notional") else float("nan")
            _last = row.last_price if hasattr(row, "last_price") else float("nan")
            positions_list.append(
                {
                    "symbol": str(row.symbol),
                    "qty": float(_qty) if pd.notna(_qty) else 0.0,
                    "avg_price": float(_avg) if pd.notna(_avg) else None,
                    "realized_pnl": float(_rpnl) if pd.notna(_rpnl) else 0.0,
                    "unrealized_pnl": float(_upnl) if pd.notna(_upnl) else 0.0,
                    "total_pnl": (
                        float(_rpnl + _upnl)
                        if pd.notna(_rpnl) and pd.notna(_upnl)
                        else None
                    ),
                    "notional": float(_notional) if pd.notna(_notional) else 0.0,
                    "last_price": float(_last) if pd.notna(_last) else None,
                }
            )

    # Build report dict
    report = {
        "schema_version": 1,
        "as_of_date": as_of.isoformat(),
        "run_id": run_id,
        "cash": {
            "start": start_cash,
            "end": cash_balance,
            "change": cash_balance - start_cash,
        },
        "pnl": {
            "total_realized": summary.get("total_realized_pnl", 0.0),
            "total_unrealized": summary.get("total_unrealized_pnl", 0.0),
            "total": summary.get(
                "total_pnl",
                summary.get("total_realized_pnl", 0.0)
                + summary.get("total_unrealized_pnl", 0.0),
            ),
        },
        "positions": positions_list,
        "summary": {
            "n_positions": summary.get("n_positions", len(positions_df)),
            "gross_exposure": summary.get("gross_exposure", 0.0),
            "net_exposure": summary.get("net_exposure", 0.0),
        },
    }

    # Add costs if provided
    if costs_breakdown:
        report["costs"] = {
            "commission_cash": costs_breakdown.get("commission_cash", 0.0),
            "spread_cash": costs_breakdown.get("spread_cash", 0.0),
            "slippage_cash": costs_breakdown.get("slippage_cash", 0.0),
            "total_cost_cash": costs_breakdown.get("total_cost_cash", 0.0),
        }

    # Add reconciliation info if provided
    if reconciliation_result:
        # Consistency flag: does accounting cash_end match broker cash in reconciliation?
        cash_end_matches_reconcile_cash: bool | None = bool(
            reconciliation_result.get("cash_match", False)
        )
        report["reconciliation"] = {
            "ok": reconciliation_result.get("ok", False),
            "cash_match": reconciliation_result.get("cash_match", False),
            "cash_diff": reconciliation_result.get("cash_diff", 0.0),
            "cash_end_matches_reconcile_cash": cash_end_matches_reconcile_cash,
        }

    # Add links if provided
    if ledger_pack_path:
        report["ledger_pack_path"] = ledger_pack_path
    if reconcile_report_path:
        report["reconcile_report_path"] = reconcile_report_path

    # Add broker meta if provided (mirrors reconciliation report)
    if broker_meta is not None:
        report["broker_meta"] = dict(broker_meta)

    # Serialize NaN/Inf to None
    report_serialized = _json_serialize_nan(report)

    # Write JSON atomically (deterministic: sort_keys=True, indent=2)
    from src.assembled_core.utils.atomic_io import atomic_write_json
    atomic_write_json(json_path, report_serialized, sort_keys=True)

    logger.info(f"Accounting report JSON written: {json_path}")
    return json_path
