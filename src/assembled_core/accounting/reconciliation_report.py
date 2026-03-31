"""Reconciliation report writer (Sprint 13 L4).

This module provides functions to write daily reconciliation reports
in CSV and JSON formats with deterministic output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _json_serialize_nan(obj: Any) -> Any:
    """JSON serializer that converts NaN/Inf to None (for deterministic JSON output).

    This function recursively processes dictionaries, lists, and values to convert
    NaN and Inf values to None, ensuring JSON compatibility.

    Args:
        obj: Object to serialize (dict, list, or scalar)

    Returns:
        Serialized object with NaN/Inf -> None
    """

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


def write_reconcile_report_csv(
    result_dict: dict,
    output_dir: Path | str,
    run_id: str,
    as_of: pd.Timestamp | str,
    *,
    ledger_cash: float | None = None,
    broker_cash: float | None = None,
    broker_meta: dict[str, Any] | None = None,
) -> Path:
    """Write reconciliation report to CSV file.

    Args:
        result_dict: Reconciliation result dictionary from reconcile_ledger_vs_broker()
        output_dir: Base output directory
        run_id: Run identifier
        as_of: Reconciliation date (UTC, tz-aware)

    Returns:
        Path to written CSV file

    Raises:
        ValueError: If result_dict is missing required keys
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
    reconcile_dir = output_path / f"reconcile_report_{run_id}"
    reconcile_dir.mkdir(parents=True, exist_ok=True)

    # Build CSV file path
    csv_path = reconcile_dir / f"reconcile_{date_str}.csv"

    # Build report DataFrame
    report_rows: list[dict[str, Any]] = []

    # Get cash values (prefer function params, fallback to result_dict)
    cash_match = result_dict.get("cash_match", False)
    cash_diff = result_dict.get("cash_diff", 0.0)

    # Use provided cash values or extract from result_dict
    if ledger_cash is None:
        ledger_cash = result_dict.get("ledger_cash")
    if broker_cash is None:
        broker_cash = result_dict.get("broker_cash")

    report_rows.append(
        {
            "type": "cash",
            "symbol": None,
            "ledger_value": ledger_cash,
            "broker_value": broker_cash,
            "diff": cash_diff,
            "match": cash_match,
        }
    )

    # Position difference rows
    position_diffs_df = result_dict.get("position_diffs_df")
    if position_diffs_df is not None and not position_diffs_df.empty:
        for _, row in position_diffs_df.iterrows():
            report_rows.append(
                {
                    "type": "position",
                    "symbol": row["symbol"],
                    "ledger_value": row["ledger_qty"],
                    "broker_value": row["broker_qty"],
                    "diff": row["diff_qty"],
                    "match": False,
                }
            )

    # Missing symbols rows
    missing_in_ledger = result_dict.get("missing_in_ledger", [])
    for symbol in missing_in_ledger:
        report_rows.append(
            {
                "type": "missing_in_ledger",
                "symbol": symbol,
                "ledger_value": None,
                "broker_value": None,
                "diff": None,
                "match": False,
            }
        )

    missing_in_broker = result_dict.get("missing_in_broker", [])
    for symbol in missing_in_broker:
        report_rows.append(
            {
                "type": "missing_in_broker",
                "symbol": symbol,
                "ledger_value": None,
                "broker_value": None,
                "diff": None,
                "match": False,
            }
        )

    # Build DataFrame
    if report_rows:
        report_df = pd.DataFrame(report_rows)

        # Deterministic sort: by abs(diff) desc, then symbol asc
        # For cash row (symbol=None), put it first
        def _sort_key_func(row: pd.Series) -> tuple[int, float, str]:
            if row["type"] == "cash":
                return (0, 0.0, "")  # Cash first
            diff_abs = abs(row["diff"]) if pd.notna(row["diff"]) else 0.0
            symbol_str = str(row["symbol"]) if pd.notna(row["symbol"]) else ""
            return (
                1,
                -diff_abs,
                symbol_str,
            )  # Position rows: by abs(diff) desc, then symbol asc

        report_df["_sort_key"] = report_df.apply(_sort_key_func, axis=1)
        report_df = report_df.sort_values("_sort_key", kind="mergesort").reset_index(
            drop=True
        )
        report_df = report_df.drop(columns=["_sort_key"])
    else:
        # Empty DataFrame with fixed schema (including broker_meta and schema_version columns)
        report_df = pd.DataFrame(
            columns=[
                "type",
                "symbol",
                "ledger_value",
                "broker_value",
                "diff",
                "match",
                "broker_view_source",
                "broker_snapshot_run_id",
                "broker_snapshot_date",
                "broker_snapshot_path",
                "schema_version",
            ]
        )

    # Always add broker_meta columns (fixed schema to prevent BI/ETL schema drift)
    # If broker_meta is None, use empty strings (consistent with CSV serialization)
    if broker_meta is not None:
        report_df["broker_view_source"] = broker_meta.get("broker_view_source")
        report_df["broker_snapshot_run_id"] = broker_meta.get("broker_snapshot_run_id")
        report_df["broker_snapshot_date"] = broker_meta.get("broker_snapshot_date")
        report_df["broker_snapshot_path"] = broker_meta.get("broker_snapshot_path")
    else:
        # Empty values for fixed schema (use empty string for consistency with CSV)
        report_df["broker_view_source"] = ""
        report_df["broker_snapshot_run_id"] = ""
        report_df["broker_snapshot_date"] = ""
        report_df["broker_snapshot_path"] = ""

    # Add schema_version as constant column (for long-term schema evolution)
    report_df["schema_version"] = 1

    # Ensure fixed column order for stability
    fixed_columns = [
        "type",
        "symbol",
        "ledger_value",
        "broker_value",
        "diff",
        "match",
        "broker_view_source",
        "broker_snapshot_run_id",
        "broker_snapshot_date",
        "broker_snapshot_path",
        "schema_version",
    ]
    report_df = report_df[fixed_columns]

    # Write CSV
    report_df.to_csv(csv_path, index=False, encoding="utf-8")

    logger.info(f"Reconciliation report CSV written to {csv_path}")

    return csv_path


def write_reconcile_report_json(
    result_dict: dict,
    output_dir: Path | str,
    run_id: str,
    as_of: pd.Timestamp | str,
    *,
    ledger_cash: float | None = None,
    broker_cash: float | None = None,
    broker_meta: dict[str, Any] | None = None,
) -> Path:
    """Write reconciliation report to JSON file.

    Args:
        result_dict: Reconciliation result dictionary from reconcile_ledger_vs_broker()
        output_dir: Base output directory
        run_id: Run identifier
        as_of: Reconciliation date (UTC, tz-aware)

    Returns:
        Path to written JSON file

    Raises:
        ValueError: If result_dict is missing required keys
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
    reconcile_dir = output_path / f"reconcile_report_{run_id}"
    reconcile_dir.mkdir(parents=True, exist_ok=True)

    # Build JSON file path
    json_path = reconcile_dir / f"reconcile_{date_str}.json"

    # Get cash values (prefer function params, fallback to result_dict)
    if ledger_cash is None:
        ledger_cash = result_dict.get("ledger_cash")
    if broker_cash is None:
        broker_cash = result_dict.get("broker_cash")

    # Build report dictionary
    report_dict = {
        "schema_version": 1,
        "reconciliation_date": as_of.isoformat(),
        "run_id": run_id,
        "ok": result_dict.get("ok", False),
        "cash": {
            "match": result_dict.get("cash_match", False),
            "ledger_cash": ledger_cash,
            "broker_cash": broker_cash,
            "diff": result_dict.get("cash_diff", None),
        },
        "positions": {
            "match": len(result_dict.get("position_diffs_df", pd.DataFrame())) == 0
            and len(result_dict.get("missing_in_ledger", [])) == 0
            and len(result_dict.get("missing_in_broker", [])) == 0,
            "n_diffs": len(result_dict.get("position_diffs_df", pd.DataFrame())),
            "n_missing_in_ledger": len(result_dict.get("missing_in_ledger", [])),
            "n_missing_in_broker": len(result_dict.get("missing_in_broker", [])),
        },
        "position_diffs": (
            result_dict.get("position_diffs_df", pd.DataFrame()).to_dict(
                orient="records"
            )
            if not result_dict.get("position_diffs_df", pd.DataFrame()).empty
            else []
        ),
        "missing_in_ledger": result_dict.get("missing_in_ledger", []),
        "missing_in_broker": result_dict.get("missing_in_broker", []),
        "message": result_dict.get("message", ""),
    }

    # Add broker_meta if provided
    if broker_meta is not None:
        report_dict["broker_meta"] = broker_meta

    # Serialize with NaN/Inf handling
    report_serialized = _json_serialize_nan(report_dict)

    # Write JSON (deterministic: sort_keys=True, indent=2, trailing newline)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_serialized, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info(f"Reconciliation report JSON written to {json_path}")

    return json_path


def write_reconcile_report_md(
    result_dict: dict,
    output_dir: Path | str,
    run_id: str,
    as_of: pd.Timestamp | str,
    *,
    ledger_cash: float | None = None,
    broker_cash: float | None = None,
    broker_meta: dict[str, Any] | None = None,
) -> Path:
    """Write reconciliation report to Markdown file (optional, human-readable).

    Args:
        result_dict: Reconciliation result dictionary from reconcile_ledger_vs_broker()
        output_dir: Base output directory
        run_id: Run identifier
        as_of: Reconciliation date (UTC, tz-aware)

    Returns:
        Path to written Markdown file
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
    reconcile_dir = output_path / f"reconcile_report_{run_id}"
    reconcile_dir.mkdir(parents=True, exist_ok=True)

    # Build Markdown file path
    md_path = reconcile_dir / f"reconcile_{date_str}.md"

    # Build report content
    lines = []
    lines.append("# Reconciliation Report")
    lines.append("")
    lines.append(f"**Date:** {date_str}")
    lines.append(f"**Run ID:** {run_id}")
    lines.append("")

    # Overall status
    ok = result_dict.get("ok", False)
    status = "[PASS]" if ok else "[FAIL]"
    lines.append(f"**Status:** {status}")
    lines.append("")

    # Broker source information
    if broker_meta is not None:
        source = broker_meta.get("broker_view_source", "unknown")
        snapshot_run_id = broker_meta.get("broker_snapshot_run_id")
        snapshot_date = broker_meta.get("broker_snapshot_date")
        snapshot_path = broker_meta.get("broker_snapshot_path")

        lines.append("## Broker Source")
        lines.append("")
        lines.append(f"- **Source:** {source}")
        if snapshot_run_id:
            lines.append(f"- **Snapshot Run ID:** {snapshot_run_id}")
        if snapshot_date:
            lines.append(f"- **Snapshot Date:** {snapshot_date}")
        if snapshot_path:
            lines.append(f"- **Snapshot Path:** {snapshot_path}")
        lines.append("")

    # Cash section
    lines.append("## Cash Reconciliation")
    lines.append("")
    cash_match = result_dict.get("cash_match", False)
    cash_diff = result_dict.get("cash_diff", 0.0)

    # Get cash values (prefer function params, fallback to result_dict)
    if ledger_cash is None:
        ledger_cash = result_dict.get("ledger_cash")
    if broker_cash is None:
        broker_cash = result_dict.get("broker_cash")

    if cash_match:
        lines.append(f"- ✅ Cash match: {ledger_cash:.2f} == {broker_cash:.2f}")
    else:
        lines.append(
            f"- ❌ Cash mismatch: ledger={ledger_cash:.2f}, broker={broker_cash:.2f}, diff={cash_diff:.2f}"
        )
    lines.append("")

    # Positions section
    lines.append("## Position Reconciliation")
    lines.append("")

    position_diffs_df = result_dict.get("position_diffs_df", pd.DataFrame())
    missing_in_ledger = result_dict.get("missing_in_ledger", [])
    missing_in_broker = result_dict.get("missing_in_broker", [])

    if (
        len(position_diffs_df) == 0
        and len(missing_in_ledger) == 0
        and len(missing_in_broker) == 0
    ):
        lines.append("- ✅ All positions match")
    else:
        if len(position_diffs_df) > 0:
            lines.append(
                f"- ❌ **{len(position_diffs_df)} position qty mismatch(es):**"
            )
            lines.append("")
            lines.append("| Symbol | Ledger Qty | Broker Qty | Diff |")
            lines.append("|--------|------------|------------|------|")
            for _, row in position_diffs_df.iterrows():
                lines.append(
                    f"| {row['symbol']} | {row['ledger_qty']:.2f} | {row['broker_qty']:.2f} | {row['diff_qty']:.2f} |"
                )
            lines.append("")

        if len(missing_in_ledger) > 0:
            lines.append(
                f"- ❌ **{len(missing_in_ledger)} symbol(s) missing in ledger:** {', '.join(missing_in_ledger[:10])}"
            )
            if len(missing_in_ledger) > 10:
                lines.append(f"  (showing first 10 of {len(missing_in_ledger)})")
            lines.append("")

        if len(missing_in_broker) > 0:
            lines.append(
                f"- ❌ **{len(missing_in_broker)} symbol(s) missing in broker:** {', '.join(missing_in_broker[:10])}"
            )
            if len(missing_in_broker) > 10:
                lines.append(f"  (showing first 10 of {len(missing_in_broker)})")
            lines.append("")

    # Message
    message = result_dict.get("message", "")
    if message:
        lines.append("## Summary")
        lines.append("")
        lines.append(message)
        lines.append("")

    # Write Markdown
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Reconciliation report Markdown written to {md_path}")

    return md_path
