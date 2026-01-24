"""Reconciliation engine: Compare ledger state vs broker snapshots (Sprint 13 L3).

This module provides functions to reconcile ledger positions/cash against
broker snapshots (paper or live), detecting mismatches and missing positions.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def reconcile_ledger_vs_broker(
    ledger_positions_df: pd.DataFrame,
    ledger_cash: float,
    broker_positions_df: pd.DataFrame,
    broker_cash: float,
    *,
    cash_tol: float = 1e-6,
    qty_tol: float = 1e-8,
    fail_fast: bool = False,
) -> dict:
    """Reconcile ledger state vs broker snapshot.

    This function compares:
    1. Cash balances (within cash_tol tolerance)
    2. Positions (symbol-by-symbol, within qty_tol tolerance)
    3. Missing positions (in ledger but not broker, or vice versa)

    Args:
        ledger_positions_df: Ledger positions DataFrame with columns: symbol, qty
            (may have additional columns like avg_price, realized_pnl, etc.)
        ledger_cash: Ledger cash balance (float)
        broker_positions_df: Broker positions DataFrame with columns: symbol, qty
            (may have additional columns)
        broker_cash: Broker cash balance (float)
        cash_tol: Cash tolerance (default: 1e-6)
            Differences <= cash_tol are treated as zero
        qty_tol: Quantity tolerance (default: 1e-8)
            Differences <= qty_tol are treated as zero
        fail_fast: If True, raise ValueError on mismatch (default: False)
            If False, return reconciliation report with ok=False

    Returns:
        Dictionary with keys:
        - ok: bool (True if all checks pass)
        - cash_match: bool (True if cash within tolerance)
        - cash_diff: float (ledger_cash - broker_cash)
        - position_diffs_df: DataFrame with columns:
            - symbol: str
            - ledger_qty: float
            - broker_qty: float
            - diff_qty: float (ledger_qty - broker_qty)
            Only includes symbols with differences > qty_tol
        - missing_in_ledger: list[str] (symbols in broker but not ledger)
        - missing_in_broker: list[str] (symbols in ledger but not broker)
        - message: str (human-readable summary)

    Raises:
        ValueError: If fail_fast=True and mismatch detected
        ValueError: If required columns (symbol, qty) are missing
    """
    # Validate inputs
    if ledger_positions_df.empty:
        ledger_positions_df = pd.DataFrame(columns=["symbol", "qty"])
    if broker_positions_df.empty:
        broker_positions_df = pd.DataFrame(columns=["symbol", "qty"])

    # Validate required columns
    required_cols = ["symbol", "qty"]
    missing_ledger = [col for col in required_cols if col not in ledger_positions_df.columns]
    missing_broker = [col for col in required_cols if col not in broker_positions_df.columns]
    if missing_ledger:
        raise ValueError(f"Missing required columns in ledger_positions_df: {missing_ledger}")
    if missing_broker:
        raise ValueError(f"Missing required columns in broker_positions_df: {missing_broker}")

    # Normalize positions: trim symbols, deterministic sort
    ledger_normalized = ledger_positions_df.copy()
    broker_normalized = broker_positions_df.copy()

    # Trim symbol strings
    ledger_normalized["symbol"] = ledger_normalized["symbol"].astype(str).str.strip()
    broker_normalized["symbol"] = broker_normalized["symbol"].astype(str).str.strip()

    # Ensure qty is float
    ledger_normalized["qty"] = ledger_normalized["qty"].astype(float)
    broker_normalized["qty"] = broker_normalized["qty"].astype(float)

    # Remove zero positions (threshold: qty_tol) - filter both sides consistently
    ledger_normalized = ledger_normalized[ledger_normalized["qty"].abs() > qty_tol].copy()
    broker_normalized = broker_normalized[broker_normalized["qty"].abs() > qty_tol].copy()

    # Deterministic sort by symbol
    ledger_normalized = ledger_normalized.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    broker_normalized = broker_normalized.sort_values("symbol", kind="mergesort").reset_index(drop=True)

    # Get symbol sets
    ledger_symbols = set(ledger_normalized["symbol"].unique())
    broker_symbols = set(broker_normalized["symbol"].unique())

    # Find missing symbols
    missing_in_ledger = sorted(list(broker_symbols - ledger_symbols))
    missing_in_broker = sorted(list(ledger_symbols - broker_symbols))

    # Check cash match
    cash_diff = ledger_cash - broker_cash
    cash_match = abs(cash_diff) <= cash_tol

    # Build position differences
    position_diffs_list = []

    # Compare common symbols
    common_symbols = sorted(list(ledger_symbols & broker_symbols))
    for symbol in common_symbols:
        ledger_row = ledger_normalized[ledger_normalized["symbol"] == symbol].iloc[0]
        broker_row = broker_normalized[broker_normalized["symbol"] == symbol].iloc[0]

        ledger_qty = float(ledger_row["qty"])
        broker_qty = float(broker_row["qty"])
        diff_qty = ledger_qty - broker_qty

        # Only include if difference exceeds tolerance
        if abs(diff_qty) > qty_tol:
            position_diffs_list.append({
                "symbol": symbol,
                "ledger_qty": ledger_qty,
                "broker_qty": broker_qty,
                "diff_qty": diff_qty,
            })

    # Build position_diffs_df
    if position_diffs_list:
        position_diffs_df = pd.DataFrame(position_diffs_list)
        position_diffs_df = position_diffs_df.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    else:
        position_diffs_df = pd.DataFrame(columns=["symbol", "ledger_qty", "broker_qty", "diff_qty"])

    # Determine overall match
    positions_match = len(position_diffs_list) == 0 and len(missing_in_ledger) == 0 and len(missing_in_broker) == 0
    ok = cash_match and positions_match

    # Build message (include key differences for fail_fast)
    message_parts = []
    if not cash_match:
        message_parts.append(f"Cash mismatch: diff={cash_diff:.6f} (ledger={ledger_cash:.6f}, broker={broker_cash:.6f})")
    if len(position_diffs_list) > 0:
        # Include all symbols with mismatches (not just first 5)
        mismatch_symbols = sorted([d["symbol"] for d in position_diffs_list])
        message_parts.append(f"Position qty mismatches: {len(position_diffs_list)} symbol(s): {mismatch_symbols}")
    if len(missing_in_ledger) > 0:
        # missing_in_ledger is already sorted
        message_parts.append(f"Missing in ledger: {len(missing_in_ledger)} symbol(s): {missing_in_ledger}")
    if len(missing_in_broker) > 0:
        # missing_in_broker is already sorted
        message_parts.append(f"Missing in broker: {len(missing_in_broker)} symbol(s): {missing_in_broker}")

    if ok:
        message = "Reconciliation OK: cash and positions match"
    else:
        message = "Reconciliation FAILED: " + "; ".join(message_parts)

    # Fail-fast if requested
    if fail_fast and not ok:
        raise ValueError(message)

    return {
        "ok": ok,
        "cash_match": cash_match,
        "cash_diff": cash_diff,
        "position_diffs_df": position_diffs_df,
        "missing_in_ledger": missing_in_ledger,
        "missing_in_broker": missing_in_broker,
        "message": message,
    }
