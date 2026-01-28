"""Broker snapshot normalization and contract (Sprint 13).

This module provides functions to normalize broker snapshots (cash + positions)
for reconciliation with ledger state.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_broker_snapshot(
    cash: float,
    positions_df: pd.DataFrame,
    *,
    qty_tol: float = 1e-8,
) -> dict[str, Any]:
    """Normalize broker snapshot for reconciliation.

    This function:
    1. Validates required columns (symbol, qty)
    2. Trims symbol strings
    3. Filters out tiny residual quantities (abs(qty) <= qty_tol)
    4. Sorts positions deterministically by symbol

    Args:
        cash: Broker cash balance
        positions_df: DataFrame with columns: symbol, qty
            (may have additional columns like avg_price, etc.)
        qty_tol: Quantity tolerance (default: 1e-8)
            Positions with abs(qty) <= qty_tol are filtered out

    Returns:
        Dictionary with:
        - cash: float (normalized cash balance)
        - positions_df: DataFrame (normalized positions, sorted by symbol)
        - metadata: dict (optional metadata from positions_df if present)

    Raises:
        ValueError: If required columns (symbol, qty) are missing
    """
    # Validate inputs
    if positions_df.empty:
        positions_df = pd.DataFrame(columns=["symbol", "qty"])

    # Validate required columns
    required_cols = ["symbol", "qty"]
    missing_cols = [col for col in required_cols if col not in positions_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in positions_df: {missing_cols}")

    # Normalize positions
    normalized = positions_df.copy()

    # Trim symbol strings
    normalized["symbol"] = normalized["symbol"].astype(str).str.strip()

    # Ensure qty is float
    normalized["qty"] = normalized["qty"].astype(float)

    # Aggregate duplicate symbols by summing qty deterministically
    # (drop all other columns to avoid ambiguous aggregation semantics)
    aggregated = (
        normalized[["symbol", "qty"]]
        .groupby("symbol", as_index=False, sort=False)["qty"]
        .sum()
    )

    # Remove zero / tiny positions (threshold: qty_tol)
    aggregated = aggregated[aggregated["qty"].abs() > qty_tol].copy()

    # Deterministic sort by symbol
    normalized = aggregated.sort_values("symbol", kind="mergesort").reset_index(drop=True)

    # Extract metadata (any additional columns beyond symbol, qty)
    metadata_cols = [col for col in positions_df.columns if col not in required_cols]
    metadata = {}
    if metadata_cols:
        # Store first row's metadata as example (or aggregate if needed)
        # For now, just note which columns are present
        metadata["additional_columns"] = metadata_cols

    return {
        "cash": float(cash),
        "positions_df": normalized,
        "metadata": metadata,
    }
