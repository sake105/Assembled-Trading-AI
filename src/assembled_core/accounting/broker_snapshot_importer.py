"""Broker snapshot importer (Sprint 13).

This module provides functions to import external broker snapshots (JSON/CSV)
and store them in the standardized snapshot format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.assembled_core.accounting.broker_snapshot import normalize_broker_snapshot
from src.assembled_core.accounting.broker_snapshot_store import (
    store_broker_snapshot_json,
    store_broker_snapshot_parquet,
)

logger = logging.getLogger(__name__)


def load_external_broker_snapshot(
    path: Path | str,
) -> tuple[float | None, pd.DataFrame]:
    """Load external broker snapshot from JSON or CSV file.

    Args:
        path: Path to snapshot file (JSON or CSV)

    Returns:
        Tuple of (cash, positions_df):
        - cash: float | None (cash balance, None if not provided)
        - positions_df: DataFrame with columns: symbol, qty

    Raises:
        ValueError: If file format is unsupported or schema is invalid
        FileNotFoundError: If file does not exist
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path_obj}")

    suffix = path_obj.suffix.lower()

    if suffix == ".json":
        return _load_external_broker_snapshot_json(path_obj)
    elif suffix == ".csv":
        return _load_external_broker_snapshot_csv(path_obj)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats: .json, .csv"
        )


def _load_external_broker_snapshot_json(path: Path) -> tuple[float | None, pd.DataFrame]:
    """Load broker snapshot from JSON file.

    Expected JSON schema:
    {
        "as_of": "YYYY-MM-DD" (optional),
        "cash": 123.4 (optional),
        "positions": [
            {"symbol": "AAPL", "qty": 1.0},
            ...
        ]
    }

    Args:
        path: Path to JSON file

    Returns:
        Tuple of (cash, positions_df)

    Raises:
        ValueError: If JSON schema is invalid
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in snapshot file {path}: {e}") from e

    # Extract cash (optional)
    cash = data.get("cash")
    if cash is not None:
        try:
            cash = float(cash)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid cash value in snapshot file {path}: {cash}") from e

    # Extract positions (required)
    if "positions" not in data:
        raise ValueError(f"Missing required field 'positions' in snapshot file {path}")

    positions_list = data["positions"]
    if not isinstance(positions_list, list):
        raise ValueError(
            f"Field 'positions' must be a list in snapshot file {path}, got {type(positions_list)}"
        )

    if len(positions_list) == 0:
        # Empty positions list is valid
        positions_df = pd.DataFrame(columns=["symbol", "qty"])
    else:
        # Validate each position entry
        validated_positions = []
        for idx, pos in enumerate(positions_list):
            if not isinstance(pos, dict):
                raise ValueError(
                    f"Position entry {idx} in snapshot file {path} must be a dict, got {type(pos)}"
                )
            if "symbol" not in pos:
                raise ValueError(
                    f"Position entry {idx} in snapshot file {path} missing required field 'symbol'"
                )
            if "qty" not in pos:
                raise ValueError(
                    f"Position entry {idx} in snapshot file {path} missing required field 'qty'"
                )

            try:
                qty = float(pos["qty"])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid qty value in position entry {idx} in snapshot file {path}: {pos['qty']}"
                ) from e

            validated_positions.append(
                {
                    "symbol": str(pos["symbol"]),
                    "qty": qty,
                }
            )

        positions_df = pd.DataFrame(validated_positions)

    return cash, positions_df


def _load_external_broker_snapshot_csv(path: Path) -> tuple[float | None, pd.DataFrame]:
    """Load broker snapshot from CSV file.

    Expected CSV schema:
    - Required columns: symbol, qty
    - Optional: cash (as separate column or single value in first row)

    Args:
        path: Path to CSV file

    Returns:
        Tuple of (cash, positions_df)

    Raises:
        ValueError: If CSV schema is invalid
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {path}: {e}") from e

    # Validate required columns
    required_cols = ["symbol", "qty"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in CSV file {path}: {missing_cols}. Found columns: {list(df.columns)}"
        )

    # Extract cash (optional, if column exists)
    cash = None
    if "cash" in df.columns:
        # Use first non-null cash value
        cash_values = df["cash"].dropna()
        if len(cash_values) > 0:
            try:
                cash = float(cash_values.iloc[0])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid cash value in CSV file {path}: {cash_values.iloc[0]}"
                ) from e

    # Extract positions
    positions_df = df[["symbol", "qty"]].copy()

    return cash, positions_df


def import_broker_snapshot(
    snapshot_path: Path | str,
    run_id: str,
    snapshot_date: pd.Timestamp | str,
    output_dir: Path | str,
    *,
    qty_tol: float = 1e-8,
    store_parquet: bool = True,
    cash_override: float | None = None,
) -> dict[str, Any]:
    """Import external broker snapshot and store in standardized format.

    This function:
    1. Loads snapshot from external file (JSON/CSV)
    2. Normalizes positions (trim, filter tiny residuals, sort)
    3. Stores in standardized layout: output/broker_snapshot_<run_id>/snapshot_<YYYY-MM-DD>.json

    Args:
        snapshot_path: Path to external snapshot file (JSON or CSV)
        run_id: Run identifier for storage namespace
        snapshot_date: Snapshot date (YYYY-MM-DD format in filename)
        output_dir: Base output directory
        qty_tol: Quantity tolerance for filtering tiny residuals (default: 1e-8)
        store_parquet: If True, also store positions as Parquet (default: True)
        cash_override: Optional cash value to override value from file (default: None)

    Returns:
        Dictionary with:
        - broker_snapshot_path: Path to stored JSON snapshot (relative to output_dir)
        - broker_positions_path: Path to stored Parquet snapshot (relative, or None if not stored)
        - cash: Final cash value used (float or None)

    Raises:
        ValueError: If snapshot file format is invalid or schema is invalid
        FileNotFoundError: If snapshot file does not exist
    """
    # Normalize paths
    snapshot_path_obj = Path(snapshot_path)
    output_dir_obj = Path(output_dir)

    # Normalize snapshot_date
    if isinstance(snapshot_date, str):
        snapshot_date = pd.to_datetime(snapshot_date, utc=True)
    if snapshot_date.tz is None:
        snapshot_date = snapshot_date.tz_localize("UTC")

    # Load external snapshot
    logger.info(f"Loading external broker snapshot from: {snapshot_path_obj}")
    cash, positions_df = load_external_broker_snapshot(snapshot_path_obj)

    # Apply cash override if provided
    if cash_override is not None:
        cash = cash_override
        logger.info(f"Using cash override: {cash}")

    # Normalize snapshot (trim, filter, sort)
    if cash is None:
        # Use 0.0 as default cash if not provided
        cash = 0.0
        logger.warning("Cash not provided in snapshot, using default: 0.0")

    normalized = normalize_broker_snapshot(
        cash=cash,
        positions_df=positions_df,
        qty_tol=qty_tol,
    )

    normalized_cash = normalized["cash"]
    normalized_positions_df = normalized["positions_df"]

    logger.info(
        f"Normalized snapshot: cash={normalized_cash}, positions={len(normalized_positions_df)}"
    )

    # Store JSON snapshot
    json_path = store_broker_snapshot_json(
        cash=normalized_cash,
        positions_df=normalized_positions_df,
        output_dir=output_dir_obj,
        run_id=run_id,
        as_of_date=snapshot_date,
    )

    # Store Parquet snapshot (optional)
    parquet_path = None
    if store_parquet and not normalized_positions_df.empty:
        parquet_path = store_broker_snapshot_parquet(
            positions_df=normalized_positions_df,
            output_dir=output_dir_obj,
            run_id=run_id,
            as_of_date=snapshot_date,
        )

    # Return paths relative to output_dir
    broker_snapshot_path = json_path.relative_to(output_dir_obj)
    broker_positions_path = parquet_path.relative_to(output_dir_obj) if parquet_path else None

    logger.info(f"Imported broker snapshot: {broker_snapshot_path}")

    return {
        "broker_snapshot_path": str(broker_snapshot_path),
        "broker_positions_path": str(broker_positions_path) if broker_positions_path else None,
        "cash": normalized_cash,
    }
