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


def _normalize_symbol(sym: Any) -> str:
    """Normalize symbol string (strip, collapse internal whitespace).

    This helper is intentionally strict but ASCII-safe:
    - Converts value to string
    - Strips leading/trailing whitespace
    - Collapses multiple internal whitespace characters into a single space
    """
    s = str(sym)
    s = s.strip()
    if not s:
        return s
    # Collapse any internal whitespace (spaces, tabs, etc.) to a single space
    return " ".join(s.split())


def _parse_float_like(value: Any) -> float | None:
    """Parse a 'float-like' value robustly.

    Rules (ASCII-only error messages):
    - None / NaN / empty string -> returns None
    - Numeric types -> float(value)
    - String with surrounding whitespace -> stripped before parsing
    - Thousands separators:
      - Patterns like "1,000" or "12,345.67" are allowed and parsed as 1000.0 / 12345.67
      - Other comma patterns (e.g. "1,2,3") raise ValueError
    - Parentheses notation:
      - "(5)" -> -5.0
      - " ( 1,000 ) " -> -1000.0
    """
    # Handle obvious None-like values up front
    if value is None:
        return None

    # Handle pandas NaN / NA by numeric check
    try:
        if isinstance(value, (int, float)):
            # pandas may pass NaN as float; treat as None
            if pd.isna(value):  # type: ignore[arg-type]
                return None
            return float(value)
    except Exception:
        # Fall through to string parsing
        pass

    # Work with string representation
    s = str(value).strip()
    if not s:
        return None

    # Parentheses notation for negatives e.g. "(5)" -> -5.0
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
        if not s:
            raise ValueError("Invalid numeric value: empty parentheses")

    # Handle thousands separators (commas)
    if "," in s:
        if "." in s:
            int_part, frac_part = s.split(".", 1)
        else:
            int_part, frac_part = s, None

        int_groups = int_part.split(",")
        # First group: 1-3 digits, remaining groups: exactly 3 digits
        if not int_groups[0].isdigit() or not (1 <= len(int_groups[0]) <= 3):
            raise ValueError("Invalid numeric value: bad thousands grouping")
        for grp in int_groups[1:]:
            if not (grp.isdigit() and len(grp) == 3):
                raise ValueError("Invalid numeric value: bad thousands grouping")

        int_clean = "".join(int_groups)
        if frac_part is not None:
            s_clean = int_clean + "." + frac_part
        else:
            s_clean = int_clean
    else:
        s_clean = s

    try:
        num = float(s_clean)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid numeric value") from exc

    if negative:
        num = -num
    return num


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


def _load_external_broker_snapshot_json(
    path: Path,
) -> tuple[float | None, pd.DataFrame]:
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

    # Extract cash (optional, may be string or numeric)
    raw_cash = data.get("cash")
    cash: float | None = None
    if raw_cash is not None:
        try:
            parsed = _parse_float_like(raw_cash)
        except ValueError as exc:
            # ASCII-only, include file path for context
            raise ValueError(f"Invalid cash value in snapshot file {path}") from exc
        cash = parsed

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

            # Normalize symbol (trim + collapse whitespace)
            symbol = _normalize_symbol(pos["symbol"])
            if not symbol:
                raise ValueError(
                    f"Position entry {idx} in snapshot file {path} has empty symbol after normalization"
                )

            # Robust qty parsing
            try:
                qty_parsed = _parse_float_like(pos["qty"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid qty value in position entry {idx} in snapshot file {path}"
                ) from exc

            if qty_parsed is None:
                raise ValueError(
                    f"Missing qty value in position entry {idx} in snapshot file {path}"
                )

            validated_positions.append(
                {
                    "symbol": symbol,
                    "qty": qty_parsed,
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
        # Read as objects to preserve raw strings for robust parsing
        df = pd.read_csv(path, dtype=object)
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
    cash: float | None = None
    if "cash" in df.columns:
        cash_series = df["cash"]
        for row_idx, raw in cash_series.items():
            if raw is None:
                continue
            s = str(raw).strip()
            if not s:
                continue
            try:
                parsed = _parse_float_like(raw)
            except ValueError as exc:
                # Include file path and row index for context
                raise ValueError(
                    f"Invalid cash value in CSV file {path} at row {row_idx}"
                ) from exc
            if parsed is not None:
                cash = parsed
                break

    # Extract and normalize positions
    positions_rows: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        raw_symbol = row.get("symbol")
        raw_qty = row.get("qty")

        symbol = _normalize_symbol(raw_symbol)
        if not symbol:
            raise ValueError(f"Empty symbol in CSV file {path} at row {row_idx}")

        try:
            qty_parsed = _parse_float_like(raw_qty)
        except ValueError as exc:
            raise ValueError(
                f"Invalid qty value in CSV file {path} at row {row_idx}"
            ) from exc

        if qty_parsed is None:
            raise ValueError(f"Missing qty value in CSV file {path} at row {row_idx}")

        positions_rows.append({"symbol": symbol, "qty": qty_parsed})

    positions_df = pd.DataFrame(positions_rows, columns=["symbol", "qty"])

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
        Dictionary (return schema stable):
        - broker_snapshot_path: Relative path to stored JSON snapshot (POSIX, relative to output_dir)
        - broker_positions_path: Relative path to stored Parquet (POSIX, or None if not stored)
        - cash: Final cash value used (float)

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

    # Return paths relative to output_dir (POSIX for portability)
    broker_snapshot_path = json_path.relative_to(output_dir_obj)
    broker_positions_path = (
        parquet_path.relative_to(output_dir_obj) if parquet_path else None
    )

    logger.info(f"Imported broker snapshot: {broker_snapshot_path}")

    return {
        "broker_snapshot_path": broker_snapshot_path.as_posix(),
        "broker_positions_path": (
            broker_positions_path.as_posix() if broker_positions_path else None
        ),
        "cash": normalized_cash,
    }
