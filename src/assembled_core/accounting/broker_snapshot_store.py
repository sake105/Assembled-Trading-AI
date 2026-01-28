"""Broker snapshot storage (Sprint 13).

This module provides functions to store and load broker snapshots (cash + positions)
in JSON and optional Parquet format.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def broker_snapshot_base_path(output_dir: Path, run_id: str) -> Path:
    """Get base path for broker snapshot storage.

    Args:
        output_dir: Base output directory
        run_id: Run identifier

    Returns:
        Path to broker snapshot directory
    """
    return output_dir / f"broker_snapshot_{run_id}"


def store_broker_snapshot_json(
    cash: float,
    positions_df: pd.DataFrame,
    output_dir: Path,
    run_id: str,
    as_of_date: pd.Timestamp | str,
) -> Path:
    """Store broker snapshot as JSON (atomic write, Windows-safe).

    Args:
        cash: Broker cash balance
        positions_df: DataFrame with columns: symbol, qty
        output_dir: Base output directory
        run_id: Run identifier
        as_of_date: Snapshot date (YYYY-MM-DD format in filename)

    Returns:
        Path to stored JSON file
    """
    # Normalize as_of_date
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date, utc=True)
    if as_of_date.tz is None:
        as_of_date = as_of_date.tz_localize("UTC")
    date_str = as_of_date.strftime("%Y-%m-%d")

    # Create snapshot directory
    snapshot_dir = broker_snapshot_base_path(output_dir, run_id)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Build snapshot dict
    snapshot = {
        # Schema version to allow future upgrades / migrations
        "schema_version": 1,
        # Store date as YYYY-MM-DD (date-only) for stability
        "as_of_date": date_str,
        "cash": cash,
        "positions": positions_df.to_dict(orient="records"),
    }

    # Write JSON deterministically (sort_keys=True, indent=2)
    snapshot_path = snapshot_dir / f"snapshot_{date_str}.json"
    
    # Atomic write (Windows-safe)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=snapshot_path.parent,
        delete=False,
        suffix=".tmp.json",
    ) as tmp_file:
        json.dump(snapshot, tmp_file, sort_keys=True, indent=2, default=str)
        tmp_path = Path(tmp_file.name)
    
    # Atomic rename (Windows-safe)
    tmp_path.replace(snapshot_path)
    
    logger.info(f"Stored broker snapshot JSON: {snapshot_path}")
    return snapshot_path


def store_broker_snapshot_parquet(
    positions_df: pd.DataFrame,
    output_dir: Path,
    run_id: str,
    as_of_date: pd.Timestamp | str,
) -> Path | None:
    """Store broker snapshot positions as Parquet (optional, atomic write).

    Args:
        positions_df: DataFrame with columns: symbol, qty
        output_dir: Base output directory
        run_id: Run identifier
        as_of_date: Snapshot date (YYYY-MM-DD format in filename)

    Returns:
        Path to stored Parquet file, or None if positions_df is empty
    """
    if positions_df.empty:
        return None

    # Normalize as_of_date
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date, utc=True)
    if as_of_date.tz is None:
        as_of_date = as_of_date.tz_localize("UTC")
    date_str = as_of_date.strftime("%Y-%m-%d")

    # Create snapshot directory
    snapshot_dir = broker_snapshot_base_path(output_dir, run_id)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Add as_of_date column
    positions_with_date = positions_df.copy()
    positions_with_date["as_of_date"] = as_of_date

    # Atomic write (Windows-safe)
    positions_path = snapshot_dir / f"positions_{date_str}.parquet"
    
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=positions_path.parent,
        delete=False,
        suffix=".tmp.parquet",
    ) as tmp_file:
        positions_with_date.to_parquet(tmp_file.name, index=False)
        tmp_path = Path(tmp_file.name)
    
    # Atomic rename (Windows-safe)
    tmp_path.replace(positions_path)
    
    logger.info(f"Stored broker snapshot Parquet: {positions_path}")
    return positions_path


def load_broker_snapshot_json(
    output_dir: Path,
    run_id: str,
    as_of_date: pd.Timestamp | str,
) -> dict[str, Any] | None:
    """Load broker snapshot from JSON.

    Args:
        output_dir: Base output directory
        run_id: Run identifier
        as_of_date: Snapshot date (YYYY-MM-DD format in filename)

    Returns:
        Dictionary with keys: as_of_date, cash, positions (list of dicts)
        Returns None if file does not exist
    """
    # Normalize as_of_date
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date, utc=True)
    if as_of_date.tz is None:
        as_of_date = as_of_date.tz_localize("UTC")
    date_str = as_of_date.strftime("%Y-%m-%d")

    snapshot_path = broker_snapshot_base_path(output_dir, run_id) / f"snapshot_{date_str}.json"

    if not snapshot_path.exists():
        # Caller decides whether this is fatal (e.g. policy=require) or a fallback case.
        logger.debug(
            "Broker snapshot JSON not found for run_id=%s, date=%s (expected path: %s)",
            run_id,
            date_str,
            snapshot_path,
        )
        return None

    try:
        with snapshot_path.open("r", encoding="utf-8") as f:
            snapshot = json.load(f)
    except json.JSONDecodeError as exc:
        # Provide clear ASCII-only context for ops/logging.
        raise ValueError(
            f"Failed to parse broker snapshot JSON at {snapshot_path}"
        ) from exc

    # Schema version handling (forward-compatible)
    schema_version = snapshot.get("schema_version", 1)
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError(
            f"Invalid schema_version in broker snapshot JSON at {snapshot_path}: {schema_version}"
        )

    return snapshot


def load_broker_snapshot_parquet(
    output_dir: Path,
    run_id: str,
    as_of_date: pd.Timestamp | str,
) -> pd.DataFrame | None:
    """Load broker snapshot positions from Parquet.

    Args:
        output_dir: Base output directory
        run_id: Run identifier
        as_of_date: Snapshot date (YYYY-MM-DD format in filename)

    Returns:
        DataFrame with columns: symbol, qty, as_of_date
        Returns None if file does not exist
    """
    # Normalize as_of_date
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date, utc=True)
    if as_of_date.tz is None:
        as_of_date = as_of_date.tz_localize("UTC")
    date_str = as_of_date.strftime("%Y-%m-%d")

    positions_path = broker_snapshot_base_path(output_dir, run_id) / f"positions_{date_str}.parquet"

    if not positions_path.exists():
        logger.debug(
            "Broker snapshot Parquet not found for run_id=%s, date=%s (expected path: %s)",
            run_id,
            date_str,
            positions_path,
        )
        return None

    try:
        return pd.read_parquet(positions_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read broker snapshot Parquet at {positions_path}"
        ) from exc
