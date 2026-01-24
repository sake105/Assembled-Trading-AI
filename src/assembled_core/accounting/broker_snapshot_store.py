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
        "as_of_date": as_of_date.isoformat(),
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
        return None
    
    with snapshot_path.open("r", encoding="utf-8") as f:
        snapshot = json.load(f)
    
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
        return None
    
    return pd.read_parquet(positions_path)
