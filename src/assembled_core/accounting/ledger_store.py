"""Ledger storage module (Sprint 13).

Provides parquet-based storage for ledger events with atomic writes and deterministic sorting.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
from typing import Literal

import pandas as pd

from src.assembled_core.accounting.ledger import REQUIRED_COLUMNS


def ledger_base_path(output_dir: Path | str, run_id: str) -> Path:
    """Get base path for ledger storage.

    Format: {output_dir}/ledger_{run_id}/

    Args:
        output_dir: Base output directory
        run_id: Run identifier

    Returns:
        Path to ledger directory
    """
    output_path = Path(output_dir)
    ledger_dir = output_path / f"ledger_{run_id}"
    return ledger_dir


def store_ledger_events_parquet(
    events_df: pd.DataFrame,
    output_dir: Path | str,
    run_id: str,
    *,
    mode: Literal["append", "replace"] = "append",
) -> Path:
    """Store ledger events to parquet file (atomic write, deterministic).

    This function:
    1. Validates events DataFrame (required columns)
    2. Loads existing events if mode="append"
    3. Merges and deduplicates by event_id
    4. Sorts deterministically (event_ts, event_id)
    5. Writes atomically (tmp -> rename, Windows-safe)

    Args:
        events_df: Ledger events DataFrame (must have REQUIRED_COLUMNS)
        output_dir: Base output directory
        run_id: Run identifier
        mode: Storage mode ("append" or "replace", default: "append")

    Returns:
        Path to stored parquet file

    Raises:
        ValueError: If events_df is empty or missing required columns
    """
    if events_df.empty:
        raise ValueError("Cannot store empty events DataFrame")

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in events_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in events_df: {missing_cols}")

    # Get ledger directory and file path
    ledger_dir = ledger_base_path(output_dir, run_id)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / "ledger_events.parquet"

    # Normalize events (ensure UTC timestamps, deterministic sort)
    events_normalized = events_df.copy()

    # Ensure event_ts is UTC-aware
    if "event_ts" in events_normalized.columns:
        events_normalized["event_ts"] = pd.to_datetime(
            events_normalized["event_ts"], utc=True
        )
        if events_normalized["event_ts"].dt.tz is None:
            events_normalized["event_ts"] = events_normalized[
                "event_ts"
            ].dt.tz_localize("UTC")

    # Load existing data if append mode
    if mode == "append" and ledger_path.exists():
        existing_df = pd.read_parquet(ledger_path)
        # Merge: concatenate
        combined = pd.concat([existing_df, events_normalized], ignore_index=True)
        # Deduplicate by event_id (keep last occurrence so corrections win)
        combined = combined.drop_duplicates(subset=["event_id"], keep="last")
        events_to_store = combined
    else:
        events_to_store = events_normalized

    # Deterministic sort: event_ts, then event_id
    events_to_store = events_to_store.sort_values(
        ["event_ts", "event_id"],
        kind="mergesort",
        ignore_index=True,
    )

    # Atomic write: write to temp file, then rename (Windows-safe)
    tmp_path = ledger_path.with_suffix(".tmp.parquet")

    try:
        events_to_store.to_parquet(tmp_path, index=False, engine="pyarrow")
        # Atomic rename: prefer pathlib.replace (os.replace internally, atomic on same volume)
        try:
            tmp_path.replace(ledger_path)
        except OSError:
            # Cross-volume fallback (different drive letters on Windows)
            logger.warning(
                "[LedgerStore] os.replace failed, falling back to shutil.move for %s",
                ledger_path,
            )
            shutil.move(str(tmp_path), str(ledger_path))
    except Exception:
        # Clean up temp file on error
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise

    return ledger_path


def load_ledger_events_parquet(
    output_dir: Path | str,
    run_id: str,
) -> pd.DataFrame:
    """Load ledger events from parquet file.

    Args:
        output_dir: Base output directory
        run_id: Run identifier

    Returns:
        Ledger events DataFrame (empty if file doesn't exist)

    Raises:
        FileNotFoundError: If ledger file doesn't exist (optional, can return empty DataFrame)
    """
    ledger_dir = ledger_base_path(output_dir, run_id)
    ledger_path = ledger_dir / "ledger_events.parquet"

    if not ledger_path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    events_df = pd.read_parquet(ledger_path)

    # Ensure event_ts is UTC-aware
    if "event_ts" in events_df.columns:
        events_df["event_ts"] = pd.to_datetime(events_df["event_ts"], utc=True)
        if events_df["event_ts"].dt.tz is None:
            events_df["event_ts"] = events_df["event_ts"].dt.tz_localize("UTC")

    return events_df


def store_daily_snapshot_parquet(
    snapshot_df: pd.DataFrame,
    output_dir: Path | str,
    run_id: str,
    as_of_date: pd.Timestamp | str,
) -> Path:
    """Store daily position/cash snapshot to parquet file.

    Args:
        snapshot_df: Snapshot DataFrame (columns: symbol, qty, avg_cost_basis, etc.)
        output_dir: Base output directory
        run_id: Run identifier
        as_of_date: Snapshot date (UTC, tz-aware)

    Returns:
        Path to stored parquet file
    """
    if snapshot_df.empty:
        raise ValueError("Cannot store empty snapshot DataFrame")

    # Normalize as_of_date
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date, utc=True)
    if as_of_date.tz is None:
        as_of_date = as_of_date.tz_localize("UTC")

    # Get ledger directory and snapshot path
    ledger_dir = ledger_base_path(output_dir, run_id)
    ledger_dir.mkdir(parents=True, exist_ok=True)

    # Format date as YYYYMMDD for filename
    date_str = as_of_date.strftime("%Y%m%d")
    snapshot_path = ledger_dir / f"positions_snapshot_{date_str}.parquet"

    # Atomic write
    tmp_path = snapshot_path.with_suffix(".tmp.parquet")

    try:
        snapshot_df.to_parquet(tmp_path, index=False, engine="pyarrow")
        try:
            tmp_path.replace(snapshot_path)
        except OSError:
            logger.warning(
                "[LedgerStore] os.replace failed, falling back to shutil.move for %s",
                snapshot_path,
            )
            shutil.move(str(tmp_path), str(snapshot_path))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise

    return snapshot_path


def list_ledger_runs(output_dir: Path | str) -> list[str]:
    """List all ledger run IDs in output directory.

    Args:
        output_dir: Base output directory

    Returns:
        List of run IDs (sorted)
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    # Find all ledger_* directories
    ledger_dirs = [
        d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("ledger_")
    ]

    # Extract run IDs
    run_ids = []
    for ledger_dir in ledger_dirs:
        # Format: ledger_{run_id}
        if ledger_dir.name.startswith("ledger_"):
            run_id = ledger_dir.name[7:]  # Remove "ledger_" prefix
            if run_id:
                run_ids.append(run_id)

    # Sort deterministically
    run_ids.sort()

    return run_ids
