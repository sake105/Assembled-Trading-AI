"""Smoke test for run_daily.py broker snapshot controls (Sprint 13).

Tests that CLI flags are accepted and broker snapshot import works (if file provided).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_daily import run_daily_eod


def test_run_daily_with_broker_snapshot_file(tmp_path: Path):
    """Test that run_daily_eod accepts broker_snapshot_file and imports snapshot."""
    # Create external JSON snapshot
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 5.0},
        ],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    # Create minimal price data (required for run_daily_eod)
    price_data = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-15", tz="UTC"),
                "symbol": "AAPL",
                "close": 150.0,
                "open": 149.0,
                "high": 151.0,
                "low": 148.0,
                "volume": 1000000,
            },
        ]
    )

    # Write price file
    price_file = tmp_path / "prices.parquet"
    price_data.to_parquet(price_file)

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call run_daily_eod with broker_snapshot_file
    # Note: This will fail if universe/price data is missing, but we test that flags are accepted
    try:
        _ = run_daily_eod(
            date_str="2025-01-15",
            price_file=price_file,
            output_dir=output_dir,
            total_capital=10000.0,
            broker_snapshot_file=external_path,
            broker_snapshot_date="2025-01-15",
            broker_snapshot_policy="prefer",
        )

        # Verify snapshot was imported (check output directory)
        snapshot_dir = output_dir / "broker_snapshot_daily_snapshot"
        assert (
            snapshot_dir.exists()
        ), "Broker snapshot directory should exist after import"

        snapshot_json = snapshot_dir / "snapshot_2025-01-15.json"
        assert snapshot_json.exists(), "Broker snapshot JSON should exist after import"

    except (FileNotFoundError, ValueError):
        # Expected if universe/price data is incomplete, but import should have been attempted
        # Check that snapshot was imported before any other errors
        snapshot_dir = output_dir / "broker_snapshot_daily_snapshot"
        if snapshot_dir.exists():
            # Import succeeded, other errors are expected (incomplete test setup)
            pass
        else:
            # Import failed or wasn't attempted - this is unexpected
            raise


def test_run_daily_broker_snapshot_policy_require_missing_file(tmp_path: Path):
    """Test that policy=require fails fast when snapshot file is missing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal price data
    price_data = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-15", tz="UTC"),
                "symbol": "AAPL",
                "close": 150.0,
                "open": 149.0,
                "high": 151.0,
                "low": 148.0,
                "volume": 1000000,
            },
        ]
    )

    price_file = tmp_path / "prices.parquet"
    price_data.to_parquet(price_file)

    # Call with policy=require and missing file
    import sys
    from io import StringIO

    # Capture stderr to check error message
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        _ = run_daily_eod(
            date_str="2025-01-15",
            price_file=price_file,
            output_dir=output_dir,
            total_capital=10000.0,
            broker_snapshot_file=tmp_path / "nonexistent.json",
            broker_snapshot_policy="require",
        )
        # Should not reach here (should exit with sys.exit(1))
        assert False, "Expected sys.exit(1) when policy=require and file missing"
    except SystemExit as e:
        # Expected: sys.exit(1) when policy=require and file missing
        assert e.code == 1, f"Expected exit code 1, got {e.code}"
    finally:
        sys.stderr = old_stderr
