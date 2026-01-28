"""Smoke tests for import_broker_snapshot.py CLI tool (Sprint 13).

Tests that the standalone CLI tool correctly imports broker snapshots.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.broker_snapshot_store import load_broker_snapshot_json


def test_cli_json_import_roundtrip(tmp_path: Path):
    """Test that JSON import via CLI creates files and roundtrip works."""
    # Create external JSON snapshot
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 100.0},
            {"symbol": "MSFT", "qty": 50.0},
        ],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    output_dir = tmp_path / "output"
    run_id = "test_cli_import"
    snapshot_date = "2025-01-15"

    # Run CLI import
    script_path = ROOT / "scripts" / "import_broker_snapshot.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(external_path),
            "--run-id",
            run_id,
            "--as-of-date",
            snapshot_date,
            "--output-dir",
            str(output_dir),
            "--store-parquet",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Verify files exist
    snapshot_dir = output_dir / f"broker_snapshot_{run_id}"
    assert snapshot_dir.exists(), "Snapshot directory should exist"

    snapshot_json = snapshot_dir / f"snapshot_{snapshot_date}.json"
    assert snapshot_json.exists(), "Snapshot JSON should exist"

    snapshot_parquet = snapshot_dir / f"positions_{snapshot_date}.parquet"
    assert snapshot_parquet.exists(), "Snapshot Parquet should exist (--store-parquet)"

    # Verify roundtrip: load the stored snapshot
    loaded = load_broker_snapshot_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=pd.Timestamp(snapshot_date, tz="UTC"),
    )

    assert loaded is not None
    assert loaded["cash"] == 10000.0
    assert len(loaded["positions"]) == 2


def test_cli_csv_import_with_cash(tmp_path: Path):
    """Test that CSV import with --cash flag sets cash correctly."""
    # Create external CSV snapshot
    external_path = tmp_path / "external_snapshot.csv"
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [100.0, 50.0],
    })
    df.to_csv(external_path, index=False)

    output_dir = tmp_path / "output"
    run_id = "test_cli_csv"
    snapshot_date = "2025-01-15"
    cash_value = 15000.0

    # Run CLI import with --cash
    script_path = ROOT / "scripts" / "import_broker_snapshot.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(external_path),
            "--run-id",
            run_id,
            "--as-of-date",
            snapshot_date,
            "--output-dir",
            str(output_dir),
            "--cash",
            str(cash_value),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Verify cash is set correctly
    loaded = load_broker_snapshot_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=pd.Timestamp(snapshot_date, tz="UTC"),
    )

    assert loaded is not None
    assert loaded["cash"] == cash_value


def test_cli_unsupported_format_exits_with_error(tmp_path: Path):
    """Test that unsupported file format exits with error code."""
    # Create unsupported file (text file, not JSON/CSV)
    external_path = tmp_path / "unsupported.txt"
    external_path.write_text("This is not a valid snapshot file")

    output_dir = tmp_path / "output"
    run_id = "test_cli_error"
    snapshot_date = "2025-01-15"

    # Run CLI import (should fail)
    script_path = ROOT / "scripts" / "import_broker_snapshot.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(external_path),
            "--run-id",
            run_id,
            "--as-of-date",
            snapshot_date,
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code is non-zero
    assert result.returncode != 0, "CLI should fail for unsupported format"

    # Verify error message is ASCII-only (no Unicode issues)
    assert all(ord(c) < 128 for c in result.stderr), "Error message should be ASCII-only"


def test_cli_missing_file_exits_with_error(tmp_path: Path):
    """Test that missing input file exits with error code."""
    output_dir = tmp_path / "output"
    run_id = "test_cli_missing"
    snapshot_date = "2025-01-15"
    nonexistent_path = tmp_path / "nonexistent.json"

    # Run CLI import (should fail)
    script_path = ROOT / "scripts" / "import_broker_snapshot.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(nonexistent_path),
            "--run-id",
            run_id,
            "--as-of-date",
            snapshot_date,
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code is non-zero
    assert result.returncode != 0, "CLI should fail for missing file"

    # Verify error message mentions file not found (check both stdout and stderr)
    output = (result.stdout + result.stderr).lower()
    assert "not found" in output or "file" in output


def test_cli_invalid_date_format_exits_with_error(tmp_path: Path):
    """Test that invalid date format exits with error code."""
    # Create valid JSON file
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [{"symbol": "AAPL", "qty": 100.0}],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    output_dir = tmp_path / "output"
    run_id = "test_cli_invalid_date"
    invalid_date = "2025-13-45"  # Invalid date

    # Run CLI import (should fail)
    script_path = ROOT / "scripts" / "import_broker_snapshot.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(external_path),
            "--run-id",
            run_id,
            "--as-of-date",
            invalid_date,
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code is non-zero
    assert result.returncode != 0, "CLI should fail for invalid date format"

    # Verify error message mentions date format (check both stdout and stderr)
    output = (result.stdout + result.stderr).lower()
    assert "date" in output or "format" in output

    # Verify error message is ASCII-only
    assert all(ord(c) < 128 for c in result.stderr), "Error message should be ASCII-only"
    assert all(ord(c) < 128 for c in result.stdout), "Output should be ASCII-only"


def test_cli_help_exits_with_zero(tmp_path: Path):
    """Test that --help gives exit code 0 and contains all flags."""
    script_path = ROOT / "scripts" / "import_broker_snapshot.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code is 0
    assert result.returncode == 0, "Help should exit with code 0"

    # Verify help text contains all expected flags
    help_text = result.stdout
    assert "--input" in help_text
    assert "--run-id" in help_text
    assert "--as-of-date" in help_text
    assert "--output-dir" in help_text
    assert "--cash" in help_text
    assert "--store-parquet" in help_text
    assert "--qty-tol" in help_text

    # Note: argparse help text may contain non-ASCII characters (e.g., in descriptions),
    # but error messages should be ASCII-only (tested separately)


def test_cli_all_error_messages_ascii_only(tmp_path: Path):
    """Test that all error messages are ASCII-only."""
    script_path = ROOT / "scripts" / "import_broker_snapshot.py"

    # Test cases that should produce errors
    test_cases = [
        # Missing file
        [
            "--input",
            str(tmp_path / "nonexistent.json"),
            "--run-id",
            "test",
            "--as-of-date",
            "2025-01-15",
        ],
        # Invalid date
        [
            "--input",
            str(tmp_path / "dummy.json"),
            "--run-id",
            "test",
            "--as-of-date",
            "2025-13-45",
        ],
    ]

    # Create dummy file for invalid date test
    dummy_file = tmp_path / "dummy.json"
    dummy_file.write_text('{"cash": 1000, "positions": []}')

    for args in test_cases:
        result = subprocess.run(
            [sys.executable, str(script_path)] + args,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )

        # Verify exit code is non-zero
        assert result.returncode != 0, f"CLI should fail for args: {args}"

        # Verify all output is ASCII-only
        combined_output = result.stdout + result.stderr
        assert all(ord(c) < 128 for c in combined_output), (
            f"Error message should be ASCII-only for args: {args}. "
            f"Output: {combined_output[:200]}"
        )
