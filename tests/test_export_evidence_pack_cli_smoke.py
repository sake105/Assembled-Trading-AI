"""Smoke tests for export_evidence_pack.py CLI tool (Sprint 13).

Tests that the standalone CLI tool correctly exports evidence packs.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json


def test_cli_export_creates_pack(tmp_path: Path) -> None:
    """Test that CLI export creates ZIP and manifest files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "cli_test"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    # Create dummy artifact files
    broker_snapshot_path = output_dir / "broker_snapshot_run" / f"snapshot_{date_str}.json"
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    reconcile_report_path = output_dir / "reconcile_run" / f"reconcile_{date_str}.json"

    for p in [broker_snapshot_path, ledger_pack_path, reconcile_report_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy content", encoding="utf-8")

    # Write evidence index
    evidence_paths = {
        "broker_snapshot_path": broker_snapshot_path,
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": reconcile_report_path,
        "accounting_report_path": None,
        "manifest_path": None,
    }

    write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=pd.Timestamp(as_of_date, tz="UTC"),
        paths=evidence_paths,
        broker_meta=None,
        reconciliation_ok=None,
    )

    # Run CLI export
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Verify output message (single status line with required fields)
    assert "OK:" in result.stdout, "Should print OK message"
    assert "pack_path=" in result.stdout
    assert "pack_manifest_path=" in result.stdout
    assert "source=" in result.stdout
    assert "n_files=" in result.stdout
    assert "required_missing=" in result.stdout
    assert "optional_missing=" in result.stdout

    # Verify ZIP file exists
    evidence_dir = output_dir / f"evidence_{run_id}"
    zip_path = evidence_dir / f"pack_{date_str}.zip"
    assert zip_path.exists(), "ZIP file should exist"

    # Verify manifest file exists
    manifest_path = evidence_dir / f"pack_manifest_{date_str}.json"
    assert manifest_path.exists(), "Manifest file should exist"

    # Verify ZIP is valid
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert len(zf.namelist()) > 0, "ZIP should contain files"


def test_cli_invalid_date_exits_with_error(tmp_path: Path) -> None:
    """Test that invalid date format causes CLI to exit with error."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "test",
            "--as-of-date",
            "invalid-date-format",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code is non-zero
    assert result.returncode != 0, "Should exit with error for invalid date"

    # Verify error message is ASCII-only
    error_output = (result.stdout + result.stderr).lower()
    assert "invalid" in error_output or "date" in error_output or "format" in error_output
    assert error_output.encode("ascii", errors="ignore").decode("ascii") == error_output


def test_cli_missing_evidence_index_exits_with_error(tmp_path: Path) -> None:
    """Test that missing evidence index causes CLI to exit with error."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "nonexistent_run",
            "--as-of-date",
            "2025-01-15",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code is non-zero
    assert result.returncode != 0, "Should exit with error for missing evidence index"

    # Verify error message is ASCII-only
    error_output = (result.stdout + result.stderr).lower()
    assert error_output.encode("ascii", errors="ignore").decode("ascii") == error_output


def test_cli_help_exits_with_zero(tmp_path: Path) -> None:
    """Test that --help exits with code 0."""
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert result.returncode == 0, "--help should exit with 0"
    assert "evidence pack" in result.stdout.lower() or "export" in result.stdout.lower()


def test_cli_strict_mode_fails_on_missing_optional(tmp_path: Path) -> None:
    """Test that --strict mode fails if optional files are missing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "strict_test"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    # Create only required file (ledger pack)
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy", encoding="utf-8")

    # Write evidence index with missing optional files
    evidence_paths = {
        "broker_snapshot_path": output_dir / "broker_snapshot_run" / f"snapshot_{date_str}.json",  # Missing
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": output_dir / "reconcile_run" / f"reconcile_{date_str}.json",  # Missing
        "accounting_report_path": None,
        "manifest_path": None,
    }

    write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=pd.Timestamp(as_of_date, tz="UTC"),
        paths=evidence_paths,
        broker_meta=None,
        reconciliation_ok=None,
    )

    # Run CLI with --strict
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
            "--strict",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Should fail with --strict when optional files are missing
    assert result.returncode != 0, "Should exit with error in --strict mode when optional files missing"


def test_cli_prints_source_and_missing_counts(tmp_path: Path) -> None:
    """CLI prints source and missing counts in success status line."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "cli_meta"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    # Create minimal required artifacts (no optional)
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    reconcile_report_path = output_dir / "reconcile_run" / f"reconcile_{date_str}.json"
    accounting_report_path = output_dir / "accounting_run" / f"accounting_{date_str}.json"

    for p in [ledger_pack_path, reconcile_report_path, accounting_report_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy", encoding="utf-8")

    evidence_paths = {
        "broker_snapshot_path": None,
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": reconcile_report_path,
        "accounting_report_path": accounting_report_path,
        "manifest_path": None,
    }

    write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=pd.Timestamp(as_of_date, tz="UTC"),
        paths=evidence_paths,
        broker_meta=None,
        reconciliation_ok=True,
    )

    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Status line should contain source and zero missing counts
    status_line = result.stdout.strip()
    assert "source=" in status_line
    assert "required_missing=0" in status_line
    assert "optional_missing=0" in status_line


def test_cli_strict_fails_on_optional_missing_with_exit_1(tmp_path: Path) -> None:
    """Strict mode fails with exit 1 when any optional files are missing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "cli_strict_optional"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    # Create only required file (ledger)
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy", encoding="utf-8")

    # Evidence index references missing optional paths
    evidence_paths = {
        "broker_snapshot_path": output_dir / "broker_snapshot_run" / f"snapshot_{date_str}.json",  # missing
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": None,
        "accounting_report_path": None,
        "manifest_path": None,
    }

    write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=pd.Timestamp(as_of_date, tz="UTC"),
        paths=evidence_paths,
        broker_meta=None,
        reconciliation_ok=None,
    )

    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
            "--strict",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert result.returncode == 1, "Strict mode should exit with code 1 when optional files are missing"
    # Error output should be ASCII-only
    error_output = (result.stdout + result.stderr)
    assert error_output.encode("ascii", errors="ignore").decode("ascii") == error_output


def test_cli_errors_ascii_only(tmp_path: Path) -> None:
    """All error messages produced by CLI are ASCII-only."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use invalid date to trigger an error
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "ascii_test",
            "--as-of-date",
            "invalid-date",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert result.returncode == 1
    error_output = (result.stdout + result.stderr)
    assert error_output.encode("ascii", errors="ignore").decode("ascii") == error_output
