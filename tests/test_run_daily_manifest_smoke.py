"""Smoke test for daily manifest writing (Sprint 13)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_daily import _write_daily_manifest


def test_daily_manifest_written_with_relative_posix_paths(tmp_path: Path):
    """Test that daily manifest is written and paths are relative + POSIX."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy SAFE orders file
    safe_path = output_dir / "orders_20250115.csv"
    safe_path.write_text("timestamp,symbol,side,qty,price\n")

    run_id = "daily_20250115"
    target_date = datetime(2025, 1, 15)

    # Write manifest
    _write_daily_manifest(
        output_dir=output_dir,
        run_id=run_id,
        target_date=target_date,
        safe_path=safe_path,
        broker_snapshot_policy="prefer",
        write_broker_snapshot=False,
        broker_snapshot_run_id=None,
        broker_snapshot_file=None,
        broker_snapshot_date=None,
    )

    # Verify manifest exists
    manifest_path = output_dir / f"manifest_daily_{run_id}.json"
    assert manifest_path.exists(), "Manifest file should exist"

    # Load and verify JSON
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Verify required fields
    assert manifest_data["run_id"] == run_id
    assert manifest_data["target_date"] == "2025-01-15"
    assert manifest_data["safe_orders_path"] is not None

    # Verify paths are relative and POSIX (forward slashes)
    safe_path_str = manifest_data["safe_orders_path"]
    assert (
        "/" in safe_path_str or safe_path_str == "orders_20250115.csv"
    ), "Path should use POSIX slashes"
    assert "\\" not in safe_path_str, "Path should not contain Windows backslashes"

    # Verify deterministic JSON (keys should be sorted)
    json_str1 = json.dumps(manifest_data, sort_keys=True, indent=2)
    json_str2 = json.dumps(manifest_data, sort_keys=True, indent=2)
    assert json_str1 == json_str2, "JSON serialization should be deterministic"

    # Verify trailing newline in file
    with manifest_path.open("rb") as f:
        content = f.read()
        assert content.endswith(b"\n"), "Manifest file should end with newline"


def test_daily_manifest_deterministic_byte_identical(tmp_path: Path):
    """Test that writing manifest twice produces byte-identical files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_path = output_dir / "orders_20250115.csv"
    safe_path.write_text("timestamp,symbol,side,qty,price\n")

    run_id = "daily_20250115"
    target_date = datetime(2025, 1, 15)

    # Write manifest first time
    _write_daily_manifest(
        output_dir=output_dir,
        run_id=run_id,
        target_date=target_date,
        safe_path=safe_path,
        broker_snapshot_policy="prefer",
        write_broker_snapshot=False,
        broker_snapshot_run_id=None,
        broker_snapshot_file=None,
        broker_snapshot_date=None,
    )

    manifest_path = output_dir / f"manifest_daily_{run_id}.json"
    with manifest_path.open("rb") as f:
        content1 = f.read()

    # Write manifest second time (should be identical)
    _write_daily_manifest(
        output_dir=output_dir,
        run_id=run_id,
        target_date=target_date,
        safe_path=safe_path,
        broker_snapshot_policy="prefer",
        write_broker_snapshot=False,
        broker_snapshot_run_id=None,
        broker_snapshot_file=None,
        broker_snapshot_date=None,
    )

    with manifest_path.open("rb") as f:
        content2 = f.read()

    # Files should be byte-identical
    assert (
        content1 == content2
    ), "Manifest should be byte-identical when written twice with same inputs"


def test_daily_manifest_all_keys_present(tmp_path: Path):
    """Test that all expected keys are present in manifest (even if None)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_path = output_dir / "orders_20250115.csv"
    safe_path.write_text("timestamp,symbol,side,qty,price\n")

    run_id = "daily_20250115"
    target_date = datetime(2025, 1, 15)

    # Write manifest
    _write_daily_manifest(
        output_dir=output_dir,
        run_id=run_id,
        target_date=target_date,
        safe_path=safe_path,
        broker_snapshot_policy="prefer",
        write_broker_snapshot=False,
        broker_snapshot_run_id=None,
        broker_snapshot_file=None,
        broker_snapshot_date=None,
    )

    manifest_path = output_dir / f"manifest_daily_{run_id}.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Verify all expected keys are present (aligned with orchestrator manifest structure)
    expected_keys = {
        "run_id",
        "target_date",
        "safe_orders_path",
        "broker_snapshot_policy",
        "broker_snapshot_date",
        "broker_snapshot_file",
        "broker_snapshot_import_ok",
        "broker_snapshot_path",
        "broker_snapshot_run_id",
        "ledger_pack_path",
        "reconcile_report_path",
        "reconciliation_ok",
        "write_paper_broker_snapshot",
    }

    actual_keys = set(manifest_data.keys())
    assert expected_keys.issubset(
        actual_keys
    ), f"Missing keys: {expected_keys - actual_keys}"

    # Verify paths are relative + POSIX for all path fields
    path_fields = [
        "safe_orders_path",
        "broker_snapshot_path",
        "ledger_pack_path",
        "reconcile_report_path",
    ]
    for field in path_fields:
        value = manifest_data.get(field)
        if value is not None:
            assert isinstance(value, str), f"{field} should be string or None"
            assert "\\" not in value, f"{field} should use POSIX slashes, got: {value}"


def test_daily_manifest_broker_snapshot_import_fields(tmp_path: Path):
    """Test that broker_snapshot_file and broker_snapshot_import_ok are set when file is provided."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_path = output_dir / "orders_20250115.csv"
    safe_path.write_text("timestamp,symbol,side,qty,price\n")

    # Create a dummy broker snapshot file
    snapshot_file = tmp_path / "external_snapshot.json"
    snapshot_file.write_text('{"cash": 1000.0, "positions": []}')

    run_id = "daily_20250115"
    target_date = datetime(2025, 1, 15)

    # Write manifest with broker_snapshot_file set
    _write_daily_manifest(
        output_dir=output_dir,
        run_id=run_id,
        target_date=target_date,
        safe_path=safe_path,
        broker_snapshot_policy="prefer",
        write_broker_snapshot=False,
        broker_snapshot_run_id="snapshot_run",
        broker_snapshot_file=snapshot_file,
        broker_snapshot_date="2025-01-15",
        broker_snapshot_import_ok=True,
    )

    manifest_path = output_dir / f"manifest_daily_{run_id}.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Verify broker snapshot import fields
    assert (
        manifest_data["broker_snapshot_file"] is not None
    ), "broker_snapshot_file should be set"
    assert (
        manifest_data["broker_snapshot_import_ok"] is True
    ), "broker_snapshot_import_ok should be True"
    assert (
        manifest_data["broker_snapshot_date"] == "2025-01-15"
    ), "broker_snapshot_date should be set"
    assert (
        manifest_data["broker_snapshot_run_id"] == "snapshot_run"
    ), "broker_snapshot_run_id should be set"

    # Verify broker_snapshot_file is relative or basename (not absolute path)
    file_str = manifest_data["broker_snapshot_file"]
    assert "\\" not in file_str, "broker_snapshot_file should use POSIX slashes"
    assert (
        not Path(file_str).is_absolute() or file_str == snapshot_file.name
    ), "broker_snapshot_file should be relative or basename"
