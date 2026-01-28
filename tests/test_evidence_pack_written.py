"""Tests for evidence pack writer (Sprint 13).

Tests that evidence packs (ZIP + manifest) are correctly created from evidence index.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json
from src.assembled_core.accounting.evidence_pack import build_evidence_pack


def test_evidence_pack_written_with_files(tmp_path: Path) -> None:
    """Evidence pack ZIP and manifest are created when files exist."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "pack_test"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    date_str = "2025-01-15"

    # Create dummy artifact files
    broker_snapshot_path = output_dir / "broker_snapshot_run" / f"snapshot_{date_str}.json"
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    reconcile_report_path = output_dir / "reconcile_run" / f"reconcile_{date_str}.json"
    reconcile_csv_path = output_dir / "reconcile_run" / f"reconcile_{date_str}.csv"
    accounting_report_path = output_dir / "accounting_report_run" / f"accounting_{date_str}.json"

    # Create directories and files
    for p in [
        broker_snapshot_path,
        ledger_pack_path,
        reconcile_report_path,
        reconcile_csv_path,
        accounting_report_path,
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy content", encoding="utf-8")

    # Write evidence index
    evidence_paths = {
        "broker_snapshot_path": broker_snapshot_path,
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": reconcile_report_path,
        "accounting_report_path": accounting_report_path,
        "manifest_path": None,
    }

    write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of,
        paths=evidence_paths,
        broker_meta={"broker_view_source": "stored_snapshot"},
        reconciliation_ok=True,
    )

    # Build evidence pack
    result = build_evidence_pack(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of,
        include_optional=True,
    )

    # Verify return values
    assert "pack_path" in result
    assert "pack_manifest_path" in result
    assert "n_files" in result
    assert result["n_files"] > 0

    # Verify ZIP file exists
    zip_path = output_dir / result["pack_path"]
    assert zip_path.exists(), f"ZIP file should exist: {zip_path}"

    # Verify manifest file exists
    manifest_path = output_dir / result["pack_manifest_path"]
    assert manifest_path.exists(), f"Manifest file should exist: {manifest_path}"

    # Verify ZIP contains expected files (including manifest)
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert len(namelist) == result["n_files"]
        
        # Check that evidence index is included
        evidence_index_name = f"evidence_{run_id}/evidence_{date_str}.json"
        assert evidence_index_name in namelist or any(
            "evidence_" in name and name.endswith(".json") for name in namelist
        ), "Evidence index should be in ZIP"
        
        # Check that pack manifest is included in ZIP
        manifest_zip_name = f"pack_manifest_{date_str}.json"
        assert manifest_zip_name in namelist, "Pack manifest should be in ZIP"

    # Verify manifest JSON structure
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data.get("schema_version") == 1
    assert manifest_data.get("run_id") == run_id
    assert manifest_data.get("as_of_date") is not None
    assert "files" in manifest_data
    assert len(manifest_data["files"]) == result["n_files"]
    # Evidence Index source: evidence index JSON must be present as source artifact
    assert manifest_data.get("source") == "evidence_index"
    evidence_index_name = f"evidence_{run_id}/evidence_{date_str}.json"
    source_entries = [
        entry for entry in manifest_data["files"]
        if entry.get("path") == evidence_index_name
    ]
    assert len(source_entries) == 1
    assert source_entries[0].get("source_type") == "evidence_index"
    
    # Verify required_missing and optional_missing fields
    assert "required_missing" in manifest_data
    assert "optional_missing" in manifest_data
    
    # Verify each file entry has required fields
    for file_entry in manifest_data["files"]:
        assert "path" in file_entry
        assert "size_bytes" in file_entry
        assert "sha256" in file_entry or file_entry.get("sha256") is None
        assert "source_type" in file_entry
        
        # Verify path is relative and POSIX (no backslashes, no .., no leading /)
        path = file_entry["path"]
        assert "\\" not in path, f"Path should not contain backslashes: {path}"
        assert ".." not in path, f"Path should not contain '..': {path}"
        assert not path.startswith("/"), f"Path should not be absolute: {path}"


def test_evidence_pack_handles_missing_optional_files(tmp_path: Path) -> None:
    """Evidence pack is created even if some optional files are missing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "pack_missing_optional"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    # Create only required files (ledger pack)
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy ledger", encoding="utf-8")

    # Write evidence index with missing optional files
    evidence_paths = {
        "broker_snapshot_path": output_dir / "broker_snapshot_run" / "snapshot_2025-01-15.json",  # Missing
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": output_dir / "reconcile_run" / "reconcile_2025-01-15.json",  # Missing
        "accounting_report_path": None,  # None
        "manifest_path": None,
    }

    write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of,
        paths=evidence_paths,
        broker_meta=None,
        reconciliation_ok=None,
    )

    # Build evidence pack (should succeed despite missing optional files)
    result = build_evidence_pack(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of,
        include_optional=True,
    )

    # Verify pack was created
    assert "pack_path" in result
    assert "pack_manifest_path" in result
    assert result["n_files"] > 0  # At least evidence index + ledger pack

    # Verify missing optional files are reported as keys (Evidence Index: reconcile/accounting are optional)
    assert "missing_optional" in result
    missing_opt = result["missing_optional"]
    assert len(missing_opt) > 0
    # Pack manifest and return use keys, not paths
    for key in missing_opt:
        assert key in ("broker_snapshot_path", "reconcile_report_path", "accounting_report_path", "manifest_path"), (
            f"missing_optional should contain keys, not paths: {missing_opt}"
        )

    # Verify ZIP exists
    zip_path = output_dir / result["pack_path"]
    assert zip_path.exists()

    # Pack manifest must list optional_missing as keys (not paths)
    with (output_dir / result["pack_manifest_path"]).open("r", encoding="utf-8") as f:
        pack_manifest = json.load(f)
    assert "optional_missing" in pack_manifest
    for key in pack_manifest["optional_missing"]:
        assert key in ("broker_snapshot_path", "reconcile_report_path", "accounting_report_path", "manifest_path"), (
            f"optional_missing should be keys: {pack_manifest['optional_missing']}"
        )


def test_evidence_pack_strict_raises_when_optional_missing(tmp_path: Path) -> None:
    """build_evidence_pack(strict=True) raises ValueError when optional files are missing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "pack_strict_optional"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy ledger", encoding="utf-8")

    evidence_paths = {
        "broker_snapshot_path": output_dir / "broker_snapshot_run" / "snapshot_2025-01-15.json",
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": output_dir / "reconcile_run" / "reconcile_2025-01-15.json",
        "accounting_report_path": None,
        "manifest_path": None,
    }

    write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of,
        paths=evidence_paths,
        broker_meta=None,
        reconciliation_ok=None,
    )

    with pytest.raises(ValueError) as exc_info:
        build_evidence_pack(
            output_dir=output_dir,
            run_id=run_id,
            as_of_date=as_of,
            include_optional=True,
            strict=True,
        )
    msg = str(exc_info.value)
    assert "optional" in msg.lower() or "missing" in msg.lower()
    assert msg.encode("ascii", errors="ignore").decode("ascii") == msg
