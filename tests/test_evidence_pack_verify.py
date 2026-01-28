"""Tests for offline verification of evidence pack ZIPs."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json  # noqa: E402
from src.assembled_core.accounting.evidence_pack import (  # noqa: E402
    build_evidence_pack,
    verify_evidence_pack_zip,
)


def _build_sample_evidence_pack(tmp_path: Path) -> tuple[Path, Path]:
    """Helper to build a minimal evidence pack and return (output_dir, zip_path)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "verify_test"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
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
        as_of_date=as_of,
        paths=evidence_paths,
        broker_meta=None,
        reconciliation_ok=None,
    )

    # Build evidence pack
    result = build_evidence_pack(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of,
        include_optional=True,
    )

    zip_path = output_dir / result["pack_path"]
    return output_dir, zip_path


def test_verify_evidence_pack_ok(tmp_path: Path) -> None:
    """verify_evidence_pack_zip returns ok=True for a valid evidence pack."""
    _, zip_path = _build_sample_evidence_pack(tmp_path)

    result = verify_evidence_pack_zip(zip_path)

    assert result["ok"] is True
    assert result["n_files"] > 0
    assert result["missing_manifest"] is False
    assert result["bad_paths"] == []
    assert result["checksum_mismatches"] == []


def test_verify_detects_checksum_mismatch(tmp_path: Path) -> None:
    """Tampering a file in the ZIP should cause checksum_mismatches."""
    output_dir, zip_path = _build_sample_evidence_pack(tmp_path)

    # Extract ZIP contents to a temp directory
    extracted_dir = tmp_path / "unzipped"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted_dir)
        namelist = zf.namelist()

    # Choose a non-manifest file to tamper with (e.g., first non-pack_manifest entry)
    target_name = None
    for name in namelist:
        if not name.startswith("pack_manifest_"):
            target_name = name
            break

    assert target_name is not None, "Expected at least one non-manifest file in pack"

    target_path = extracted_dir / target_name
    assert target_path.exists()

    # Overwrite the target file with different content
    target_path.write_text("tampered content", encoding="utf-8")

    # Rebuild ZIP from extracted directory (keeping same names)
    # This leaves the manifest JSON unchanged, so checksums will now mismatch
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in extracted_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(extracted_dir).as_posix()
                zf.write(file_path, arcname=arcname)

    # Verify again: should detect checksum mismatch
    result = verify_evidence_pack_zip(zip_path)

    assert result["ok"] is False
    assert target_name in result["checksum_mismatches"]


def test_verify_detects_illegal_paths(tmp_path: Path) -> None:
    """verify_evidence_pack_zip flags illegal ZIP entry paths."""
    zip_path = tmp_path / "bad_paths.zip"

    # Create a minimal pack manifest with one good file and one bad path entry
    manifest = {
        "schema_version": 1,
        "run_id": "bad_paths",
        "as_of_date": "2025-01-15T00:00:00+00:00",
        "source": None,
        "source_path": None,
        "files": [
            {"path": "good_file.txt", "size_bytes": 3, "sha256": None, "source_type": "other"},
        ],
        "required_missing": [],
        "optional_missing": [],
        "tool_version": "test",
    }

    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Good file
        zf.writestr("good_file.txt", b"123")
        # Pack manifest
        zf.writestr("pack_manifest_2025-01-15.json", manifest_bytes)
        # Illegal path entries
        zf.writestr("../evil.txt", b"bad")
        zf.writestr(r"bad\path.txt", b"bad")

    result = verify_evidence_pack_zip(zip_path)

    assert result["ok"] is False
    assert result["missing_manifest"] is False
    # Illegal names should be reported in bad_paths
    assert "../evil.txt" in result["bad_paths"]
