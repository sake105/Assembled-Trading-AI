"""Tests for evidence pack manifest fallback (Evidence Index -> Manifest).

These tests verify that build_evidence_pack() can fall back to orchestrator
manifests when no Evidence Index JSON exists, and that required/optional
rules and source metadata are applied correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_pack import (  # noqa: E402
    build_evidence_pack,
    collect_evidence_files,
)


def _write_manifest(path: Path, payload: dict) -> None:
    """Write JSON manifest with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def test_collect_uses_manifest_when_evidence_index_missing(tmp_path: Path) -> None:
    """collect_evidence_files falls back to manifest when evidence index is missing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "manifest_run"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    date_str = "2025-01-15"

    # Create dummy artifacts referenced by manifest
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    reconcile_report_path = output_dir / "reconcile_run" / f"reconcile_{date_str}.json"
    accounting_report_path = (
        output_dir / "accounting_run" / f"accounting_{date_str}.json"
    )

    for p in [ledger_pack_path, reconcile_report_path, accounting_report_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy", encoding="utf-8")

    # Write orchestrator-like manifest (no evidence index present)
    manifest_payload = {
        "schema_version": 1,
        "freq": "1d",
        "ledger_pack_path": "ledger_run/ledger_events.parquet",
        "reconcile_report_path": f"reconcile_run/reconcile_{date_str}.json",
        "accounting_report_path": f"accounting_run/accounting_{date_str}.json",
        "broker_snapshot_path": None,
        "evidence_index_path": None,
    }
    manifest_path = output_dir / "run_manifest_1d.json"
    _write_manifest(manifest_path, manifest_payload)

    # No evidence index written on purpose

    result = collect_evidence_files(output_dir, run_id, as_of)

    assert result["source"] == "manifest"
    assert result["evidence_index_path"] is None
    assert result["manifest_path"] == manifest_path

    # Source path should be relative POSIX path
    assert result["source_path"] == "run_manifest_1d.json"

    # Files must include ledger pack and manifest itself
    file_paths = [zp for _, zp in result["files"]]
    assert "ledger_run/ledger_events.parquet" in file_paths
    assert "run_manifest_1d.json" in file_paths


def test_manifest_fallback_required_missing_raises(tmp_path: Path) -> None:
    """Manifest fallback raises when required paths are missing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "manifest_missing"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    date_str = "2025-01-15"

    # Create only some of the required files (e.g., ledger only)
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy", encoding="utf-8")

    # Manifest references all three required paths, but only ledger exists
    manifest_payload = {
        "schema_version": 1,
        "freq": "1d",
        "ledger_pack_path": "ledger_run/ledger_events.parquet",
        "reconcile_report_path": f"reconcile_run/reconcile_{date_str}.json",  # missing
        "accounting_report_path": f"accounting_run/accounting_{date_str}.json",  # missing
        "broker_snapshot_path": None,
        "evidence_index_path": None,
    }
    manifest_path = output_dir / "run_manifest_1d.json"
    _write_manifest(manifest_path, manifest_payload)

    # No evidence index written on purpose

    # build_evidence_pack should fail-fast with ValueError mentioning run_id and as_of_date
    try:
        build_evidence_pack(
            output_dir=output_dir,
            run_id=run_id,
            as_of_date=as_of,
            include_optional=True,
        )
        raise AssertionError(
            "build_evidence_pack should have raised ValueError for missing required files"
        )
    except ValueError as exc:
        msg = str(exc)
        assert "run_id=manifest_missing" in msg
        assert f"as_of_date={date_str}" in msg
        # required_missing are keys (not paths); manifest fallback requires reconcile + accounting
        assert "reconcile_report_path" in msg or "accounting_report_path" in msg


def test_manifest_fallback_sets_source_fields(tmp_path: Path) -> None:
    """Pack manifest reflects manifest fallback as source and source_path."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "manifest_source_meta"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    date_str = "2025-01-15"

    # Create all required artifacts + one optional broker snapshot
    broker_snapshot_path = (
        output_dir / "broker_snapshot_run" / f"snapshot_{date_str}.json"
    )
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    reconcile_report_path = output_dir / "reconcile_run" / f"reconcile_{date_str}.json"
    accounting_report_path = (
        output_dir / "accounting_run" / f"accounting_{date_str}.json"
    )

    for p in [
        broker_snapshot_path,
        ledger_pack_path,
        reconcile_report_path,
        accounting_report_path,
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy", encoding="utf-8")

    manifest_payload = {
        "schema_version": 1,
        "freq": "1d",
        "ledger_pack_path": "ledger_run/ledger_events.parquet",
        "reconcile_report_path": f"reconcile_run/reconcile_{date_str}.json",
        "accounting_report_path": f"accounting_run/accounting_{date_str}.json",
        "broker_snapshot_path": f"broker_snapshot_run/snapshot_{date_str}.json",
        "evidence_index_path": None,
    }
    manifest_path = output_dir / "run_manifest_1d.json"
    _write_manifest(manifest_path, manifest_payload)

    # No evidence index written on purpose

    result = build_evidence_pack(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of,
        include_optional=True,
    )

    # Load pack manifest JSON and inspect source fields
    pack_manifest_path = output_dir / result["pack_manifest_path"]
    with pack_manifest_path.open("r", encoding="utf-8") as f:
        pack_manifest = json.load(f)

    assert pack_manifest.get("source") == "manifest"
    assert pack_manifest.get("source_path") == "run_manifest_1d.json"

    # source_path must match exactly one files[] entry (source artifact in pack)
    manifest_entries = [
        entry
        for entry in pack_manifest["files"]
        if entry.get("path") == "run_manifest_1d.json"
    ]
    assert len(manifest_entries) == 1
    assert manifest_entries[0].get("path") == pack_manifest["source_path"]
    assert manifest_entries[0].get("source_type") == "manifest"

    # ZIP must also contain the manifest exactly once
    zip_path = output_dir / result["pack_path"]
    assert zip_path.exists()
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
    manifest_names = [name for name in namelist if name == "run_manifest_1d.json"]
    assert len(manifest_names) == 1

    # Count fields: manifest source has 3 required (ledger, reconcile, accounting) and
    # 1 optional present (broker snapshot), no missing keys.
    assert pack_manifest.get("required_present_count") == 3
    assert pack_manifest.get("required_missing_count") == 0
    assert pack_manifest.get("optional_present_count") == 1
    assert pack_manifest.get("optional_missing_count") == 0
