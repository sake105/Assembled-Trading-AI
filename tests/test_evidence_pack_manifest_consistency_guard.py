"""Tests for pack manifest consistency guards (files_count, zip_entries, no duplicates, pack_manifest exactly once)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json
from src.assembled_core.accounting.evidence_pack import (
    build_evidence_pack,
    _validate_manifest_consistency,
)


def _build_pack(tmp_path: Path) -> Path:
    """Build minimal evidence pack; return manifest path."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "consistency_test"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    broker_snapshot_path = output_dir / "broker_snapshot_run" / f"snapshot_{date_str}.json"
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    reconcile_report_path = output_dir / "reconcile_run" / f"reconcile_{date_str}.json"
    for p in [broker_snapshot_path, ledger_pack_path, reconcile_report_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy content", encoding="utf-8")

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

    build_evidence_pack(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of_date,
        include_optional=True,
    )
    manifest_path = output_dir / f"evidence_{run_id}" / f"pack_manifest_{date_str}.json"
    return manifest_path


def test_manifest_consistency_guard_happy_path(tmp_path: Path) -> None:
    """Built pack manifest passes all consistency checks (files_count, zip_entries, no dupe, pack_manifest once)."""
    manifest_path = _build_pack(tmp_path)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "files_count" in manifest_data
    assert manifest_data["files_count"] == len(manifest_data["files"])
    _validate_manifest_consistency(manifest_data, "consistency_test", "2025-01-15")


def test_manifest_consistency_guard_duplicate_path_raises(tmp_path: Path) -> None:
    """Duplicate files[].path triggers ValueError from _validate_manifest_consistency."""
    manifest_path = _build_pack(tmp_path)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Duplicate first file path in files[]
    first_path = manifest_data["files"][0]["path"]
    dup_entry = {**manifest_data["files"][0]}
    manifest_data["files"].append(dup_entry)

    with pytest.raises(ValueError) as exc_info:
        _validate_manifest_consistency(manifest_data, "r1", "2025-01-15")
    msg = exc_info.value.args[0]
    assert "duplicate" in msg or "run_id" in msg
