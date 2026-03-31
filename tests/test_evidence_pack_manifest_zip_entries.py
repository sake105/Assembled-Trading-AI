"""Tests for pack manifest zip_entries and zip_entries_count (offline audit without ZIP listing)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json
from src.assembled_core.accounting.evidence_pack import build_evidence_pack


def _build_pack(tmp_path: Path) -> tuple[Path, Path]:
    """Build minimal evidence pack; return (output_dir, manifest_path)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "zip_entries_test"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    broker_snapshot_path = (
        output_dir / "broker_snapshot_run" / f"snapshot_{date_str}.json"
    )
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

    _result = build_evidence_pack(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of_date,
        include_optional=True,
    )
    evidence_dir = output_dir / f"evidence_{run_id}"
    manifest_path = evidence_dir / f"pack_manifest_{date_str}.json"
    return output_dir, manifest_path


def test_manifest_has_zip_entries_and_count(tmp_path: Path) -> None:
    """Pack manifest contains zip_entries (sorted) and zip_entries_count; invariants hold."""
    _output_dir, manifest_path = _build_pack(tmp_path)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "zip_entries" in manifest_data
    assert "zip_entries_count" in manifest_data
    zip_entries = manifest_data["zip_entries"]
    zip_entries_count = manifest_data["zip_entries_count"]

    assert isinstance(zip_entries, list)
    assert isinstance(zip_entries_count, int)
    assert zip_entries_count == len(zip_entries)
    assert zip_entries == sorted(zip_entries), "zip_entries must be sorted"

    # pack_manifest_*.json must be in zip_entries
    manifest_names = [
        e for e in zip_entries if e.startswith("pack_manifest_") and e.endswith(".json")
    ]
    assert len(manifest_names) >= 1, "pack_manifest_*.json must appear in zip_entries"

    # files[] length should match zip_entries_count
    files_list = manifest_data.get("files", [])
    assert len(files_list) == zip_entries_count

    # required_keys / optional_keys exist, are lists, sorted, allowed keys only
    assert "required_keys" in manifest_data
    assert "optional_keys" in manifest_data
    required_keys = manifest_data["required_keys"]
    optional_keys = manifest_data["optional_keys"]
    assert isinstance(required_keys, list)
    assert isinstance(optional_keys, list)
    assert required_keys == sorted(required_keys)
    assert optional_keys == sorted(optional_keys)
    from src.assembled_core.accounting.evidence_pack import (
        REQUIRED_KEYS_BY_SOURCE,
        OPTIONAL_KEYS_BY_SOURCE,
    )

    src = manifest_data.get("source")
    allowed_req = set(REQUIRED_KEYS_BY_SOURCE.get(src, []))
    allowed_opt = set(OPTIONAL_KEYS_BY_SOURCE.get(src, []))
    for k in required_keys:
        assert k in allowed_req, f"required_keys must only contain allowed keys: {k!r}"
    for k in optional_keys:
        assert k in allowed_opt, f"optional_keys must only contain allowed keys: {k!r}"
    # evidence_index source: required_keys == ["ledger_pack_path"]
    assert manifest_data.get("source") == "evidence_index"
    assert required_keys == ["ledger_pack_path"]
