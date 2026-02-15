"""Stable schema test for export_evidence_pack.py default (JSON) output.

Asserts the JSON output has all documented keys, schema_version=1, ok=true,
error_code="", details={}, and that two runs produce identical bytes (deterministic).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json
from src.assembled_core.accounting.evidence_pack import build_evidence_pack

REQUIRED_KEYS = [
    "schema_version",
    "ok",
    "error_code",
    "details",
    "tool_version",
    "pack_path",
    "pack_manifest_path",
    "source",
    "source_path",
    "n_files",
    "required_missing_count",
    "optional_missing_count",
    "out_zip_path",
    "out_manifest_path",
    "output_dir",
    "output_dir_resolved",
    "pack_path_resolved",
    "pack_manifest_path_resolved",
    "zip_entries_count",
    "files_count",
    "pack_manifest_schema_version",
]


def _build_minimal_output_dir(tmp_path: Path) -> Path:
    """Build minimal output dir with evidence index and run pack (like CLI smoke)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "schema_stable_export"
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
    return output_dir


def test_export_json_schema_stable(tmp_path: Path) -> None:
    """Export default JSON: required keys present, schema_version=1, ok=true, deterministic."""
    output_dir = _build_minimal_output_dir(tmp_path)
    script_path = ROOT / "scripts" / "export_evidence_pack.py"

    run1 = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "schema_stable_export",
            "--as-of-date",
            "2025-01-15",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        cwd=str(ROOT),
    )
    assert run1.returncode == 0, f"export should exit 0: stderr={run1.stderr!r}"
    out1_bytes = run1.stdout
    out1 = json.loads(out1_bytes.decode("utf-8"))

    for key in REQUIRED_KEYS:
        assert key in out1, f"Missing key in export JSON: {key}"

    assert out1["schema_version"] == 1
    assert out1["ok"] is True
    assert out1["error_code"] == ""
    assert out1["details"] == {}

    # Export JSON includes zip_entries_count, files_count, pack_manifest_schema_version from manifest (ints)
    assert "zip_entries_count" in out1
    assert "files_count" in out1
    assert "pack_manifest_schema_version" in out1
    assert isinstance(out1["zip_entries_count"], int)
    assert isinstance(out1["files_count"], int)
    assert out1["pack_manifest_schema_version"] == 1

    # Pack manifest must contain zip_entries_count (offline audit)
    manifest_path = output_dir / "evidence_schema_stable_export" / "pack_manifest_2025-01-15.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "zip_entries_count" in manifest_data
    assert manifest_data["zip_entries_count"] == len(manifest_data.get("zip_entries", []))
    # Export JSON source/source_path must match pack manifest (single source of truth)
    assert out1["source"] == manifest_data.get("source")
    assert out1["source_path"] == manifest_data.get("source_path")

    run2 = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "schema_stable_export",
            "--as-of-date",
            "2025-01-15",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        cwd=str(ROOT),
    )
    assert run2.returncode == 0
    out2_bytes = run2.stdout
    assert out1_bytes == out2_bytes, "Two runs must produce identical JSON bytes (deterministic)"
