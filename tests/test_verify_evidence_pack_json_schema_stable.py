"""Stable schema test for verify_evidence_pack.py --json output.

Asserts the JSON output has all documented keys, schema_version=1, error_code="" for OK zip,
and that two runs produce identical bytes (deterministic).
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
    "zip_path",
    "ok",
    "error_code",
    "missing_manifest",
    "n_files",
    "zip_entries_count",
    "manifest_files_count",
    "zip_compression",
    "tool_version",
    "source",
    "source_path",
    "bad_paths_count",
    "missing_entries_count",
    "paths_not_in_zip_entries_count",
    "checksum_mismatches_count",
    "details",
]


def _build_ok_zip(tmp_path: Path) -> Path:
    """Build a minimal valid evidence pack ZIP and return its path."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "schema_stable_ok"
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

    result = build_evidence_pack(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of_date,
        include_optional=True,
    )
    return output_dir / result["pack_path"]


def test_verify_json_schema_stable_ok_zip(tmp_path: Path) -> None:
    """Against an OK zip: all keys present, schema_version=1, error_code=\"\", output deterministic."""
    zip_path = _build_ok_zip(tmp_path)
    script_path = ROOT / "scripts" / "verify_evidence_pack.py"

    run1 = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(zip_path), "--json"],
        capture_output=True,
        cwd=str(ROOT),
    )
    assert run1.returncode == 0, f"verify --json should exit 0: stderr={run1.stderr!r}"
    out1_bytes = run1.stdout
    out1 = json.loads(out1_bytes.decode("utf-8"))

    for key in REQUIRED_KEYS:
        assert key in out1, f"Missing key in --json output: {key}"

    assert out1["schema_version"] == 1
    assert out1["error_code"] == ""
    assert out1["ok"] is True
    assert out1["missing_manifest"] is False
    assert out1["bad_paths_count"] == 0
    assert out1["missing_entries_count"] == 0
    assert out1["checksum_mismatches_count"] == 0
    assert out1["source"] in ("evidence_index", "manifest")
    assert isinstance(out1["source_path"], str) and len(out1["source_path"]) > 0
    assert out1["zip_compression"] in ("deflated", "stored")
    assert isinstance(out1["tool_version"], str) and len(out1["tool_version"]) > 0
    assert out1["tool_version"].encode("ascii", errors="ignore").decode("ascii") == out1["tool_version"]
    assert "details" in out1
    assert isinstance(out1["details"], dict)
    assert out1["details"] == {}

    run2 = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(zip_path), "--json"],
        capture_output=True,
        cwd=str(ROOT),
    )
    assert run2.returncode == 0
    out2_bytes = run2.stdout
    assert out1_bytes == out2_bytes, "Two runs must produce identical JSON bytes (deterministic)"
