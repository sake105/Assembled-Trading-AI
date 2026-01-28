"""Evidence pack rejects directory references (hardening).

Evidence Pack expects file paths, not directory paths. If the Evidence Index (or
manifest fallback) references a directory, build must not attempt to open it.
Instead:
- required directory -> treat as missing_required key -> build_evidence_pack raises ValueError
- optional directory -> treat as missing_optional key -> build succeeds (strict=False)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json
from src.assembled_core.accounting.evidence_pack import build_evidence_pack


def test_required_directory_treated_as_missing_required(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "dir_required"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    date_str = "2025-01-15"

    # Ledger pack is REQUIRED for evidence_index source, but we (wrongly) provide a directory.
    ledger_pack_dir = output_dir / "ledger_run"
    ledger_pack_dir.mkdir(parents=True, exist_ok=True)

    evidence_paths = {
        "broker_snapshot_path": None,
        "ledger_pack_path": ledger_pack_dir,  # WRONG: directory
        "reconcile_report_path": None,
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
        build_evidence_pack(output_dir=output_dir, run_id=run_id, as_of_date=as_of)

    msg = str(exc_info.value)
    assert f"run_id={run_id}" in msg
    assert f"as_of_date={date_str}" in msg
    assert "ledger_pack_path" in msg
    assert msg.encode("ascii", errors="ignore").decode("ascii") == msg


def test_optional_directory_treated_as_missing_optional(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "dir_optional"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    # Required file exists
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy ledger", encoding="utf-8")

    # Optional broker_snapshot_path is (wrongly) a directory that exists
    broker_snapshot_dir = output_dir / "broker_snapshot_run"
    broker_snapshot_dir.mkdir(parents=True, exist_ok=True)

    evidence_paths = {
        "broker_snapshot_path": broker_snapshot_dir,  # WRONG: directory (optional for evidence_index)
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": None,
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

    result = build_evidence_pack(output_dir=output_dir, run_id=run_id, as_of_date=as_of)
    assert "missing_optional" in result
    assert "broker_snapshot_path" in result["missing_optional"]

    # Pack manifest must not include the directory as a file entry
    pack_manifest_path = output_dir / result["pack_manifest_path"]
    with pack_manifest_path.open("r", encoding="utf-8") as f:
        pack_manifest = json.load(f)

    file_paths = [entry.get("path") for entry in pack_manifest.get("files", [])]
    # The directory itself should never be included
    assert "broker_snapshot_run" not in file_paths
    # Still must include the required ledger file
    assert "ledger_run/ledger_events.parquet" in file_paths

