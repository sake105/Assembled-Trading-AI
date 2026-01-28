"""Tests for evidence index writer and integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json


def test_evidence_index_written_with_expected_paths(tmp_path: Path) -> None:
    """Evidence index JSON is written with expected keys and relative POSIX paths."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "evidence_run"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    # Create some dummy files to reference
    broker_snapshot_path = output_dir / "broker_snapshot_run" / "snapshot_2025-01-15.json"
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    reconcile_report_path = output_dir / "reconcile_run" / "reconcile_2025-01-15.json"
    accounting_report_path = output_dir / "accounting_report_run" / "accounting_2025-01-15.json"
    manifest_path = output_dir / "run_manifest_1d.json"

    for p in [
        broker_snapshot_path,
        ledger_pack_path,
        reconcile_report_path,
        accounting_report_path,
        manifest_path,
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")

    paths = {
        "broker_snapshot_path": broker_snapshot_path,
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": reconcile_report_path,
        "accounting_report_path": accounting_report_path,
        "manifest_path": manifest_path,
    }

    evidence_path = write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of,
        paths=paths,
        broker_meta={"broker_view_source": "stored_snapshot"},
        reconciliation_ok=True,
    )

    assert evidence_path.exists()
    assert evidence_path.name == "evidence_2025-01-15.json"
    assert evidence_path.parent.name == f"evidence_{run_id}"

    with evidence_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("schema_version") == 1
    assert data.get("run_id") == run_id
    assert "paths" in data
    paths_block = data["paths"]

    # All expected path keys present
    expected_keys = {
        "broker_snapshot_path",
        "ledger_pack_path",
        "reconcile_report_path",
        "accounting_report_path",
        "manifest_path",
    }
    assert expected_keys.issubset(paths_block.keys())

    # Paths should be relative to output_dir and use POSIX slashes
    for key, value in paths_block.items():
        if value is None:
            continue
        assert "\\" not in value, f"{key} should use POSIX slashes"
        assert not Path(value).is_absolute(), f"{key} should be relative: {value}"


def test_evidence_index_deterministic_bytes(tmp_path: Path) -> None:
    """Writing the same evidence index twice produces byte-identical output."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "evidence_deterministic"
    as_of_date = "2025-01-15"

    paths = {
        "broker_snapshot_path": output_dir / "broker_snapshot_run" / "snapshot_2025-01-15.json",
        "ledger_pack_path": output_dir / "ledger_run" / "ledger_events.parquet",
        "reconcile_report_path": output_dir / "reconcile_run" / "reconcile_2025-01-15.json",
        "accounting_report_path": output_dir / "accounting_report_run" / "accounting_2025-01-15.json",
        "manifest_path": output_dir / "run_manifest_1d.json",
    }

    # First write
    path1 = write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of_date,
        paths=paths,
        broker_meta=None,
        reconciliation_ok=None,
    )
    content1 = path1.read_bytes()

    # Second write (same inputs) -> identical bytes
    path2 = write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of_date,
        paths=paths,
        broker_meta=None,
        reconciliation_ok=None,
    )
    content2 = path2.read_bytes()

    assert content1 == content2

    data = json.loads(content1.decode("utf-8"))
    assert data.get("schema_version") == 1
    assert data.get("run_id") == run_id
    assert data.get("as_of_date") == "2025-01-15"
    assert "paths" in data
    assert set(data["paths"].keys()) >= {
        "broker_snapshot_path",
        "ledger_pack_path",
        "reconcile_report_path",
        "accounting_report_path",
        "manifest_path",
    }

