"""Tests for Evidence Index manifest_path backfill from orchestrator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json
from src.assembled_core.pipeline.orchestrator import (
    _backfill_evidence_index_manifest_path,
    _manifest_path_str,
)


def test_evidence_index_manifest_path_backfill_sets_relative_posix_path(
    tmp_path: Path,
) -> None:
    """Backfill sets paths.manifest_path to a relative POSIX path and is byte-deterministic."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "evidence_backfill"
    as_of_date = "2025-01-15"

    # Prepare minimal paths block without manifest_path
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy", encoding="utf-8")

    paths = {
        "broker_snapshot_path": None,
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": None,
        "accounting_report_path": None,
        "manifest_path": None,
    }

    # Write initial Evidence Index JSON (no manifest_path yet)
    evidence_path = write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of_date,
        paths=paths,
        broker_meta=None,
        reconciliation_ok=None,
    )

    # Write a manifest file (content irrelevant for backfill)
    manifest_path = output_dir / "run_manifest_1d.json"
    manifest_payload = {"schema_version": 1, "freq": "1d"}
    manifest_path.write_text(
        json.dumps(manifest_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    # Simulate ledger_result as returned from ledger_integration (relative paths)
    ledger_result = {
        "evidence_index_path": _manifest_path_str(evidence_path, base_dir=output_dir),
    }

    # First backfill run
    _backfill_evidence_index_manifest_path(
        base_dir=output_dir,
        ledger_result=ledger_result,
        manifest_path=manifest_path,
    )
    content1 = evidence_path.read_bytes()

    data1 = json.loads(content1.decode("utf-8"))
    assert "paths" in data1
    manifest_value = data1["paths"].get("manifest_path")
    assert manifest_value is not None
    # Relative, POSIX, no backslashes, not absolute
    assert "\\" not in manifest_value
    assert not Path(manifest_value).is_absolute()

    # Second backfill run with same inputs should be byte-identical
    _backfill_evidence_index_manifest_path(
        base_dir=output_dir,
        ledger_result=ledger_result,
        manifest_path=manifest_path,
    )
    content2 = evidence_path.read_bytes()

    assert content1 == content2
