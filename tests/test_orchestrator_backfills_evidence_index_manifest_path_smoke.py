"""Smoke test that orchestrator backfills manifest_path into Evidence Index after manifest write.

This test exercises the minimal "manifest writer path" without running a full EOD pipeline.
"""

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


def test_orchestrator_backfills_manifest_path_into_evidence_index(tmp_path: Path) -> None:
    """After manifest write, orchestrator backfill updates paths.manifest_path in Evidence Index."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "orchestrator_backfill_smoke"
    as_of_date = "2025-01-15"

    # Minimal Evidence Index: only ledger_pack_path is required; manifest_path is None initially.
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy ledger", encoding="utf-8")

    paths = {
        "broker_snapshot_path": None,
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": None,
        "accounting_report_path": None,
        "manifest_path": None,
    }

    evidence_path = write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of_date,
        paths=paths,
        broker_meta=None,
        reconciliation_ok=None,
    )

    # Manifest writer path: orchestrator writes run_manifest_1d.json
    manifest_path = output_dir / "run_manifest_1d.json"
    manifest_payload = {"schema_version": 1, "freq": "1d"}
    manifest_path.write_text(
        json.dumps(manifest_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    # Simulate ledger_result from ledger_integration (relative evidence_index_path)
    ledger_result = {
        "evidence_index_path": _manifest_path_str(evidence_path, base_dir=output_dir),
    }

    # Call orchestrator backfill helper
    _backfill_evidence_index_manifest_path(
        base_dir=output_dir,
        ledger_result=ledger_result,
        manifest_path=manifest_path,
    )

    # Evidence Index must now contain paths.manifest_path with a relative POSIX path
    content = evidence_path.read_bytes()
    data = json.loads(content.decode("utf-8"))

    assert "paths" in data
    manifest_value = data["paths"].get("manifest_path")
    assert manifest_value is not None
    assert "\\" not in manifest_value
    assert not Path(manifest_value).is_absolute()
    # For this smoke test we expect exactly "run_manifest_1d.json"
    assert manifest_value == "run_manifest_1d.json"

