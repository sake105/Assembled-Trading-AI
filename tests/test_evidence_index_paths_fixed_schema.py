"""Evidence index: paths object has fixed schema (all keys always present, null if missing)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import PATHS_KEYS, write_evidence_index_json


def test_evidence_index_paths_contains_all_keys_with_null(tmp_path: Path) -> None:
    """Write index without some paths; JSON paths object contains all keys (missing -> null)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "ledger" / "ledger.parquet"
    paths = {
        "ledger_pack_path": ledger_path,
        # Omit broker_snapshot_path, reconcile_report_path, accounting_report_path, manifest_path
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("x", encoding="utf-8")

    out_path = write_evidence_index_json(
        output_dir=output_dir,
        run_id="fixed_schema",
        as_of_date="2025-01-15",
        paths=paths,
        broker_meta=None,
        reconciliation_ok=None,
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    p = data.get("paths")
    assert p is not None
    for key in PATHS_KEYS:
        assert key in p, f"paths must contain key: {key}"
    assert p.get("ledger_pack_path") is not None
    assert p.get("broker_snapshot_path") is None
    assert p.get("reconcile_report_path") is None
    assert p.get("accounting_report_path") is None
    assert p.get("manifest_path") is None


def test_evidence_index_paths_determinism_two_writes_same_bytes(tmp_path: Path) -> None:
    """Two writes with same inputs produce identical JSON bytes."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "ledger" / "ledger.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("dummy", encoding="utf-8")
    paths = {
        "broker_snapshot_path": None,
        "ledger_pack_path": ledger_path,
        "reconcile_report_path": None,
        "accounting_report_path": None,
        "manifest_path": None,
    }

    out1 = write_evidence_index_json(
        output_dir=output_dir,
        run_id="det",
        as_of_date="2025-01-16",
        paths=paths,
        broker_meta=None,
        reconciliation_ok=None,
    )
    bytes1 = out1.read_bytes()

    out2 = write_evidence_index_json(
        output_dir=output_dir,
        run_id="det",
        as_of_date="2025-01-16",
        paths=paths,
        broker_meta=None,
        reconciliation_ok=None,
    )
    bytes2 = out2.read_bytes()

    assert bytes1 == bytes2, "Two writes with same inputs must yield identical bytes"
