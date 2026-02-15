"""Tests for Evidence Index accounting_report_path backfill from orchestrator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json
from src.assembled_core.pipeline.orchestrator import (
    _backfill_evidence_index_accounting_path,
    _manifest_path_str,
)


def test_evidence_index_accounting_backfill_sets_relative_posix(tmp_path: Path) -> None:
    """Evidence Index without accounting_report_path; backfill sets path (relative, POSIX, no backslashes)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "evidence_accounting_backfill"
    as_of_date = "2025-01-15"

    # Known accounting report path (relative to output_dir)
    accounting_rel = Path("evidence_run") / "accounting_report_abc" / "report.csv"
    accounting_abs = output_dir / accounting_rel
    accounting_abs.parent.mkdir(parents=True, exist_ok=True)
    accounting_abs.write_text("dummy", encoding="utf-8")

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

    evidence_path = write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of_date,
        paths=paths,
        broker_meta=None,
        reconciliation_ok=None,
    )

    evidence_index_rel = _manifest_path_str(evidence_path, base_dir=output_dir)
    ledger_result = {
        "evidence_index_path": evidence_index_rel,
        "accounting_report_path": str(accounting_rel),
    }

    _backfill_evidence_index_accounting_path(base_dir=output_dir, ledger_result=ledger_result)

    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert "paths" in data
    value = data["paths"].get("accounting_report_path")
    assert value is not None
    assert "\\" not in value
    assert not Path(value).is_absolute()
    assert value == accounting_rel.as_posix()


def test_evidence_index_accounting_backfill_byte_determinism(tmp_path: Path) -> None:
    """Two backfill runs produce identical Evidence Index bytes."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "determinism"
    as_of_date = "2025-01-16"
    accounting_rel = Path("evidence_det") / "accounting_report_x" / "report.csv"
    accounting_abs = output_dir / accounting_rel
    accounting_abs.parent.mkdir(parents=True, exist_ok=True)
    accounting_abs.write_text("x", encoding="utf-8")

    ledger_pack_path = output_dir / "ledger" / "ledger.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("x", encoding="utf-8")

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

    evidence_index_rel = _manifest_path_str(evidence_path, base_dir=output_dir)
    ledger_result = {
        "evidence_index_path": evidence_index_rel,
        "accounting_report_path": str(accounting_rel),
    }

    _backfill_evidence_index_accounting_path(base_dir=output_dir, ledger_result=ledger_result)
    content1 = evidence_path.read_bytes()

    _backfill_evidence_index_accounting_path(base_dir=output_dir, ledger_result=ledger_result)
    content2 = evidence_path.read_bytes()

    assert content1 == content2


def test_evidence_index_accounting_backfill_does_not_overwrite(tmp_path: Path) -> None:
    """Backfill only sets when missing; does not overwrite existing accounting_report_path."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "no_overwrite"
    as_of_date = "2025-01-17"
    existing_value = "existing/accounting/report.csv"

    ledger_pack_path = output_dir / "ledger" / "ledger.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("x", encoding="utf-8")

    paths = {
        "broker_snapshot_path": None,
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": None,
        "accounting_report_path": existing_value,
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

    evidence_index_rel = _manifest_path_str(evidence_path, base_dir=output_dir)
    ledger_result = {
        "evidence_index_path": evidence_index_rel,
        "accounting_report_path": "other/accounting_report_new/report.csv",
    }

    _backfill_evidence_index_accounting_path(base_dir=output_dir, ledger_result=ledger_result)

    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert data["paths"].get("accounting_report_path") == existing_value
