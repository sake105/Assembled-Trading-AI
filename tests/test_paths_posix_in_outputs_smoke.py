"""Smoke test: path fields in Evidence Index and Pack Manifest JSON are POSIX (no backslashes).

Uses stdlib + existing modules only; runs in under 1s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json
from src.assembled_core.accounting.evidence_pack import build_evidence_pack


def test_evidence_index_and_pack_manifest_paths_posix(tmp_path: Path) -> None:
    """Evidence Index and Pack Manifest JSON path fields contain no backslashes and no absolute paths."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "posix_smoke"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy", encoding="utf-8")

    evidence_paths = {
        "broker_snapshot_path": None,
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": None,
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
    evidence_index_path = (
        output_dir / f"evidence_{run_id}" / f"evidence_{date_str}.json"
    )
    with open(evidence_index_path, encoding="utf-8") as f:
        evidence_index = json.load(f)
    paths_obj = evidence_index.get("paths") or {}
    for key, val in paths_obj.items():
        if val is not None and isinstance(val, str):
            assert "\\" not in val, (
                f"Evidence index paths.{key} must not contain backslash: {val!r}"
            )
            if val and not val.startswith("/"):
                p = Path(val)
                assert not p.is_absolute(), (
                    f"Evidence index paths.{key} should be relative: {val!r}"
                )

    result = build_evidence_pack(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of_date,
        include_optional=True,
    )
    pack_manifest_path = output_dir / result["pack_manifest_path"]
    with open(pack_manifest_path, encoding="utf-8") as f:
        pack_manifest = json.load(f)
    source_path = pack_manifest.get("source_path")
    if source_path is not None and isinstance(source_path, str):
        assert "\\" not in source_path, (
            f"Pack manifest source_path must not contain backslash: {source_path!r}"
        )
        if source_path and not source_path.startswith("/"):
            assert not Path(source_path).is_absolute()
    for entry in pack_manifest.get("files") or []:
        path_val = entry.get("path")
        if path_val is not None and isinstance(path_val, str):
            assert "\\" not in path_val, (
                f"Pack manifest files[].path must not contain backslash: {path_val!r}"
            )
            if path_val and not path_val.startswith("/"):
                assert not Path(path_val).is_absolute(), (
                    f"Pack manifest file path should be relative: {path_val!r}"
                )
