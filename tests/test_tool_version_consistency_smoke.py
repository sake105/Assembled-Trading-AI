"""Smoke test: tool_version is single authority (assembled_core.__version__) across Evidence Index, Pack, Verify, Export."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core import __version__ as CORE_VERSION
from src.assembled_core.accounting.evidence_index import write_evidence_index_json
from src.assembled_core.accounting.evidence_pack import build_evidence_pack


def test_tool_version_consistent_across_evidence_artifacts(tmp_path: Path) -> None:
    """Evidence Index, Pack manifest, Verify JSON, and Export JSON all report same tool_version."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "tool_ver"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    ledger_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("dummy", encoding="utf-8")

    paths = {
        "broker_snapshot_path": None,
        "ledger_pack_path": ledger_path,
        "reconcile_report_path": None,
        "accounting_report_path": None,
        "manifest_path": None,
    }

    write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=pd.Timestamp(as_of_date, tz="UTC"),
        paths=paths,
        broker_meta=None,
        reconciliation_ok=None,
    )

    evidence_index_path = (
        output_dir / f"evidence_{run_id}" / f"evidence_{date_str}.json"
    )
    with evidence_index_path.open("r", encoding="utf-8") as f:
        evidence_index_data = json.load(f)
    index_tool_version = evidence_index_data.get("tool_version")
    assert index_tool_version == CORE_VERSION, (
        "Evidence Index tool_version must match core"
    )

    result = build_evidence_pack(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of_date,
        include_optional=True,
    )
    pack_manifest_path = output_dir / result["pack_manifest_path"]
    with pack_manifest_path.open("r", encoding="utf-8") as f:
        pack_manifest_data = json.load(f)
    pack_tool_version = pack_manifest_data.get("tool_version")
    assert pack_tool_version == CORE_VERSION, (
        "Pack manifest tool_version must match core"
    )

    zip_path = output_dir / result["pack_path"]
    verify_script = ROOT / "scripts" / "verify_evidence_pack.py"
    verify_result = subprocess.run(
        [sys.executable, str(verify_script), "--zip", str(zip_path), "--json"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert verify_result.returncode == 0
    verify_data = json.loads(verify_result.stdout)
    verify_tool_version = verify_data.get("tool_version")
    assert verify_tool_version == CORE_VERSION, (
        "Verify JSON tool_version must match core"
    )

    export_script = ROOT / "scripts" / "export_evidence_pack.py"
    export_result = subprocess.run(
        [
            sys.executable,
            str(export_script),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert export_result.returncode == 0
    export_data = json.loads(export_result.stdout)
    export_tool_version = export_data.get("tool_version")
    assert export_tool_version == CORE_VERSION, (
        "Export JSON tool_version must match core"
    )

    assert (
        index_tool_version
        == pack_tool_version
        == verify_tool_version
        == export_tool_version
    ), "All tool_version values must be identical"
