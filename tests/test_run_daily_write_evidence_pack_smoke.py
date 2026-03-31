"""Smoke tests for run_daily.py --write-evidence-pack (Evidence Pack in Daily flow)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def test_cli_accepts_write_evidence_pack() -> None:
    """CLI accepts --write-evidence-pack flag (argparse smoke)."""
    script = ROOT / "scripts" / "run_daily.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0
    assert (
        "write-evidence-pack" in result.stdout or "write_evidence_pack" in result.stdout
    )


def test_daily_manifest_contains_evidence_pack_fields(tmp_path: Path) -> None:
    """Daily manifest includes evidence_index_path, evidence_pack_path, evidence_pack_manifest_path, write_evidence_pack.
    Daily does not run ledger/accounting; paths are expected to be None.
    """
    import run_daily

    _write_daily_manifest = run_daily._write_daily_manifest

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_path = output_dir / "orders_20250115.csv"
    safe_path.write_text("timestamp,symbol,side,qty,price\n", encoding="utf-8")

    _write_daily_manifest(
        output_dir=output_dir,
        run_id="daily_20250115",
        target_date=datetime(2025, 1, 15),
        safe_path=safe_path,
        broker_snapshot_policy="prefer",
        write_broker_snapshot=False,
        broker_snapshot_run_id=None,
        broker_snapshot_file=None,
        broker_snapshot_date=None,
        broker_snapshot_import_ok=None,
        write_evidence_pack=True,
        evidence_index_path=None,
        evidence_pack_path=None,
        evidence_pack_manifest_path=None,
    )

    manifest_path = output_dir / "manifest_daily_daily_20250115.json"
    assert manifest_path.exists()
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert "evidence_index_path" in data
    assert "evidence_pack_path" in data
    assert "evidence_pack_manifest_path" in data
    assert "write_evidence_pack" in data
    assert data["write_evidence_pack"] is True
    # Daily currently does not build ledger -> evidence pack paths remain None (no false expectations)
    assert data["evidence_index_path"] is None
    assert data["evidence_pack_path"] is None
    assert data["evidence_pack_manifest_path"] is None


def test_daily_manifest_evidence_pack_paths_when_set(tmp_path: Path) -> None:
    """Daily manifest stores evidence pack paths (relative POSIX) when provided."""
    import run_daily

    _write_daily_manifest = run_daily._write_daily_manifest

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_path = output_dir / "orders_20250115.csv"
    safe_path.write_text("timestamp,symbol,side,qty,price\n", encoding="utf-8")

    _write_daily_manifest(
        output_dir=output_dir,
        run_id="daily_20250115",
        target_date=datetime(2025, 1, 15),
        safe_path=safe_path,
        broker_snapshot_policy="prefer",
        write_broker_snapshot=False,
        broker_snapshot_run_id=None,
        broker_snapshot_file=None,
        broker_snapshot_date=None,
        broker_snapshot_import_ok=None,
        write_evidence_pack=True,
        evidence_index_path="evidence_daily_20250115/evidence_2025-01-15.json",
        evidence_pack_path="evidence_daily_20250115/pack_2025-01-15.zip",
        evidence_pack_manifest_path="evidence_daily_20250115/pack_manifest_2025-01-15.json",
    )

    manifest_path = output_dir / "manifest_daily_daily_20250115.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert (
        data["evidence_index_path"]
        == "evidence_daily_20250115/evidence_2025-01-15.json"
    )
    assert data["evidence_pack_path"] == "evidence_daily_20250115/pack_2025-01-15.zip"
    assert (
        data["evidence_pack_manifest_path"]
        == "evidence_daily_20250115/pack_manifest_2025-01-15.json"
    )
    assert "\\" not in data["evidence_pack_path"]
