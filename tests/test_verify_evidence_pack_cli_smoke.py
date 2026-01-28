"""Smoke tests for verify_evidence_pack.py CLI (offline Evidence Pack validation)."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json
from src.assembled_core.accounting.evidence_pack import build_evidence_pack


def _build_valid_pack(tmp_path: Path) -> Path:
    """Build a minimal valid evidence pack and return path to ZIP."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "cli_verify_ok"
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
    zip_path: Path = output_dir / result["pack_path"]
    return zip_path


def test_cli_verify_ok_exits_zero(tmp_path: Path) -> None:
    """Valid evidence pack ZIP: CLI exits 0 and prints OK."""
    zip_path = _build_valid_pack(tmp_path)
    script_path = ROOT / "scripts" / "verify_evidence_pack.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(zip_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"Expected exit 0: stderr={result.stderr}"
    assert "OK:" in result.stdout
    assert "ok=True" in result.stdout or "n_files=" in result.stdout


def test_cli_verify_fail_exits_one(tmp_path: Path) -> None:
    """Invalid or tampered ZIP: CLI exits 1 and prints FAIL or ERROR."""
    zip_path = _build_valid_pack(tmp_path)
    extracted_dir = tmp_path / "unzipped"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted_dir)
        namelist = zf.namelist()

    target_name = None
    for name in namelist:
        if not name.startswith("pack_manifest_"):
            target_name = name
            break
    assert target_name is not None

    target_path = extracted_dir / target_name
    target_path.write_text("tampered content", encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in extracted_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(extracted_dir).as_posix()
                zf.write(file_path, arcname=arcname)

    script_path = ROOT / "scripts" / "verify_evidence_pack.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(zip_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 1, "Expected exit 1 for tampered pack"
    assert "FAIL:" in result.stdout or "ERROR:" in result.stderr


def test_cli_verify_json_output_is_valid_and_deterministic(tmp_path: Path) -> None:
    """--json produces valid, deterministic JSON; two runs yield identical bytes."""
    zip_path = _build_valid_pack(tmp_path)
    script_path = ROOT / "scripts" / "verify_evidence_pack.py"
    cmd = [sys.executable, str(script_path), "--zip", str(zip_path), "--json"]

    result1 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    result2 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    assert result1.returncode == 0 and result2.returncode == 0
    out1 = result1.stdout
    out2 = result2.stdout
    assert out1 == out2, "Two runs must produce identical JSON bytes"

    data = json.loads(out1)
    assert data.get("schema_version") == 1
    assert "zip_path" in data
    assert data.get("error_code") == ""
    assert "ok" in data
    assert "n_files" in data
    assert "missing_manifest" in data
    assert "bad_paths_count" in data
    assert "checksum_mismatches_count" in data
    assert data["ok"] is True
    assert out1.endswith("\n")


def test_cli_verify_json_error_code_file_not_found(tmp_path: Path) -> None:
    """--json with missing file returns error_code FILE_NOT_FOUND."""
    script_path = ROOT / "scripts" / "verify_evidence_pack.py"
    nonexistent = tmp_path / "nonexistent.zip"
    assert not nonexistent.exists()

    result = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(nonexistent), "--json"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data.get("schema_version") == 1
    assert data.get("zip_path") == str(nonexistent)
    assert data.get("ok") is False
    assert data.get("error_code") == "FILE_NOT_FOUND"


def test_cli_verify_json_error_code_checksum_mismatch(tmp_path: Path) -> None:
    """--json with tampered ZIP returns error_code CHECKSUM_MISMATCH (or FAIL with error_code)."""
    zip_path = _build_valid_pack(tmp_path)
    extracted_dir = tmp_path / "unzipped"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted_dir)
        namelist = zf.namelist()

    target_name = next((n for n in namelist if not n.startswith("pack_manifest_")), None)
    assert target_name is not None
    (extracted_dir / target_name).parent.mkdir(parents=True, exist_ok=True)
    (extracted_dir / target_name).write_text("tampered content", encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in extracted_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(extracted_dir).as_posix()
                zf.write(file_path, arcname=arcname)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_evidence_pack.py"), "--zip", str(zip_path), "--json"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data.get("schema_version") == 1
    assert data.get("ok") is False
    assert data.get("error_code") == "CHECKSUM_MISMATCH"


def test_cli_errors_ascii_only(tmp_path: Path) -> None:
    """All status and error lines are ASCII-only."""
    script_path = ROOT / "scripts" / "verify_evidence_pack.py"
    nonexistent = tmp_path / "nonexistent.zip"
    assert not nonexistent.exists()

    result = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(nonexistent)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    ascii_only = combined.encode("ascii", errors="ignore").decode("ascii")
    assert combined == ascii_only, "Output must be ASCII-only"
