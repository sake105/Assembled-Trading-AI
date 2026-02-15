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
    """Valid evidence pack ZIP: CLI exits 0; default output is JSON."""
    zip_path = _build_valid_pack(tmp_path)
    script_path = ROOT / "scripts" / "verify_evidence_pack.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(zip_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"Expected exit 0: stderr={result.stderr}"
    data = json.loads(result.stdout)
    assert data.get("ok") is True
    assert data.get("schema_version") == 1


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
    data = json.loads(result.stdout)
    assert data.get("ok") is False
    assert "error_code" in data


def test_cli_verify_text_produces_ascii_lines(tmp_path: Path) -> None:
    """--text produces human-readable OK/FAIL ASCII line on stdout."""
    zip_path = _build_valid_pack(tmp_path)
    script_path = ROOT / "scripts" / "verify_evidence_pack.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(zip_path), "--text"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0
    assert result.stdout.strip().startswith("OK:")
    assert "ok=True" in result.stdout or "n_files=" in result.stdout
    assert result.stdout.encode("ascii", errors="ignore").decode("ascii") == result.stdout


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
    assert "details" in data
    assert isinstance(data["details"], dict)
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

    # Human-readable output (--text) must include error_code= for clarity
    result_human = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(nonexistent), "--text"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result_human.returncode == 1
    assert "error_code=FILE_NOT_FOUND" in result_human.stderr


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

    # Human-readable output (--text) must include error_code=
    result_human = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_evidence_pack.py"), "--zip", str(zip_path), "--text"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result_human.returncode == 1
    assert "error_code=" in result_human.stdout or "error_code=" in result_human.stderr
    assert "CHECKSUM_MISMATCH" in result_human.stdout or "CHECKSUM_MISMATCH" in result_human.stderr

    # Details should include bounded list of checksum mismatches (max 20)
    details = data.get("details") or {}
    mismatches = details.get("checksum_mismatches") or []
    assert isinstance(mismatches, list)
    assert 1 <= len(mismatches) <= 20


def test_cli_verify_json_details_bad_paths(tmp_path: Path) -> None:
    """--json exposes details.bad_paths (max 20 entries) for illegal paths."""
    zip_path = tmp_path / "bad_paths_pack.zip"
    # Create a minimal ZIP with an illegal path to trigger bad_paths.
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../evil.txt", "x")

    script_path = ROOT / "scripts" / "verify_evidence_pack.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(zip_path), "--json"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    # Missing manifest + bad paths -> exit 1, but JSON still returned
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data.get("schema_version") == 1
    assert data.get("ok") is False

    details = data.get("details") or {}
    bad_paths = details.get("bad_paths") or []
    assert isinstance(bad_paths, list)
    assert 1 <= len(bad_paths) <= 20


def test_cli_verify_json_relative_zip_path_resolved(tmp_path: Path) -> None:
    """--json with relative zip path: zip_path_resolved is absolute and file exists."""
    zip_path = _build_valid_pack(tmp_path)
    # Use relative path from tmp_path so that zip_path_resolved normalizes it
    try:
        rel = zip_path.relative_to(tmp_path)
    except ValueError:
        rel = zip_path.name
    script_path = ROOT / "scripts" / "verify_evidence_pack.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(rel), "--json"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data.get("ok") is True
    assert "zip_path_resolved" in data
    resolved = data["zip_path_resolved"]
    assert Path(resolved).is_absolute()
    assert Path(resolved).exists()
    assert "\\" not in resolved, "zip_path_resolved must be ASCII/POSIX (no backslashes)"


def test_cli_verify_fail_on_warn_exit_one_when_paths_not_in_zip_entries(tmp_path: Path) -> None:
    """When paths_not_in_zip_entries_count > 0, --fail-on-warn causes exit 1."""
    zip_path = _build_valid_pack(tmp_path)
    extracted_dir = tmp_path / "unzipped"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted_dir)
    manifest_path = extracted_dir / "pack_manifest_2025-01-15.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    zip_entries = list(manifest_data.get("zip_entries", []))
    assert len(zip_entries) >= 2
    manifest_data["zip_entries"] = [e for e in zip_entries if e != zip_entries[1]]
    manifest_data["zip_entries_count"] = len(manifest_data["zip_entries"])
    manifest_path.write_text(json.dumps(manifest_data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in extracted_dir.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(extracted_dir).as_posix())
    script_path = ROOT / "scripts" / "verify_evidence_pack.py"
    proc = subprocess.run(
        [sys.executable, str(script_path), "--zip", str(zip_path), "--json", "--fail-on-warn"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 1, f"--fail-on-warn with paths_not_in_zip_entries should exit 1: stdout={proc.stdout!r}"


def test_cli_verify_fail_on_warn_help() -> None:
    """--fail-on-warn flag is present in --help."""
    script_path = ROOT / "scripts" / "verify_evidence_pack.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0
    assert "fail-on-warn" in result.stdout


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
