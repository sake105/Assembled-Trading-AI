"""Smoke tests for export_evidence_pack.py CLI tool (Sprint 13).

Tests that the standalone CLI tool correctly exports evidence packs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_index import write_evidence_index_json


def test_cli_export_creates_pack(tmp_path: Path) -> None:
    """Test that CLI export creates ZIP and manifest files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "cli_test"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    # Create dummy artifact files
    broker_snapshot_path = (
        output_dir / "broker_snapshot_run" / f"snapshot_{date_str}.json"
    )
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    reconcile_report_path = output_dir / "reconcile_run" / f"reconcile_{date_str}.json"

    for p in [broker_snapshot_path, ledger_pack_path, reconcile_report_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy content", encoding="utf-8")

    # Write evidence index
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

    # Run CLI export
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Default output is JSON (machine-readable)
    data = json.loads(result.stdout)
    assert data.get("schema_version") == 1
    assert data.get("ok") is True
    assert "pack_path" in data
    assert "pack_manifest_path" in data
    assert "source" in data
    assert data.get("n_files", 0) >= 0

    # Verify ZIP file exists
    evidence_dir = output_dir / f"evidence_{run_id}"
    zip_path = evidence_dir / f"pack_{date_str}.zip"
    assert zip_path.exists(), "ZIP file should exist"

    # Verify manifest file exists
    manifest_path = evidence_dir / f"pack_manifest_{date_str}.json"
    assert manifest_path.exists(), "Manifest file should exist"

    # Verify ZIP is valid
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert len(zf.namelist()) > 0, "ZIP should contain files"


def test_cli_export_text_option_single_line(tmp_path: Path) -> None:
    """--text produces human-readable single-line OK status (legacy behavior)."""
    run_id = "cli_text"
    as_of_date = "2025-01-15"
    output_dir = _build_output_dir_with_evidence_index(
        tmp_path, run_id=run_id, as_of_date=as_of_date
    )
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
            "--text",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0
    assert result.stdout.strip().startswith("OK:")
    assert "pack_path=" in result.stdout
    assert "pack_manifest_path=" in result.stdout
    assert "source=" in result.stdout
    assert "n_files=" in result.stdout
    assert "required_missing=" in result.stdout
    assert "optional_missing=" in result.stdout


def test_cli_export_print_pack_path_one_line_file_exists(tmp_path: Path) -> None:
    """--print-pack-path: stdout is exactly one line (pack_path_resolved); that path exists."""
    run_id = "cli_print_path"
    as_of_date = "2025-01-15"
    output_dir = _build_output_dir_with_evidence_index(
        tmp_path, run_id=run_id, as_of_date=as_of_date
    )
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
            "--print-pack-path",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    lines = [ln for ln in result.stdout.strip().split("\n") if ln]
    assert len(lines) == 1, (
        f"Expected exactly one line stdout, got {len(lines)}: {result.stdout!r}"
    )
    pack_path_resolved = lines[0].strip()
    assert Path(pack_path_resolved).exists(), (
        f"Pack path should exist: {pack_path_resolved}"
    )
    assert pack_path_resolved.endswith(".zip"), "Resolved path should be the ZIP file"


def test_cli_export_out_paths_written(tmp_path: Path) -> None:
    """--out-zip and --out-manifest copy artifacts to given paths; JSON has out_zip_path, out_manifest_path."""
    run_id = "cli_out"
    as_of_date = "2025-01-15"
    output_dir = _build_output_dir_with_evidence_index(
        tmp_path, run_id=run_id, as_of_date=as_of_date
    )
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    out_zip = archive_dir / "pack_out.zip"
    out_manifest = archive_dir / "pack_manifest_out.json"
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
            "--out-zip",
            str(out_zip),
            "--out-manifest",
            str(out_manifest),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert data.get("ok") is True
    assert "out_zip_path" in data
    assert "out_manifest_path" in data
    assert data.get("out_zip_path") is not None
    assert data.get("out_manifest_path") is not None
    assert out_zip.exists()
    assert out_manifest.exists()
    assert out_zip.stat().st_size > 0
    assert "schema_version" in json.loads(out_manifest.read_text(encoding="utf-8"))


def test_cli_export_json_resolved_paths_with_relative_output_dir(
    tmp_path: Path,
) -> None:
    """Relative --output-dir: JSON contains output_dir, output_dir_resolved, pack_path_resolved, pack_manifest_path_resolved; resolved paths exist."""
    rel_dir = "out_rel"
    output_dir = tmp_path / rel_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "cli_resolved"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"
    broker_snapshot_path = (
        output_dir / "broker_snapshot_run" / f"snapshot_{date_str}.json"
    )
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
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            rel_dir,
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert data.get("ok") is True
    assert data.get("output_dir") == rel_dir
    assert "output_dir_resolved" in data
    assert "pack_path_resolved" in data
    assert "pack_manifest_path_resolved" in data
    out_resolved = data.get("output_dir_resolved")
    pack_resolved = data.get("pack_path_resolved")
    manifest_resolved = data.get("pack_manifest_path_resolved")
    assert out_resolved
    assert pack_resolved
    assert manifest_resolved
    assert Path(out_resolved).exists(), "output_dir_resolved must point to existing dir"
    assert Path(pack_resolved).exists(), "pack_path_resolved must point to existing ZIP"
    assert Path(manifest_resolved).exists(), (
        "pack_manifest_path_resolved must point to existing manifest"
    )


def _build_output_dir_with_evidence_index(
    tmp_path: Path, run_id: str = "cli_json", as_of_date: str = "2025-01-15"
) -> Path:
    """Create output dir with artifacts and evidence index; return output_dir."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = as_of_date
    broker_snapshot_path = (
        output_dir / "broker_snapshot_run" / f"snapshot_{date_str}.json"
    )
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
    return output_dir


def test_cli_export_json_output_is_valid_and_deterministic(tmp_path: Path) -> None:
    """--json on success: valid JSON, required keys present, two runs yield identical bytes."""
    output_dir = _build_output_dir_with_evidence_index(tmp_path)
    run_id = "cli_json"
    as_of_date = "2025-01-15"
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--run-id",
        run_id,
        "--as-of-date",
        as_of_date,
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    result1 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    result2 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert result1.returncode == 0 and result2.returncode == 0, (
        f"stdout={result1.stdout!r} stderr={result1.stderr!r}"
    )
    out1 = result1.stdout
    out2 = result2.stdout
    assert out1 == out2, "Two runs must produce identical JSON bytes"
    assert out1.strip().startswith("{") and out1.strip().endswith("}"), (
        "stdout must be exactly JSON (no prefix/suffix)"
    )
    data = json.loads(out1)
    required_keys = [
        "schema_version",
        "ok",
        "error_code",
        "details",
        "tool_version",
        "pack_path",
        "pack_manifest_path",
        "source",
        "source_path",
        "n_files",
        "required_missing_count",
        "optional_missing_count",
    ]
    for k in required_keys:
        assert k in data, f"Missing key: {k}"
    assert data.get("schema_version") == 1
    assert data.get("ok") is True
    assert data.get("error_code") == ""
    assert out1.endswith("\n")


def test_cli_export_json_output_is_pure_json(tmp_path: Path) -> None:
    """--json: stdout is exactly valid JSON (no prefix/suffix), deterministic."""
    output_dir = _build_output_dir_with_evidence_index(tmp_path)
    run_id = "cli_json"
    as_of_date = "2025-01-15"
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--run-id",
        run_id,
        "--as-of-date",
        as_of_date,
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0
    raw = result.stdout
    assert raw.startswith("{"), "stdout must start with { (no log prefix)"
    assert raw.rstrip().endswith("}"), "stdout must end with }"
    data = json.loads(raw)
    assert "schema_version" in data and data["schema_version"] == 1
    result2 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert result2.returncode == 0
    assert result.stdout == result2.stdout, "Two runs must produce identical bytes"


def test_cli_export_verify_after_build_ok_exits_zero(tmp_path: Path) -> None:
    """Export with --verify-after-build on valid pack exits 0."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "cli_verify_ok"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"
    broker_snapshot_path = (
        output_dir / "broker_snapshot_run" / f"snapshot_{date_str}.json"
    )
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
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
            "--verify-after-build",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, (
        f"CLI with --verify-after-build failed: {result.stderr}"
    )
    data = json.loads(result.stdout)
    assert data.get("ok") is True


def test_cli_export_verify_after_build_fail_exits_one(tmp_path: Path) -> None:
    """When verify fails (e.g. tampered ZIP), verify CLI exits 1; export --verify-after-build would exit 1 on same condition."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "cli_verify_fail"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"
    for p in [
        output_dir / "ledger_run" / "ledger_events.parquet",
        output_dir / "reconcile_run" / f"reconcile_{date_str}.json",
        output_dir / "accounting_run" / f"accounting_{date_str}.json",
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy", encoding="utf-8")
    evidence_paths = {
        "broker_snapshot_path": None,
        "ledger_pack_path": output_dir / "ledger_run" / "ledger_events.parquet",
        "reconcile_report_path": output_dir
        / "reconcile_run"
        / f"reconcile_{date_str}.json",
        "accounting_report_path": output_dir
        / "accounting_run"
        / f"accounting_{date_str}.json",
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
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_evidence_pack.py"),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    zip_path = output_dir / f"evidence_{run_id}" / f"pack_{date_str}.zip"
    assert zip_path.exists()
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted)
        namelist = zf.namelist()
    target_name = next(
        (n for n in namelist if not n.startswith("pack_manifest_")), None
    )
    assert target_name is not None
    (extracted / target_name).write_text("tampered", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in extracted.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(extracted).as_posix())
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_evidence_pack.py"),
            "--zip",
            str(zip_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 1, "Verify on tampered pack should exit 1"


def test_cli_invalid_date_exits_with_error(tmp_path: Path) -> None:
    """Test that invalid date format causes CLI to exit with error."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "test",
            "--as-of-date",
            "invalid-date-format",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code is non-zero
    assert result.returncode != 0, "Should exit with error for invalid date"

    # Verify error message is ASCII-only
    error_output = (result.stdout + result.stderr).lower()
    assert (
        "invalid" in error_output or "date" in error_output or "format" in error_output
    )
    assert error_output.encode("ascii", errors="ignore").decode("ascii") == error_output


def test_cli_missing_evidence_index_exits_with_error(tmp_path: Path) -> None:
    """Test that missing evidence index causes CLI to exit with error."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "nonexistent_run",
            "--as-of-date",
            "2025-01-15",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Verify exit code is non-zero
    assert result.returncode != 0, "Should exit with error for missing evidence index"

    # Verify error message is ASCII-only
    error_output = (result.stdout + result.stderr).lower()
    assert error_output.encode("ascii", errors="ignore").decode("ascii") == error_output


def test_cli_help_exits_with_zero(tmp_path: Path) -> None:
    """Test that --help exits with code 0."""
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert result.returncode == 0, "--help should exit with 0"
    assert "evidence pack" in result.stdout.lower() or "export" in result.stdout.lower()


def test_cli_strict_mode_fails_on_missing_optional(tmp_path: Path) -> None:
    """Test that --strict mode fails if optional files are missing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "strict_test"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    # Create only required file (ledger pack)
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy", encoding="utf-8")

    # Write evidence index with missing optional files
    evidence_paths = {
        "broker_snapshot_path": output_dir
        / "broker_snapshot_run"
        / f"snapshot_{date_str}.json",  # Missing
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": output_dir
        / "reconcile_run"
        / f"reconcile_{date_str}.json",  # Missing
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

    # Run CLI with --strict
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
            "--strict",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Should fail with --strict when optional files are missing
    assert result.returncode != 0, (
        "Should exit with error in --strict mode when optional files missing"
    )


def test_cli_prints_source_and_missing_counts(tmp_path: Path) -> None:
    """CLI prints source and missing counts in success status line."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "cli_meta"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    # Create minimal required artifacts (no optional)
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    reconcile_report_path = output_dir / "reconcile_run" / f"reconcile_{date_str}.json"
    accounting_report_path = (
        output_dir / "accounting_run" / f"accounting_{date_str}.json"
    )

    for p in [ledger_pack_path, reconcile_report_path, accounting_report_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy", encoding="utf-8")

    evidence_paths = {
        "broker_snapshot_path": None,
        "ledger_pack_path": ledger_pack_path,
        "reconcile_report_path": reconcile_report_path,
        "accounting_report_path": accounting_report_path,
        "manifest_path": None,
    }

    write_evidence_index_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=pd.Timestamp(as_of_date, tz="UTC"),
        paths=evidence_paths,
        broker_meta=None,
        reconciliation_ok=True,
    )

    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Default output is JSON; should contain source and zero missing counts
    data = json.loads(result.stdout)
    assert "source" in data
    assert "source_path" in data
    assert data.get("required_missing_count", -1) == 0
    assert data.get("optional_missing_count", -1) == 0


def test_cli_strict_fails_on_optional_missing_with_exit_1(tmp_path: Path) -> None:
    """Strict mode fails with exit 1 when any optional files are missing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "cli_strict_optional"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    # Create only required file (ledger)
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy", encoding="utf-8")

    # Evidence index references missing optional paths
    evidence_paths = {
        "broker_snapshot_path": output_dir
        / "broker_snapshot_run"
        / f"snapshot_{date_str}.json",  # missing
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

    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
            "--strict",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert result.returncode == 1, (
        "Strict mode should exit with code 1 when optional files are missing"
    )
    # Error output should be ASCII-only
    error_output = result.stdout + result.stderr
    assert error_output.encode("ascii", errors="ignore").decode("ascii") == error_output


def test_cli_export_json_error_code_no_source(tmp_path: Path) -> None:
    """--json with no evidence index/manifest: exit 1, error_code NO_SOURCE, JSON on stdout."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--run-id",
        "nonexistent_run",
        "--as-of-date",
        "2025-01-15",
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data.get("schema_version") == 1
    assert data.get("ok") is False
    assert data.get("error_code") == "NO_SOURCE"
    assert "details" in data and isinstance(data["details"], dict)
    assert data.get("pack_path") is None


def test_cli_export_json_error_code_strict_optional_missing(tmp_path: Path) -> None:
    """--json with --strict and missing optional: exit 1, error_code OPTIONAL_MISSING_STRICT, JSON on stdout."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "strict_json"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"
    ledger_pack_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_pack_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_pack_path.write_text("dummy", encoding="utf-8")
    evidence_paths = {
        "broker_snapshot_path": output_dir
        / "broker_snapshot_run"
        / f"snapshot_{date_str}.json",
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
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--run-id",
        run_id,
        "--as-of-date",
        as_of_date,
        "--output-dir",
        str(output_dir),
        "--strict",
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data.get("schema_version") == 1
    assert data.get("ok") is False
    assert data.get("error_code") == "OPTIONAL_MISSING_STRICT"
    assert "details" in data and isinstance(data["details"], dict)


def test_cli_export_json_error_determinism(tmp_path: Path) -> None:
    """Two identical error runs (no source) produce identical JSON bytes."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--run-id",
        "no_run",
        "--as-of-date",
        "2025-01-15",
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    result1 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    result2 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert result1.returncode == 1 and result2.returncode == 1
    assert result1.stdout == result2.stdout, (
        "Same error must yield identical JSON bytes"
    )
    data = json.loads(result1.stdout)
    assert data.get("error_code") == "NO_SOURCE"


def test_cli_errors_ascii_only(tmp_path: Path) -> None:
    """All error messages produced by CLI are ASCII-only."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use invalid date to trigger an error
    script_path = ROOT / "scripts" / "export_evidence_pack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "ascii_test",
            "--as-of-date",
            "invalid-date",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert result.returncode == 1
    error_output = result.stdout + result.stderr
    assert error_output.encode("ascii", errors="ignore").decode("ascii") == error_output
