"""Unit tests for batch health checks in check_health.py.

Tests cover:
- Finding batch directories
- Checking latest batch status
- Checking failure rate
- Checking missing manifests
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Import functions from check_health (need to handle script imports)
# We'll import via sys.path manipulation
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_health import (
    check_batch_failure_rate,
    check_batch_latest_status,
    check_batch_missing_manifests,
    find_batch_directories,
    load_batch_manifest,
    load_batch_summary_csv,
    run_batch_health_checks,
)


def test_find_batch_directories(tmp_path: Path) -> None:
    """Test finding batch directories."""
    # Create batch directories
    batch1 = tmp_path / "batch1"
    batch1.mkdir()
    (batch1 / "batch_manifest.json").write_text('{"batch_name": "batch1"}', encoding="utf-8")

    batch2 = tmp_path / "batch2"
    batch2.mkdir()
    (batch2 / "batch_manifest.json").write_text('{"batch_name": "batch2"}', encoding="utf-8")

    # Non-batch directory
    non_batch = tmp_path / "not_a_batch"
    non_batch.mkdir()

    batch_dirs = find_batch_directories(tmp_path)

    assert len(batch_dirs) == 2
    assert batch1 in batch_dirs
    assert batch2 in batch_dirs


def test_load_batch_manifest(tmp_path: Path) -> None:
    """Test loading batch manifest."""
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()

    manifest_path = batch_dir / "batch_manifest.json"
    manifest_data = {
        "batch_name": "test_batch",
        "run_results_summary": {
            "total_runs": 10,
            "success_count": 8,
            "failed_count": 2,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    manifest = load_batch_manifest(batch_dir)

    assert manifest is not None
    assert manifest["batch_name"] == "test_batch"
    assert manifest["run_results_summary"]["total_runs"] == 10


def test_load_batch_summary_csv(tmp_path: Path) -> None:
    """Test loading batch summary CSV."""
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()

    summary_csv = batch_dir / "batch_summary.csv"
    df = pd.DataFrame(
        {
            "run_id": ["run1", "run2", "run3"],
            "status": ["success", "success", "failed"],
            "runtime_sec": [10.0, 20.0, 15.0],
        }
    )
    df.to_csv(summary_csv, index=False)

    loaded_df = load_batch_summary_csv(batch_dir)

    assert loaded_df is not None
    assert len(loaded_df) == 3
    assert "status" in loaded_df.columns


def test_check_batch_latest_status_success(tmp_path: Path) -> None:
    """Test batch status check with all successes."""
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()

    manifest = {
        "batch_name": "test_batch",
        "run_results_summary": {
            "total_runs": 5,
            "success_count": 5,
            "failed_count": 0,
        },
        "finished_at": datetime.utcnow().isoformat(),
    }
    manifest_path = batch_dir / "batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    check = check_batch_latest_status(batch_dir)

    assert check.status == "OK"
    assert check.value == "5/5"


def test_check_batch_latest_status_failures(tmp_path: Path) -> None:
    """Test batch status check with failures."""
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()

    manifest = {
        "batch_name": "test_batch",
        "run_results_summary": {
            "total_runs": 10,
            "success_count": 7,
            "failed_count": 3,
        },
        "finished_at": datetime.utcnow().isoformat(),
    }
    manifest_path = batch_dir / "batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    check = check_batch_latest_status(batch_dir)

    assert check.status == "WARN"
    assert "3/10" in check.details


def test_check_batch_latest_status_all_failed(tmp_path: Path) -> None:
    """Test batch status check with all failures."""
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()

    manifest = {
        "batch_name": "test_batch",
        "run_results_summary": {
            "total_runs": 5,
            "success_count": 0,
            "failed_count": 5,
        },
    }
    manifest_path = batch_dir / "batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    check = check_batch_latest_status(batch_dir)

    assert check.status == "CRITICAL"
    assert "All 5 runs failed" in check.details


def test_check_batch_failure_rate_ok(tmp_path: Path) -> None:
    """Test failure rate check within threshold."""
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()

    # Create summary CSV
    summary_csv = batch_dir / "batch_summary.csv"
    df = pd.DataFrame(
        {
            "run_id": ["run1", "run2", "run3", "run4", "run5"],
            "status": ["success", "success", "success", "success", "failed"],
        }
    )
    df.to_csv(summary_csv, index=False)

    check = check_batch_failure_rate(batch_dir, max_failure_rate=0.2)

    assert check.status == "OK"
    assert check.value == 0.2  # 1/5 = 20%


def test_check_batch_failure_rate_warn(tmp_path: Path) -> None:
    """Test failure rate check exceeding threshold."""
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()

    # Create summary CSV with 30% failure rate
    summary_csv = batch_dir / "batch_summary.csv"
    df = pd.DataFrame(
        {
            "run_id": ["run1", "run2", "run3", "run4", "run5", "run6", "run7", "run8", "run9", "run10"],
            "status": ["success", "success", "success", "success", "success", "success", "failed", "failed", "failed", "failed"],
        }
    )
    df.to_csv(summary_csv, index=False)

    check = check_batch_failure_rate(batch_dir, max_failure_rate=0.2)

    assert check.status == "WARN"
    assert check.value == 0.4  # 4/10 = 40%


def test_check_batch_missing_manifests_none_missing(tmp_path: Path) -> None:
    """Test missing manifests check with all manifests present."""
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()
    runs_dir = batch_dir / "runs"
    runs_dir.mkdir()

    # Create batch manifest
    manifest = {
        "batch_name": "test_batch",
        "run_results_summary": {
            "run_ids": ["run1", "run2", "run3"],
        },
    }
    manifest_path = batch_dir / "batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    # Create run directories with manifests
    for run_id in ["run1", "run2", "run3"]:
        run_dir = runs_dir / f"0000_{run_id}"
        run_dir.mkdir()
        run_manifest = run_dir / "run_manifest.json"
        run_manifest.write_text(f'{{"run_id": "{run_id}"}}', encoding="utf-8")

    check = check_batch_missing_manifests(batch_dir)

    assert check.status == "OK"
    assert check.value == 0


def test_check_batch_missing_manifests_some_missing(tmp_path: Path) -> None:
    """Test missing manifests check with some manifests missing."""
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()
    runs_dir = batch_dir / "runs"
    runs_dir.mkdir()

    # Create batch manifest
    manifest = {
        "batch_name": "test_batch",
        "run_results_summary": {
            "run_ids": ["run1", "run2", "run3", "run4", "run5"],
        },
    }
    manifest_path = batch_dir / "batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    # Create only 3 run directories with manifests
    for run_id in ["run1", "run2", "run3"]:
        run_dir = runs_dir / f"0000_{run_id}"
        run_dir.mkdir()
        run_manifest = run_dir / "run_manifest.json"
        run_manifest.write_text(f'{{"run_id": "{run_id}"}}', encoding="utf-8")

    check = check_batch_missing_manifests(batch_dir)

    assert check.status == "WARN" or check.status == "CRITICAL"
    assert check.value == 2  # run4 and run5 missing


def test_run_batch_health_checks_with_batches(tmp_path: Path, monkeypatch) -> None:
    """Test running batch health checks with existing batches."""
    # Create argparse namespace
    class Args:
        batch_root = None
        batch_max_failure_rate = 0.2
        skip_batch_if_missing = False

    args = Args()

    # Create batch directory structure
    batch_dir = tmp_path / "test_batch"
    batch_dir.mkdir()
    runs_dir = batch_dir / "runs"
    runs_dir.mkdir()

    # Create batch manifest
    manifest = {
        "batch_name": "test_batch",
        "finished_at": datetime.utcnow().isoformat(),
        "run_results_summary": {
            "total_runs": 3,
            "success_count": 2,
            "failed_count": 1,
            "run_ids": ["run1", "run2", "run3"],
        },
    }
    manifest_path = batch_dir / "batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    # Create batch summary CSV
    summary_csv = batch_dir / "batch_summary.csv"
    df = pd.DataFrame(
        {
            "run_id": ["run1", "run2", "run3"],
            "status": ["success", "success", "failed"],
        }
    )
    df.to_csv(summary_csv, index=False)

    # Create run directories with manifests
    for run_id in ["run1", "run2", "run3"]:
        run_subdir = runs_dir / f"0000_{run_id}"
        run_subdir.mkdir()
        run_manifest = run_subdir / "run_manifest.json"
        run_manifest.write_text(f'{{"run_id": "{run_id}"}}', encoding="utf-8")

    checks = run_batch_health_checks(tmp_path, args)

    # Should have at least: batch_root_exists, batch_directories_found, latest_status, failure_rate, missing_manifests
    assert len(checks) >= 5
    assert any(c.name == "batch_root_exists" for c in checks)
    assert any(c.name == "batch_directories_found" for c in checks)
    assert any(c.name == "batch_latest_status" for c in checks)
    assert any(c.name == "batch_failure_rate" for c in checks)
    assert any(c.name == "batch_missing_manifests" for c in checks)


def test_run_batch_health_checks_no_batches(tmp_path: Path) -> None:
    """Test running batch health checks with no batches."""
    class Args:
        batch_root = None
        batch_max_failure_rate = 0.2
        skip_batch_if_missing = False

    args = Args()

    checks = run_batch_health_checks(tmp_path, args)

    # Should have batch_root_exists and batch_directories_found (WARN)
    assert len(checks) >= 2
    assert any(c.name == "batch_root_exists" and c.status == "OK" for c in checks)
    assert any(c.name == "batch_directories_found" and c.status == "WARN" for c in checks)

