"""Unit tests for batch runner manifest generation.

Tests cover:
- Run manifest writing/reading (roundtrip)
- Batch manifest writing/reading (roundtrip)
- Stable ordering in manifests
- Config hash computation
- Artifact collection
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.assembled_core.experiments.batch_config import BatchConfig, RunSpec
from src.assembled_core.experiments.batch_runner import (
    BatchResult,
    RunResult,
    _collect_run_artifacts,
    _compute_config_hash,
    _write_batch_manifest,
    _write_run_manifest,
)


def test_compute_config_hash_deterministic() -> None:
    """Test that config hash is deterministic."""
    config1 = {"freq": "1d", "strategy": "test", "param": 100}
    config2 = {"strategy": "test", "freq": "1d", "param": 100}  # Different order

    hash1 = _compute_config_hash(config1)
    hash2 = _compute_config_hash(config2)

    # Should be the same despite different key order
    assert hash1 == hash2
    assert len(hash1) == 16  # First 16 chars of SHA256


def test_collect_run_artifacts_stable_ordering(tmp_path: Path) -> None:
    """Test that artifact collection returns stable ordering."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Create artifacts in non-alphabetical order
    (run_dir / "zebra.csv").write_text("data")
    (run_dir / "alpha.json").write_text("{}")
    (run_dir / "beta.md").write_text("# Report")

    artifacts = _collect_run_artifacts(run_dir)

    # Should be sorted
    assert artifacts == sorted(artifacts)
    assert len(artifacts) == 3


def test_write_run_manifest_roundtrip(tmp_path: Path) -> None:
    """Test that run manifest can be written and read back."""
    run_output_dir = tmp_path / "run"
    run_output_dir.mkdir()

    run_spec = RunSpec(
        id="test_run",
        bundle_path=Path("config/bundle.yaml"),
        start_date="2015-01-01",
        end_date="2020-12-31",
        tags=["test"],
        overrides={"param": 42},
    )

    base_args = {"freq": "1d", "strategy": "test"}

    result = RunResult(
        run_id="test_run",
        status="success",
        output_dir=run_output_dir,
        runtime_sec=10.5,
        run_index=0,
    )

    start_time = datetime.utcnow()
    end_time = datetime.utcnow()

    # Write manifest
    _write_run_manifest(
        result=result,
        run_spec=run_spec,
        base_args=base_args,
        run_output_dir=run_output_dir,
        start_time=start_time,
        end_time=end_time,
    )

    # Read back
    manifest_path = run_output_dir / "run_manifest.json"
    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify structure
    assert manifest["run_id"] == "test_run"
    assert manifest["status"] == "success"
    assert manifest["runtime_sec"] == 10.5
    assert manifest["run_index"] == 0
    assert "config_hash" in manifest
    assert "git_commit_hash" in manifest  # May be None if no git
    assert "started_at" in manifest
    assert "finished_at" in manifest
    assert "artifacts" in manifest
    # Handle path separator differences (Windows vs Unix)
    assert manifest["run_spec"]["bundle_path"].replace("\\", "/") == "config/bundle.yaml"


def test_write_batch_manifest_roundtrip(tmp_path: Path) -> None:
    """Test that batch manifest can be written and read back."""
    batch_output_dir = tmp_path / "batch"
    batch_output_dir.mkdir()

    config = BatchConfig(
        batch_name="test_batch",
        description="Test batch",
        output_root=Path("output"),
        base_args={"freq": "1d"},
        runs=[],
        seed=42,
    )

    run_specs = [
        RunSpec(
            id="run1",
            bundle_path=Path("config/bundle.yaml"),
            start_date="2015-01-01",
            end_date="2020-12-31",
        ),
        RunSpec(
            id="run2",
            bundle_path=Path("config/bundle.yaml"),
            start_date="2016-01-01",
            end_date="2021-12-31",
        ),
    ]

    batch_result = BatchResult(
        batch_name="test_batch",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        total_runtime_sec=100.0,
        run_results=[
            RunResult(run_id="run1", status="success", output_dir=tmp_path / "run1", runtime_sec=50.0),
            RunResult(run_id="run2", status="success", output_dir=tmp_path / "run2", runtime_sec=50.0),
        ],
    )

    # Write manifest
    _write_batch_manifest(
        batch_result=batch_result,
        config=config,
        batch_output_dir=batch_output_dir,
        run_specs=run_specs,
    )

    # Read back
    manifest_path = batch_output_dir / "batch_manifest.json"
    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify structure
    assert manifest["batch_name"] == "test_batch"
    assert manifest["description"] == "Test batch"
    assert manifest["seed"] == 42
    assert manifest["total_runtime_sec"] == 100.0
    assert "config_hash" in manifest
    assert "git_commit_hash" in manifest
    assert len(manifest["expanded_runs"]) == 2
    assert manifest["expanded_runs"][0]["run_id"] == "run1"
    assert manifest["expanded_runs"][1]["run_id"] == "run2"
    assert manifest["run_results_summary"]["total_runs"] == 2
    assert manifest["run_results_summary"]["success_count"] == 2


def test_batch_manifest_stable_ordering(tmp_path: Path) -> None:
    """Test that batch manifest has stable ordering for runs."""
    batch_output_dir = tmp_path / "batch"
    batch_output_dir.mkdir()

    config = BatchConfig(
        batch_name="test",
        description="Test",
        output_root=Path("output"),
        base_args={},
        runs=[],
    )

    # Create runs in non-alphabetical order
    run_specs = [
        RunSpec(id="run3", bundle_path=Path("b.yaml"), start_date="2015-01-01", end_date="2020-12-31"),
        RunSpec(id="run1", bundle_path=Path("a.yaml"), start_date="2015-01-01", end_date="2020-12-31"),
        RunSpec(id="run2", bundle_path=Path("b.yaml"), start_date="2015-01-01", end_date="2020-12-31"),
    ]

    batch_result = BatchResult(
        batch_name="test",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        total_runtime_sec=10.0,
        run_results=[
            RunResult(run_id="run3", status="success", runtime_sec=3.0),
            RunResult(run_id="run1", status="success", runtime_sec=4.0),
            RunResult(run_id="run2", status="success", runtime_sec=3.0),
        ],
    )

    _write_batch_manifest(
        batch_result=batch_result,
        config=config,
        batch_output_dir=batch_output_dir,
        run_specs=run_specs,
    )

    # Read back
    manifest_path = batch_output_dir / "batch_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Expanded runs should be in the order of run_specs (not sorted)
    assert manifest["expanded_runs"][0]["run_id"] == "run3"
    assert manifest["expanded_runs"][1]["run_id"] == "run1"
    assert manifest["expanded_runs"][2]["run_id"] == "run2"

    # But manifest keys should be sorted (due to sort_keys=True)
    keys = list(manifest.keys())
    assert keys == sorted(keys)


def test_run_manifest_includes_artifacts(tmp_path: Path) -> None:
    """Test that run manifest includes artifact list."""
    run_output_dir = tmp_path / "run"
    run_output_dir.mkdir()
    (run_output_dir / "result.csv").write_text("data")
    (run_output_dir / "report.md").write_text("# Report")
    (run_output_dir / "nested").mkdir(parents=True, exist_ok=True)
    (run_output_dir / "nested" / "data.json").write_text("{}")

    run_spec = RunSpec(
        id="test",
        bundle_path=Path("bundle.yaml"),
        start_date="2015-01-01",
        end_date="2020-12-31",
    )

    result = RunResult(
        run_id="test",
        status="success",
        output_dir=run_output_dir,
        runtime_sec=5.0,
    )

    _write_run_manifest(
        result=result,
        run_spec=run_spec,
        base_args={},
        run_output_dir=run_output_dir,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
    )

    manifest_path = run_output_dir / "run_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    artifacts = manifest["artifacts"]
    assert len(artifacts) >= 3
    assert "result.csv" in artifacts
    assert "report.md" in artifacts
    assert "nested/data.json" in artifacts
    # Should be sorted
    assert artifacts == sorted(artifacts)

