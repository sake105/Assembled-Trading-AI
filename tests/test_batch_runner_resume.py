"""Tests for batch runner resume functionality."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_load_existing_manifest(tmp_path: Path) -> None:
    """Test loading an existing manifest."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import RunConfig, load_existing_manifest, write_run_manifest

    run_output_dir = tmp_path / "run1"
    run_output_dir.mkdir(parents=True)

    run_cfg = RunConfig(
        id="run1",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        start_capital=100000.0,
    )

    started_at = datetime.utcnow()
    finished_at = datetime.utcnow()

    # Write manifest
    write_run_manifest(
        run_id="run1",
        run_cfg=run_cfg,
        run_output_dir=run_output_dir,
        status="success",
        started_at=started_at,
        finished_at=finished_at,
        runtime_sec=150.5,
        exit_code=0,
        seed=42,
    )

    # Load manifest
    manifest = load_existing_manifest(run_output_dir)
    assert manifest is not None
    assert manifest["run_id"] == "run1"
    assert manifest["status"] == "success"
    assert manifest["runtime_sec"] == 150.5
    assert manifest["exit_code"] == 0


def test_load_existing_manifest_not_found(tmp_path: Path) -> None:
    """Test loading manifest when it doesn't exist."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import load_existing_manifest

    run_output_dir = tmp_path / "run1"
    run_output_dir.mkdir(parents=True)

    # No manifest exists
    manifest = load_existing_manifest(run_output_dir)
    assert manifest is None


def test_resume_skip_successful_run(tmp_path: Path) -> None:
    """Test that resume skips successful runs."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import BatchConfig, RunConfig, run_batch

    run_cfg = RunConfig(
        id="run1",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        start_capital=100000.0,
    )

    batch_cfg = BatchConfig(
        batch_name="test_batch",
        output_root=tmp_path / "output",
        seed=42,
        runs=[run_cfg],
    )

    batch_output_root = batch_cfg.output_root / "batch"
    run_output_dir = batch_output_root / run_cfg.id
    run_output_dir.mkdir(parents=True)

    # Write a successful manifest
    from scripts.batch_runner import write_run_manifest
    started_at = datetime.utcnow()
    finished_at = datetime.utcnow()

    write_run_manifest(
        run_id=run_cfg.id,
        run_cfg=run_cfg,
        run_output_dir=run_output_dir,
        status="success",
        started_at=started_at,
        finished_at=finished_at,
        runtime_sec=150.5,
        exit_code=0,
        seed=42,
    )

    # Run with resume - should skip
    exit_code = run_batch(batch_cfg, max_workers=1, dry_run=False, resume=True)
    assert exit_code == 0


def test_resume_skip_failed_run(tmp_path: Path) -> None:
    """Test that resume skips failed runs by default."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import BatchConfig, RunConfig, run_batch

    run_cfg = RunConfig(
        id="run1",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        start_capital=100000.0,
    )

    batch_cfg = BatchConfig(
        batch_name="test_batch",
        output_root=tmp_path / "output",
        seed=42,
        runs=[run_cfg],
    )

    batch_output_root = batch_cfg.output_root / "batch"
    run_output_dir = batch_output_root / run_cfg.id
    run_output_dir.mkdir(parents=True)

    # Write a failed manifest
    from scripts.batch_runner import write_run_manifest
    started_at = datetime.utcnow()
    finished_at = datetime.utcnow()

    write_run_manifest(
        run_id=run_cfg.id,
        run_cfg=run_cfg,
        run_output_dir=run_output_dir,
        status="failed",
        started_at=started_at,
        finished_at=finished_at,
        runtime_sec=50.0,
        exit_code=1,
        seed=42,
        error="Test error",
    )

    # Run with resume (without rerun-failed) - should skip failed
    exit_code = run_batch(batch_cfg, max_workers=1, dry_run=False, resume=True, rerun_failed=False)
    # Exit code should be 1 (because run is still failed, even if skipped)
    assert exit_code == 1


def test_resume_rerun_failed(tmp_path: Path) -> None:
    """Test that --rerun-failed reruns failed runs."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import BatchConfig, RunConfig, run_single_backtest

    run_cfg = RunConfig(
        id="run1",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        start_capital=100000.0,
    )

    batch_cfg = BatchConfig(
        batch_name="test_batch",
        output_root=tmp_path / "output",
        seed=42,
        runs=[run_cfg],
    )

    batch_output_root = batch_cfg.output_root / "batch"
    run_output_dir = batch_output_root / run_cfg.id
    run_output_dir.mkdir(parents=True)

    # Write a failed manifest
    from scripts.batch_runner import write_run_manifest
    started_at = datetime.utcnow()
    finished_at = datetime.utcnow()

    write_run_manifest(
        run_id=run_cfg.id,
        run_cfg=run_cfg,
        run_output_dir=run_output_dir,
        status="failed",
        started_at=started_at,
        finished_at=finished_at,
        runtime_sec=50.0,
        exit_code=1,
        seed=42,
        error="Test error",
    )

    # Run with resume + rerun-failed - should attempt rerun (dry-run mode)
    status, runtime_sec, exit_code, error = run_single_backtest(
        run_cfg=run_cfg,
        batch_output_root=batch_output_root,
        seed=42,
        dry_run=True,  # Don't actually run, just test logic
        resume=True,
        rerun_failed=True,
    )

    # Should not be skipped (will be "skipped" due to dry_run, but logic should attempt rerun)
    # In dry_run mode, it returns "skipped", but the resume check happens before
    # So if rerun_failed is True, it should proceed to dry_run check
    assert status == "skipped"  # Because dry_run=True


def test_resume_no_manifest_runs_normally(tmp_path: Path) -> None:
    """Test that resume doesn't affect runs without existing manifest."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import RunConfig, run_single_backtest

    run_cfg = RunConfig(
        id="run1",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        start_capital=100000.0,
    )

    batch_output_root = tmp_path / "output" / "batch"
    batch_output_root.mkdir(parents=True)

    # No manifest exists - should run normally (dry-run mode)
    status, runtime_sec, exit_code, error = run_single_backtest(
        run_cfg=run_cfg,
        batch_output_root=batch_output_root,
        seed=42,
        dry_run=True,
        resume=True,
        rerun_failed=False,
    )

    # Should proceed normally (dry_run returns "skipped")
    assert status == "skipped"


def test_manifest_contains_required_fields(tmp_path: Path) -> None:
    """Test that manifest contains required fields for resume functionality."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import RunConfig, write_run_manifest

    run_output_dir = tmp_path / "run1"
    run_output_dir.mkdir(parents=True)

    run_cfg = RunConfig(
        id="run1",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        start_capital=100000.0,
    )

    started_at = datetime.utcnow()
    finished_at = datetime.utcnow()

    write_run_manifest(
        run_id="run1",
        run_cfg=run_cfg,
        run_output_dir=run_output_dir,
        status="failed",
        started_at=started_at,
        finished_at=finished_at,
        runtime_sec=150.5,
        exit_code=1,
        seed=42,
        error="Test error message",
    )

    # Check manifest file
    manifest_path = run_output_dir / "run_manifest.json"
    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check required fields
    assert "status" in manifest
    assert "started_at" in manifest
    assert "finished_at" in manifest
    assert "error" in manifest
    assert manifest["status"] == "failed"
    assert manifest["error"] == "Test error message"

