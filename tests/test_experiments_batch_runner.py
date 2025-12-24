"""Unit tests for batch runner module.

Tests cover:
- expand_run_specs (config expansion)
- run_batch_serial (serial execution with mocked backtest calls)
- BatchResult generation
- Output file generation (JSON + CSV)
- Error handling (fail_fast, exceptions)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


from src.assembled_core.experiments.batch_config import BatchConfig, RunSpec
from src.assembled_core.experiments.batch_runner import (
    BatchResult,
    RunResult,
    expand_run_specs,
    run_batch_serial,
)


def test_expand_run_specs_individual_runs(tmp_path: Path) -> None:
    """Test expand_run_specs with individual runs."""
    config = BatchConfig(
        batch_name="test_batch",
        description="Test",
        output_root=tmp_path / "output",
        runs=[
            RunSpec(
                id="run1",
                bundle_path=tmp_path / "bundle.yaml",
                start_date="2015-01-01",
                end_date="2020-12-31",
            ),
            RunSpec(
                id="run2",
                bundle_path=tmp_path / "bundle.yaml",
                start_date="2015-01-01",
                end_date="2020-12-31",
            ),
        ],
    )

    run_specs = expand_run_specs(config)

    assert len(run_specs) == 2
    assert run_specs[0].id == "run1"
    assert run_specs[1].id == "run2"


def test_expand_run_specs_grid_search(tmp_path: Path) -> None:
    """Test expand_run_specs with grid search."""
    config = BatchConfig(
        batch_name="test_batch",
        description="Test",
        output_root=tmp_path / "output",
        base_args={
            "bundle_path": str(tmp_path / "bundle.yaml"),
            "start_date": "2015-01-01",
            "end_date": "2020-12-31",
            "freq": "1d",
        },
        grid={
            "max_gross_exposure": [0.6, 0.8],
            "commission_bps": [0.0, 0.5],
        },
    )

    run_specs = expand_run_specs(config)

    # 2 exposures * 2 commission values = 4 combinations
    assert len(run_specs) == 4


def test_run_batch_serial_success(tmp_path: Path) -> None:
    """Test run_batch_serial with successful mocked backtest."""
    run_specs = [
        RunSpec(
            id="run1",
            bundle_path=tmp_path / "bundle.yaml",
            start_date="2015-01-01",
            end_date="2020-12-31",
        ),
    ]

    def mock_backtest_fn(run_spec: RunSpec, base_args: dict, output_dir: Path) -> RunResult:
        """Mock backtest function that always succeeds."""
        return RunResult(
            run_id=run_spec.id,
            status="success",
            output_dir=output_dir,
            runtime_sec=1.5,
        )

    result = run_batch_serial(
        run_specs=run_specs,
        batch_name="test_batch",
        output_root=tmp_path / "output",
        base_args={"freq": "1d"},
        repo_root=tmp_path,
        backtest_fn=mock_backtest_fn,
    )

    assert result.batch_name == "test_batch"
    assert result.success_count == 1
    assert result.failed_count == 0
    assert len(result.run_results) == 1
    assert result.run_results[0].status == "success"
    assert result.run_results[0].run_id == "run1"


def test_run_batch_serial_failure(tmp_path: Path) -> None:
    """Test run_batch_serial with failed mocked backtest."""
    run_specs = [
        RunSpec(
            id="run1",
            bundle_path=tmp_path / "bundle.yaml",
            start_date="2015-01-01",
            end_date="2020-12-31",
        ),
    ]

    def mock_backtest_fn(run_spec: RunSpec, base_args: dict, output_dir: Path) -> RunResult:
        """Mock backtest function that always fails."""
        return RunResult(
            run_id=run_spec.id,
            status="failed",
            output_dir=output_dir,
            runtime_sec=0.5,
            error="Mock error",
        )

    result = run_batch_serial(
        run_specs=run_specs,
        batch_name="test_batch",
        output_root=tmp_path / "output",
        base_args={"freq": "1d"},
        repo_root=tmp_path,
        backtest_fn=mock_backtest_fn,
    )

    assert result.success_count == 0
    assert result.failed_count == 1
    assert result.run_results[0].status == "failed"
    assert result.run_results[0].error == "Mock error"
    assert result.error_summary is not None


def test_run_batch_serial_multiple_runs(tmp_path: Path) -> None:
    """Test run_batch_serial with multiple runs."""
    run_specs = [
        RunSpec(
            id="run1",
            bundle_path=tmp_path / "bundle.yaml",
            start_date="2015-01-01",
            end_date="2020-12-31",
        ),
        RunSpec(
            id="run2",
            bundle_path=tmp_path / "bundle.yaml",
            start_date="2015-01-01",
            end_date="2020-12-31",
        ),
        RunSpec(
            id="run3",
            bundle_path=tmp_path / "bundle.yaml",
            start_date="2015-01-01",
            end_date="2020-12-31",
        ),
    ]

    call_count = 0

    def mock_backtest_fn(run_spec: RunSpec, base_args: dict, output_dir: Path) -> RunResult:
        """Mock backtest function that tracks call count."""
        nonlocal call_count
        call_count += 1
        return RunResult(
            run_id=run_spec.id,
            status="success",
            output_dir=output_dir,
            runtime_sec=1.0,
        )

    result = run_batch_serial(
        run_specs=run_specs,
        batch_name="test_batch",
        output_root=tmp_path / "output",
        base_args={"freq": "1d"},
        repo_root=tmp_path,
        backtest_fn=mock_backtest_fn,
    )

    assert call_count == 3
    assert result.success_count == 3
    assert len(result.run_results) == 3


def test_run_batch_serial_fail_fast(tmp_path: Path) -> None:
    """Test run_batch_serial with fail_fast enabled."""
    run_specs = [
        RunSpec(
            id="run1",
            bundle_path=tmp_path / "bundle.yaml",
            start_date="2015-01-01",
            end_date="2020-12-31",
        ),
        RunSpec(
            id="run2",
            bundle_path=tmp_path / "bundle.yaml",
            start_date="2015-01-01",
            end_date="2020-12-31",
        ),
        RunSpec(
            id="run3",
            bundle_path=tmp_path / "bundle.yaml",
            start_date="2015-01-01",
            end_date="2020-12-31",
        ),
    ]

    call_count = 0

    def mock_backtest_fn(run_spec: RunSpec, base_args: dict, output_dir: Path) -> RunResult:
        """Mock backtest function that fails on run2."""
        nonlocal call_count
        call_count += 1
        status = "failed" if run_spec.id == "run2" else "success"
        return RunResult(
            run_id=run_spec.id,
            status=status,
            output_dir=output_dir,
            runtime_sec=1.0,
            error="Mock error" if status == "failed" else None,
        )

    result = run_batch_serial(
        run_specs=run_specs,
        batch_name="test_batch",
        output_root=tmp_path / "output",
        base_args={"freq": "1d"},
        repo_root=tmp_path,
        fail_fast=True,
        backtest_fn=mock_backtest_fn,
    )

    # Should stop after run2 fails
    assert call_count == 2
    assert len(result.run_results) == 2
    assert result.run_results[0].status == "success"
    assert result.run_results[1].status == "failed"


def test_run_batch_serial_deterministic_ordering(tmp_path: Path) -> None:
    """Test that run_batch_serial executes runs in deterministic order."""
    run_specs = [
        RunSpec(id="run3", bundle_path=tmp_path / "bundle.yaml", start_date="2015-01-01", end_date="2020-12-31"),
        RunSpec(id="run1", bundle_path=tmp_path / "bundle.yaml", start_date="2015-01-01", end_date="2020-12-31"),
        RunSpec(id="run2", bundle_path=tmp_path / "bundle.yaml", start_date="2015-01-01", end_date="2020-12-31"),
    ]

    execution_order = []

    def mock_backtest_fn(run_spec: RunSpec, base_args: dict, output_dir: Path) -> RunResult:
        """Mock backtest function that tracks execution order."""
        execution_order.append(run_spec.id)
        return RunResult(
            run_id=run_spec.id,
            status="success",
            output_dir=output_dir,
            runtime_sec=1.0,
        )

    result = run_batch_serial(
        run_specs=run_specs,
        batch_name="test_batch",
        output_root=tmp_path / "output",
        base_args={"freq": "1d"},
        repo_root=tmp_path,
        backtest_fn=mock_backtest_fn,
    )

    # Should execute in order provided (not sorted)
    assert execution_order == ["run3", "run1", "run2"]
    assert [r.run_id for r in result.run_results] == ["run3", "run1", "run2"]


def test_run_batch_serial_output_files(tmp_path: Path) -> None:
    """Test that run_batch_serial generates output files."""
    run_specs = [
        RunSpec(
            id="run1",
            bundle_path=tmp_path / "bundle.yaml",
            start_date="2015-01-01",
            end_date="2020-12-31",
        ),
    ]

    def mock_backtest_fn(run_spec: RunSpec, base_args: dict, output_dir: Path) -> RunResult:
        """Mock backtest function."""
        return RunResult(
            run_id=run_spec.id,
            status="success",
            output_dir=output_dir,
            runtime_sec=1.5,
        )

    run_batch_serial(
        run_specs=run_specs,
        batch_name="test_batch",
        output_root=tmp_path / "output",
        base_args={"freq": "1d"},
        repo_root=tmp_path,
        backtest_fn=mock_backtest_fn,
    )

    # Check JSON summary
    summary_json = tmp_path / "output" / "test_batch" / "batch_summary.json"
    assert summary_json.exists()

    with summary_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["batch_name"] == "test_batch"
    assert data["success_count"] == 1
    assert len(data["runs"]) == 1
    assert data["runs"][0]["run_id"] == "run1"

    # Check CSV summary
    summary_csv = tmp_path / "output" / "test_batch" / "batch_summary.csv"
    assert summary_csv.exists()

    with summary_csv.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 2  # Header + 1 run
        assert "run_id" in lines[0]
        assert "run1" in lines[1]


def test_batch_result_post_init() -> None:
    """Test BatchResult.__post_init__ computes counts correctly."""
    result = BatchResult(
        batch_name="test",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        total_runtime_sec=10.0,
        run_results=[
            RunResult(run_id="run1", status="success", runtime_sec=1.0),
            RunResult(run_id="run2", status="success", runtime_sec=2.0),
            RunResult(run_id="run3", status="failed", runtime_sec=0.5, error="Error"),
            RunResult(run_id="run4", status="skipped", runtime_sec=0.0),
        ],
    )

    assert result.success_count == 2
    assert result.failed_count == 1
    assert result.skipped_count == 1
    assert result.error_summary is not None
    assert "1 errors" in result.error_summary


def test_run_batch_serial_with_seed(tmp_path: Path) -> None:
    """Test that run_batch_serial sets random seed if provided."""
    run_specs = [
        RunSpec(
            id="run1",
            bundle_path=tmp_path / "bundle.yaml",
            start_date="2015-01-01",
            end_date="2020-12-31",
        ),
    ]

    def mock_backtest_fn(run_spec: RunSpec, base_args: dict, output_dir: Path) -> RunResult:
        """Mock backtest function."""
        return RunResult(
            run_id=run_spec.id,
            status="success",
            output_dir=output_dir,
            runtime_sec=1.0,
        )

    run_batch_serial(
        run_specs=run_specs,
        batch_name="test_batch",
        output_root=tmp_path / "output",
        base_args={"freq": "1d"},
        repo_root=tmp_path,
        seed=42,
        backtest_fn=mock_backtest_fn,
    )

    # Should complete successfully (seed is set, but doesn't affect mock)
    # Note: We don't check result here, just verify seed is set without error

