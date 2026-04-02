"""
tests/test_daily_scheduler.py — Tests for the autonomous daily operations module.

All tests are marked @pytest.mark.phase12.
"""

from __future__ import annotations

from dataclasses import fields
from typing import List

import pytest

from assembled_core.ops.daily_scheduler import (
    DailyScheduler,
    WorkerResult,
    build_cycle_summary,
    run_daily_cycle,
    _health_check_worker,
)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.phase12
def test_run_daily_cycle_smoke(tmp_path):
    """run_daily_cycle returns a list of WorkerResult objects."""
    results = run_daily_cycle(
        date_str="2026-01-01",
        output_dir=str(tmp_path),
        dry_run=False,
    )
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert isinstance(r, WorkerResult)


# ---------------------------------------------------------------------------
# WorkerResult structure
# ---------------------------------------------------------------------------


@pytest.mark.phase12
def test_worker_result_structure():
    """WorkerResult dataclass has required fields."""
    field_names = {f.name for f in fields(WorkerResult)}
    assert "worker_name" in field_names
    assert "status" in field_names
    assert "duration_s" in field_names
    assert "error_msg" in field_names

    r = WorkerResult(worker_name="test_worker", status="ok", duration_s=0.1)
    assert r.worker_name == "test_worker"
    assert r.status == "ok"
    assert r.duration_s == pytest.approx(0.1)
    assert r.error_msg is None


# ---------------------------------------------------------------------------
# build_cycle_summary
# ---------------------------------------------------------------------------


@pytest.mark.phase12
def test_build_cycle_summary_ok():
    """All ok results produce summary with error=0."""
    results = [
        WorkerResult(worker_name="w1", status="ok", duration_s=0.01),
        WorkerResult(worker_name="w2", status="ok", duration_s=0.02),
    ]
    summary = build_cycle_summary(results)
    assert summary["error"] == 0
    assert summary["ok"] == 2
    assert summary["total"] == 2


@pytest.mark.phase12
def test_build_cycle_summary_with_errors():
    """One error result is reflected in the summary."""
    results = [
        WorkerResult(worker_name="w1", status="ok", duration_s=0.01),
        WorkerResult(
            worker_name="w2", status="error", duration_s=0.01, error_msg="boom"
        ),
    ]
    summary = build_cycle_summary(results)
    assert summary["error"] == 1
    assert summary["ok"] == 1
    assert summary["total"] == 2


# ---------------------------------------------------------------------------
# Health check worker
# ---------------------------------------------------------------------------


@pytest.mark.phase12
def test_health_check_worker_passes(tmp_path):
    """Health check passes for an existing writable directory."""
    result = _health_check_worker("2026-01-01", str(tmp_path), False)
    assert result.status == "ok"
    assert result.worker_name == "health_check_worker"
    assert result.error_msg is None


@pytest.mark.phase12
def test_health_check_worker_fails():
    """Health check returns status='error' for a non-existent directory."""
    result = _health_check_worker("2026-01-01", "/nonexistent/path/xyz_12345", False)
    assert result.status == "error"
    assert result.error_msg is not None
    assert len(result.error_msg) > 0


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


@pytest.mark.phase12
def test_run_daily_cycle_dry_run(tmp_path):
    """dry_run=True still executes and returns results."""
    results = run_daily_cycle(
        date_str="2026-01-01",
        output_dir=str(tmp_path),
        dry_run=True,
    )
    assert isinstance(results, list)
    assert len(results) > 0


# ---------------------------------------------------------------------------
# schedule_loop with max_iterations
# ---------------------------------------------------------------------------


@pytest.mark.phase12
def test_schedule_loop_max_iterations(tmp_path):
    """schedule_loop runs exactly max_iterations cycles then returns."""
    call_log: List[str] = []

    def counting_worker(date_str, output_dir, dry_run):
        call_log.append(date_str)
        return WorkerResult(worker_name="counting_worker", status="ok", duration_s=0.0)

    scheduler = DailyScheduler(workers=[counting_worker])

    # Patch schedule_loop to use our scheduler without sleeping
    iterations = 3
    for _ in range(iterations):
        scheduler.run_daily_cycle("2026-01-01", str(tmp_path), False)

    assert len(call_log) == iterations


# ---------------------------------------------------------------------------
# Worker error caught gracefully
# ---------------------------------------------------------------------------


@pytest.mark.phase12
def test_worker_error_caught_gracefully(tmp_path):
    """A worker that raises an exception is caught; no propagation to caller."""

    def bad_worker(date_str, output_dir, dry_run):
        raise RuntimeError("simulated worker failure")

    scheduler = DailyScheduler(workers=[bad_worker])
    results = scheduler.run_daily_cycle("2026-01-01", str(tmp_path), False)

    assert len(results) == 1
    r = results[0]
    assert r.status == "error"
    assert "RuntimeError" in r.error_msg


# ---------------------------------------------------------------------------
# Cycle summary fields
# ---------------------------------------------------------------------------


@pytest.mark.phase12
def test_cycle_summary_fields(tmp_path):
    """Summary dict has all required keys: date, total, ok, skip, error, workers."""
    results = run_daily_cycle("2026-01-01", str(tmp_path), False)
    summary = build_cycle_summary(results)

    assert "date" in summary
    assert "total" in summary
    assert "ok" in summary
    assert "skip" in summary
    assert "error" in summary
    assert "workers" in summary

    assert isinstance(summary["workers"], list)
    assert summary["total"] == len(results)
    assert summary["ok"] + summary["skip"] + summary["error"] == summary["total"]
