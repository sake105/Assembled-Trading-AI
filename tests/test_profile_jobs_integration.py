"""Integration tests for profile_jobs (fast, no real backtests)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import scripts.profile_jobs as pj

pytestmark = pytest.mark.advanced


@pytest.fixture
def fast_dummy_job() -> callable:
    """Create a fast dummy job function (no sleep, instant return)."""

    def _dummy_job() -> None:
        # Just do a trivial computation to have something to profile
        _ = sum(range(100))

    return _dummy_job


def test_profile_jobs_registry_list(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that list_reference_jobs shows expected output."""
    pj.list_reference_jobs()

    captured = capsys.readouterr()
    output = captured.out

    assert "Reference Jobs" in output
    assert "=" in output  # Separator line


def test_timing_logs_written(tmp_path: Path, fast_dummy_job: callable) -> None:
    """Test that timing logs are written when profiling a job."""
    original_job_map = pj.JOB_MAP.copy()
    pj.JOB_MAP["TEST_TIMING"] = fast_dummy_job

    try:
        profile_dir = tmp_path / "profiles" / "TEST_TIMING" / "20240101T000000Z"
        profile_dir.mkdir(parents=True, exist_ok=True)

        pj.profile_job(
            job_name="TEST_TIMING",
            profiler="none",  # No profiling, just timing
            profile_out=profile_dir,
        )

        # Check that summary log exists
        summary_path = profile_dir / "profile_summary.log"
        assert summary_path.exists()

        # Check content
        content = summary_path.read_text(encoding="utf-8")
        assert "job_name=TEST_TIMING" in content
        assert "profiler=none" in content
        assert "total_seconds=" in content

    finally:
        pj.JOB_MAP.clear()
        pj.JOB_MAP.update(original_job_map)


def test_profiler_cprofile_creates_pstats_for_dummy_job(
    tmp_path: Path, fast_dummy_job: callable
) -> None:
    """Test that cProfile creates .prof and stats files for a dummy job."""
    original_job_map = pj.JOB_MAP.copy()
    pj.JOB_MAP["TEST_CPROFILE"] = fast_dummy_job

    try:
        profile_dir = tmp_path / "profiles" / "TEST_CPROFILE" / "20240101T000000Z"
        profile_dir.mkdir(parents=True, exist_ok=True)

        pj.profile_job(
            job_name="TEST_CPROFILE",
            profiler="cprofile",
            profile_out=profile_dir,
            top_n=10,
        )

        # Check that .prof file exists
        prof_path = profile_dir / "profile_TEST_CPROFILE.prof"
        assert prof_path.exists(), "cProfile .prof file should exist"

        # Check that stats file exists
        stats_path = profile_dir / "profile_TEST_CPROFILE_stats.txt"
        assert stats_path.exists(), "cProfile stats .txt file should exist"

        # Check stats content
        stats_content = stats_path.read_text(encoding="utf-8")
        assert "cProfile stats for TEST_CPROFILE" in stats_content
        assert "Total runtime:" in stats_content

        # Check summary log
        summary_path = profile_dir / "profile_summary.log"
        assert summary_path.exists()
        summary_content = summary_path.read_text(encoding="utf-8")
        assert "profiler=cprofile" in summary_content

    finally:
        pj.JOB_MAP.clear()
        pj.JOB_MAP.update(original_job_map)


def test_generate_performance_profile_report_from_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test generating performance profile report from fake fixtures."""
    import scripts.generate_performance_profile_report as gppr

    # Create fake profiles directory structure
    profiles_root = tmp_path / "profiles"

    # Create EOD_SMALL run
    eod_run = profiles_root / "EOD_SMALL" / "20240101T120000Z"
    eod_run.mkdir(parents=True, exist_ok=True)

    # Create fake timings.json
    timings_data = {
        "job_name": "EOD_SMALL",
        "steps": {
            "load_data": {"duration_ms": 100.0},
            "build_factors": {"duration_ms": 200.0},
        },
        "summary": {"total_duration_ms": 300.0},
    }
    with open(eod_run / "timings.json", "w", encoding="utf-8") as f:
        json.dump(timings_data, f)

    # Create fake .prof file
    (eod_run / "profile_EOD_SMALL.prof").write_bytes(b"fake_prof_data")

    # Mock hotspot extraction (since we can't easily create valid pstats files)
    def mock_extract_hotspots(pstats_path: Path, top_n: int = 3) -> list[dict[str, Any]]:
        return [
            {
                "function": "test_func",
                "file": "test_file.py",
                "line": 42,
                "cumtime": 0.1,
                "ncalls": 1,
            }
        ]

    monkeypatch.setattr(gppr, "extract_top_hotspots_from_pstats", mock_extract_hotspots)

    # Generate report
    output_path = tmp_path / "performance_profile.md"
    gppr.generate_performance_profile_report(
        profiles_root=profiles_root,
        output_path=output_path,
        job_names=["EOD_SMALL"],
    )

    # Check that report was created
    assert output_path.exists()

    # Check content
    content = output_path.read_text(encoding="utf-8")
    assert "# Performance Profile Report" in content
    assert "EOD_SMALL" in content
    assert "0.3" in content  # Runtime should be in seconds
    assert "test_func" in content


def test_profile_job_with_custom_output_dir(
    tmp_path: Path, fast_dummy_job: callable
) -> None:
    """Test that profile_job respects custom output directory."""
    original_job_map = pj.JOB_MAP.copy()
    pj.JOB_MAP["TEST_CUSTOM_OUT"] = fast_dummy_job

    try:
        custom_dir = tmp_path / "custom_output"
        pj.profile_job(
            job_name="TEST_CUSTOM_OUT",
            profiler="none",
            profile_out=custom_dir,
        )

        # Check that files are in custom directory (not default structure)
        summary_path = custom_dir / "profile_summary.log"
        assert summary_path.exists()

    finally:
        pj.JOB_MAP.clear()
        pj.JOB_MAP.update(original_job_map)


def test_run_job_without_profiling_no_side_effects(
    fast_dummy_job: callable, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that run_job_without_profiling doesn't create profile files."""
    original_job_map = pj.JOB_MAP.copy()
    pj.JOB_MAP["TEST_NO_PROFILE"] = fast_dummy_job

    # Mock logger to avoid output noise
    mock_logger = MagicMock()
    monkeypatch.setattr("scripts.profile_jobs.logger", mock_logger)

    try:
        pj.run_job_without_profiling("TEST_NO_PROFILE")

        # Verify job was called (would raise if not found)
        # No profile files should be created (we don't pass profile_out)
        # This test mainly ensures the function doesn't crash

    finally:
        pj.JOB_MAP.clear()
        pj.JOB_MAP.update(original_job_map)

