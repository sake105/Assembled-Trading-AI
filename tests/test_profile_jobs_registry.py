"""Unit tests for profile_jobs registry and list functionality."""

from __future__ import annotations

import pytest

import scripts.profile_jobs as pj

pytestmark = pytest.mark.advanced


def test_list_reference_jobs(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that list_reference_jobs prints job information."""
    pj.list_reference_jobs()

    captured = capsys.readouterr()
    output = captured.out

    assert "Reference Jobs" in output
    assert "EOD_SMALL" in output or "BACKTEST_MEDIUM" in output or "ML_JOB" in output
    assert "Total:" in output


def test_parse_args_list_flag() -> None:
    """Test that --list flag is parsed correctly."""
    args = pj.parse_args(["--list"])
    assert args.list is True


def test_parse_args_job_selection() -> None:
    """Test that job selection works."""
    # Test with valid job
    args = pj.parse_args(["--job", "EOD_SMALL"])
    assert args.job == "EOD_SMALL"

    # Test with profiler flag
    args = pj.parse_args(["--job", "BACKTEST_MEDIUM", "--profiler", "cprofile"])
    assert args.job == "BACKTEST_MEDIUM"
    assert args.profiler == "cprofile"


def test_parse_args_profiler_options() -> None:
    """Test that profiler options are parsed correctly."""
    args = pj.parse_args(["--job", "EOD_SMALL", "--profiler", "pyinstrument"])
    assert args.profiler == "pyinstrument"

    args = pj.parse_args(["--job", "EOD_SMALL", "--profiler", "none"])
    assert args.profiler == "none"


def test_job_map_contains_reference_jobs() -> None:
    """Test that JOB_MAP contains expected reference jobs."""
    assert "EOD_SMALL" in pj.JOB_MAP or "BACKTEST_MEDIUM" in pj.JOB_MAP

    # All jobs should be callable
    for job_name, job_func in pj.JOB_MAP.items():
        assert callable(job_func), f"Job {job_name} should be callable"


def test_reference_jobs_list_not_empty() -> None:
    """Test that REFERENCE_JOBS list is not empty."""
    assert len(pj.REFERENCE_JOBS) > 0

    # Check structure of reference jobs
    for ref_job in pj.REFERENCE_JOBS:
        assert hasattr(ref_job, "name")
        assert hasattr(ref_job, "description")
        assert ref_job.name in pj.JOB_MAP

