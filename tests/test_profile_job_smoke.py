# tests/test_profile_job_smoke.py
"""Smoke tests for profile_job.py (dry-run and argument parsing)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_profile_job_help():
    """Test that profile_job.py shows help."""
    result = subprocess.run(
        [sys.executable, "scripts/profile_job.py", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    
    assert result.returncode == 0
    assert "Profile reference benchmark jobs" in result.stdout
    assert "--job" in result.stdout
    assert "--profiler" in result.stdout


def test_profile_job_dry_run_eod_small():
    """Test dry-run for EOD_SMALL."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profile_job.py",
            "--job", "EOD_SMALL",
            "--profiler", "cprofile",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    
    assert result.returncode == 0
    assert "DRY RUN" in result.stdout or "dry-run" in result.stdout.lower()


def test_profile_job_dry_run_backtest_medium():
    """Test dry-run for BACKTEST_MEDIUM."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profile_job.py",
            "--job", "BACKTEST_MEDIUM",
            "--profiler", "cprofile",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    
    assert result.returncode == 0


def test_profile_job_dry_run_ml_job():
    """Test dry-run for ML_JOB."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profile_job.py",
            "--job", "ML_JOB",
            "--profiler", "cprofile",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    
    assert result.returncode == 0


def test_profile_job_invalid_job():
    """Test that invalid job raises error."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profile_job.py",
            "--job", "INVALID_JOB",
            "--profiler", "cprofile",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower() or "error" in result.stderr.lower()


def test_profile_job_invalid_profiler():
    """Test that invalid profiler raises error."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profile_job.py",
            "--job", "EOD_SMALL",
            "--profiler", "invalid_profiler",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower() or "error" in result.stderr.lower()


def test_profile_job_missing_required_args():
    """Test that missing required arguments raises error."""
    result = subprocess.run(
        [sys.executable, "scripts/profile_job.py"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

