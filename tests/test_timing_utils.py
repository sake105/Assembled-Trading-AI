# tests/test_timing_utils.py
"""Tests for timing utilities."""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.assembled_core.utils.timing import (
    load_timings_json,
    timed_step,
    write_timings_json,
)


def test_timed_step_basic():
    """Test basic timed_step functionality."""
    timings = {}
    
    with timed_step("test_step", timings):
        time.sleep(0.01)  # Small delay to ensure measurable duration
    
    assert "test_step" in timings
    assert "duration_ms" in timings["test_step"]
    assert "start_ts" in timings["test_step"]
    assert "end_ts" in timings["test_step"]
    assert timings["test_step"]["duration_ms"] > 0


def test_timed_step_meta():
    """Test timed_step with metadata."""
    timings = {}
    
    with timed_step("test_step", timings, meta={"symbols": 100, "rows": 5000}):
        time.sleep(0.01)
    
    assert "test_step" in timings
    assert "meta" in timings["test_step"]
    assert timings["test_step"]["meta"]["symbols"] == 100
    assert timings["test_step"]["meta"]["rows"] == 5000


def test_write_timings_json():
    """Test writing timings to JSON file."""
    timings = {
        "step1": {"duration_ms": 123.45, "start_ts": "2024-01-01T10:00:00", "end_ts": "2024-01-01T10:00:01"},
        "step2": {"duration_ms": 67.89, "start_ts": "2024-01-01T10:00:01", "end_ts": "2024-01-01T10:00:02"},
    }
    
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "timings.json"
        
        write_timings_json(
            timings=timings,
            output_path=output_path,
            job_name="test_job",
            job_meta={"strategy": "trend_baseline", "freq": "1d"},
        )
        
        assert output_path.exists()
        
        # Verify file can be loaded
        loaded = load_timings_json(output_path)
        assert loaded["job_name"] == "test_job"
        assert loaded["job_meta"]["strategy"] == "trend_baseline"
        assert "summary" in loaded
        assert loaded["summary"]["total_steps"] == 2
        assert loaded["summary"]["total_duration_ms"] == pytest.approx(191.34, abs=0.1)


def test_load_timings_json():
    """Test loading timings from JSON file."""
    timings = {
        "step1": {"duration_ms": 123.45, "start_ts": "2024-01-01T10:00:00", "end_ts": "2024-01-01T10:00:01"},
    }
    
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "timings.json"
        write_timings_json(timings, output_path, job_name="test_job")
        
        loaded = load_timings_json(output_path)
        assert loaded["job_name"] == "test_job"
        assert "step1" in loaded["steps"]


def test_timed_step_nested():
    """Test nested timed_step calls."""
    timings = {}
    
    with timed_step("outer", timings):
        with timed_step("inner", timings):
            time.sleep(0.01)
    
    assert "outer" in timings
    assert "inner" in timings
    assert timings["outer"]["duration_ms"] >= timings["inner"]["duration_ms"]
