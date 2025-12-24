"""Unit tests for timing utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from src.assembled_core.utils.timing import (
    load_timings_json,
    timed_step,
    write_timings_json,
)


def test_timed_step_basic() -> None:
    """Test basic timed_step functionality."""
    timings: dict[str, Any] = {}

    with timed_step("test_step", timings):
        time.sleep(0.01)  # Sleep for ~10ms

    assert "test_step" in timings
    step_data = timings["test_step"]
    assert "start_ts" in step_data
    assert "end_ts" in step_data
    assert "start_ts_epoch" in step_data
    assert "end_ts_epoch" in step_data
    assert "duration_ms" in step_data
    assert step_data["duration_ms"] >= 10.0  # At least 10ms
    assert step_data["duration_ms"] < 100.0  # But less than 100ms (with some margin)


def test_timed_step_with_meta() -> None:
    """Test timed_step with metadata."""
    timings: dict[str, Any] = {}
    meta = {"symbols": 100, "rows": 1000}

    with timed_step("test_step_with_meta", timings, meta=meta):
        pass

    assert "test_step_with_meta" in timings
    step_data = timings["test_step_with_meta"]
    assert "meta" in step_data
    assert step_data["meta"] == meta


def test_timed_step_monotonic_timestamps() -> None:
    """Test that timestamps are monotonic (end >= start)."""
    timings: dict[str, Any] = {}

    with timed_step("test_monotonic", timings):
        pass

    step_data = timings["test_monotonic"]
    assert step_data["end_ts_epoch"] >= step_data["start_ts_epoch"]
    assert step_data["duration_ms"] >= 0.0


def test_timed_step_multiple_steps() -> None:
    """Test timing multiple steps."""
    timings: dict[str, Any] = {}

    with timed_step("step1", timings):
        time.sleep(0.005)

    with timed_step("step2", timings):
        time.sleep(0.005)

    assert "step1" in timings
    assert "step2" in timings
    assert timings["step1"]["duration_ms"] > 0
    assert timings["step2"]["duration_ms"] > 0


def test_write_timings_json(tmp_path: Path) -> None:
    """Test writing timings to JSON file."""
    timings = {
        "step1": {
            "start_ts": "2024-01-01T10:00:00",
            "end_ts": "2024-01-01T10:00:01",
            "duration_ms": 1000.0,
        },
        "step2": {
            "start_ts": "2024-01-01T10:00:01",
            "end_ts": "2024-01-01T10:00:02",
            "duration_ms": 500.0,
        },
    }

    output_path = tmp_path / "timings.json"
    write_timings_json(timings, output_path, job_name="test_job")

    assert output_path.exists()

    # Load and verify
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "job_name" in data
    assert data["job_name"] == "test_job"
    assert "steps" in data
    assert "summary" in data
    assert data["summary"]["total_steps"] == 2
    assert data["summary"]["total_duration_ms"] == 1500.0
    assert data["summary"]["avg_duration_ms"] == 750.0
    assert data["summary"]["min_duration_ms"] == 500.0
    assert data["summary"]["max_duration_ms"] == 1000.0


def test_write_timings_json_with_meta(tmp_path: Path) -> None:
    """Test writing timings with job metadata."""
    timings = {"step1": {"duration_ms": 100.0}}

    output_path = tmp_path / "timings_meta.json"
    job_meta = {"config": {"freq": "1d", "symbols": 100}}

    write_timings_json(timings, output_path, job_name="test_job", job_meta=job_meta)

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "job_meta" in data
    assert data["job_meta"] == job_meta


def test_load_timings_json(tmp_path: Path) -> None:
    """Test loading timings from JSON file."""
    # Create a test JSON file
    test_data = {
        "job_name": "test_job",
        "steps": {
            "step1": {"duration_ms": 100.0},
            "step2": {"duration_ms": 200.0},
        },
    }

    output_path = tmp_path / "timings_load.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)

    loaded = load_timings_json(output_path)

    assert loaded == test_data
    assert loaded["job_name"] == "test_job"
    assert len(loaded["steps"]) == 2


def test_timings_json_serializable() -> None:
    """Test that timing data is JSON serializable."""
    timings: dict[str, Any] = {}

    with timed_step("test_serialize", timings):
        pass

    # Should not raise when converting to JSON
    json_str = json.dumps(timings)
    assert json_str

    # Should be able to parse back
    parsed = json.loads(json_str)
    assert "test_serialize" in parsed


def test_write_timings_empty_steps(tmp_path: Path) -> None:
    """Test writing timings with no steps."""
    timings = {}
    output_path = tmp_path / "timings_empty.json"

    write_timings_json(timings, output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "steps" in data
    assert data["steps"] == {}
    assert "summary" in data
    assert data["summary"]["total_steps"] == 0
    assert data["summary"]["total_duration_ms"] == 0.0

