"""Unit tests for generate_performance_profile_report.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.generate_performance_profile_report as gppr


@pytest.fixture
def fake_profiles_dir(tmp_path: Path) -> Path:
    """Create a fake profiles directory structure with timings and pstats."""
    profiles_root = tmp_path / "profiles"
    
    # Create EOD_SMALL run
    eod_run = profiles_root / "EOD_SMALL" / "20240101T120000Z"
    eod_run.mkdir(parents=True, exist_ok=True)
    
    # Create fake timings.json
    timings_data = {
        "job_name": "EOD_SMALL",
        "steps": {
            "load_data": {"duration_ms": 100.0, "start_ts": "2024-01-01T12:00:00", "end_ts": "2024-01-01T12:00:00"},
            "build_factors": {"duration_ms": 200.0, "start_ts": "2024-01-01T12:00:01", "end_ts": "2024-01-01T12:00:01"},
            "signals": {"duration_ms": 50.0, "start_ts": "2024-01-01T12:00:02", "end_ts": "2024-01-01T12:00:02"},
        },
        "summary": {
            "total_steps": 3,
            "total_duration_ms": 350.0,
            "avg_duration_ms": 116.67,
            "min_duration_ms": 50.0,
            "max_duration_ms": 200.0,
        },
    }
    with open(eod_run / "timings.json", "w", encoding="utf-8") as f:
        json.dump(timings_data, f)
    
    # Create fake .prof file (binary format, but we'll just check if it exists)
    # In real tests, we'd need actual pstats data, but for now we'll mock the extraction
    (eod_run / "profile_EOD_SMALL.prof").write_bytes(b"fake_prof_data")
    
    # Create BACKTEST_MEDIUM run (newer timestamp)
    bt_run = profiles_root / "BACKTEST_MEDIUM" / "20240101T130000Z"
    bt_run.mkdir(parents=True, exist_ok=True)
    
    bt_timings = {
        "job_name": "BACKTEST_MEDIUM",
        "steps": {
            "load_data": {"duration_ms": 500.0},
            "backtest_engine": {"duration_ms": 2000.0},
            "metrics": {"duration_ms": 300.0},
        },
        "summary": {
            "total_duration_ms": 2800.0,
        },
    }
    with open(bt_run / "timings.json", "w", encoding="utf-8") as f:
        json.dump(bt_timings, f)
    
    (bt_run / "profile_BACKTEST_MEDIUM.prof").write_bytes(b"fake_prof_data")
    
    return profiles_root


@pytest.fixture
def fake_pstats_data(tmp_path: Path) -> Path:
    """Create a fake .prof file for testing hotspot extraction."""
    # Note: Creating actual valid pstats files is complex.
    # In a real scenario, we'd need to use cProfile to generate one.
    # For this test, we'll mock the extraction function.
    prof_file = tmp_path / "test.prof"
    prof_file.write_bytes(b"fake_prof_data")
    return prof_file


def test_find_latest_job_run(fake_profiles_dir: Path) -> None:
    """Test finding the latest job run."""
    eod_run = gppr.find_latest_job_run(fake_profiles_dir, "EOD_SMALL")
    assert eod_run is not None
    assert eod_run.name == "20240101T120000Z"
    
    bt_run = gppr.find_latest_job_run(fake_profiles_dir, "BACKTEST_MEDIUM")
    assert bt_run is not None
    assert bt_run.name == "20240101T130000Z"
    
    # Non-existent job
    missing = gppr.find_latest_job_run(fake_profiles_dir, "NONEXISTENT")
    assert missing is None


def test_load_timings_json(fake_profiles_dir: Path) -> None:
    """Test loading timings.json."""
    eod_run = fake_profiles_dir / "EOD_SMALL" / "20240101T120000Z"
    timings_path = eod_run / "timings.json"
    
    timings_data = gppr.load_timings_json(timings_path)
    assert timings_data is not None
    assert "steps" in timings_data
    assert "summary" in timings_data
    assert timings_data["summary"]["total_duration_ms"] == 350.0


def test_extract_runtime_from_timings(fake_profiles_dir: Path) -> None:
    """Test extracting runtime from timings data."""
    eod_run = fake_profiles_dir / "EOD_SMALL" / "20240101T120000Z"
    timings_path = eod_run / "timings.json"
    timings_data = gppr.load_timings_json(timings_path)
    
    runtime = gppr.extract_runtime_from_timings(timings_data)
    assert runtime is not None
    assert abs(runtime - 0.35) < 0.001  # 350ms = 0.35s


def test_format_step_breakdown(fake_profiles_dir: Path) -> None:
    """Test formatting step breakdown."""
    eod_run = fake_profiles_dir / "EOD_SMALL" / "20240101T120000Z"
    timings_path = eod_run / "timings.json"
    timings_data = gppr.load_timings_json(timings_path)
    
    breakdown = gppr.format_step_breakdown(timings_data)
    assert "Step" in breakdown
    assert "Duration" in breakdown
    assert "load_data" in breakdown
    assert "build_factors" in breakdown
    assert "signals" in breakdown


def test_extract_top_hotspots_from_pstats_missing_file(tmp_path: Path) -> None:
    """Test hotspot extraction when file doesn't exist."""
    missing_prof = tmp_path / "missing.prof"
    hotspots = gppr.extract_top_hotspots_from_pstats(missing_prof)
    assert hotspots == []


def test_generate_report_markdown(tmp_path: Path) -> None:
    """Test generating Markdown report."""
    output_path = tmp_path / "test_report.md"
    
    jobs_data = {
        "EOD_SMALL": {
            "runtime": 0.35,
            "latest_run": "20240101T120000Z",
            "step_breakdown": "| Step | Duration (ms) |\n|------|---------------|\n| load_data | 100.0 |",
            "hotspots": [
                {"function": "load_prices", "file": "prices_ingest.py", "line": 42, "cumtime": 0.1, "ncalls": 1},
            ],
        },
        "BACKTEST_MEDIUM": {
            "runtime": 2.8,
            "latest_run": "20240101T130000Z",
            "step_breakdown": "| Step | Duration (ms) |\n|------|---------------|\n| backtest_engine | 2000.0 |",
            "hotspots": [
                {"function": "run_portfolio_backtest", "file": "backtest_engine.py", "line": 100, "cumtime": 2.0, "ncalls": 1},
            ],
        },
    }
    
    gppr.generate_report_markdown(jobs_data, output_path)
    
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    
    # Check structure
    assert "# Performance Profile Report" in content
    assert "## Overview" in content
    assert "## EOD_SMALL" in content
    assert "## BACKTEST_MEDIUM" in content
    assert "EOD_SMALL" in content
    assert "0.35" in content
    assert "2.8" in content


def test_generate_performance_profile_report_with_data(fake_profiles_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generating report from fake profiles directory."""
    # Mock hotspot extraction (since we can't easily create valid pstats files)
    def mock_extract_hotspots(pstats_path: Path, top_n: int = 3) -> list[dict[str, Any]]:
        if "EOD_SMALL" in str(pstats_path):
            return [
                {"function": "load_prices", "file": "prices_ingest.py", "line": 42, "cumtime": 0.05, "ncalls": 1},
                {"function": "add_features", "file": "ta_features.py", "line": 100, "cumtime": 0.15, "ncalls": 1},
                {"function": "generate_signals", "file": "rules_trend.py", "line": 50, "cumtime": 0.03, "ncalls": 1},
            ]
        elif "BACKTEST_MEDIUM" in str(pstats_path):
            return [
                {"function": "run_portfolio_backtest", "file": "backtest_engine.py", "line": 200, "cumtime": 1.8, "ncalls": 1},
                {"function": "compute_metrics", "file": "metrics.py", "line": 150, "cumtime": 0.25, "ncalls": 1},
                {"function": "load_prices", "file": "prices_ingest.py", "line": 42, "cumtime": 0.4, "ncalls": 1},
            ]
        return []
    
    monkeypatch.setattr(gppr, "extract_top_hotspots_from_pstats", mock_extract_hotspots)
    
    output_path = tmp_path / "performance_profile.md"
    
    gppr.generate_performance_profile_report(
        profiles_root=fake_profiles_dir,
        output_path=output_path,
        job_names=["EOD_SMALL", "BACKTEST_MEDIUM"],
    )
    
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    
    # Check that both jobs are included
    assert "EOD_SMALL" in content
    assert "BACKTEST_MEDIUM" in content
    assert "0.35" in content or "350" in content  # Runtime should appear
    assert "2.8" in content or "2800" in content  # Runtime should appear
    
    # Check hotspots
    assert "load_prices" in content
    assert "run_portfolio_backtest" in content


def test_generate_performance_profile_report_missing_dir(tmp_path: Path) -> None:
    """Test generating report when profiles directory doesn't exist."""
    missing_profiles = tmp_path / "missing_profiles"
    output_path = tmp_path / "performance_profile.md"
    
    gppr.generate_performance_profile_report(
        profiles_root=missing_profiles,
        output_path=output_path,
    )
    
    # Should create placeholder report
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "# Performance Profile Report" in content
    assert "EOD_SMALL" in content
    assert "BACKTEST_MEDIUM" in content
    assert "ML_JOB" in content

