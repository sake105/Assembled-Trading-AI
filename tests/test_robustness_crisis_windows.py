"""Tests for crisis windows evaluation (RB4).

These tests verify that crisis windows are evaluated deterministically:
- Deterministic window ordering
- Correct date range slicing
- Pass/fail flags work correctly
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.assembled_core.qa.robustness import (
    export_crisis_windows_results,
    get_standard_crisis_windows,
    run_crisis_windows,
)


def test_get_standard_crisis_windows():
    """Test that standard crisis windows are returned in deterministic order."""
    windows = get_standard_crisis_windows()

    # Verify structure
    assert len(windows) >= 3  # At least GFC, COVID, 2022_RATES

    # Verify each window has required keys
    for window in windows:
        assert "name" in window
        assert "start" in window
        assert "end" in window
        assert isinstance(window["name"], str)
        assert isinstance(window["start"], str)
        assert isinstance(window["end"], str)

    # Verify deterministic ordering (sorted by start, then name)
    for i in range(len(windows) - 1):
        w1 = windows[i]
        w2 = windows[i + 1]
        assert w1["start"] <= w2["start"]
        if w1["start"] == w2["start"]:
            assert w1["name"] <= w2["name"]


def test_get_standard_crisis_windows_contains_expected():
    """Test that standard windows contain expected crisis periods."""
    windows = get_standard_crisis_windows()

    window_names = [w["name"] for w in windows]
    assert "GFC" in window_names
    assert "COVID" in window_names
    assert "2022_RATES" in window_names


def test_run_crisis_windows_deterministic():
    """Test that same input produces identical window order."""
    def backtest_fn(config):
        # Simulate different performance for different date ranges
        start = config.get("start_date", "")
        if "2007" in start or "2008" in start:
            return {"sharpe": -0.5, "max_drawdown": -0.40, "cagr": -0.20}
        elif "2020" in start:
            return {"sharpe": -1.0, "max_drawdown": -0.35, "cagr": -0.15}
        elif "2022" in start:
            return {"sharpe": 0.2, "max_drawdown": -0.25, "cagr": 0.05}
        return {"sharpe": 0.0, "max_drawdown": -0.10, "cagr": 0.0}

    base_config: dict[str, Any] = {"strategy": "ema"}
    windows = get_standard_crisis_windows()

    # Run twice
    results1 = run_crisis_windows(
        backtest_fn=backtest_fn,
        base_config=base_config,
        windows=windows,
        deterministic=True,
    )

    results2 = run_crisis_windows(
        backtest_fn=backtest_fn,
        base_config=base_config,
        windows=windows,
        deterministic=True,
    )

    # Assert identical results
    pd.testing.assert_frame_equal(results1, results2)

    # Verify we have results for all windows
    assert len(results1) == len(windows)

    # Verify required columns
    assert "window_name" in results1.columns
    assert "window_start" in results1.columns
    assert "window_end" in results1.columns
    assert "pass_max_dd" in results1.columns
    assert "pass_sharpe" in results1.columns
    assert "pass_overall" in results1.columns


def test_run_crisis_windows_deterministic_ordering():
    """Test that windows are sorted by start date, then name."""
    def backtest_fn(config):
        return {"sharpe": 0.0, "max_drawdown": -0.10}

    base_config: dict[str, Any] = {}
    windows = [
        {"name": "B", "start": "2020-01-01", "end": "2020-12-31"},
        {"name": "A", "start": "2020-01-01", "end": "2020-12-31"},  # Same start, different name
        {"name": "C", "start": "2019-01-01", "end": "2019-12-31"},  # Earlier start
    ]

    results = run_crisis_windows(
        backtest_fn=backtest_fn,
        base_config=base_config,
        windows=windows,
        deterministic=True,
    )

    # Verify ordering: C (2019) first, then A and B (2020, sorted by name)
    assert results["window_name"].iloc[0] == "C"
    assert results["window_start"].iloc[0] == "2019-01-01"
    # A and B should be sorted by name
    assert results["window_name"].iloc[1] == "A"
    assert results["window_name"].iloc[2] == "B"


def test_run_crisis_windows_pass_fail_flags():
    """Test that pass/fail flags are computed correctly."""
    def backtest_fn(config):
        # Return metrics that will test pass/fail logic
        return {
            "sharpe": -0.5,  # Below sharpe_floor (-1.0) -> pass_sharpe = True
            "max_drawdown": -0.35,  # Below max_dd_threshold (-0.30) -> pass_max_dd = False
            "cagr": -0.20,
        }

    base_config: dict[str, Any] = {}
    windows = [
        {"name": "test", "start": "2020-01-01", "end": "2020-12-31"},
    ]

    results = run_crisis_windows(
        backtest_fn=backtest_fn,
        base_config=base_config,
        windows=windows,
        max_dd_threshold=-0.30,
        sharpe_floor=-1.0,
        deterministic=True,
    )

    row = results.iloc[0]

    # max_dd = -0.35 < -0.30 (threshold) -> pass_max_dd = False
    assert row["pass_max_dd"] is False

    # sharpe = -0.5 >= -1.0 (floor) -> pass_sharpe = True
    assert row["pass_sharpe"] is True

    # pass_overall = False (because pass_max_dd is False)
    assert row["pass_overall"] is False


def test_run_crisis_windows_pass_fail_both_pass():
    """Test that pass_overall is True when both conditions pass."""
    def backtest_fn(config):
        return {
            "sharpe": 0.5,  # Above sharpe_floor (-1.0) -> pass_sharpe = True
            "max_drawdown": -0.20,  # Above max_dd_threshold (-0.30) -> pass_max_dd = True
        }

    base_config: dict[str, Any] = {}
    windows = [
        {"name": "test", "start": "2020-01-01", "end": "2020-12-31"},
    ]

    results = run_crisis_windows(
        backtest_fn=backtest_fn,
        base_config=base_config,
        windows=windows,
        max_dd_threshold=-0.30,
        sharpe_floor=-1.0,
        deterministic=True,
    )

    row = results.iloc[0]
    assert row["pass_max_dd"] is True
    assert row["pass_sharpe"] is True
    assert row["pass_overall"] is True


def test_run_crisis_windows_handles_failures():
    """Test that failed windows are handled gracefully."""
    call_count = [0]

    def backtest_fn(config):
        call_count[0] += 1
        if call_count[0] == 2:  # Fail second window
            raise ValueError("Simulated failure")
        return {"sharpe": 0.0, "max_drawdown": -0.10}

    base_config: dict[str, Any] = {}
    windows = [
        {"name": "window1", "start": "2020-01-01", "end": "2020-12-31"},
        {"name": "window2", "start": "2021-01-01", "end": "2021-12-31"},
        {"name": "window3", "start": "2022-01-01", "end": "2022-12-31"},
    ]

    results = run_crisis_windows(
        backtest_fn=backtest_fn,
        base_config=base_config,
        windows=windows,
        deterministic=True,
    )

    # Should have 3 rows (one per window)
    assert len(results) == 3

    # One row should have error
    assert "error" in results.columns
    assert results["error"].notna().sum() == 1

    # Failed window should have all pass flags = False
    failed_row = results[results["error"].notna()].iloc[0]
    assert failed_row["pass_max_dd"] is False
    assert failed_row["pass_sharpe"] is False
    assert failed_row["pass_overall"] is False


def test_run_crisis_windows_empty_windows():
    """Test that empty windows list raises ValueError."""
    def backtest_fn(config):
        return {"sharpe": 0.0}

    base_config: dict[str, Any] = {}

    with pytest.raises(ValueError, match="windows must not be empty"):
        run_crisis_windows(
            backtest_fn=backtest_fn,
            base_config=base_config,
            windows=[],
            deterministic=True,
        )


def test_run_crisis_windows_date_range_slicing():
    """Test that backtest is restricted to window date range."""
    call_log = []

    def backtest_fn(config):
        # Log the date range used
        start = config.get("start_date")
        end = config.get("end_date")
        call_log.append((start, end))
        return {"sharpe": 0.0, "max_drawdown": -0.10}

    base_config: dict[str, Any] = {}
    windows = [
        {"name": "window1", "start": "2020-01-01", "end": "2020-06-30"},
        {"name": "window2", "start": "2021-01-01", "end": "2021-06-30"},
    ]

    results = run_crisis_windows(
        backtest_fn=backtest_fn,
        base_config=base_config,
        windows=windows,
        deterministic=True,
    )

    # Verify backtest was called with correct date ranges
    assert len(call_log) == 2
    assert call_log[0] == ("2020-01-01", "2020-06-30")
    assert call_log[1] == ("2021-01-01", "2021-06-30")


def test_export_crisis_windows_results_smoke(tmp_path):
    """Test that export_crisis_windows_results produces stable CSV file."""
    from pathlib import Path

    # Create synthetic results
    results_df = pd.DataFrame({
        "window_name": ["GFC", "COVID"],
        "window_start": ["2007-12-01", "2020-02-20"],
        "window_end": ["2009-06-30", "2020-04-30"],
        "sharpe": [-0.5, -1.0],
        "max_drawdown": [-0.40, -0.35],
        "pass_max_dd": [False, False],
        "pass_sharpe": [True, False],
        "pass_overall": [False, False],
    })

    # Export
    csv_path = export_crisis_windows_results(
        results_df=results_df,
        output_dir=Path(tmp_path),
        run_id="test_run",
    )

    # Verify file exists
    assert csv_path.exists()

    # Verify CSV is readable
    results_read = pd.read_csv(csv_path)
    assert len(results_read) == 2
    assert "window_name" in results_read.columns
    assert "pass_overall" in results_read.columns

    # Verify deterministic ordering (sorted by window_start, then window_name)
    assert results_read["window_start"].iloc[0] == "2007-12-01"
    assert results_read["window_start"].iloc[1] == "2020-02-20"
