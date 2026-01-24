"""Smoke tests for walk-forward runner (RB1).

These tests verify that the walk-forward runner produces stable outputs
with synthetic data and that key columns don't contain NaNs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.qa.walk_forward import (
    export_walk_forward_results,
    make_walk_forward_splits,
    run_walk_forward,
)


def test_run_walk_forward_smoke():
    """Smoke test: run_walk_forward with synthetic data produces stable schema."""
    # Create synthetic price data (3 years, daily)
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D", tz="UTC")
    prices_df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "close": 100.0 + (pd.Series(range(len(dates))) * 0.1),  # Simple trend
        }
    )

    # Generate splits
    splits = make_walk_forward_splits(
        prices_df=prices_df,
        n_splits=3,
        train_days=252,
        test_days=63,
        seed=0,
    )

    # Create simple backtest function (returns synthetic metrics)
    def backtest_fn(train_start, train_end, test_start, test_end):
        """Simple backtest function that returns synthetic metrics."""
        return {
            "sharpe": 1.5,
            "cagr": 0.15,
            "max_drawdown": -0.10,
            "total_return": 0.12,
        }

    # Run walk-forward
    wf_result = run_walk_forward(
        backtest_fn=backtest_fn,
        splits=splits,
        config=None,
        deterministic_seed=0,
    )

    # Verify result structure
    assert "splits" in wf_result
    assert "metrics" in wf_result
    assert "summary_df" in wf_result
    assert "oos_first_metrics" in wf_result

    # Verify splits
    assert len(wf_result["splits"]) == len(splits)
    assert wf_result["splits"] == splits

    # Verify metrics structure
    metrics = wf_result["metrics"]
    assert "mean_sharpe" in metrics
    assert "mean_cagr" in metrics
    assert "mean_max_drawdown" in metrics
    assert "n_splits" in metrics
    assert "n_successful_splits" in metrics

    # Verify OOS-first metrics
    oos_metrics = wf_result["oos_first_metrics"]
    assert "oos_mean_sharpe" in oos_metrics
    assert "oos_mean_cagr" in oos_metrics
    assert "oos_mean_max_dd" in oos_metrics
    assert "oos_win_rate" in oos_metrics

    # Verify no NaNs in key metrics (all splits succeeded)
    assert not math.isnan(oos_metrics["oos_mean_sharpe"])
    assert not math.isnan(oos_metrics["oos_mean_cagr"])
    assert not math.isnan(oos_metrics["oos_mean_max_dd"])


def test_run_walk_forward_no_nans_in_key_columns():
    """Test that key columns in summary_df don't contain NaNs for successful splits."""
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D", tz="UTC")
    prices_df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "close": 100.0,
        }
    )

    splits = make_walk_forward_splits(
        prices_df=prices_df,
        n_splits=3,
        train_days=252,
        test_days=63,
        seed=0,
    )

    def backtest_fn(train_start, train_end, test_start, test_end):
        return {
            "sharpe": 1.0,
            "cagr": 0.10,
            "max_drawdown": -0.05,
            "total_return": 0.08,
        }

    wf_result = run_walk_forward(
        backtest_fn=backtest_fn,
        splits=splits,
        config=None,
        deterministic_seed=0,
    )

    # Convert summary_df to DataFrame for easier checking
    summary_df = pd.DataFrame(wf_result["summary_df"])

    # Check successful splits (status == "success")
    successful = summary_df[summary_df["status"] == "success"]

    # Key columns should not be NaN for successful splits
    assert not successful["sharpe"].isna().any()
    assert not successful["cagr"].isna().any()
    assert not successful["max_drawdown"].isna().any()
    assert not successful["total_return"].isna().any()


def test_export_walk_forward_results_smoke(tmp_path: Path):
    """Smoke test: export_walk_forward_results produces stable files."""
    # Create synthetic result
    splits = [
        {
            "split_index": 0,
            "train_start": "2020-01-01T00:00:00+00:00",
            "train_end": "2020-12-31T00:00:00+00:00",
            "test_start": "2021-01-01T00:00:00+00:00",
            "test_end": "2021-03-31T00:00:00+00:00",
            "n_train": 252,
            "n_test": 63,
        }
    ]

    wf_result = {
        "splits": splits,
        "metrics": {
            "mean_sharpe": 1.5,
            "std_sharpe": 0.2,
            "mean_cagr": 0.15,
            "n_splits": 1,
            "n_successful_splits": 1,
            "n_failed_splits": 0,
        },
        "summary_df": [
            {
                "split_index": 0,
                "train_start": "2020-01-01T00:00:00+00:00",
                "train_end": "2020-12-31T00:00:00+00:00",
                "test_start": "2021-01-01T00:00:00+00:00",
                "test_end": "2021-03-31T00:00:00+00:00",
                "n_train": 252,
                "n_test": 63,
                "status": "success",
                "sharpe": 1.5,
                "cagr": 0.15,
                "max_drawdown": -0.10,
                "total_return": 0.12,
            }
        ],
        "oos_first_metrics": {
            "oos_mean_sharpe": 1.5,
            "oos_mean_cagr": 0.15,
            "oos_mean_max_dd": -0.10,
            "oos_win_rate": 1.0,
        },
    }

    # Export results
    exported = export_walk_forward_results(
        wf_result=wf_result,
        output_dir=tmp_path,
        run_id="test_run",
    )

    # Verify files exist
    assert exported["splits_json"].exists()
    assert exported["summary_csv"].exists()
    assert exported["metrics_json"].exists()

    # Verify splits.json is valid JSON with sort_keys
    with exported["splits_json"].open("r", encoding="utf-8") as f:
        splits_data = json.load(f)
    assert isinstance(splits_data, list)
    assert len(splits_data) == 1

    # Verify wf_metrics.json is valid JSON
    with exported["metrics_json"].open("r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert "aggregated_metrics" in metrics_data
    assert "oos_first_metrics" in metrics_data

    # Verify wf_summary.csv is valid CSV
    summary_df = pd.read_csv(exported["summary_csv"])
    assert len(summary_df) == 1
    assert "split_index" in summary_df.columns
    assert "sharpe" in summary_df.columns


def test_run_walk_forward_handles_failures():
    """Test that run_walk_forward handles failed splits gracefully."""
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D", tz="UTC")
    prices_df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "close": 100.0,
        }
    )

    splits = make_walk_forward_splits(
        prices_df=prices_df,
        n_splits=3,
        train_days=252,
        test_days=63,
        seed=0,
    )

    # Backtest function that fails for split_index 1
    def backtest_fn(train_start, train_end, test_start, test_end):
        # Check if this is split 1 (by checking test_start)
        # This is a simplified check - in real scenario, we'd track split_index
        if test_start.year == 2021:  # Approximate check
            raise ValueError("Simulated failure")
        return {
            "sharpe": 1.0,
            "cagr": 0.10,
            "max_drawdown": -0.05,
            "total_return": 0.08,
        }

    wf_result = run_walk_forward(
        backtest_fn=backtest_fn,
        splits=splits,
        config=None,
        deterministic_seed=0,
    )

    # Should have some successful and some failed splits
    summary_df = pd.DataFrame(wf_result["summary_df"])
    assert len(summary_df) == len(splits)
    assert (summary_df["status"] == "success").sum() > 0
    assert (summary_df["status"] == "failed").sum() > 0

    # Failed splits should have error message
    failed = summary_df[summary_df["status"] == "failed"]
    assert "error" in failed.columns
    assert not failed["error"].isna().all()
