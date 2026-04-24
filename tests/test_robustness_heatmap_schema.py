"""Tests for heatmap table schema and pivot correctness (RB2).

These tests verify that heatmap tables have correct shape, handle missing
combinations deterministically, and produce valid pivot tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import pytest; pytest.importorskip('src.assembled_core.qa.robustness')
from src.assembled_core.qa.robustness import (
    build_heatmap_table,
    export_robustness_sweep_results,
    run_param_grid_sweep,
)


def test_build_heatmap_table_correct_shape():
    """Test that heatmap table has correct pivot shape."""

    # Create results with 2x3 grid
    def backtest_fn(config):
        return {"sharpe": float(config["x"] + config["y"])}

    base_config = {}
    grid = {
        "x": [1, 2],
        "y": [10, 20, 30],
    }

    results_df = run_param_grid_sweep(
        backtest_fn=backtest_fn,
        base_config=base_config,
        grid=grid,
        deterministic=True,
    )

    # Build heatmap
    heatmap_df = build_heatmap_table(
        results_df=results_df,
        x_param="x",
        y_param="y",
        metric="sharpe",
    )

    # Verify shape: 3 rows (y values) x 2 columns (x values)
    assert heatmap_df.shape == (3, 2)

    # Verify index is y values
    assert list(heatmap_df.index) == [10, 20, 30]

    # Verify columns are x values
    assert list(heatmap_df.columns) == [1, 2]


def test_build_heatmap_table_missing_combinations():
    """Test that missing combinations are represented as NaN deterministically."""
    # Create results with some missing combinations
    results_df = pd.DataFrame(
        {
            "x": [1, 1, 2],  # Missing: x=2, y=20
            "y": [10, 20, 10],
            "sharpe": [1.0, 1.2, 1.1],
        }
    )

    heatmap_df = build_heatmap_table(
        results_df=results_df,
        x_param="x",
        y_param="y",
        metric="sharpe",
    )

    # Should have 2 rows (y=10, y=20) and 2 columns (x=1, x=2)
    assert heatmap_df.shape == (2, 2)

    # Missing combination (x=2, y=20) should be NaN
    assert pd.isna(heatmap_df.loc[20, 2])

    # Existing combinations should have values
    assert not pd.isna(heatmap_df.loc[10, 1])
    assert not pd.isna(heatmap_df.loc[10, 2])
    assert not pd.isna(heatmap_df.loc[20, 1])


def test_build_heatmap_table_deterministic_ordering():
    """Test that heatmap table is sorted deterministically."""
    results_df = pd.DataFrame(
        {
            "x": [3, 1, 2, 1, 2, 3],
            "y": [30, 10, 20, 20, 10, 30],
            "sharpe": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        }
    )

    heatmap_df = build_heatmap_table(
        results_df=results_df,
        x_param="x",
        y_param="y",
        metric="sharpe",
    )

    # Verify index is sorted
    assert list(heatmap_df.index) == sorted([10, 20, 30])

    # Verify columns are sorted
    assert list(heatmap_df.columns) == sorted([1, 2, 3])


def test_build_heatmap_table_duplicate_combinations():
    """Test that duplicate combinations use first value (aggfunc='first')."""
    results_df = pd.DataFrame(
        {
            "x": [1, 1, 1],
            "y": [10, 10, 10],  # Duplicate combination
            "sharpe": [1.0, 1.5, 1.2],  # First value (1.0) should be used
        }
    )

    heatmap_df = build_heatmap_table(
        results_df=results_df,
        x_param="x",
        y_param="y",
        metric="sharpe",
    )

    # Should have single value
    assert heatmap_df.shape == (1, 1)
    assert heatmap_df.loc[10, 1] == 1.0  # First value


def test_build_heatmap_table_missing_metric():
    """Test that missing metric raises ValueError."""
    results_df = pd.DataFrame(
        {
            "x": [1, 2],
            "y": [10, 20],
            "cagr": [0.10, 0.15],
        }
    )

    with pytest.raises(ValueError, match="metric 'sharpe' not found"):
        build_heatmap_table(
            results_df=results_df,
            x_param="x",
            y_param="y",
            metric="sharpe",
        )


def test_build_heatmap_table_missing_params():
    """Test that missing parameter columns raise ValueError."""
    results_df = pd.DataFrame(
        {
            "x": [1, 2],
            "sharpe": [1.0, 1.1],
        }
    )

    with pytest.raises(ValueError, match="y_param 'y' not found"):
        build_heatmap_table(
            results_df=results_df,
            x_param="x",
            y_param="y",
            metric="sharpe",
        )


def test_build_heatmap_table_string_params():
    """Test that string parameter values work correctly."""
    results_df = pd.DataFrame(
        {
            "strategy": ["ema", "sma"],
            "freq": ["1d", "5min"],
            "sharpe": [1.0, 1.1],
        }
    )

    heatmap_df = build_heatmap_table(
        results_df=results_df,
        x_param="strategy",
        y_param="freq",
        metric="sharpe",
    )

    # Verify shape
    assert heatmap_df.shape == (2, 2)

    # Verify string values are preserved
    assert "ema" in heatmap_df.columns
    assert "sma" in heatmap_df.columns
    assert "1d" in heatmap_df.index
    assert "5min" in heatmap_df.index


def test_export_robustness_sweep_results_smoke(tmp_path: Path):
    """Test that export_robustness_sweep_results produces stable files."""
    # Create synthetic results
    results_df = pd.DataFrame(
        {
            "ma_fast": [10, 10, 20, 20],
            "ma_slow": [50, 100, 50, 100],
            "sharpe": [1.0, 1.2, 1.1, 1.3],
            "cagr": [0.10, 0.12, 0.11, 0.13],
        }
    )

    # Build heatmap
    heatmap_df = build_heatmap_table(
        results_df=results_df,
        x_param="ma_fast",
        y_param="ma_slow",
        metric="sharpe",
    )

    heatmap_tables = {"ma_fast_ma_slow": heatmap_df}

    # Create plateau info
    from src.assembled_core.qa.robustness import detect_plateau

    plateau_info = detect_plateau(
        results_df=results_df,
        metric="sharpe",
        top_k=3,
        epsilon=0.05,
    )

    # Export
    exported = export_robustness_sweep_results(
        results_df=results_df,
        heatmap_tables=heatmap_tables,
        plateau_info=plateau_info,
        output_dir=tmp_path,
        run_id="test_run",
    )

    # Verify files exist
    assert exported["results_csv"].exists()
    assert exported["plateau_json"].exists()
    assert "ma_fast_ma_slow" in exported["heatmap_csvs"]

    # Verify results CSV
    results_read = pd.read_csv(exported["results_csv"])
    assert len(results_read) == 4
    assert "ma_fast" in results_read.columns
    assert "sharpe" in results_read.columns

    # Verify heatmap CSV
    heatmap_read = pd.read_csv(exported["heatmap_csvs"]["ma_fast_ma_slow"], index_col=0)
    assert heatmap_read.shape == (2, 2)  # 2 ma_slow values x 2 ma_fast values

    # Verify plateau JSON
    import json

    with exported["plateau_json"].open("r", encoding="utf-8") as f:
        plateau_read = json.load(f)
    assert "plateau_size" in plateau_read
    assert "best_metric" in plateau_read
