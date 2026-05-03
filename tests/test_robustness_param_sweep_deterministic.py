"""Tests for deterministic parameter sweep (RB2).

These tests verify that parameter sweeps are generated deterministically:
- Same input -> same output order
- Deterministic combination generation
- Plateau detection is reproducible
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

pytest.importorskip('src.assembled_core.qa.robustness')
from src.assembled_core.qa.robustness import (
    detect_plateau,
    run_param_grid_sweep,
)


def test_run_param_grid_sweep_deterministic():
    """Test that same input produces identical output order."""

    # Define simple backtest function
    def backtest_fn(config):
        return {
            "sharpe": float(config.get("ma_fast", 0) + config.get("ma_slow", 0))
            / 100.0,
            "cagr": 0.10,
            "max_drawdown": -0.05,
            "turnover": 0.5,
        }

    base_config = {"strategy": "ema"}
    grid = {
        "ma_fast": [10, 20],
        "ma_slow": [50, 100, 200],
    }

    # Run sweep twice
    results1 = run_param_grid_sweep(
        backtest_fn=backtest_fn,
        base_config=base_config,
        grid=grid,
        deterministic=True,
    )

    results2 = run_param_grid_sweep(
        backtest_fn=backtest_fn,
        base_config=base_config,
        grid=grid,
        deterministic=True,
    )

    # Assert identical results
    pd.testing.assert_frame_equal(results1, results2)

    # Verify we have 2 * 3 = 6 combinations
    assert len(results1) == 6

    # Verify parameter columns exist
    assert "ma_fast" in results1.columns
    assert "ma_slow" in results1.columns

    # Verify metric columns exist
    assert "sharpe" in results1.columns
    assert "cagr" in results1.columns

    # Verify deterministic ordering (sorted by parameters)
    assert results1["ma_fast"].tolist() == [10, 10, 10, 20, 20, 20]
    assert results1["ma_slow"].tolist() == [50, 100, 200, 50, 100, 200]


def test_run_param_grid_sweep_deterministic_ordering():
    """Test that parameter combinations are generated in deterministic order."""

    def backtest_fn(config):
        return {"sharpe": 1.0}

    base_config = {}
    grid = {
        "param_a": [1, 2],
        "param_b": [10, 20],
    }

    results = run_param_grid_sweep(
        backtest_fn=backtest_fn,
        base_config=base_config,
        grid=grid,
        deterministic=True,
    )

    # Verify lexicographic ordering (param_a first, then param_b)
    expected_order = [
        (1, 10),
        (1, 20),
        (2, 10),
        (2, 20),
    ]

    actual_order = list(zip(results["param_a"], results["param_b"], strict=True))
    assert actual_order == expected_order


def test_run_param_grid_sweep_empty_grid():
    """Test that empty grid raises ValueError."""

    def backtest_fn(config):
        return {"sharpe": 1.0}

    base_config: dict[str, Any] = {}

    with pytest.raises(ValueError, match="grid must not be empty"):
        run_param_grid_sweep(
            backtest_fn=backtest_fn,
            base_config=base_config,
            grid={},
            deterministic=True,
        )


def test_run_param_grid_sweep_handles_failures():
    """Test that failed backtests are handled gracefully."""
    call_count = [0]

    def backtest_fn(config):
        call_count[0] += 1
        if call_count[0] == 2:  # Fail second call
            raise ValueError("Simulated failure")
        return {"sharpe": 1.0, "cagr": 0.10}

    base_config: dict[str, Any] = {}
    grid = {"param": [1, 2, 3]}

    results = run_param_grid_sweep(
        backtest_fn=backtest_fn,
        base_config=base_config,
        grid=grid,
        deterministic=True,
    )

    # Should have 3 rows (one per parameter value)
    assert len(results) == 3

    # One row should have error
    assert "error" in results.columns
    assert results["error"].notna().sum() == 1


def test_detect_plateau_reproducible():
    """Test that plateau detection is reproducible."""
    # Create synthetic results with clear plateau
    results_df = pd.DataFrame(
        {
            "ma_fast": [10, 10, 20, 20, 30, 30],
            "ma_slow": [50, 100, 50, 100, 50, 100],
            "sharpe": [1.0, 1.2, 1.15, 1.18, 0.8, 0.9],  # Best is 1.2, plateau at 1.15+
        }
    )

    # Run plateau detection twice
    plateau1 = detect_plateau(
        results_df=results_df,
        metric="sharpe",
        top_k=3,
        epsilon=0.05,
    )

    plateau2 = detect_plateau(
        results_df=results_df,
        metric="sharpe",
        top_k=3,
        epsilon=0.05,
    )

    # Assert identical results
    assert plateau1 == plateau2

    # Verify structure
    assert "plateau_size" in plateau1
    assert "plateau_fraction" in plateau1
    assert "best_metric" in plateau1
    assert "robust_score" in plateau1
    assert "plateau_threshold" in plateau1
    assert "top_k_combinations" in plateau1

    # Verify best_metric is correct
    assert plateau1["best_metric"] == 1.2

    # Verify plateau includes combinations within 5% of best (1.2 * 0.95 = 1.14)
    # So 1.2, 1.18, 1.15 should be in plateau (3 combinations)
    assert plateau1["plateau_size"] >= 3


def test_detect_plateau_negative_metrics():
    """Test plateau detection with negative metrics (e.g., max_drawdown)."""
    results_df = pd.DataFrame(
        {
            "param": [1, 2, 3],
            "max_drawdown": [-0.20, -0.10, -0.15],  # Best (least negative) is -0.10
        }
    )

    plateau = detect_plateau(
        results_df=results_df,
        metric="max_drawdown",
        top_k=2,
        epsilon=0.05,
    )

    # Best metric should be -0.10 (least negative)
    assert plateau["best_metric"] == -0.10

    # Plateau threshold should be -0.10 * 1.05 = -0.105 (for negative metrics)
    assert plateau["plateau_threshold"] <= -0.10


def test_detect_plateau_empty_results():
    """Test plateau detection with empty or all-NaN results."""
    results_df = pd.DataFrame(
        {
            "param": [1, 2],
            "sharpe": [math.nan, math.nan],
        }
    )

    plateau = detect_plateau(
        results_df=results_df,
        metric="sharpe",
        top_k=5,
        epsilon=0.05,
    )

    assert plateau["plateau_size"] == 0
    assert plateau["plateau_fraction"] == 0.0
    assert plateau["best_metric"] is None
    assert plateau["robust_score"] == 0.0
    assert plateau["top_k_combinations"] == []


def test_detect_plateau_top_k_combinations():
    """Test that top_k_combinations contains correct number of entries."""
    results_df = pd.DataFrame(
        {
            "param": [1, 2, 3, 4, 5],
            "sharpe": [1.0, 1.5, 1.2, 1.3, 1.1],
        }
    )

    plateau = detect_plateau(
        results_df=results_df,
        metric="sharpe",
        top_k=3,
        epsilon=0.05,
    )

    # Should have 3 top combinations
    assert len(plateau["top_k_combinations"]) == 3

    # Top combination should have highest sharpe
    top_combos = plateau["top_k_combinations"]
    assert top_combos[0]["sharpe"] == 1.5  # Highest

    # Verify all top_k entries are dicts with param and sharpe
    for combo in top_combos:
        assert "param" in combo
        assert "sharpe" in combo
