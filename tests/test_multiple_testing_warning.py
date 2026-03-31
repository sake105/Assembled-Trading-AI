"""Tests for multiple testing warnings (RB5).

These tests verify that multiple testing warnings are built correctly:
- Heuristic detects inflated best metric
- Warnings are deterministic
- Edge cases (empty results, missing columns)
"""

from __future__ import annotations


import pandas as pd
import pytest

from src.assembled_core.qa.robustness import build_multiple_testing_warnings


def test_build_multiple_testing_warnings_inflated():
    """Test that inflated best metric triggers warning."""
    # Create synthetic results with inflated best Sharpe
    results_df = pd.DataFrame(
        {
            "param": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "sharpe": [
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                5.0,
            ],  # Inflated best
        }
    )

    warnings = build_multiple_testing_warnings(results_df, metric_col="sharpe")

    # Should detect warning
    assert warnings["n_trials"] == 15
    assert warnings["best_metric"] == 5.0
    assert warnings["median_metric"] == pytest.approx(1.2, abs=0.1)
    assert warnings["metric_spread"] == pytest.approx(5.0 - 1.2, abs=0.1)
    assert warnings["warning_inflated"] is True
    assert "warning" in warnings["warning_message"].lower()


def test_build_multiple_testing_warnings_no_warning():
    """Test that small spread or few trials don't trigger warning."""
    # Small number of trials
    results_df = pd.DataFrame(
        {
            "param": [1, 2, 3],
            "sharpe": [1.0, 1.1, 1.2],
        }
    )

    warnings = build_multiple_testing_warnings(results_df, metric_col="sharpe")

    assert warnings["n_trials"] == 3
    assert warnings["warning_inflated"] is False

    # Small spread (even with many trials)
    results_df2 = pd.DataFrame(
        {
            "param": list(range(1, 21)),
            "sharpe": [1.0] * 20,  # All same (spread = 0)
        }
    )

    warnings2 = build_multiple_testing_warnings(results_df2, metric_col="sharpe")

    assert warnings2["n_trials"] == 20
    assert warnings2["warning_inflated"] is False


def test_build_multiple_testing_warnings_deterministic():
    """Test that same input produces identical warnings."""
    results_df = pd.DataFrame(
        {
            "param": list(range(1, 16)),
            "sharpe": [0.5 + i * 0.1 for i in range(15)],
        }
    )

    warnings1 = build_multiple_testing_warnings(results_df, metric_col="sharpe")
    warnings2 = build_multiple_testing_warnings(results_df, metric_col="sharpe")

    # Should be identical
    assert warnings1 == warnings2


def test_build_multiple_testing_warnings_missing_column():
    """Test that missing metric column returns empty warnings."""
    results_df = pd.DataFrame(
        {
            "param": [1, 2, 3],
            "other_metric": [1.0, 2.0, 3.0],
        }
    )

    warnings = build_multiple_testing_warnings(results_df, metric_col="sharpe")

    assert warnings["n_trials"] == 0
    assert warnings["best_metric"] is None
    assert warnings["warning_inflated"] is False


def test_build_multiple_testing_warnings_all_nan():
    """Test that all-NaN metrics return empty warnings."""
    results_df = pd.DataFrame(
        {
            "param": [1, 2, 3],
            "sharpe": [float("nan"), float("nan"), float("nan")],
        }
    )

    warnings = build_multiple_testing_warnings(results_df, metric_col="sharpe")

    assert warnings["n_trials"] == 0
    assert warnings["best_metric"] is None
    assert warnings["warning_inflated"] is False


def test_build_multiple_testing_warnings_some_nan():
    """Test that some NaN values are filtered out."""
    results_df = pd.DataFrame(
        {
            "param": [1, 2, 3, 4, 5],
            "sharpe": [1.0, float("nan"), 1.2, float("nan"), 1.5],
        }
    )

    warnings = build_multiple_testing_warnings(results_df, metric_col="sharpe")

    # Should only count non-NaN values
    assert warnings["n_trials"] == 3
    assert warnings["best_metric"] == 1.5
    assert warnings["median_metric"] == 1.2


def test_build_multiple_testing_warnings_custom_metric():
    """Test that custom metric column works."""
    results_df = pd.DataFrame(
        {
            "param": list(range(1, 16)),
            "cagr": [0.05 + i * 0.01 for i in range(15)],
        }
    )

    warnings = build_multiple_testing_warnings(results_df, metric_col="cagr")

    assert warnings["n_trials"] == 15
    assert warnings["best_metric"] == pytest.approx(0.19, abs=0.01)
    assert warnings["median_metric"] == pytest.approx(0.12, abs=0.01)


def test_build_multiple_testing_warnings_threshold_edge_case():
    """Test threshold edge cases (n_trials=10, spread=2.0)."""
    # Exactly at threshold (should trigger warning)
    results_df = pd.DataFrame(
        {
            "param": list(range(1, 11)),  # n_trials = 10
            "sharpe": [1.0] * 9 + [3.0],  # spread = 2.0
        }
    )

    warnings = build_multiple_testing_warnings(results_df, metric_col="sharpe")

    assert warnings["n_trials"] == 10
    assert warnings["metric_spread"] == pytest.approx(2.0, abs=0.01)
    assert warnings["warning_inflated"] is True

    # Just below threshold (spread = 1.99)
    results_df2 = pd.DataFrame(
        {
            "param": list(range(1, 11)),
            "sharpe": [1.0] * 9 + [2.99],  # spread = 1.99
        }
    )

    warnings2 = build_multiple_testing_warnings(results_df2, metric_col="sharpe")

    assert warnings2["metric_spread"] == pytest.approx(1.99, abs=0.01)
    assert warnings2["warning_inflated"] is False
