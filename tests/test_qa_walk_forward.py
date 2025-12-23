"""Tests for Walk-Forward Analysis (B3).

These tests verify that walk-forward split generation and backtest execution
work correctly for both expanding and rolling window modes.
"""

from __future__ import annotations

import pytest

import pandas as pd

from src.assembled_core.qa.walk_forward import (
    WalkForwardConfig,
    generate_walk_forward_splits,
    run_walk_forward_backtest,
)

pytestmark = pytest.mark.advanced


@pytest.fixture
def sample_date_range() -> pd.Series:
    """Create sample date range for testing (100 days)."""
    return pd.date_range(
        start="2024-01-01",
        end="2024-04-09",  # 100 days
        freq="D",
        tz="UTC",
    )


def test_generate_walk_forward_splits_expanding_basic():
    """Test expanding window split generation with basic config."""
    config = WalkForwardConfig(
        start_date=pd.Timestamp("2024-01-01", tz="UTC"),
        end_date=pd.Timestamp("2024-04-09", tz="UTC"),  # 100 days
        train_window_days=None,  # Expanding (not used in expanding mode)
        test_window_days=20,
        mode="expanding",
        step_size_days=20,
        min_train_periods=40,
        min_test_periods=15,
    )

    splits = generate_walk_forward_splits(
        start_date=config.start_date,
        end_date=config.end_date,
        config=config,
    )

    assert len(splits) > 0, "Should generate at least one split"

    # Verify splits don't overlap (test windows)
    for i, split in enumerate(splits):
        assert split.test_start < split.test_end, "Test window should be valid"
        assert split.train_start < split.train_end, "Train window should be valid"
        assert split.train_end <= split.test_start, (
            "Train should end before test starts"
        )
        assert split.split_index == i, "Split index should match list index"

        # Verify expanding window: train_start should always be start_date
        assert split.train_start == config.start_date, (
            f"Expanding window: train_start should be {config.start_date}"
        )

        # Verify train window grows (for expanding mode)
        if i > 0:
            assert split.train_end >= splits[i - 1].train_end, (
                "Expanding window: train_end should grow or stay same"
            )

        # Verify test windows don't overlap (when step_size >= test_window)
        if i > 0:
            prev_split = splits[i - 1]
            assert split.test_start >= prev_split.test_end, (
                "Test windows should not overlap"
            )

    # Verify all test windows are within date range
    for split in splits:
        assert split.test_start >= config.start_date, (
            f"Test start {split.test_start} should be >= start_date {config.start_date}"
        )
        assert split.test_end <= config.end_date + pd.Timedelta(days=1), (
            f"Test end {split.test_end} should be <= end_date {config.end_date}"
        )


def test_generate_walk_forward_splits_rolling_basic():
    """Test rolling window split generation with basic config."""
    config = WalkForwardConfig(
        start_date=pd.Timestamp("2024-01-01", tz="UTC"),
        end_date=pd.Timestamp("2024-04-09", tz="UTC"),  # 100 days
        train_window_days=40,
        test_window_days=20,
        mode="rolling",
        step_size_days=20,
        min_train_periods=30,
        min_test_periods=15,
    )

    splits = generate_walk_forward_splits(
        start_date=config.start_date,
        end_date=config.end_date,
        config=config,
    )

    assert len(splits) > 0, "Should generate at least one split"

    # Verify rolling window: train window should be fixed size
    for i, split in enumerate(splits):
        assert split.test_start < split.test_end, "Test window should be valid"
        assert split.train_start < split.train_end, "Train window should be valid"
        assert split.train_end <= split.test_start, (
            "Train should end before test starts"
        )

        # Verify rolling window: train window should be approximately fixed size
        train_days = (split.train_end - split.train_start).days
        assert train_days >= config.min_train_periods, (
            f"Train window should be at least {config.min_train_periods} days"
        )

        # Verify test windows don't overlap (when step_size >= test_window)
        if i > 0:
            prev_split = splits[i - 1]
            assert split.test_start >= prev_split.test_end, (
                "Test windows should not overlap"
            )

    # Verify all test windows are within date range
    for split in splits:
        assert split.test_start >= config.start_date, (
            f"Test start {split.test_start} should be >= start_date {config.start_date}"
        )
        assert split.test_end <= config.end_date + pd.Timedelta(days=1), (
            f"Test end {split.test_end} should be <= end_date {config.end_date}"
        )


def test_generate_walk_forward_splits_validation():
    """Test that split generation validates config correctly."""
    # Test: invalid mode without train_window_days
    with pytest.raises(ValueError, match="train_window_days must be provided"):
        config = WalkForwardConfig(
            start_date=pd.Timestamp("2024-01-01", tz="UTC"),
            end_date=pd.Timestamp("2024-04-09", tz="UTC"),
            train_window_days=None,
            test_window_days=20,
            mode="rolling",  # Requires train_window_days
        )
        generate_walk_forward_splits(config.start_date, config.end_date, config)

    # Test: insufficient data
    with pytest.raises(ValueError, match="Insufficient data"):
        config = WalkForwardConfig(
            start_date=pd.Timestamp("2024-01-01", tz="UTC"),
            end_date=pd.Timestamp("2024-01-10", tz="UTC"),  # Only 10 days
            train_window_days=252,
            test_window_days=63,
            mode="rolling",
            min_train_periods=252,
            min_test_periods=63,
        )
        generate_walk_forward_splits(config.start_date, config.end_date, config)

    # Test: overlap not allowed but step_size < test_window
    with pytest.raises(ValueError, match="step_size_days.*must be >= test_window_days"):
        config = WalkForwardConfig(
            start_date=pd.Timestamp("2024-01-01", tz="UTC"),
            end_date=pd.Timestamp("2024-04-09", tz="UTC"),
            train_window_days=40,
            test_window_days=20,
            mode="rolling",
            step_size_days=10,  # < test_window_days, but overlap_allowed=False
            overlap_allowed=False,
        )
        generate_walk_forward_splits(config.start_date, config.end_date, config)


def test_run_walk_forward_backtest_aggregates_metrics():
    """Test that run_walk_forward_backtest correctly aggregates metrics."""

    # Create dummy backtest function that returns deterministic metrics
    def dummy_backtest_fn(
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> dict[str, float | int]:
        # Return deterministic metrics based on split timing
        # Use test_start day-of-year as a simple hash
        day_of_year = test_start.dayofyear

        return {
            "test_sharpe": float(day_of_year % 10) / 10.0,  # 0.0 to 0.9
            "test_return": float(day_of_year % 20) / 100.0,  # 0.0 to 0.19
            "test_max_dd": -float(day_of_year % 15) / 100.0,  # -0.14 to 0.0
            "test_trades": day_of_year % 50,  # 0 to 49
        }

    config = WalkForwardConfig(
        start_date=pd.Timestamp("2024-01-01", tz="UTC"),
        end_date=pd.Timestamp("2024-04-09", tz="UTC"),  # 100 days
        train_window_days=40,
        test_window_days=20,
        mode="rolling",
        step_size_days=20,
        min_train_periods=30,
        min_test_periods=15,
    )

    result = run_walk_forward_backtest(
        config=config,
        backtest_fn=dummy_backtest_fn,
    )

    # Verify result structure
    assert len(result.window_results) > 0, "Should have at least one window result"
    assert len(result.summary_df) == len(result.window_results), (
        "Summary DataFrame should have one row per window result"
    )

    # Verify aggregated metrics
    assert "n_splits" in result.aggregated_metrics
    assert "n_successful_splits" in result.aggregated_metrics
    assert result.aggregated_metrics["n_splits"] == len(result.window_results)
    assert result.aggregated_metrics["n_successful_splits"] > 0

    # Verify metric aggregations exist
    assert "mean_test_sharpe" in result.aggregated_metrics
    assert "std_test_sharpe" in result.aggregated_metrics
    assert "min_test_sharpe" in result.aggregated_metrics
    assert "max_test_sharpe" in result.aggregated_metrics

    # Verify aggregated values are reasonable
    mean_sharpe = result.aggregated_metrics["mean_test_sharpe"]
    min_sharpe = result.aggregated_metrics["min_test_sharpe"]
    max_sharpe = result.aggregated_metrics["max_test_sharpe"]

    assert min_sharpe <= mean_sharpe + 1e-10, (
        f"Mean ({mean_sharpe}) should be >= min ({min_sharpe})"
    )
    assert mean_sharpe <= max_sharpe + 1e-10, (
        f"Mean ({mean_sharpe}) should be <= max ({max_sharpe})"
    )

    # Verify summary DataFrame structure
    assert "split_index" in result.summary_df.columns
    assert "test_start" in result.summary_df.columns
    assert "test_end" in result.summary_df.columns
    assert "status" in result.summary_df.columns

    # Verify all successful splits have metrics
    successful_splits = result.summary_df[result.summary_df["status"] == "success"]
    if len(successful_splits) > 0:
        assert "test_sharpe" in successful_splits.columns
        assert "test_return" in successful_splits.columns


def test_run_walk_forward_backtest_handles_failures():
    """Test that run_walk_forward_backtest handles failed splits gracefully."""
    failure_count = [0]

    def failing_backtest_fn(
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> dict[str, float | int]:
        # Fail on every other split
        split_idx = (test_start - pd.Timestamp("2024-01-01", tz="UTC")).days // 20
        if split_idx % 2 == 0:
            failure_count[0] += 1
            raise ValueError(f"Simulated failure for split {split_idx}")

        return {
            "test_sharpe": 1.0,
            "test_return": 0.1,
            "test_trades": 10,
        }

    config = WalkForwardConfig(
        start_date=pd.Timestamp("2024-01-01", tz="UTC"),
        end_date=pd.Timestamp("2024-04-09", tz="UTC"),
        train_window_days=40,
        test_window_days=20,
        mode="rolling",
        step_size_days=20,
        min_train_periods=30,
        min_test_periods=15,
    )

    result = run_walk_forward_backtest(
        config=config,
        backtest_fn=failing_backtest_fn,
    )

    # Verify some splits succeeded and some failed
    assert result.aggregated_metrics["n_failed_splits"] > 0, (
        "Should have some failed splits"
    )
    assert result.aggregated_metrics["n_successful_splits"] > 0, (
        "Should have some successful splits"
    )

    # Verify failed splits are marked correctly
    failed_splits = result.summary_df[result.summary_df["status"] == "failed"]
    assert len(failed_splits) == result.aggregated_metrics["n_failed_splits"]

    # Verify error messages are recorded
    for _, row in failed_splits.iterrows():
        # The error message should be in the window_result
        split_idx = int(row["split_index"])
        window_result = result.window_results[split_idx]
        assert window_result.status == "failed"
        assert window_result.error_message is not None


def test_generate_walk_forward_splits_max_splits():
    """Test that max_splits limit is respected."""
    config = WalkForwardConfig(
        start_date=pd.Timestamp("2024-01-01", tz="UTC"),
        end_date=pd.Timestamp("2024-04-09", tz="UTC"),
        train_window_days=40,
        test_window_days=20,
        mode="rolling",
        step_size_days=20,
        min_train_periods=30,
        min_test_periods=15,
        max_splits=3,  # Limit to 3 splits
    )

    splits = generate_walk_forward_splits(
        start_date=config.start_date,
        end_date=config.end_date,
        config=config,
    )

    assert len(splits) <= config.max_splits, (
        f"Should generate at most {config.max_splits} splits, got {len(splits)}"
    )


def test_generate_walk_forward_splits_no_splits_error():
    """Test that appropriate error is raised when no splits can be generated."""
    # Config that requires too much data (will fail early with insufficient data)
    with pytest.raises(ValueError, match="Insufficient data"):
        config = WalkForwardConfig(
            start_date=pd.Timestamp("2024-01-01", tz="UTC"),
            end_date=pd.Timestamp("2024-01-15", tz="UTC"),  # Only 15 days
            train_window_days=252,  # Requires 252 days
            test_window_days=63,  # Requires 63 days
            mode="rolling",
            min_train_periods=252,
            min_test_periods=63,
        )
        generate_walk_forward_splits(config.start_date, config.end_date, config)

    # Config that might pass validation but generate no splits due to window constraints
    # This case is harder to trigger because validation catches it early
    # But we can test with a config that will produce no valid splits after filtering
    with pytest.raises(
        ValueError
    ):  # Accept either "Insufficient data" or "No valid splits"
        config = WalkForwardConfig(
            start_date=pd.Timestamp("2024-01-01", tz="UTC"),
            end_date=pd.Timestamp("2024-02-29", tz="UTC"),  # 60 days
            train_window_days=50,
            test_window_days=30,
            mode="rolling",
            step_size_days=30,
            min_train_periods=50,  # Requires 50 days, but with rolling window we need 50+30=80
            min_test_periods=30,
        )
        # This will either fail validation or generate no splits
        generate_walk_forward_splits(config.start_date, config.end_date, config)
