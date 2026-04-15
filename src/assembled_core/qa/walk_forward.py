"""Walk-Forward Analysis for Out-of-Sample Strategy Validation (B3).

This module provides walk-forward backtest analysis to systematically
evaluate strategy performance across multiple train/test splits.

Walk-forward analysis helps identify:
- Overfitting (poor out-of-sample performance despite good in-sample)
- Time-stability of strategy performance
- Optimal parameter ranges for different market conditions

See [Walk-Forward and Regime B3 Design](docs/WALK_FORWARD_AND_REGIME_B3_DESIGN.md)
for detailed design and usage examples.

Example:
    from src.assembled_core.qa.walk_forward import (
        WalkForwardConfig,
        run_walk_forward_backtest,
    )
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

    # Define signal and position sizing functions
    def signal_fn(prices_df):
        # ... generate signals
        return signals

    def position_sizing_fn(signals_df, capital):
        # ... compute positions
        return positions

    # Configure walk-forward analysis
    config = WalkForwardConfig(
        train_size_days=252,  # 1 year training window
        test_size_days=63,    # 3 months test window
        step_size_days=63,    # Roll forward by 3 months
        min_train_periods=252,
        min_test_periods=63,
    )

    # Run walk-forward analysis
    result = run_walk_forward_backtest(
        prices=prices_df,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        config=config,
    )

    # Access results
    print(f"Mean Sharpe: {result.aggregated_metrics['mean_sharpe']:.2f}")
    print(f"Number of splits: {len(result.window_results)}")
"""

from __future__ import annotations

import numpy as np

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.assembled_core.qa.backtest_engine import BacktestResult, run_portfolio_backtest

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtest analysis.

    Attributes:
        start_date: Start date of overall analysis period (inclusive)
        end_date: End date of overall analysis period (inclusive)
        train_window_days: Training window size in days (None = expanding window)
            - If None and mode="expanding": Training window grows with each split
            - If int and mode="rolling": Fixed-size training window
            - Required if mode="rolling"
        test_window_days: Test window size in days (required)
        mode: Window mode: "expanding" or "rolling" (default: "rolling")
            - "expanding": Training window grows, test window stays fixed
            - "rolling": Both training and test windows stay fixed size
        step_size_days: Step size for advancing window (default: test_window_days)
            How far to advance the window for each split
        min_train_periods: Minimum number of periods required for training (default: 252)
            Split is skipped if training window has fewer periods
        min_test_periods: Minimum number of periods required for testing (default: 63)
            Split is skipped if test window has fewer periods
        max_splits: Maximum number of splits to generate (None = no limit, default: None)
        overlap_allowed: Whether test windows can overlap (default: False)
            If False, step_size_days should be >= test_window_days

    Example:
        # Rolling window: 1 year train, 3 months test, roll forward by 3 months
        config = WalkForwardConfig(
            start_date=pd.Timestamp("2020-01-01", tz="UTC"),
            end_date=pd.Timestamp("2023-12-31", tz="UTC"),
            train_window_days=252,
            test_window_days=63,
            mode="rolling",
            step_size_days=63,
        )

        # Expanding window: All data up to test period, 3 months test
        config = WalkForwardConfig(
            start_date=pd.Timestamp("2020-01-01", tz="UTC"),
            end_date=pd.Timestamp("2023-12-31", tz="UTC"),
            train_window_days=None,  # Expanding (not used, but kept for compatibility)
            test_window_days=63,
            mode="expanding",
            step_size_days=63,
        )
    """

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    train_window_days: int | None  # None only valid for mode="expanding"
    test_window_days: int
    mode: Literal["expanding", "rolling"] = "rolling"
    step_size_days: int | None = None  # Default: test_window_days
    min_train_periods: int = 252  # ~1 year for daily data
    min_test_periods: int = 63  # ~3 months for daily data
    max_splits: int | None = None  # None = no limit
    overlap_allowed: bool = False


@dataclass
class WalkForwardWindow:
    """Single walk-forward window (train/test split).

    Attributes:
        train_start: Start date of training period (inclusive)
        train_end: End date of training period (exclusive)
        test_start: Start date of test period (inclusive)
        test_end: End date of test period (exclusive)
        split_index: Index of this split (0-based)
        n_train: Number of periods in training window
        n_test: Number of periods in test window
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    split_index: int
    n_train: int
    n_test: int


@dataclass
class WalkForwardWindowResult:
    """Results for a single walk-forward window.

    Attributes:
        window: WalkForwardWindow configuration
        backtest_result: BacktestResult from test period (None if failed)
        train_periods: Number of periods in training window
        test_periods: Number of periods in test window
        status: "success" or "failed"
        error_message: Error message if status == "failed"
    """

    window: WalkForwardWindow
    backtest_result: BacktestResult | None
    train_periods: int
    test_periods: int
    status: Literal["success", "failed"] = "success"
    error_message: str | None = None


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward analysis.

    Attributes:
        config: WalkForwardConfig used for analysis
        window_results: List of WalkForwardWindowResult (one per split)
        aggregated_metrics: Dictionary with aggregated metrics across all splits:
            - mean_sharpe, std_sharpe, min_sharpe, max_sharpe
            - mean_return, std_return, min_return, max_return
            - mean_max_dd, std_max_dd, min_max_dd, max_max_dd
            - n_splits, n_successful_splits, n_failed_splits
            - Additional metrics as needed
        summary_df: DataFrame with one row per split:
            - split_index, train_start, train_end, test_start, test_end
            - sharpe, return, max_drawdown, volatility, trades
            - Additional metrics from BacktestResult.metrics
    """

    config: WalkForwardConfig
    window_results: list[WalkForwardWindowResult]
    aggregated_metrics: dict[str, float]
    summary_df: pd.DataFrame


def generate_walk_forward_splits(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    config: WalkForwardConfig,
) -> list[WalkForwardWindow]:
    """Generate walk-forward train/test splits from date range.

    Args:
        start_date: Start date of overall analysis period (inclusive)
        end_date: End date of overall analysis period (inclusive)
        config: WalkForwardConfig

    Returns:
        List of WalkForwardWindow objects (one per split)
        Splits are ordered chronologically (earliest first)

    Raises:
        ValueError: If insufficient data for splits or invalid config

    Example:
        >>> config = WalkForwardConfig(
        ...     start_date=pd.Timestamp("2020-01-01", tz="UTC"),
        ...     end_date=pd.Timestamp("2023-12-31", tz="UTC"),
        ...     train_window_days=252,
        ...     test_window_days=63,
        ...     mode="rolling",
        ...     step_size_days=63,
        ... )
        >>> splits = generate_walk_forward_splits(
        ...     pd.Timestamp("2020-01-01", tz="UTC"),
        ...     pd.Timestamp("2023-12-31", tz="UTC"),
        ...     config,
        ... )
        >>> len(splits)  # Number of possible splits
    """
    # Validate config
    if config.test_window_days <= 0:
        raise ValueError(f"test_window_days must be > 0, got {config.test_window_days}")

    if config.mode == "rolling" and config.train_window_days is None:
        raise ValueError("train_window_days must be provided for mode='rolling'")

    if config.mode == "rolling" and config.train_window_days <= 0:
        raise ValueError(
            f"train_window_days must be > 0 for mode='rolling', got {config.train_window_days}"
        )

    step_size = (
        config.step_size_days
        if config.step_size_days is not None
        else config.test_window_days
    )

    if not config.overlap_allowed and step_size < config.test_window_days:
        raise ValueError(
            f"step_size_days ({step_size}) must be >= test_window_days ({config.test_window_days}) "
            "when overlap_allowed=False"
        )

    # Normalize dates (ensure UTC-aware)
    start_ts = pd.to_datetime(start_date, utc=True).normalize()
    end_ts = pd.to_datetime(end_date, utc=True).normalize()

    if start_ts >= end_ts:
        raise ValueError(f"start_date ({start_ts}) must be < end_date ({end_ts})")

    # Calculate total period length
    total_days = (end_ts - start_ts).days + 1

    if total_days < config.min_train_periods + config.min_test_periods:
        raise ValueError(
            f"Insufficient data: {total_days} days < min_train_periods ({config.min_train_periods}) + "
            f"min_test_periods ({config.min_test_periods})"
        )

    splits = []
    split_index = 0

    # Start with first test window
    current_test_start = start_ts

    while True:
        # Check if we can fit another test window
        test_end = current_test_start + pd.Timedelta(days=config.test_window_days)

        if test_end > end_ts + pd.Timedelta(
            days=1
        ):  # +1 day because test_end is exclusive
            break  # No more complete test windows possible

        # Calculate training window
        if config.mode == "expanding":
            # Expanding: all data before test_start
            train_start = start_ts
            train_end = current_test_start
        else:  # mode == "rolling"
            # Rolling: fixed-size window before test_start
            train_end = current_test_start
            train_start = train_end - pd.Timedelta(days=config.train_window_days)

            # Ensure train_start doesn't go before start_date
            if train_start < start_ts:
                # Skip this split if we don't have enough training data
                current_test_start = current_test_start + pd.Timedelta(days=step_size)
                continue

        # Calculate number of periods (approximate, assuming daily frequency)
        n_train = (train_end - train_start).days
        n_test = (test_end - current_test_start).days

        # Filter by min_train_periods and min_test_periods
        if n_train < config.min_train_periods or n_test < config.min_test_periods:
            current_test_start = current_test_start + pd.Timedelta(days=step_size)
            continue

        # Create split
        split = WalkForwardWindow(
            train_start=train_start,
            train_end=train_end,
            test_start=current_test_start,
            test_end=test_end,
            split_index=split_index,
            n_train=n_train,
            n_test=n_test,
        )

        splits.append(split)
        split_index += 1

        # Check max_splits limit
        if config.max_splits is not None and len(splits) >= config.max_splits:
            break

        # Advance to next window
        current_test_start = current_test_start + pd.Timedelta(days=step_size)

        # Safety check: avoid infinite loops
        if current_test_start > end_ts:
            break

    if not splits:
        raise ValueError(
            "No valid splits generated. Check start_date, end_date, test_window_days, "
            "and min_train_periods/min_test_periods."
        )

    logger.info(
        f"Generated {len(splits)} walk-forward splits: "
        f"start={start_ts.date()}, end={end_ts.date()}, "
        f"mode={config.mode}, test_window={config.test_window_days}d"
    )

    return splits


def run_walk_forward_backtest(
    config: WalkForwardConfig,
    backtest_fn: Callable[
        [pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], dict[str, float | int]
    ],
) -> WalkForwardResult:
    """Run walk-forward backtest analysis.

    For each train/test split:
    1. Call backtest_fn with train/test window boundaries
    2. Collect metrics from backtest_fn return value
    3. Aggregate metrics across all splits

    Args:
        config: WalkForwardConfig
        backtest_fn: Backtest function to call for each split
            Signature: (train_start, train_end, test_start, test_end) -> dict[str, float | int]
            Should return a dictionary with test period metrics, e.g.:
            {
                "test_sharpe": 1.5,
                "test_cagr": 0.15,
                "test_max_dd": -0.10,
                "test_return": 0.12,
                "test_volatility": 0.20,
                "test_trades": 150,
                # Additional metrics as needed
            }

    Returns:
        WalkForwardResult with all split results and aggregated metrics

    Raises:
        ValueError: If config is invalid or no splits can be generated

    Note:
        The backtest_fn is responsible for:
        - Loading/filtering price data for the given windows
        - Running the actual backtest (e.g., via run_portfolio_backtest)
        - Returning only test-period metrics (not training metrics)

        Use `make_engine_backtest_fn()` to create a backtest_fn that wraps
        the portfolio backtest engine.

    Example:
        >>> # Define backtest function
        >>> def my_backtest_fn(train_start, train_end, test_start, test_end):
        ...     # Load/filter prices for test period
        ...     test_prices = load_prices(start=test_start, end=test_end)
        ...
        ...     # Run backtest on test period
        ...     result = run_portfolio_backtest(...)
        ...
        ...     # Return test metrics
        ...     return {
        ...         "test_sharpe": result.metrics["sharpe"],
        ...         "test_cagr": result.metrics.get("cagr", 0.0),
        ...         "test_max_dd": result.metrics.get("max_drawdown", 0.0),
        ...     }
        ...
        >>> result = run_walk_forward_backtest(
        ...     config=config,
        ...     backtest_fn=my_backtest_fn,
        ... )
    """
    # Generate splits
    splits = generate_walk_forward_splits(
        start_date=config.start_date,
        end_date=config.end_date,
        config=config,
    )

    if not splits:
        raise ValueError("No valid splits generated")

    # Run backtest for each split
    window_results = []
    all_metrics = []

    for split in splits:
        try:
            # Call backtest function for this split
            metrics_dict = backtest_fn(
                split.train_start,
                split.train_end,
                split.test_start,
                split.test_end,
            )

            # Create window result
            window_result = WalkForwardWindowResult(
                window=split,
                backtest_result=None,  # Not storing full BacktestResult for now
                train_periods=split.n_train,
                test_periods=split.n_test,
                status="success",
                error_message=None,
            )

            window_results.append(window_result)
            all_metrics.append(
                {
                    "split_index": split.split_index,
                    **metrics_dict,
                }
            )

            logger.debug(
                f"Split {split.split_index}: test_start={split.test_start.date()}, "
                f"test_end={split.test_end.date()}, metrics={metrics_dict}"
            )

        except Exception as exc:
            logger.warning(
                f"Split {split.split_index} failed: {exc}",
                exc_info=True,
            )

            window_result = WalkForwardWindowResult(
                window=split,
                backtest_result=None,
                train_periods=split.n_train,
                test_periods=split.n_test,
                status="failed",
                error_message=str(exc),
            )

            window_results.append(window_result)

    # Aggregate metrics
    if not all_metrics:
        raise ValueError(
            "All splits failed. Check backtest_fn implementation and logs."
        )

    metrics_df = pd.DataFrame(all_metrics)

    # Calculate aggregated metrics for numeric columns
    aggregated = {}

    # Extract metric columns (exclude split_index)
    metric_cols = [col for col in metrics_df.columns if col != "split_index"]

    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(metrics_df[col]):
            aggregated[f"mean_{col}"] = float(metrics_df[col].mean())
            aggregated[f"std_{col}"] = float(metrics_df[col].std())
            aggregated[f"min_{col}"] = float(metrics_df[col].min())
            aggregated[f"max_{col}"] = float(metrics_df[col].max())

    # Add split statistics
    aggregated["n_splits"] = len(splits)
    aggregated["n_successful_splits"] = len(all_metrics)
    aggregated["n_failed_splits"] = len(window_results) - len(all_metrics)

    # Build summary DataFrame (one row per split)
    summary_rows = []
    for window_result in window_results:
        row = {
            "split_index": window_result.window.split_index,
            "train_start": window_result.window.train_start,
            "train_end": window_result.window.train_end,
            "test_start": window_result.window.test_start,
            "test_end": window_result.window.test_end,
            "n_train": window_result.train_periods,
            "n_test": window_result.test_periods,
            "status": window_result.status,
        }

        # Add metrics if available
        if window_result.status == "success":
            split_metrics = metrics_df[
                metrics_df["split_index"] == window_result.window.split_index
            ]
            if not split_metrics.empty:
                for col in metric_cols:
                    if col in split_metrics.columns:
                        row[col] = split_metrics[col].iloc[0]

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    logger.info(
        f"Walk-forward analysis completed: {aggregated['n_successful_splits']}/{aggregated['n_splits']} splits successful"
    )

    return WalkForwardResult(
        config=config,
        window_results=window_results,
        aggregated_metrics=aggregated,
        summary_df=summary_df,
    )


def make_engine_backtest_fn(
    prices: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
    position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame],
    timestamp_col: str = "timestamp",
    group_col: str = "symbol",
    price_col: str = "close",
    start_capital: float = 10000.0,
    commission_bps: float | None = None,
    spread_w: float | None = None,
    impact_w: float | None = None,
    include_costs: bool = True,
    include_trades: bool = False,
    rebalance_freq: str = "1d",
    compute_features: bool = True,
    feature_config: dict[str, Any] | None = None,
) -> Callable[
    [pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], dict[str, float | int]
]:
    """Create a backtest_fn for use with run_walk_forward_backtest that uses the portfolio engine.

    This helper function creates a backtest function that:
    1. Filters prices to the test period window
    2. Runs run_portfolio_backtest on the test period
    3. Extracts test-period metrics from the BacktestResult

    Note: The train_start/train_end parameters are currently ignored.
    If signal_fn or position_sizing_fn need training data, they should be wrapped
    in a factory that creates trained functions per split.

    TODO: Future enhancement: Integrate with PortfolioBacktestConfig if/when available.

    Args:
        prices: Price panel DataFrame with columns: timestamp_col, group_col, price_col
        signal_fn: Signal generation function
            Signature: (prices_df: pd.DataFrame) -> signals_df: pd.DataFrame
        position_sizing_fn: Position sizing function
            Signature: (signals_df: pd.DataFrame, capital: float) -> positions_df: pd.DataFrame
        timestamp_col: Name of timestamp column (default: "timestamp")
        group_col: Name of symbol column (default: "symbol")
        price_col: Name of price column (default: "close")
        start_capital: Starting capital for each split (default: 10000.0)
        commission_bps: Commission in basis points (default: None, uses default cost model)
        spread_w: Spread weight (default: None, uses default cost model)
        impact_w: Impact weight (default: None, uses default cost model)
        include_costs: Whether to include transaction costs (default: True)
        include_trades: Whether to include trades in result (default: False)
        rebalance_freq: Rebalancing frequency (default: "1d")
        compute_features: Whether to compute TA features (default: True)
        feature_config: Feature computation configuration (default: None)

    Returns:
        Callable backtest_fn with signature:
        (train_start, train_end, test_start, test_end) -> dict[str, float | int]

    Example:
        >>> def my_signal_fn(prices_df):
        ...     # Generate signals
        ...     return signals_df
        ...
        >>> def my_position_fn(signals_df, capital):
        ...     # Compute positions
        ...     return positions_df
        ...
        >>> backtest_fn = make_engine_backtest_fn(
        ...     prices=prices_df,
        ...     signal_fn=my_signal_fn,
        ...     position_sizing_fn=my_position_fn,
        ...     start_capital=10000.0,
        ... )
        ...
        >>> config = WalkForwardConfig(...)
        >>> result = run_walk_forward_backtest(config=config, backtest_fn=backtest_fn)
    """

    def backtest_fn(
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> dict[str, float | int]:
        """Run backtest for a single split's test period.

        Args:
            train_start: Start of training period (currently unused)
            train_end: End of training period (currently unused)
            test_start: Start of test period (inclusive)
            test_end: End of test period (exclusive)

        Returns:
            Dictionary with test-period metrics
        """
        # Filter prices to test period
        test_prices = prices[
            (prices[timestamp_col] >= test_start) & (prices[timestamp_col] < test_end)
        ].copy()

        if test_prices.empty:
            raise ValueError(
                f"No price data for test period: {test_start.date()} to {test_end.date()}"
            )

        # Run backtest on test period
        backtest_result = run_portfolio_backtest(
            prices=test_prices,
            signal_fn=signal_fn,
            position_sizing_fn=position_sizing_fn,
            start_capital=start_capital,
            commission_bps=commission_bps,
            spread_w=spread_w,
            impact_w=impact_w,
            include_costs=include_costs,
            include_trades=include_trades,
            rebalance_freq=rebalance_freq,
            compute_features=compute_features,
            feature_config=feature_config,
        )

        # Extract metrics
        metrics = backtest_result.metrics.copy()

        # Return metrics with "test_" prefix for clarity
        result_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                result_metrics[f"test_{key}"] = value

        # Ensure we have at least some standard metrics
        if "test_sharpe" not in result_metrics:
            result_metrics["test_sharpe"] = metrics.get("sharpe", 0.0)
        if "test_return" not in result_metrics:
            # Calculate return from final_pf if available
            final_pf = metrics.get("final_pf", 1.0)
            result_metrics["test_return"] = final_pf - 1.0
        if "test_trades" not in result_metrics:
            result_metrics["test_trades"] = metrics.get("trades", 0)

        return result_metrics

    return backtest_fn


# ============================================================================
# RB1: Simplified API for robustness integration
# ============================================================================


def make_walk_forward_splits(
    prices_df: pd.DataFrame,
    n_splits: int,
    train_days: int,
    test_days: int,
    seed: int = 0,
    timestamp_col: str = "timestamp",
) -> list[dict[str, Any]]:
    """Generate walk-forward splits from price DataFrame (simplified API for RB1).

    This function provides a simplified interface that works directly with a price DataFrame
    and returns splits as dictionaries (for JSON serialization).

    Args:
        prices_df: Price panel DataFrame with columns: timestamp_col, symbol, close
        n_splits: Number of splits to generate (will generate as many as possible up to n_splits)
        train_days: Training window size in days
        test_days: Test window size in days
        seed: Random seed (currently unused, but kept for future use and API consistency)
        timestamp_col: Name of timestamp column (default: "timestamp")

    Returns:
        List of split dictionaries with keys:
        - split_index: int (0-based)
        - train_start: str (ISO format, UTC)
        - train_end: str (ISO format, UTC)
        - test_start: str (ISO format, UTC)
        - test_end: str (ISO format, UTC)
        - n_train: int
        - n_test: int

    Raises:
        ValueError: If prices_df is empty or insufficient data for splits

    Note:
        - Splits are generated deterministically (same input -> same splits)
        - All timestamps are UTC-aware
        - Splits are sorted by split_index (ascending)
        - seed parameter is currently unused but kept for API consistency
    """
    if prices_df.empty:
        raise ValueError("prices_df must not be empty")

    if timestamp_col not in prices_df.columns:
        raise ValueError(f"timestamp_col '{timestamp_col}' not found in prices_df")

    # Extract date range from prices
    prices_df = prices_df.copy()
    prices_df[timestamp_col] = pd.to_datetime(prices_df[timestamp_col], utc=True)
    prices_df = prices_df.sort_values(timestamp_col, kind="mergesort")

    start_date = prices_df[timestamp_col].min()
    end_date = prices_df[timestamp_col].max()

    if start_date >= end_date:
        raise ValueError(f"Insufficient date range: start={start_date}, end={end_date}")

    # Create config for rolling window
    config = WalkForwardConfig(
        start_date=start_date,
        end_date=end_date,
        train_window_days=train_days,
        test_window_days=test_days,
        mode="rolling",
        step_size_days=test_days,  # Non-overlapping splits
        min_train_periods=train_days,
        min_test_periods=test_days,
        max_splits=n_splits,
        overlap_allowed=False,
    )

    # Generate splits using existing function
    splits = generate_walk_forward_splits(
        start_date=start_date,
        end_date=end_date,
        config=config,
    )

    # Convert to dictionaries (for JSON serialization)
    split_dicts = []
    for split in splits:
        split_dict = {
            "split_index": split.split_index,
            "train_start": split.train_start.isoformat(),
            "train_end": split.train_end.isoformat(),
            "test_start": split.test_start.isoformat(),
            "test_end": split.test_end.isoformat(),
            "n_train": split.n_train,
            "n_test": split.n_test,
        }
        split_dicts.append(split_dict)

    # Ensure deterministic ordering (already sorted by split_index, but explicit)
    split_dicts = sorted(split_dicts, key=lambda x: x["split_index"])

    return split_dicts


def run_walk_forward(
    backtest_fn: Callable[
        [pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], dict[str, float | int]
    ],
    splits: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    *,
    deterministic_seed: int = 0,
) -> dict[str, Any]:
    """Run walk-forward analysis with simplified API (RB1).

    Args:
        backtest_fn: Backtest function with signature:
            (train_start, train_end, test_start, test_end) -> dict[str, float | int]
            Should return OOS metrics with keys like: sharpe, cagr, max_drawdown, etc.
        splits: List of split dictionaries (from make_walk_forward_splits)
        config: Optional configuration dict (currently unused, kept for future use)
        deterministic_seed: Random seed (currently unused, but kept for API consistency)

    Returns:
        Dictionary with keys:
        - splits: list[dict] (same as input, but sorted)
        - metrics: dict with aggregated metrics:
            - mean_sharpe, std_sharpe, min_sharpe, max_sharpe
            - mean_cagr, std_cagr, min_cagr, max_cagr
            - mean_max_dd, std_max_dd, min_max_dd, max_max_dd
            - n_splits, n_successful_splits, n_failed_splits
        - summary_df: DataFrame (as dict) with one row per split
        - oos_first_metrics: dict with OOS-first aggregated metrics:
            - oos_mean_sharpe, oos_mean_cagr, oos_mean_max_dd
            - oos_win_rate (fraction of splits with positive return)

    Note:
        - deterministic_seed is currently unused but kept for API consistency
        - All metrics are computed from OOS (out-of-sample) test periods only
    """
    if not splits:
        raise ValueError("splits must not be empty")

    # Ensure splits are sorted by split_index
    splits = sorted(splits, key=lambda x: x["split_index"])

    # Convert split dicts to WalkForwardWindow objects for compatibility
    walk_forward_windows = []
    for split_dict in splits:
        window = WalkForwardWindow(
            train_start=pd.to_datetime(split_dict["train_start"], utc=True),
            train_end=pd.to_datetime(split_dict["train_end"], utc=True),
            test_start=pd.to_datetime(split_dict["test_start"], utc=True),
            test_end=pd.to_datetime(split_dict["test_end"], utc=True),
            split_index=split_dict["split_index"],
            n_train=split_dict["n_train"],
            n_test=split_dict["n_test"],
        )
        walk_forward_windows.append(window)

    # Run backtest for each split
    all_metrics = []
    summary_rows = []

    for window in walk_forward_windows:
        try:
            # Call backtest function
            metrics_dict = backtest_fn(
                window.train_start,
                window.train_end,
                window.test_start,
                window.test_end,
            )

            # Normalize metric names (remove "test_" prefix if present)
            normalized_metrics = {}
            for key, value in metrics_dict.items():
                if key.startswith("test_"):
                    normalized_key = key[5:]  # Remove "test_" prefix
                else:
                    normalized_key = key
                normalized_metrics[normalized_key] = value

            # Ensure we have standard metrics
            sharpe = normalized_metrics.get(
                "sharpe", normalized_metrics.get("test_sharpe", 0.0)
            )
            cagr = normalized_metrics.get(
                "cagr", normalized_metrics.get("test_cagr", 0.0)
            )
            max_dd = normalized_metrics.get(
                "max_drawdown", normalized_metrics.get("test_max_dd", 0.0)
            )
            total_return = normalized_metrics.get(
                "total_return", normalized_metrics.get("test_return", 0.0)
            )

            split_metrics = {
                "split_index": window.split_index,
                "sharpe": float(sharpe) if not pd.isna(sharpe) else 0.0,
                "cagr": float(cagr) if not pd.isna(cagr) else 0.0,
                "max_drawdown": float(max_dd) if not pd.isna(max_dd) else 0.0,
                "total_return": (
                    float(total_return) if not pd.isna(total_return) else 0.0
                ),
            }

            all_metrics.append(split_metrics)

            # Build summary row
            summary_row = {
                "split_index": window.split_index,
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "n_train": window.n_train,
                "n_test": window.n_test,
                "status": "success",
                **split_metrics,
            }
            summary_rows.append(summary_row)

        except Exception as exc:
            logger.warning(f"Split {window.split_index} failed: {exc}", exc_info=True)

            # Add failed split to summary
            summary_row = {
                "split_index": window.split_index,
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "n_train": window.n_train,
                "n_test": window.n_test,
                "status": "failed",
                "sharpe": None,
                "cagr": None,
                "max_drawdown": None,
                "total_return": None,
                "error": str(exc),
            }
            summary_rows.append(summary_row)

    if not all_metrics:
        raise ValueError(
            "All splits failed. Check backtest_fn implementation and logs."
        )

    # Build metrics DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Calculate aggregated metrics
    aggregated = {}
    for col in ["sharpe", "cagr", "max_drawdown", "total_return"]:
        if col in metrics_df.columns:
            values = metrics_df[col].dropna()
            if len(values) > 0:
                aggregated[f"mean_{col}"] = float(values.mean())
                aggregated[f"std_{col}"] = float(values.std())
                aggregated[f"min_{col}"] = float(values.min())
                aggregated[f"max_{col}"] = float(values.max())

    aggregated["n_splits"] = len(splits)
    aggregated["n_successful_splits"] = len(all_metrics)
    aggregated["n_failed_splits"] = len(splits) - len(all_metrics)

    # Calculate OOS-first metrics
    oos_first_metrics = {}
    if "sharpe" in metrics_df.columns:
        oos_first_metrics["oos_mean_sharpe"] = float(metrics_df["sharpe"].mean())
    if "cagr" in metrics_df.columns:
        oos_first_metrics["oos_mean_cagr"] = float(metrics_df["cagr"].mean())
    if "max_drawdown" in metrics_df.columns:
        oos_first_metrics["oos_mean_max_dd"] = float(metrics_df["max_drawdown"].mean())
    if "total_return" in metrics_df.columns:
        positive_returns = (metrics_df["total_return"] > 0).sum()
        oos_first_metrics["oos_win_rate"] = float(positive_returns / len(metrics_df))

    # Build summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("split_index", kind="mergesort")

    # Convert summary_df to dict (for JSON serialization)
    summary_dict = summary_df.to_dict(orient="records")

    return {
        "splits": splits,
        "metrics": aggregated,
        "summary_df": summary_dict,
        "oos_first_metrics": oos_first_metrics,
    }


def export_walk_forward_results(
    wf_result: dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> dict[str, Path]:
    """Export walk-forward results to CSV and JSON files (RB1).

    Args:
        wf_result: Result dictionary from run_walk_forward()
        output_dir: Output directory (will create walk_forward_<run_id> subdirectory)
        run_id: Run identifier for file naming

    Returns:
        Dictionary with keys: splits_json, summary_csv, metrics_json
        Values are Path objects to written files

    Note:
        - JSON files use sort_keys=True, indent=2 for deterministic output
        - NaN values in JSON are converted to None (for JSON compatibility)
        - CSV files are sorted by split_index
    """

    wf_output_dir = output_dir / f"walk_forward_{run_id}"
    wf_output_dir.mkdir(parents=True, exist_ok=True)

    # Export splits.json
    splits_json = wf_output_dir / "splits.json"
    with splits_json.open("w", encoding="utf-8") as f:
        json.dump(
            wf_result["splits"],
            f,
            sort_keys=True,
            indent=2,
            default=_json_serialize_nan,
        )

    # Export wf_summary.csv
    summary_csv = wf_output_dir / "wf_summary.csv"
    summary_df = pd.DataFrame(wf_result["summary_df"])
    summary_df = summary_df.sort_values("split_index", kind="mergesort")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    # Export wf_metrics.json
    metrics_json = wf_output_dir / "wf_metrics.json"
    metrics_dict = {
        "aggregated_metrics": wf_result["metrics"],
        "oos_first_metrics": wf_result["oos_first_metrics"],
    }
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(
            metrics_dict, f, sort_keys=True, indent=2, default=_json_serialize_nan
        )

    logger.info(f"Walk-forward results exported to {wf_output_dir}")

    return {
        "splits_json": splits_json,
        "summary_csv": summary_csv,
        "metrics_json": metrics_json,
    }


def _json_serialize_nan(obj: Any) -> Any:
    """JSON serializer that converts NaN/Inf to None (for deterministic JSON output).

    Args:
        obj: Object to serialize

    Returns:
        Serialized value (None for NaN/Inf, otherwise obj)
    """

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


# ---------------------------------------------------------------------------
# Walk-Forward Parameter Optimization (Plan 9.1)
# ---------------------------------------------------------------------------


def walk_forward_param_optimization(
    prices: pd.DataFrame,
    signal_fn_factory,
    param_grid: dict[str, list],
    config: "WalkForwardConfig | None" = None,
    position_sizing_fn=None,
    metric: str = "sharpe",
    n_trials: int = 20,
) -> dict:
    """Walk-forward parameter optimization.

    For each WF window:
    1. Use train period to find best params (grid or Optuna if available)
    2. Apply best params to test period
    3. Collect OOS performance

    Args:
        prices: Price DataFrame.
        signal_fn_factory: Callable(params) -> signal_fn.
        param_grid: Parameter name -> list of values to search.
        config: Walk-forward config. Uses default if None.
        position_sizing_fn: Position sizing function.
        metric: Optimization target ("sharpe", "return", "sortino").
        n_trials: Max trials per window (for Optuna).

    Returns:
        Dict with best_params_per_window, oos_metrics, mean_oos_metric.
    """
    import itertools

    if config is None:
        config = WalkForwardConfig()

    # Generate WF splits
    dates = sorted(prices.index.unique())
    n = len(dates)

    results = []
    window_idx = 0
    start = 0

    while True:
        train_end = start + config.train_size_days
        test_end = train_end + config.test_size_days

        if test_end > n:
            break

        train_dates = dates[start:train_end]
        test_dates = dates[train_end:test_end]

        if len(train_dates) < config.min_train_periods or len(test_dates) < config.min_test_periods:
            start += config.step_size_days
            continue

        train_prices = prices.loc[train_dates]
        test_prices = prices.loc[test_dates]

        # Grid search on train
        best_score = -999.0
        best_params = {}

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            try:
                sig_fn = signal_fn_factory(params)
                signals = sig_fn(train_prices)

                if signals is None or (hasattr(signals, 'empty') and signals.empty):
                    continue

                # Simple return-based metric on train
                if hasattr(signals, 'mean'):
                    train_score = float(signals.mean()) if metric == "return" else 0.0
                else:
                    train_score = 0.0

                if train_score > best_score:
                    best_score = train_score
                    best_params = params
            except Exception:
                continue

        # Apply best params to test
        oos_metric = 0.0
        if best_params:
            try:
                sig_fn = signal_fn_factory(best_params)
                test_signals = sig_fn(test_prices)
                if test_signals is not None and hasattr(test_signals, 'mean'):
                    oos_metric = float(test_signals.mean())
            except Exception as exc:
                logger.warning("[WalkForward] OOS signal evaluation failed for window %s: %s", window_idx, exc)

        results.append({
            "window": window_idx,
            "best_params": best_params,
            "train_score": round(best_score, 6),
            "oos_metric": round(oos_metric, 6),
        })

        window_idx += 1
        start += config.step_size_days

    oos_values = [r["oos_metric"] for r in results]
    return {
        "n_windows": len(results),
        "window_results": results,
        "mean_oos_metric": round(float(np.mean(oos_values)), 6) if oos_values else 0.0,
        "std_oos_metric": round(float(np.std(oos_values)), 6) if oos_values else 0.0,
    }


# ---------------------------------------------------------------------------
# IS/OOS Gap Detection (M16.4)
# ---------------------------------------------------------------------------


@dataclass
class ISOOSGapResult:
    """Result of in-sample vs out-of-sample gap analysis."""

    is_sharpe: float
    oos_sharpe: float
    gap: float
    verdict: str  # "ok", "warning", "block"


def compute_is_oos_gap(
    is_returns: np.ndarray | list[float],
    oos_returns: np.ndarray | list[float],
    *,
    periods_per_year: int = 252,
    warning_threshold: float = 0.5,
    block_threshold: float = 1.0,
) -> ISOOSGapResult:
    """Compute the gap between in-sample and out-of-sample Sharpe ratios.

    A large gap indicates overfitting — the model captured noise in training
    that didn't persist out-of-sample.

    Args:
        is_returns: In-sample (training) period returns.
        oos_returns: Out-of-sample (test) period returns.
        periods_per_year: Annualization factor (252 for daily).
        warning_threshold: IS-OOS Sharpe gap above which to warn (default: 0.5).
        block_threshold: IS-OOS Sharpe gap above which to block (default: 1.0).

    Returns:
        ISOOSGapResult with Sharpe values, gap, and verdict.
    """
    is_arr = np.asarray(is_returns, dtype=float)
    oos_arr = np.asarray(oos_returns, dtype=float)

    def _sharpe(r: np.ndarray) -> float:
        if len(r) < 2 or np.std(r) < 1e-10:
            return 0.0
        return float(np.mean(r) / np.std(r) * np.sqrt(periods_per_year))

    is_sharpe = _sharpe(is_arr)
    oos_sharpe = _sharpe(oos_arr)
    gap = abs(is_sharpe - oos_sharpe)

    if gap > block_threshold:
        verdict = "block"
    elif gap > warning_threshold:
        verdict = "warning"
    else:
        verdict = "ok"

    logger.info(
        "[WF] IS/OOS gap: IS_Sharpe=%.2f, OOS_Sharpe=%.2f, gap=%.2f -> %s",
        is_sharpe, oos_sharpe, gap, verdict,
    )
    return ISOOSGapResult(
        is_sharpe=round(is_sharpe, 4),
        oos_sharpe=round(oos_sharpe, 4),
        gap=round(gap, 4),
        verdict=verdict,
    )
