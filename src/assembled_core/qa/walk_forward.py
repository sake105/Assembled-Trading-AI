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

import logging
from collections.abc import Callable
from dataclasses import dataclass
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
