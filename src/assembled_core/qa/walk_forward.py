"""Walk-Forward Analysis for backtesting strategies.

This module provides functions to split time series into train/validation/test windows
and execute repeated backtests over these windows to evaluate strategy performance
in a time-series cross-validation manner.

Key concepts:
- Train window: Data used to train/optimize the strategy
- Test window: Data used to evaluate the strategy (out-of-sample)
- Rolling window: Fixed-size train window that moves forward
- Expanding window: Train window grows over time, test window stays fixed
- Embargo: Time period after test window that cannot be used for training (prevents data leakage)
- Purging: Removal of overlap between train and test windows (prevents data leakage)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.assembled_core.costs import CostModel, get_default_cost_model
from src.assembled_core.qa.backtest_engine import BacktestResult, run_portfolio_backtest
from src.assembled_core.qa.metrics import PerformanceMetrics, compute_all_metrics


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis.
    
    Attributes:
        train_size: Size of training window (number of periods or timedelta string, e.g., "252D" for 1 year)
        test_size: Size of test window (number of periods or timedelta string, e.g., "63D" for 1 quarter)
        step_size: Step size for moving windows (number of periods or timedelta string, e.g., "21D" for 1 month)
        window_type: "rolling" (fixed train size) or "expanding" (train grows over time)
        embargo_periods: Number of periods to embargo after test window (default: 0)
        purge_periods: Number of periods to purge between train and test (default: 0)
        min_train_periods: Minimum number of periods required for training (default: 1)
        min_test_periods: Minimum number of periods required for testing (default: 1)
        start_date: Optional start date for analysis (filters data before this date)
        end_date: Optional end date for analysis (filters data after this date)
    """
    train_size: int | str  # e.g., 252 (days) or "252D" (timedelta)
    test_size: int | str   # e.g., 63 (days) or "63D" (timedelta)
    step_size: int | str   # e.g., 21 (days) or "21D" (timedelta)
    window_type: str = "rolling"  # "rolling" or "expanding"
    embargo_periods: int = 0  # Periods to embargo after test window
    purge_periods: int = 0    # Periods to purge between train and test
    min_train_periods: int = 1
    min_test_periods: int = 1
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None


@dataclass
class WalkForwardWindow:
    """Represents a single train/test window in walk-forward analysis.
    
    Attributes:
        train_start: Start timestamp of training window
        train_end: End timestamp of training window
        test_start: Start timestamp of test window
        test_end: End timestamp of test window
        train_data: Training data DataFrame
        test_data: Test data DataFrame
        window_index: Index of this window (0-based)
    """
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    window_index: int


@dataclass
class WalkForwardWindowResult:
    """Result for a single walk-forward window.
    
    Attributes:
        window_index: Index of this window (0-based)
        train_start: Start timestamp of training window
        train_end: End timestamp of training window
        test_start: Start timestamp of test window
        test_end: End timestamp of test window
        is_metrics: In-sample (train) performance metrics
        oos_metrics: Out-of-sample (test) performance metrics
        backtest_result: BacktestResult from test window backtest
    """
    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    is_metrics: PerformanceMetrics | None
    oos_metrics: PerformanceMetrics
    backtest_result: BacktestResult


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis.
    
    Attributes:
        config: WalkForwardConfig used for analysis
        windows: List of WalkForwardWindow objects
        window_results: List of WalkForwardWindowResult objects (one per window)
        summary_metrics: Aggregated metrics across all windows
            - is_mean_sharpe: Mean in-sample Sharpe ratio
            - oos_mean_sharpe: Mean out-of-sample Sharpe ratio
            - is_mean_cagr: Mean in-sample CAGR
            - oos_mean_cagr: Mean out-of-sample CAGR
            - is_mean_final_pf: Mean in-sample final PF
            - oos_mean_final_pf: Mean out-of-sample final PF
            - oos_std_sharpe: Std dev of out-of-sample Sharpe
            - oos_std_cagr: Std dev of out-of-sample CAGR
            - oos_win_rate: Percentage of windows with positive OOS returns
            - oos_mean_max_drawdown_pct: Mean OOS maximum drawdown
            - total_periods: Total number of periods tested
            - total_trades: Total number of trades across all windows
            - num_windows: Number of windows
    """
    config: WalkForwardConfig
    windows: list[WalkForwardWindow]
    window_results: list[WalkForwardWindowResult]
    summary_metrics: dict[str, float | int | None]


def _parse_window_size(size: int | str, freq: str) -> int:
    """Parse window size (int or timedelta string) to number of periods.
    
    Args:
        size: Window size as int (periods) or str (timedelta, e.g., "252D")
        freq: Frequency string ("1d" or "5min")
    
    Returns:
        Number of periods
    """
    if isinstance(size, int):
        return size
    
    # Parse timedelta string (e.g., "252D", "63D", "21D")
    if isinstance(size, str):
        try:
            td = pd.Timedelta(size)
            # Approximate periods based on frequency
            if freq == "1d":
                # Daily: 1 period per day
                return int(td.days)
            elif freq == "5min":
                # 5min: ~78 periods per trading day (6.5 hours * 12 periods/hour)
                days = td.days + td.seconds / 86400
                return int(days * 78)
            else:
                # Default: assume daily
                return int(td.days)
        except Exception:
            raise ValueError(f"Invalid window size format: {size}. Use int or timedelta string (e.g., '252D')")
    
    raise ValueError(f"Window size must be int or str, got {type(size)}")


def _split_time_series(
    data: pd.DataFrame,
    config: WalkForwardConfig,
    freq: str
) -> list[WalkForwardWindow]:
    """Split time series into train/test windows according to walk-forward configuration.
    
    Args:
        data: DataFrame with 'timestamp' column (must be sorted)
        config: WalkForwardConfig
        freq: Frequency string ("1d" or "5min")
    
    Returns:
        List of WalkForwardWindow objects
    """
    if data.empty:
        return []
    
    # Ensure data is sorted by timestamp
    data = data.sort_values("timestamp").reset_index(drop=True)
    
    # Filter by date range if specified
    if config.start_date is not None:
        data = data[data["timestamp"] >= config.start_date]
    if config.end_date is not None:
        data = data[data["timestamp"] <= config.end_date]
    
    if data.empty:
        return []
    
    # Parse window sizes
    train_size = _parse_window_size(config.train_size, freq)
    test_size = _parse_window_size(config.test_size, freq)
    step_size = _parse_window_size(config.step_size, freq)
    
    timestamps = data["timestamp"].unique()
    timestamps = pd.Series(timestamps).sort_values().reset_index(drop=True)
    
    windows = []
    window_idx = 0
    
    # Start from first timestamp
    current_start = timestamps.iloc[0]
    max_timestamp = timestamps.iloc[-1]
    
    while current_start <= max_timestamp:
        # Find train window end
        train_start_idx = timestamps[timestamps >= current_start].index[0]
        train_end_idx = min(train_start_idx + train_size - 1, len(timestamps) - 1)
        train_end = timestamps.iloc[train_end_idx]
        
        # Apply purge: skip purge_periods after train_end
        if config.purge_periods > 0:
            purge_end_idx = min(train_end_idx + config.purge_periods, len(timestamps) - 1)
            test_start_idx = purge_end_idx + 1
        else:
            test_start_idx = train_end_idx + 1
        
        if test_start_idx >= len(timestamps):
            break  # No more test data available
        
        # Find test window
        test_start = timestamps.iloc[test_start_idx]
        test_end_idx = min(test_start_idx + test_size - 1, len(timestamps) - 1)
        test_end = timestamps.iloc[test_end_idx]
        
        # Check minimum periods
        train_periods = len(timestamps[(timestamps >= current_start) & (timestamps <= train_end)])
        test_periods = len(timestamps[(timestamps >= test_start) & (timestamps <= test_end)])
        
        if train_periods < config.min_train_periods or test_periods < config.min_test_periods:
            # Skip this window if it doesn't meet minimum requirements
            current_start = timestamps.iloc[min(train_start_idx + step_size, len(timestamps) - 1)]
            continue
        
        # Extract train and test data
        train_mask = (data["timestamp"] >= current_start) & (data["timestamp"] <= train_end)
        test_mask = (data["timestamp"] >= test_start) & (data["timestamp"] <= test_end)
        
        train_data = data[train_mask].copy()
        test_data = data[test_mask].copy()
        
        if train_data.empty or test_data.empty:
            current_start = timestamps.iloc[min(train_start_idx + step_size, len(timestamps) - 1)]
            continue
        
        # Create window
        window = WalkForwardWindow(
            train_start=current_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_data=train_data,
            test_data=test_data,
            window_index=window_idx
        )
        windows.append(window)
        window_idx += 1
        
        # Move to next window
        if config.window_type == "rolling":
            # Rolling: fixed train size, move forward by step_size
            current_start = timestamps.iloc[min(train_start_idx + step_size, len(timestamps) - 1)]
        else:  # expanding
            # Expanding: train grows, move test window forward by step_size
            # Train start stays at beginning, train end moves forward
            test_start_idx_next = min(test_start_idx + step_size, len(timestamps) - 1)
            if test_start_idx_next >= len(timestamps):
                break
            current_start = timestamps.iloc[0]  # Keep train start at beginning
    
    return windows


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
    position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame],
    config: WalkForwardConfig,
    start_capital: float = 10000.0,
    freq: str = "1d",
    cost_model: CostModel | None = None,
    include_costs: bool = True,
    include_trades: bool = False,
    include_signals: bool = False,
    include_targets: bool = False,
    compute_features: bool = True,
    feature_config: dict[str, Any] | None = None,
    compute_is_metrics: bool = True
) -> WalkForwardResult:
    """Run walk-forward analysis with repeated backtests over train/test windows.
    
    This function:
    1. Splits the price data into train/test windows according to config
    2. For each window:
       a. Optionally runs backtest on train window (IS - In-Sample)
       b. Runs backtest on test window (OOS - Out-of-Sample)
       c. Computes performance metrics for both IS and OOS using qa.metrics
    3. Aggregates results across all windows
    
    Args:
        prices: DataFrame with columns: timestamp, symbol, close (and optionally open, high, low, volume)
        signal_fn: Function that takes prices DataFrame and returns signals DataFrame
            with columns: timestamp, symbol, direction, score
        position_sizing_fn: Function that takes signals DataFrame and capital, returns
            target positions DataFrame with columns: symbol, target_weight, target_qty
        config: WalkForwardConfig with window configuration
        start_capital: Starting capital for each backtest window
        freq: Frequency string ("1d" or "5min")
        cost_model: Optional CostModel for transaction costs (default: from get_default_cost_model)
        include_costs: Whether to include transaction costs in backtest
        include_trades: Whether to include trades in BacktestResult
        include_signals: Whether to include signals in BacktestResult
        include_targets: Whether to include target positions in BacktestResult
        compute_features: Whether to compute TA features before signal generation
        feature_config: Optional configuration for feature computation
        compute_is_metrics: Whether to compute in-sample (train) metrics (default: True)
    
    Returns:
        WalkForwardResult with windows, window results (IS/OOS metrics), and summary metrics
    
    Example:
        ```python
        from src.assembled_core.qa.walk_forward import WalkForwardConfig, run_walk_forward_backtest
        from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
        from src.assembled_core.portfolio.position_sizing import compute_target_positions
        
        config = WalkForwardConfig(
            train_size=252,  # 1 year training
            test_size=63,    # 1 quarter testing
            step_size=21,    # Move forward 1 month
            window_type="rolling"
        )
        
        result = run_walk_forward_backtest(
            prices=price_df,
            signal_fn=generate_trend_signals_from_prices,
            position_sizing_fn=compute_target_positions,
            config=config,
            start_capital=10000.0,
            freq="1d"
        )
        
        print(f"OOS Mean Sharpe: {result.summary_metrics['oos_mean_sharpe']:.2f}")
        print(f"OOS Win Rate: {result.summary_metrics['oos_win_rate']:.1%}")
        ```
    """
    # Split into windows
    windows = _split_time_series(prices, config, freq)
    
    if not windows:
        raise ValueError("No valid windows found. Check data range and window configuration.")
    
    # Use default cost model if not provided
    if cost_model is None:
        cost_model = get_default_cost_model()
    
    # Run backtest for each window
    window_results = []
    
    for window in windows:
        # Step 1: Run backtest on test window (OOS - Out-of-Sample)
        oos_backtest = run_portfolio_backtest(
            prices=window.test_data,
            signal_fn=signal_fn,
            position_sizing_fn=position_sizing_fn,
            start_capital=start_capital,
            cost_model=cost_model,
            include_costs=include_costs,
            include_trades=include_trades,
            include_signals=include_signals,
            include_targets=include_targets,
            rebalance_freq=freq,
            compute_features=compute_features,
            feature_config=feature_config
        )
        
        # Step 2: Compute OOS metrics using qa.metrics
        oos_metrics = compute_all_metrics(
            equity=oos_backtest.equity,
            trades=oos_backtest.trades,
            start_capital=start_capital,
            freq=freq,
            risk_free_rate=0.0
        )
        
        # Step 3: Optionally compute IS metrics (train window)
        is_metrics = None
        if compute_is_metrics and not window.train_data.empty:
            try:
                # Run backtest on train window (IS - In-Sample)
                is_backtest = run_portfolio_backtest(
                    prices=window.train_data,
                    signal_fn=signal_fn,
                    position_sizing_fn=position_sizing_fn,
                    start_capital=start_capital,
                    cost_model=cost_model,
                    include_costs=include_costs,
                    include_trades=False,  # Don't need trades for IS metrics
                    include_signals=False,
                    include_targets=False,
                    rebalance_freq=freq,
                    compute_features=compute_features,
                    feature_config=feature_config
                )
                
                # Compute IS metrics using qa.metrics
                is_metrics = compute_all_metrics(
                    equity=is_backtest.equity,
                    trades=None,  # IS metrics typically don't include trades
                    start_capital=start_capital,
                    freq=freq,
                    risk_free_rate=0.0
                )
            except Exception:
                # If IS backtest fails, continue without IS metrics
                is_metrics = None
        
        # Create window result
        window_result = WalkForwardWindowResult(
            window_index=window.window_index,
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            backtest_result=oos_backtest
        )
        window_results.append(window_result)
    
    # Aggregate summary metrics
    summary_metrics = _aggregate_walk_forward_metrics(window_results)
    
    return WalkForwardResult(
        config=config,
        windows=windows,
        window_results=window_results,
        summary_metrics=summary_metrics
    )


def _aggregate_walk_forward_metrics(
    window_results: list[WalkForwardWindowResult]
) -> dict[str, float | int | None]:
    """Aggregate metrics across multiple walk-forward window results.
    
    Args:
        window_results: List of WalkForwardWindowResult objects
    
    Returns:
        Dictionary with aggregated IS and OOS metrics
    """
    if not window_results:
        return {}
    
    # Extract OOS metrics
    oos_final_pfs = []
    oos_sharpes = []
    oos_cagrs = []
    oos_max_dd_pcts = []
    oos_total_trades = 0
    oos_total_periods = 0
    
    # Extract IS metrics (if available)
    is_final_pfs = []
    is_sharpes = []
    is_cagrs = []
    
    for window_result in window_results:
        # OOS metrics
        oos_metrics = window_result.oos_metrics
        oos_final_pfs.append(oos_metrics.final_pf)
        
        if oos_metrics.sharpe_ratio is not None and not pd.isna(oos_metrics.sharpe_ratio):
            oos_sharpes.append(oos_metrics.sharpe_ratio)
        
        if oos_metrics.cagr is not None and not pd.isna(oos_metrics.cagr):
            oos_cagrs.append(oos_metrics.cagr)
        
        if oos_metrics.max_drawdown_pct is not None and not pd.isna(oos_metrics.max_drawdown_pct):
            oos_max_dd_pcts.append(oos_metrics.max_drawdown_pct)
        
        oos_total_trades += oos_metrics.total_trades if oos_metrics.total_trades is not None else 0
        oos_total_periods += oos_metrics.periods
        
        # IS metrics (if available)
        if window_result.is_metrics is not None:
            is_metrics = window_result.is_metrics
            is_final_pfs.append(is_metrics.final_pf)
            
            if is_metrics.sharpe_ratio is not None and not pd.isna(is_metrics.sharpe_ratio):
                is_sharpes.append(is_metrics.sharpe_ratio)
            
            if is_metrics.cagr is not None and not pd.isna(is_metrics.cagr):
                is_cagrs.append(is_metrics.cagr)
    
    # Calculate aggregated OOS metrics
    oos_mean_final_pf = float(pd.Series(oos_final_pfs).mean()) if oos_final_pfs else 0.0
    oos_std_final_pf = float(pd.Series(oos_final_pfs).std()) if len(oos_final_pfs) > 1 else 0.0
    
    oos_mean_sharpe = float(pd.Series(oos_sharpes).mean()) if oos_sharpes else None
    oos_std_sharpe = float(pd.Series(oos_sharpes).std()) if len(oos_sharpes) > 1 else None
    
    oos_mean_cagr = float(pd.Series(oos_cagrs).mean()) if oos_cagrs else None
    oos_std_cagr = float(pd.Series(oos_cagrs).std()) if len(oos_cagrs) > 1 else None
    
    oos_mean_max_dd_pct = float(pd.Series(oos_max_dd_pcts).mean()) if oos_max_dd_pcts else None
    
    # OOS Win rate: percentage of windows with positive returns
    oos_win_rate = float(sum(1 for pf in oos_final_pfs if pf > 1.0) / len(oos_final_pfs)) if oos_final_pfs else 0.0
    
    # Calculate aggregated IS metrics (if available)
    is_mean_final_pf = float(pd.Series(is_final_pfs).mean()) if is_final_pfs else None
    is_mean_sharpe = float(pd.Series(is_sharpes).mean()) if is_sharpes else None
    is_mean_cagr = float(pd.Series(is_cagrs).mean()) if is_cagrs else None
    
    return {
        # IS (In-Sample) metrics
        "is_mean_final_pf": is_mean_final_pf,
        "is_mean_sharpe": is_mean_sharpe,
        "is_mean_cagr": is_mean_cagr,
        # OOS (Out-of-Sample) metrics
        "oos_mean_final_pf": oos_mean_final_pf,
        "oos_std_final_pf": oos_std_final_pf,
        "oos_mean_sharpe": oos_mean_sharpe,
        "oos_std_sharpe": oos_std_sharpe,
        "oos_mean_cagr": oos_mean_cagr,
        "oos_std_cagr": oos_std_cagr,
        "oos_mean_max_drawdown_pct": oos_mean_max_dd_pct,
        "oos_win_rate": oos_win_rate,
        # Totals
        "total_periods": oos_total_periods,
        "total_trades": oos_total_trades,
        "num_windows": len(window_results)
    }

