"""Event Study Engine for analyzing price reactions to events.

This module provides functions for event study analysis:
- Extracting price windows around events
- Computing normal and abnormal returns
- Aggregating results across events

Part of Phase C3: Event Study Framework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_event_window_prices(
    prices_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_before: int = 20,
    window_after: int = 40,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    price_col: str = "close",
    event_id_col: str | None = None,
    event_type_col: str = "event_type",
) -> pd.DataFrame:
    """Extract price windows around events.

    For each event, extracts prices from `window_before` days before to `window_after` days after
    the event. Returns a "stacked" DataFrame with one row per (event, relative_day).

    Args:
        prices_df: Panel DataFrame with at least timestamp, symbol, close (and optionally
            other columns like factors). Must be sorted by symbol, then timestamp.
        events_df: DataFrame with timestamp, symbol, event_type (and optionally event_id).
            If event_id is not present, it will be generated.
        window_before: Number of days before event to include (default: 20)
        window_after: Number of days after event to include (default: 40)
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        price_col: Column name for price (default: "close")
        event_id_col: Column name for event ID (default: None, will generate if missing)
        event_type_col: Column name for event type (default: "event_type")

    Returns:
        DataFrame with columns:
        - event_id: Unique event identifier
        - symbol: Symbol
        - event_type: Event type
        - event_timestamp: Event timestamp
        - rel_day: Relative day (-window_before to +window_after, 0 = event day)
        - timestamp: Actual timestamp for this day
        - close: Price at this timestamp (or price_col value)
        - Additional columns from prices_df (e.g., open, high, low, volume, factors)

    Raises:
        KeyError: If required columns are missing
        ValueError: If timestamps are not UTC-aware

    Example:
        >>> events = pd.DataFrame({
        ...     "timestamp": [pd.Timestamp("2024-01-15", tz="UTC")],
        ...     "symbol": ["AAPL"],
        ...     "event_type": ["earnings"]
        ... })
        >>> prices = load_prices(...)  # Panel with timestamp, symbol, close
        >>> windows = build_event_window_prices(prices, events, window_before=5, window_after=5)
        >>> # Result has rows for rel_day = -5, -4, ..., 0, ..., +5
    """
    # Validate inputs
    required_price_cols = [timestamp_col, group_col, price_col]
    for col in required_price_cols:
        if col not in prices_df.columns:
            raise KeyError(f"Required column '{col}' not found in prices_df")

    required_event_cols = [timestamp_col, group_col, event_type_col]
    for col in required_event_cols:
        if col not in events_df.columns:
            raise KeyError(f"Required column '{col}' not found in events_df")

    # Ensure timestamps are UTC-aware
    if prices_df[timestamp_col].dtype != "datetime64[ns, UTC]":
        prices_df = prices_df.copy()
        prices_df[timestamp_col] = pd.to_datetime(prices_df[timestamp_col], utc=True)

    if events_df[timestamp_col].dtype != "datetime64[ns, UTC]":
        events_df = events_df.copy()
        events_df[timestamp_col] = pd.to_datetime(events_df[timestamp_col], utc=True)

    # Generate event_id if not present
    events_work = events_df.copy()
    if event_id_col is None:
        # Check if "event_id" column exists
        if "event_id" in events_work.columns:
            event_id_col = "event_id"
        else:
            # Generate event_id
            events_work["event_id"] = (
                events_work[event_type_col].astype(str)
                + "_"
                + events_work[group_col].astype(str)
                + "_"
                + events_work[timestamp_col].dt.strftime("%Y%m%d")
            )
            # Make unique by adding index if duplicates
            if events_work["event_id"].duplicated().any():
                events_work["event_id"] = (
                    events_work["event_id"] + "_" + events_work.index.astype(str)
                )
            event_id_col = "event_id"
    elif event_id_col not in events_work.columns:
        # event_id_col specified but not found - generate it
        events_work["event_id"] = (
            events_work[event_type_col].astype(str)
            + "_"
            + events_work[group_col].astype(str)
            + "_"
            + events_work[timestamp_col].dt.strftime("%Y%m%d")
        )
        # Make unique by adding index if duplicates
        if events_work["event_id"].duplicated().any():
            events_work["event_id"] = (
                events_work["event_id"] + "_" + events_work.index.astype(str)
            )
        event_id_col = "event_id"

    # Sort prices by symbol, then timestamp
    prices_sorted = prices_df.sort_values([group_col, timestamp_col]).reset_index(
        drop=True
    )

    # Build event windows
    all_windows = []

    for _, event_row in events_work.iterrows():
        event_symbol = event_row[group_col]
        event_timestamp = event_row[timestamp_col]
        event_type = event_row[event_type_col]
        event_id = event_row[event_id_col]

        # Filter prices for this symbol
        symbol_prices = prices_sorted[prices_sorted[group_col] == event_symbol].copy()

        if symbol_prices.empty:
            continue

        # Find event day index
        event_idx = symbol_prices[symbol_prices[timestamp_col] == event_timestamp].index

        if len(event_idx) == 0:
            # Event timestamp not found in prices, try to find closest
            time_diffs = (symbol_prices[timestamp_col] - event_timestamp).abs()
            closest_idx = time_diffs.idxmin()
            closest_diff = time_diffs.loc[closest_idx]

            # Only use if within 1 day
            if closest_diff > pd.Timedelta(days=1):
                continue

            event_idx = pd.Index([closest_idx])

        event_idx = event_idx[0]
        event_row_idx = symbol_prices.index.get_loc(event_idx)

        # Extract window: from (event_row_idx - window_before) to (event_row_idx + window_after)
        start_idx = max(0, event_row_idx - window_before)
        end_idx = min(len(symbol_prices), event_row_idx + window_after + 1)

        window_prices = symbol_prices.iloc[start_idx:end_idx].copy()

        if window_prices.empty:
            continue

        # Calculate relative day
        event_day_timestamp = symbol_prices.iloc[event_row_idx][timestamp_col]
        window_prices["rel_day"] = (
            window_prices[timestamp_col] - event_day_timestamp
        ).dt.days

        # Add event metadata
        window_prices["event_id"] = event_id
        window_prices["event_type"] = event_type
        window_prices["event_timestamp"] = event_timestamp
        # Ensure group_col exists (it should already, but make sure)
        if group_col not in window_prices.columns:
            window_prices[group_col] = event_symbol

        # Reorder columns: event metadata first, then price data
        cols_order = [
            "event_id",
            group_col,
            "event_type",
            "event_timestamp",
            "rel_day",
            timestamp_col,
        ]
        # Add price_col and other columns from prices_df
        other_cols = [c for c in window_prices.columns if c not in cols_order]
        cols_order.extend(other_cols)

        # Keep only existing columns
        cols_order = [c for c in cols_order if c in window_prices.columns]
        window_prices = window_prices[cols_order]

        all_windows.append(window_prices)

    if not all_windows:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "event_id",
                group_col,
                "event_type",
                "event_timestamp",
                "rel_day",
                timestamp_col,
                price_col,
            ]
        )

    result = pd.concat(all_windows, ignore_index=True)

    # Ensure rel_day is integer
    result["rel_day"] = result["rel_day"].astype(int)

    return result.sort_values(["event_id", "rel_day"]).reset_index(drop=True)


def compute_event_returns(
    event_windows_df: pd.DataFrame,
    price_col: str = "close",
    benchmark_col: str | None = None,
    return_type: str = "log",
    rel_day_col: str = "rel_day",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Compute normal and abnormal returns for event windows.

    Computes returns for each event and relative day. If benchmark_col is provided,
    also computes abnormal returns (return - benchmark_return).

    Args:
        event_windows_df: Output from build_event_window_prices()
        price_col: Column name for price (default: "close")
        benchmark_col: Column name for benchmark price (default: None).
            If provided, abnormal returns will be computed.
        return_type: "log" for log returns, "simple" for simple returns (default: "log")
        rel_day_col: Column name for relative day (default: "rel_day")
        timestamp_col: Column name for timestamp (default: "timestamp")

    Returns:
        DataFrame with columns:
        - event_id, symbol, event_type, event_timestamp, rel_day (from input)
        - timestamp: Actual timestamp
        - event_return: Return for this relative day
        - abnormal_return: Abnormal return (if benchmark_col provided)
        - Additional columns from input (e.g., close, factors)

    Raises:
        KeyError: If required columns are missing
        ValueError: If return_type is not "log" or "simple"

    Example:
        >>> windows = build_event_window_prices(prices, events)
        >>> returns = compute_event_returns(windows, price_col="close")
        >>> # Returns has event_return column with log returns
        >>>
        >>> # With benchmark
        >>> returns = compute_event_returns(windows, price_col="close", benchmark_col="spy_close")
        >>> # Returns has both event_return and abnormal_return columns
    """
    if return_type not in ["log", "simple"]:
        raise ValueError(f"return_type must be 'log' or 'simple', got '{return_type}'")

    # Validate inputs
    required_cols = [price_col, rel_day_col, "event_id"]
    for col in required_cols:
        if col not in event_windows_df.columns:
            raise KeyError(f"Required column '{col}' not found in event_windows_df")

    if benchmark_col is not None and benchmark_col not in event_windows_df.columns:
        raise KeyError(f"benchmark_col '{benchmark_col}' not found in event_windows_df")

    result = event_windows_df.copy()

    # Compute returns per event
    result["event_return"] = np.nan

    for event_id in result["event_id"].unique():
        event_mask = result["event_id"] == event_id
        event_data = result[event_mask].sort_values(rel_day_col)

        if len(event_data) < 2:
            continue

        prices = event_data[price_col].values

        if return_type == "log":
            # Log returns: ln(price[t] / price[t-1])
            returns = np.diff(np.log(prices))
            # First day has no return (NaN)
            returns = np.concatenate([[np.nan], returns])
        else:
            # Simple returns: (price[t] / price[t-1]) - 1
            returns = np.diff(prices) / prices[:-1]
            # First day has no return (NaN)
            returns = np.concatenate([[np.nan], returns])

        result.loc[event_mask, "event_return"] = returns

    # Compute abnormal returns if benchmark provided
    if benchmark_col is not None:
        result["abnormal_return"] = np.nan

        for event_id in result["event_id"].unique():
            event_mask = result["event_id"] == event_id
            event_data = result[event_mask].sort_values(rel_day_col)

            if len(event_data) < 2:
                continue

            event_returns = event_data["event_return"].values
            benchmark_prices = event_data[benchmark_col].values

            # Compute benchmark returns
            if return_type == "log":
                benchmark_returns = np.diff(np.log(benchmark_prices))
                benchmark_returns = np.concatenate([[np.nan], benchmark_returns])
            else:
                benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]
                benchmark_returns = np.concatenate([[np.nan], benchmark_returns])

            # Abnormal return = event_return - benchmark_return
            abnormal_returns = event_returns - benchmark_returns
            result.loc[event_mask, "abnormal_return"] = abnormal_returns

    return result


def aggregate_event_study(
    returns_df: pd.DataFrame,
    use_abnormal: bool = True,
    return_col: str | None = None,
    rel_day_col: str = "rel_day",
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Aggregate event returns across events.

    Computes average returns, cumulative returns, and confidence intervals
    for each relative day across all events.

    Args:
        returns_df: Output from compute_event_returns()
        use_abnormal: If True, use abnormal_return; if False, use event_return (default: True)
        return_col: Explicit column name for returns (default: None, auto-detect)
        rel_day_col: Column name for relative day (default: "rel_day")
        confidence_level: Confidence level for intervals (default: 0.95)

    Returns:
        DataFrame with columns:
        - rel_day: Relative day
        - avg_ret: Average return across events
        - std_ret: Standard deviation of returns
        - cum_ret: Cumulative return (sum from first day to this day)
        - n_events: Number of events with valid data for this day
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - se: Standard error (std / sqrt(n))

    Raises:
        KeyError: If required columns are missing
        ValueError: If no valid return column found

    Example:
        >>> returns = compute_event_returns(windows, price_col="close", benchmark_col="spy_close")
        >>> aggregated = aggregate_event_study(returns, use_abnormal=True)
        >>> # Result has avg_ret, cum_ret, n_events, ci_lower, ci_upper for each rel_day
    """
    # Determine return column
    if return_col is None:
        if use_abnormal and "abnormal_return" in returns_df.columns:
            return_col = "abnormal_return"
        elif "event_return" in returns_df.columns:
            return_col = "event_return"
        else:
            raise ValueError(
                "No return column found. Expected 'abnormal_return' or 'event_return'. "
                "Set use_abnormal=False or provide return_col explicitly."
            )

    if return_col not in returns_df.columns:
        raise KeyError(f"Return column '{return_col}' not found in returns_df")

    if rel_day_col not in returns_df.columns:
        raise KeyError(f"Relative day column '{rel_day_col}' not found in returns_df")

    # Group by relative day and aggregate
    grouped = returns_df.groupby(rel_day_col)[return_col]

    # Compute statistics
    result = pd.DataFrame(
        {
            "rel_day": grouped.mean().index,
            "avg_ret": grouped.mean().values,
            "std_ret": grouped.std().values,
            "n_events": grouped.count().values,
        }
    )

    # Compute standard error
    result["se"] = result["std_ret"] / np.sqrt(result["n_events"])

    # Compute confidence intervals (using z-score for normal distribution)
    # For large n, use z-score; for small n, could use t-distribution
    # Using z-score for simplicity (1.96 for 95% CI)
    try:
        from scipy import stats

        z_score = stats.norm.ppf((1 + confidence_level) / 2)
    except ImportError:
        # Fallback: use approximate z-scores for common confidence levels
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        z_score = z_scores.get(confidence_level, 1.96)  # Default to 95% CI
    result["ci_lower"] = result["avg_ret"] - z_score * result["se"]
    result["ci_upper"] = result["avg_ret"] + z_score * result["se"]

    # Compute cumulative return
    result = result.sort_values("rel_day")
    result["cum_ret"] = result["avg_ret"].cumsum()

    # Reset index
    result = result.reset_index(drop=True)

    return result
