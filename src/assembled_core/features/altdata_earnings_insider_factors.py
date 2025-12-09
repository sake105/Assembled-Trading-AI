"""Alt-Data Factors: Earnings and Insider Activity.

This module implements Phase B1 factors that transform earnings and insider events
into time-series factors for factor analysis.

**Important:** Price data comes from LocalParquetPriceDataSource (local Parquet files),
NOT from Finnhub. Finnhub is used only for events.

**Factor Columns Generated:**

**Earnings Factors:**
- `earnings_eps_surprise_last`: Last EPS surprise percentage (most recent earnings event)
- `earnings_revenue_surprise_last`: Last revenue surprise percentage
- `earnings_positive_surprise_flag`: Binary flag (1 if last surprise was positive, 0 otherwise)
- `earnings_negative_surprise_flag`: Binary flag (1 if last surprise was negative, 0 otherwise)
- `post_earnings_drift_return_{window_days}d`: Forward return after earnings announcement

**Insider Activity Factors:**
- `insider_net_notional_{lookback_days}d`: Net insider notional (buy - sell) over lookback window
- `insider_buy_count_{lookback_days}d`: Number of insider buy transactions
- `insider_sell_count_{lookback_days}d`: Number of insider sell transactions
- `insider_buy_sell_ratio_{lookback_days}d`: Ratio of buys to sells (count-based)
- `insider_net_notional_normalized_{lookback_days}d`: Net notional normalized by market cap proxy

All factors are computed per symbol and aligned with the price DataFrame timestamps.
Missing values (NaN) occur when no events are available for a given symbol/date.

Integration:
- Compatible with build_core_ta_factors() and other Phase A factors
- Can be merged with price DataFrame using timestamp & symbol
- Designed for use in Phase C1/C2 factor analysis workflows
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_earnings_surprise_factors(
    events_earnings: pd.DataFrame,
    prices: pd.DataFrame,
    window_days: int = 20,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    price_col: str = "close",
) -> pd.DataFrame:
    """Build earnings surprise factors from earnings events and price data.
    
    Transforms earnings events into time-series factors that can be used in factor analysis.
    Price data must come from LocalParquetPriceDataSource (local Parquet files), not from Finnhub.
    
    Args:
        events_earnings: DataFrame with earnings events (events_earnings_df data contract)
            Required columns: timestamp, symbol, event_type, event_id
            Optional columns: eps_actual, eps_estimate, revenue_actual, revenue_estimate
        prices: DataFrame with price data (Panel format)
            Required columns: timestamp, symbol, close
            Optional columns: open, high, low, volume
        window_days: Window for post-earnings drift return calculation (default: 20)
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        price_col: Column name for price (default: "close")
    
    Returns:
        DataFrame with columns:
        - timestamp, symbol (from prices)
        - earnings_eps_surprise_last: Last EPS surprise percentage
        - earnings_revenue_surprise_last: Last revenue surprise percentage
        - earnings_positive_surprise_flag: 1 if last EPS surprise > 0, else 0
        - earnings_negative_surprise_flag: 1 if last EPS surprise < 0, else 0
        - post_earnings_drift_return_{window_days}d: Forward return after earnings
        
        Sorted by symbol, then timestamp.
        Factors are NaN where no earnings events are available.
    
    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrames are empty or invalid
    """
    # Validate inputs
    required_price_cols = [timestamp_col, group_col, price_col]
    missing_price_cols = [col for col in required_price_cols if col not in prices.columns]
    if missing_price_cols:
        raise KeyError(
            f"Missing required columns in prices: {', '.join(missing_price_cols)}. "
            f"Available: {list(prices.columns)}"
        )
    
    required_event_cols = [timestamp_col, group_col, "event_type"]
    missing_event_cols = [col for col in required_event_cols if col not in events_earnings.columns]
    if missing_event_cols:
        raise KeyError(
            f"Missing required columns in events_earnings: {', '.join(missing_event_cols)}. "
            f"Available: {list(events_earnings.columns)}"
        )
    
    if prices.empty:
        raise ValueError("prices DataFrame is empty")
    
    # Prepare prices DataFrame
    result = prices.copy()
    
    # Ensure timestamps are UTC-aware datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)
    
    if not events_earnings.empty:
        if not pd.api.types.is_datetime64_any_dtype(events_earnings[timestamp_col]):
            events_earnings = events_earnings.copy()
            events_earnings[timestamp_col] = pd.to_datetime(events_earnings[timestamp_col], utc=True)
    
    # Sort by symbol and timestamp (required for merge_asof later)
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    if events_earnings.empty:
        logger.warning("events_earnings is empty. Returning prices with NaN factors.")
        # Add empty factor columns
        result["earnings_eps_surprise_last"] = np.nan
        result["earnings_revenue_surprise_last"] = np.nan
        result["earnings_positive_surprise_flag"] = 0.0
        result["earnings_negative_surprise_flag"] = 0.0
        result[f"post_earnings_drift_return_{window_days}d"] = np.nan
        return result
    
    # Filter events to earnings type
    events_filtered = events_earnings[
        events_earnings["event_type"] == "earnings"
    ].copy()
    
    if events_filtered.empty:
        logger.warning("No earnings events found. Returning prices with NaN factors.")
        result["earnings_eps_surprise_last"] = np.nan
        result["earnings_revenue_surprise_last"] = np.nan
        result["earnings_positive_surprise_flag"] = 0.0
        result["earnings_negative_surprise_flag"] = 0.0
        result[f"post_earnings_drift_return_{window_days}d"] = np.nan
        return result
    
    # Compute surprise metrics for each event
    events_with_surprise = events_filtered.copy()
    
    # EPS Surprise
    if "eps_actual" in events_with_surprise.columns and "eps_estimate" in events_with_surprise.columns:
        eps_actual = pd.to_numeric(events_with_surprise["eps_actual"], errors="coerce")
        eps_estimate = pd.to_numeric(events_with_surprise["eps_estimate"], errors="coerce")
        
        # Compute surprise percentage: (actual - estimate) / abs(estimate)
        # Handle division by zero and NaN
        eps_surprise_pct = np.where(
            (eps_estimate.notna()) & (eps_estimate != 0),
            (eps_actual - eps_estimate) / np.abs(eps_estimate) * 100,
            np.nan
        )
        events_with_surprise["eps_surprise_pct"] = eps_surprise_pct
    else:
        # Use existing eps_surprise_pct if available
        if "eps_surprise_pct" not in events_with_surprise.columns:
            events_with_surprise["eps_surprise_pct"] = np.nan
    
    # Revenue Surprise
    if "revenue_actual" in events_with_surprise.columns and "revenue_estimate" in events_with_surprise.columns:
        revenue_actual = pd.to_numeric(events_with_surprise["revenue_actual"], errors="coerce")
        revenue_estimate = pd.to_numeric(events_with_surprise["revenue_estimate"], errors="coerce")
        
        revenue_surprise_pct = np.where(
            (revenue_estimate.notna()) & (revenue_estimate != 0),
            (revenue_actual - revenue_estimate) / np.abs(revenue_estimate) * 100,
            np.nan
        )
        events_with_surprise["revenue_surprise_pct"] = revenue_surprise_pct
    else:
        if "revenue_surprise_pct" not in events_with_surprise.columns:
            events_with_surprise["revenue_surprise_pct"] = np.nan
    
    # Sort events by symbol and timestamp
    events_with_surprise = events_with_surprise.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    # For each price row, find the most recent earnings event up to that date
    # Use merge_asof to align events with prices (forward-fill last event)
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    # Create a temporary DataFrame with events for merge_asof
    events_for_merge = events_with_surprise[[
        group_col,
        timestamp_col,
        "eps_surprise_pct",
        "revenue_surprise_pct",
    ]].copy()
    
    # Use merge_asof to forward-fill the last earnings event to each price date
    # This gives us the "last earnings surprise" up to each date
    # merge_asof requires both DataFrames to be sorted by the join key (timestamp_col)
    # When using by=group_col, we need to ensure proper sorting within each group
    events_sorted = events_for_merge.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    # Ensure result is sorted by [group_col, timestamp_col] (required for merge_asof with by=)
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    # Perform merge_asof per group to avoid sorting issues with multiple groups
    merged_parts = []
    for symbol in result[group_col].unique():
        symbol_prices = result[result[group_col] == symbol].copy()
        symbol_events = events_sorted[events_sorted[group_col] == symbol].copy()
        
        if symbol_events.empty:
            # No events for this symbol, add NaN columns
            symbol_prices["eps_surprise_pct"] = np.nan
            symbol_prices["revenue_surprise_pct"] = np.nan
        else:
            # Both DataFrames are sorted by timestamp for this symbol
            # Ensure group_col is preserved (merge_asof might drop it)
            symbol_prices = symbol_prices.sort_values(timestamp_col).reset_index(drop=True)
            symbol_events = symbol_events.sort_values(timestamp_col).reset_index(drop=True)
            
            # merge_asof only merges on timestamp, group_col must be preserved
            symbol_prices = pd.merge_asof(
                symbol_prices,
                symbol_events[[timestamp_col, "eps_surprise_pct", "revenue_surprise_pct"]],
                on=timestamp_col,
                direction="backward",
                allow_exact_matches=True,
            )
        
        merged_parts.append(symbol_prices)
    
    # Combine all symbols
    result = pd.concat(merged_parts, ignore_index=True).sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    # Rename merged columns
    result = result.rename(columns={
        "eps_surprise_pct": "earnings_eps_surprise_last",
        "revenue_surprise_pct": "earnings_revenue_surprise_last",
    })
    
    # Create surprise flags
    result["earnings_positive_surprise_flag"] = (
        (result["earnings_eps_surprise_last"] > 0).astype(float)
    )
    result["earnings_negative_surprise_flag"] = (
        (result["earnings_eps_surprise_last"] < 0).astype(float)
    )
    
    # Compute post-earnings drift return
    # For each earnings event date, compute forward return over window_days
    # This factor is only non-NaN on earnings announcement dates
    
    # Initialize column with NaN
    result[f"post_earnings_drift_return_{window_days}d"] = np.nan
    
    # Group prices by symbol for forward return calculation
    grouped_prices = result.groupby(group_col, group_keys=False)
    
    # Compute forward return: log(price[t+window_days] / price[t])
    future_price = grouped_prices[price_col].shift(-window_days)
    current_price = result[price_col]
    
    forward_return = np.log(future_price / current_price)
    
    # Only set forward return for dates that have earnings events
    # Create a mask: True for dates that have earnings events
    earnings_dates = events_with_surprise[[group_col, timestamp_col]].drop_duplicates()
    
    # Merge to find earnings event dates in result
    earnings_mask = pd.merge(
        result[[group_col, timestamp_col]],
        earnings_dates,
        on=[group_col, timestamp_col],
        how="inner"
    )
    
    if not earnings_mask.empty:
        # Set forward return only for earnings event dates
        earnings_indices = result.index[
            result.set_index([group_col, timestamp_col]).index.isin(
                earnings_mask.set_index([group_col, timestamp_col]).index
            )
        ]
        result.loc[earnings_indices, f"post_earnings_drift_return_{window_days}d"] = (
            forward_return.loc[earnings_indices].values
        )
    
    # Count earnings event dates
    n_earnings_dates = len(earnings_mask) if not earnings_mask.empty else 0
    
    logger.info(
        f"Built earnings surprise factors for {result[group_col].nunique()} symbols, "
        f"{len(result)} rows. {n_earnings_dates} earnings event dates found."
    )
    
    return result.sort_values([group_col, timestamp_col]).reset_index(drop=True)


def build_insider_activity_factors(
    events_insider: pd.DataFrame,
    prices: pd.DataFrame,
    lookback_days: int = 60,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    price_col: str = "close",
) -> pd.DataFrame:
    """Build insider activity factors from insider events and price data.
    
    Aggregates insider transactions over a rolling window to create time-series factors.
    Price data must come from LocalParquetPriceDataSource (local Parquet files), not from Finnhub.
    
    Args:
        events_insider: DataFrame with insider events (events_insider_df data contract)
            Required columns: timestamp, symbol, event_type, event_id
            Optional columns: usd_notional, direction, shares, price
        prices: DataFrame with price data (Panel format)
            Required columns: timestamp, symbol, close
            Optional columns: volume (for normalization)
        lookback_days: Rolling window size in days (default: 60)
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        price_col: Column name for price (default: "close")
    
    Returns:
        DataFrame with columns:
        - timestamp, symbol (from prices)
        - insider_net_notional_{lookback_days}d: Net insider notional (buy - sell) over window
        - insider_buy_count_{lookback_days}d: Number of buy transactions
        - insider_sell_count_{lookback_days}d: Number of sell transactions
        - insider_buy_sell_ratio_{lookback_days}d: Ratio of buys to sells (count-based)
        - insider_net_notional_normalized_{lookback_days}d: Net notional / (close * volume) proxy
        
        Sorted by symbol, then timestamp.
        Factors are NaN where no insider events are available.
    
    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrames are empty or invalid
    """
    # Validate inputs
    required_price_cols = [timestamp_col, group_col, price_col]
    missing_price_cols = [col for col in required_price_cols if col not in prices.columns]
    if missing_price_cols:
        raise KeyError(
            f"Missing required columns in prices: {', '.join(missing_price_cols)}. "
            f"Available: {list(prices.columns)}"
        )
    
    required_event_cols = [timestamp_col, group_col, "event_type"]
    missing_event_cols = [col for col in required_event_cols if col not in events_insider.columns]
    if missing_event_cols:
        raise KeyError(
            f"Missing required columns in events_insider: {', '.join(missing_event_cols)}. "
            f"Available: {list(events_insider.columns)}"
        )
    
    if prices.empty:
        raise ValueError("prices DataFrame is empty")
    
    # Prepare prices DataFrame
    result = prices.copy()
    
    # Ensure timestamps are UTC-aware datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)
    
    if not events_insider.empty:
        if not pd.api.types.is_datetime64_any_dtype(events_insider[timestamp_col]):
            events_insider = events_insider.copy()
            events_insider[timestamp_col] = pd.to_datetime(events_insider[timestamp_col], utc=True)
    
    # Sort by symbol and timestamp
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    if events_insider.empty:
        logger.warning("events_insider is empty. Returning prices with NaN factors.")
        # Add empty factor columns
        result[f"insider_net_notional_{lookback_days}d"] = np.nan
        result[f"insider_buy_count_{lookback_days}d"] = 0.0
        result[f"insider_sell_count_{lookback_days}d"] = 0.0
        result[f"insider_buy_sell_ratio_{lookback_days}d"] = np.nan
        result[f"insider_net_notional_normalized_{lookback_days}d"] = np.nan
        return result
    
    # Filter to insider events
    events_filtered = events_insider[
        events_insider["event_type"].str.startswith("insider_", na=False)
    ].copy()
    
    if events_filtered.empty:
        logger.warning("No insider events found. Returning prices with NaN factors.")
        result[f"insider_net_notional_{lookback_days}d"] = np.nan
        result[f"insider_buy_count_{lookback_days}d"] = 0.0
        result[f"insider_sell_count_{lookback_days}d"] = 0.0
        result[f"insider_buy_sell_ratio_{lookback_days}d"] = np.nan
        result[f"insider_net_notional_normalized_{lookback_days}d"] = np.nan
        return result
    
    # Prepare events: ensure direction and usd_notional are available
    if "direction" not in events_filtered.columns:
        # Infer direction from event_type
        events_filtered["direction"] = events_filtered["event_type"].str.replace("insider_", "")
    
    if "usd_notional" not in events_filtered.columns:
        # Try to compute from shares * price
        if "shares" in events_filtered.columns and "price" in events_filtered.columns:
            shares = pd.to_numeric(events_filtered["shares"], errors="coerce")
            price = pd.to_numeric(events_filtered["price"], errors="coerce")
            events_filtered["usd_notional"] = shares * price
        else:
            events_filtered["usd_notional"] = np.nan
    
    # Sort events by symbol and timestamp
    events_filtered = events_filtered.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    # For each price row, aggregate insider events in the lookback window
    # We'll use a rolling window approach: for each price date, sum events in [date - lookback_days, date]
    
    # Create a helper function to aggregate events per symbol/date
    def aggregate_insider_events_per_date(
        symbol: str,
        date: pd.Timestamp,
        events_df: pd.DataFrame,
        lookback: int,
    ) -> dict:
        """Aggregate insider events for a specific symbol and date."""
        # Filter events for this symbol
        symbol_events = events_df[events_df[group_col] == symbol].copy()
        
        if symbol_events.empty:
            return {
                "net_notional": 0.0,
                "buy_count": 0,
                "sell_count": 0,
                "buy_sell_ratio": np.nan,
            }
        
        # Filter to events within lookback window
        window_start = date - pd.Timedelta(days=lookback)
        window_events = symbol_events[
            (symbol_events[timestamp_col] >= window_start) &
            (symbol_events[timestamp_col] <= date)
        ]
        
        if window_events.empty:
            return {
                "net_notional": 0.0,
                "buy_count": 0,
                "sell_count": 0,
                "buy_sell_ratio": np.nan,
            }
        
        # Aggregate by direction
        buy_events = window_events[window_events["direction"] == "buy"]
        sell_events = window_events[window_events["direction"] == "sell"]
        
        buy_notional = pd.to_numeric(buy_events["usd_notional"], errors="coerce").sum()
        sell_notional = pd.to_numeric(sell_events["usd_notional"], errors="coerce").sum()
        
        buy_count = len(buy_events)
        sell_count = len(sell_events)
        
        net_notional = buy_notional - sell_notional
        
        # Buy/sell ratio (count-based)
        if sell_count > 0:
            buy_sell_ratio = buy_count / sell_count
        elif buy_count > 0:
            buy_sell_ratio = np.inf  # Only buys, no sells
        else:
            buy_sell_ratio = np.nan  # No events
        
        return {
            "net_notional": net_notional if not pd.isna(net_notional) else 0.0,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_sell_ratio": buy_sell_ratio,
        }
    
    # For each symbol, compute rolling aggregations over lookback window
    factors_list = []
    
    for symbol in result[group_col].unique():
        symbol_prices = result[result[group_col] == symbol].copy()
        symbol_events = events_filtered[events_filtered[group_col] == symbol].copy()
        
        if symbol_events.empty:
            # No events for this symbol
            symbol_prices[f"insider_net_notional_{lookback_days}d"] = 0.0
            symbol_prices[f"insider_buy_count_{lookback_days}d"] = 0.0
            symbol_prices[f"insider_sell_count_{lookback_days}d"] = 0.0
            symbol_prices[f"insider_buy_sell_ratio_{lookback_days}d"] = np.nan
            symbol_prices[f"insider_net_notional_normalized_{lookback_days}d"] = np.nan
            factors_list.append(symbol_prices)
            continue
        
        # Set timestamp as index for rolling operations
        symbol_prices_indexed = symbol_prices.set_index(timestamp_col).sort_index()
        symbol_events_indexed = symbol_events.set_index(timestamp_col).sort_index()
        
        # Create buy and sell notional series (one value per event date)
        buy_events = symbol_events_indexed[symbol_events_indexed["direction"] == "buy"]
        sell_events = symbol_events_indexed[symbol_events_indexed["direction"] == "sell"]
        
        buy_notional_series = pd.to_numeric(buy_events["usd_notional"], errors="coerce").fillna(0.0)
        sell_notional_series = pd.to_numeric(sell_events["usd_notional"], errors="coerce").fillna(0.0)
        
        buy_count_series = pd.Series(1, index=buy_events.index)
        sell_count_series = pd.Series(1, index=sell_events.index)
        
        # Aggregate multiple events on the same date (sum notional, count events)
        buy_notional_agg = buy_notional_series.groupby(level=0).sum()
        sell_notional_agg = sell_notional_series.groupby(level=0).sum()
        buy_count_agg = buy_count_series.groupby(level=0).count()
        sell_count_agg = sell_count_series.groupby(level=0).count()
        
        # Reindex to price dates (fill missing dates with 0)
        buy_notional_aligned = buy_notional_agg.reindex(symbol_prices_indexed.index, fill_value=0.0)
        sell_notional_aligned = sell_notional_agg.reindex(symbol_prices_indexed.index, fill_value=0.0)
        buy_count_aligned = buy_count_agg.reindex(symbol_prices_indexed.index, fill_value=0.0)
        sell_count_aligned = sell_count_agg.reindex(symbol_prices_indexed.index, fill_value=0.0)
        
        # Compute rolling sums over lookback_days window (time-based)
        net_notional_rolling = (
            buy_notional_aligned.rolling(window=f"{lookback_days}D", min_periods=0).sum() -
            sell_notional_aligned.rolling(window=f"{lookback_days}D", min_periods=0).sum()
        )
        
        buy_count_rolling = buy_count_aligned.rolling(window=f"{lookback_days}D", min_periods=0).sum()
        sell_count_rolling = sell_count_aligned.rolling(window=f"{lookback_days}D", min_periods=0).sum()
        
        # Buy/sell ratio (count-based)
        buy_sell_ratio = np.where(
            sell_count_rolling > 0,
            buy_count_rolling / sell_count_rolling,
            np.where(buy_count_rolling > 0, np.inf, np.nan)
        )
        
        # Assign factors
        symbol_prices_indexed[f"insider_net_notional_{lookback_days}d"] = net_notional_rolling.values
        symbol_prices_indexed[f"insider_buy_count_{lookback_days}d"] = buy_count_rolling.values.astype(float)
        symbol_prices_indexed[f"insider_sell_count_{lookback_days}d"] = sell_count_rolling.values.astype(float)
        symbol_prices_indexed[f"insider_buy_sell_ratio_{lookback_days}d"] = buy_sell_ratio
        
        # Normalized notional (if volume available)
        if "volume" in symbol_prices_indexed.columns:
            volume = pd.to_numeric(symbol_prices_indexed["volume"], errors="coerce")
            close = pd.to_numeric(symbol_prices_indexed[price_col], errors="coerce")
            market_cap_proxy = close * volume
            normalized_notional = np.where(
                (market_cap_proxy > 0) & (market_cap_proxy.notna()),
                net_notional_rolling / market_cap_proxy,
                np.nan
            )
            symbol_prices_indexed[f"insider_net_notional_normalized_{lookback_days}d"] = normalized_notional
        else:
            symbol_prices_indexed[f"insider_net_notional_normalized_{lookback_days}d"] = np.nan
        
        # Reset index
        symbol_prices_with_factors = symbol_prices_indexed.reset_index()
        factors_list.append(symbol_prices_with_factors)
    
    # Combine all symbols
    if factors_list:
        result = pd.concat(factors_list, ignore_index=True)
    else:
        # No symbols processed (shouldn't happen, but handle gracefully)
        result[f"insider_net_notional_{lookback_days}d"] = 0.0
        result[f"insider_buy_count_{lookback_days}d"] = 0.0
        result[f"insider_sell_count_{lookback_days}d"] = 0.0
        result[f"insider_buy_sell_ratio_{lookback_days}d"] = np.nan
        result[f"insider_net_notional_normalized_{lookback_days}d"] = np.nan
    
    # Sort by symbol and timestamp
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    logger.info(
        f"Built insider activity factors for {result[group_col].nunique()} symbols, "
        f"{len(result)} rows. Lookback window: {lookback_days} days."
    )
    
    return result

