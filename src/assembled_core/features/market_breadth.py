"""Market Breadth and Risk-On/Risk-Off Indicators module.

This module implements Phase A, Sprint A3 from the Advanced Analytics & Factor Labs roadmap.
It provides market-wide indicators that describe the state of the entire universe:

- Market Breadth (fraction of stocks above moving average)
- Advance/Decline Line (cumulative net advances)
- Risk-On/Risk-Off Indicators (optional, sector-based)

All indicators are computed at the universe level (aggregated across all symbols)
and returned as time-series DataFrames with one row per timestamp.

Integration:
- Works with panel price data (multiple symbols over time)
- Designed for regime detection and market state analysis
- Compatible with factor research and ML feature engineering
- Primary use: Research notebooks, factor analysis, regime detection workflows

Usage:
    # Compute market breadth for entire universe
    breadth = compute_market_breadth_ma(prices, ma_window=50)
    
    # Compute advance/decline line
    ad_line = compute_advance_decline_line(prices)
    
    # Combine for regime detection
    market_state = pd.merge(breadth, ad_line, on="timestamp")
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_market_breadth_ma(
    prices: pd.DataFrame,
    ma_window: int = 50,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Compute market breadth: fraction of symbols above moving average.
    
    This indicator measures market participation and strength:
    - High values (>0.7): Broad participation, strong market
    - Low values (<0.3): Narrow participation, weak market
    
    Args:
        prices: DataFrame with price data (panel format: multiple symbols over time)
            Required columns: timestamp_col, group_col, price_col
        ma_window: Moving average window in days (default: 50)
        price_col: Column name for price data (default: "close")
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
    
    Returns:
        DataFrame with columns:
        - timestamp: Timestamp (UTC)
        - fraction_above_ma_{ma_window}: Fraction of symbols above MA (0.0 to 1.0)
        - count_above_ma: Number of symbols above MA
        - count_total: Total number of symbols with data at that timestamp
        
        One row per timestamp, sorted by timestamp.
    
    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty or invalid
    """
    # Validate input
    required_cols = [timestamp_col, group_col, price_col]
    missing_cols = [col for col in required_cols if col not in prices.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available columns: {list(prices.columns)}"
        )
    
    if prices.empty:
        raise ValueError("Input DataFrame is empty")
    
    result_df = prices.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col], utc=True)
    
    # Sort by symbol and timestamp
    result_df = result_df.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    # Compute moving average per symbol
    from src.assembled_core.features.ta_features import add_moving_averages
    
    # Temporarily rename columns if needed for add_moving_averages
    if group_col != "symbol" or timestamp_col != "timestamp":
        temp_df = result_df.copy()
        rename_map = {}
        if group_col != "symbol":
            rename_map[group_col] = "symbol"
        if timestamp_col != "timestamp":
            rename_map[timestamp_col] = "timestamp"
        temp_df = temp_df.rename(columns=rename_map)
        temp_df = add_moving_averages(temp_df, windows=(ma_window,), price_col=price_col)
        ma_col = f"ma_{ma_window}"
        if ma_col in temp_df.columns:
            result_df[ma_col] = temp_df[ma_col].reindex(result_df.index)
    else:
        result_df = add_moving_averages(result_df, windows=(ma_window,), price_col=price_col)
    
    ma_col = f"ma_{ma_window}"
    if ma_col not in result_df.columns:
        raise ValueError(f"Failed to compute moving average column {ma_col}")
    
    # For each timestamp, compute fraction of symbols above MA
    breadth_data = []
    
    for timestamp in sorted(result_df[timestamp_col].unique()):
        timestamp_data = result_df[result_df[timestamp_col] == timestamp]
        
        # Filter rows where both price and MA are available
        valid_mask = (
            timestamp_data[price_col].notna() &
            timestamp_data[ma_col].notna()
        )
        valid_data = timestamp_data[valid_mask]
        
        if len(valid_data) == 0:
            continue
        
        # Count symbols above MA
        above_ma = (valid_data[price_col] > valid_data[ma_col]).sum()
        total_count = len(valid_data)
        fraction = above_ma / total_count if total_count > 0 else 0.0
        
        breadth_data.append({
            timestamp_col: timestamp,
            f"fraction_above_ma_{ma_window}": fraction,
            "count_above_ma": above_ma,
            "count_total": total_count,
        })
    
    breadth_df = pd.DataFrame(breadth_data)
    
    if breadth_df.empty:
        logger.warning("No market breadth data computed. Check input data.")
        return pd.DataFrame(columns=[timestamp_col, f"fraction_above_ma_{ma_window}", "count_above_ma", "count_total"])
    
    # Sort by timestamp
    breadth_df = breadth_df.sort_values(timestamp_col).reset_index(drop=True)
    
    logger.info(
        f"Computed market breadth for {len(breadth_df)} timestamps, "
        f"average symbols per timestamp: {breadth_df['count_total'].mean():.1f}"
    )
    
    return breadth_df


def compute_advance_decline_line(
    prices: pd.DataFrame,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Compute Advance/Decline Line for the universe.
    
    The A/D Line is a market breadth indicator that measures the net difference
    between advancing and declining stocks. A rising A/D Line indicates broad
    market participation.
    
    Args:
        prices: DataFrame with price data (panel format: multiple symbols over time)
            Required columns: timestamp_col, group_col, price_col
        price_col: Column name for price data (default: "close")
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
    
    Returns:
        DataFrame with columns:
        - timestamp: Timestamp (UTC)
        - advances: Number of symbols with positive return on this day
        - declines: Number of symbols with negative return on this day
        - net_advances: advances - declines
        - ad_line: Cumulative sum of net_advances (Advance/Decline Line)
        - ad_line_normalized: A/D Line normalized to start at 0 (first value = 0)
        
        One row per timestamp, sorted by timestamp.
    
    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty or invalid
    """
    # Validate input
    required_cols = [timestamp_col, group_col, price_col]
    missing_cols = [col for col in required_cols if col not in prices.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available columns: {list(prices.columns)}"
        )
    
    if prices.empty:
        raise ValueError("Input DataFrame is empty")
    
    result_df = prices.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col], utc=True)
    
    # Sort by symbol and timestamp
    result_df = result_df.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    # Compute daily returns per symbol
    grouped_price = result_df.groupby(group_col, group_keys=False)[price_col]
    daily_returns = grouped_price.pct_change()
    
    result_df["_daily_return"] = daily_returns
    
    # Aggregate by timestamp: count advances and declines
    ad_data = []
    
    for timestamp in sorted(result_df[timestamp_col].unique()):
        timestamp_data = result_df[result_df[timestamp_col] == timestamp]
        
        # Filter rows with valid returns (not NaN)
        valid_returns = timestamp_data["_daily_return"].dropna()
        
        if len(valid_returns) == 0:
            continue
        
        # Count advances (positive returns) and declines (negative returns)
        advances = (valid_returns > 0).sum()
        declines = (valid_returns < 0).sum()
        net_advances = advances - declines
        
        ad_data.append({
            timestamp_col: timestamp,
            "advances": advances,
            "declines": declines,
            "net_advances": net_advances,
            "count_total": len(valid_returns),
        })
    
    ad_df = pd.DataFrame(ad_data)
    
    if ad_df.empty:
        logger.warning("No advance/decline data computed. Check input data.")
        return pd.DataFrame(columns=[timestamp_col, "advances", "declines", "net_advances", "ad_line", "ad_line_normalized"])
    
    # Sort by timestamp
    ad_df = ad_df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Compute cumulative A/D Line
    ad_df["ad_line"] = ad_df["net_advances"].cumsum()
    
    # Normalize: start at 0 (first value = 0)
    ad_df["ad_line_normalized"] = ad_df["ad_line"] - ad_df["ad_line"].iloc[0]
    
    logger.info(
        f"Computed Advance/Decline Line for {len(ad_df)} timestamps, "
        f"average symbols per timestamp: {ad_df['count_total'].mean():.1f}"
    )
    
    return ad_df


def compute_risk_on_off_indicator(
    prices: pd.DataFrame,
    sector_col: str | None = None,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Compute Risk-On/Risk-Off indicator based on sector classification.
    
    This is a placeholder implementation that computes simple ratios of
    advancing vs. declining stocks. For a full implementation, sector
    classifications (cyclical vs. defensive) would be required.
    
    Args:
        prices: DataFrame with price data (panel format)
            Required columns: timestamp_col, group_col, price_col
        sector_col: Optional column name for sector classification
            If provided, computes separate metrics for cyclical vs. defensive sectors
            (Not implemented in this placeholder)
        price_col: Column name for price data (default: "close")
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
    
    Returns:
        DataFrame with columns:
        - timestamp: Timestamp (UTC)
        - risk_on_ratio: Ratio of advancing to declining stocks (simple proxy)
        - risk_off_ratio: Inverse ratio
        - risk_on_off_score: Normalized score (-1 = risk-off, +1 = risk-on)
        
        One row per timestamp, sorted by timestamp.
    
    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty
    """
    # Validate input
    required_cols = [timestamp_col, group_col, price_col]
    missing_cols = [col for col in required_cols if col not in prices.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available columns: {list(prices.columns)}"
        )
    
    if prices.empty:
        raise ValueError("Input DataFrame is empty")
    
    result_df = prices.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col], utc=True)
    
    # Sort by symbol and timestamp
    result_df = result_df.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    # Compute daily returns per symbol
    grouped_price = result_df.groupby(group_col, group_keys=False)[price_col]
    daily_returns = grouped_price.pct_change()
    
    result_df["_daily_return"] = daily_returns
    
    # Aggregate by timestamp
    risk_data = []
    
    for timestamp in sorted(result_df[timestamp_col].unique()):
        timestamp_data = result_df[result_df[timestamp_col] == timestamp]
        
        # Filter rows with valid returns
        valid_returns = timestamp_data["_daily_return"].dropna()
        
        if len(valid_returns) == 0:
            continue
        
        # Count advances and declines
        advances = (valid_returns > 0).sum()
        declines = (valid_returns < 0).sum()
        
        # Compute ratios
        total = advances + declines
        if total > 0:
            risk_on_ratio = advances / total
            risk_off_ratio = declines / total
            # Score: -1 (all declines) to +1 (all advances)
            risk_on_off_score = (advances - declines) / total
        else:
            risk_on_ratio = 0.5
            risk_off_ratio = 0.5
            risk_on_off_score = 0.0
        
        risk_data.append({
            timestamp_col: timestamp,
            "risk_on_ratio": risk_on_ratio,
            "risk_off_ratio": risk_off_ratio,
            "risk_on_off_score": risk_on_off_score,
            "count_total": len(valid_returns),
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    if risk_df.empty:
        logger.warning("No risk-on/risk-off data computed. Check input data.")
        return pd.DataFrame(columns=[timestamp_col, "risk_on_ratio", "risk_off_ratio", "risk_on_off_score"])
    
    # Sort by timestamp
    risk_df = risk_df.sort_values(timestamp_col).reset_index(drop=True)
    
    if sector_col is not None:
        logger.info("Sector-based risk-on/risk-off classification not yet implemented. Using simple ratio proxy.")
    
    logger.info(
        f"Computed Risk-On/Risk-Off indicator for {len(risk_df)} timestamps"
    )
    
    return risk_df

