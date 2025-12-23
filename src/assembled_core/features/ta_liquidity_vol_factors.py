"""Liquidity and Volatility Factors module.

This module implements Phase A, Sprint A2 from the Advanced Analytics & Factor Labs roadmap.
It provides liquidity and volatility-based factors:

- Realized Volatility (various windows)
- Volatility of Volatility (Vol-of-Vol)
- Turnover and Liquidity Proxies (volume, spread, illiquidity scores)

All factors are designed to work with the standard price data format:
- Columns: timestamp (UTC), symbol, close (required)
- Optional: high, low, volume, freefloat
- Sorted by symbol, then timestamp
- Panel format (multiple symbols over time)

Integration:
- Builds on existing ta_features.py functions (log returns)
- Designed for factor research and ML feature engineering
- Compatible with backtest engine and EOD pipeline
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_realized_volatility(
    prices: pd.DataFrame,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add realized volatility columns to price DataFrame.

    Computes rolling standard deviation of log returns for different windows.

    Args:
        prices: DataFrame with price data
            Required columns: timestamp_col, group_col, price_col
        price_col: Column name for price data (default: "close")
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        windows: List of window sizes in days (default: [20, 60])
            Each window will create a column rv_{window}

    Returns:
        DataFrame with original columns plus realized volatility columns:
        - rv_{window} for each window in windows
        - All computed per group (symbol)

    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty or invalid
    """
    if windows is None:
        windows = [20, 60]

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

    result = prices.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)

    # Sort by group and timestamp
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)

    # Compute log returns per symbol (similar to add_log_returns in ta_features.py)
    # Sort by group and timestamp for stable calculation
    temp = result.sort_values([group_col, timestamp_col])

    # Log prices
    log_price = np.log(temp[price_col].astype("float64"))

    # Log returns: diff of log prices per symbol
    log_returns = log_price.groupby(temp[group_col]).diff()

    # Reindex to match original index
    log_returns = log_returns.reindex(result.index)

    # Compute realized volatility for each window
    # Need to sort again after adding log_returns
    result["_log_return_temp"] = log_returns
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)

    for window in windows:
        # Rolling standard deviation of log returns per symbol
        rv = (
            result.groupby(group_col, group_keys=False)["_log_return_temp"]
            .rolling(window=window, min_periods=min(5, window // 4))
            .std()
            .reset_index(level=0, drop=True)
        )

        # Annualize (multiply by sqrt(252) for daily data)
        rv_annualized = rv * np.sqrt(252)

        # Reindex to match original order
        result[f"rv_{window}"] = rv_annualized.reindex(result.index).astype("float64")

    # Remove temporary column
    result = result.drop(columns=["_log_return_temp"])

    logger.info(f"Added realized volatility factors for windows: {windows}")

    return result


def add_vol_of_vol(
    prices: pd.DataFrame,
    rv_cols: list[str] | None = None,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    vol_window: int = 60,
) -> pd.DataFrame:
    """
    Add Volatility-of-Volatility (Vol-of-Vol) factors.

    Computes rolling standard deviation of realized volatility over a longer time period.
    This captures the stability/variability of volatility itself.

    Args:
        prices: DataFrame with price data (must already have realized volatility columns)
        rv_cols: List of realized volatility column names to compute Vol-of-Vol for
            (default: None, will auto-detect rv_* columns)
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        vol_window: Window size for Vol-of-Vol calculation (default: 60 days)
            Creates columns vov_{rv_col}_{vol_window}

    Returns:
        DataFrame with original columns plus Vol-of-Vol columns:
        - vov_{rv_col}_{vol_window} for each rv_col

    Raises:
        KeyError: If required columns or rv_cols are missing
    """
    result = prices.copy()

    # Auto-detect rv columns if not provided
    if rv_cols is None:
        rv_cols = [col for col in result.columns if col.startswith("rv_")]

    if not rv_cols:
        logger.warning(
            "No realized volatility columns found. Skipping Vol-of-Vol calculation."
        )
        return result

    # Validate required columns
    if group_col not in result.columns:
        raise KeyError(f"Missing required column: {group_col}")

    # Sort by group and timestamp
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)

    # Compute Vol-of-Vol for each realized volatility column
    for rv_col in rv_cols:
        if rv_col not in result.columns:
            logger.warning(f"Realized volatility column {rv_col} not found. Skipping.")
            continue

        # Rolling standard deviation of realized volatility
        vov = (
            result.groupby(group_col, group_keys=False)[rv_col]
            .rolling(window=vol_window, min_periods=min(10, vol_window // 6))
            .std()
            .reset_index(level=0, drop=True)
        )

        # Extract window number from rv_col name (e.g., "rv_20" -> "20")
        rv_window = rv_col.replace("rv_", "")
        result[f"vov_{rv_window}_{vol_window}"] = vov.astype("float64")

    logger.info(
        f"Added Vol-of-Vol factors for {len(rv_cols)} RV columns with window {vol_window}"
    )

    return result


def add_turnover_and_liquidity_proxies(
    prices: pd.DataFrame,
    volume_col: str = "volume",
    freefloat_col: str | None = None,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Add turnover and liquidity proxy factors.

    Computes various liquidity-related factors:
    - Turnover (volume / freefloat, if freefloat available)
    - Volume z-score (normalized volume per symbol)
    - Spread proxy ((high - low) / close, if high/low available)

    Args:
        prices: DataFrame with price data
            Required columns: timestamp_col, group_col
            Optional: volume_col, freefloat_col, high, low, close
        volume_col: Column name for volume data (default: "volume")
        freefloat_col: Optional column name for free float market cap or shares
            If provided, computes turnover = volume / freefloat
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")

    Returns:
        DataFrame with original columns plus liquidity proxy columns:
        - turnover (if freefloat_col provided)
        - volume_zscore (always, if volume_col available)
        - spread_proxy (if high/low/close available)

    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty
    """
    result = prices.copy()

    # Validate input
    required_cols = [timestamp_col, group_col]
    missing_cols = [col for col in required_cols if col not in result.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available columns: {list(result.columns)}"
        )

    if result.empty:
        raise ValueError("Input DataFrame is empty")

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)

    # Sort by group and timestamp
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)

    # 1. Turnover = volume / freefloat (if freefloat available)
    if (
        freefloat_col is not None
        and freefloat_col in result.columns
        and volume_col in result.columns
    ):
        # Avoid division by zero
        turnover = result[volume_col] / result[freefloat_col].replace(0, np.nan)
        result["turnover"] = turnover.astype("float64")
        logger.info("Added turnover factor (volume / freefloat)")
    else:
        if freefloat_col is not None:
            logger.warning(
                f"Freefloat column {freefloat_col} not found. Skipping turnover calculation."
            )

    # 2. Volume z-score (rolling z-score of volume per symbol)
    if volume_col in result.columns:
        grouped_volume = result.groupby(group_col, group_keys=False)[volume_col]

        # Rolling mean and std over 60 days (or available period)
        rolling_mean = (
            grouped_volume.rolling(window=60, min_periods=10)
            .mean()
            .reset_index(level=0, drop=True)
        )

        rolling_std = (
            grouped_volume.rolling(window=60, min_periods=10)
            .std()
            .reset_index(level=0, drop=True)
        )

        # Z-score: (volume - mean) / std
        volume_zscore = (result[volume_col] - rolling_mean) / rolling_std

        # Handle division by zero
        volume_zscore = np.where(rolling_std > 1e-10, volume_zscore, 0.0)

        result["volume_zscore"] = volume_zscore.astype("float64")
        logger.info("Added volume_zscore factor")
    else:
        logger.warning(
            f"Volume column {volume_col} not found. Skipping volume_zscore calculation."
        )

    # 3. Spread proxy = (high - low) / close (if high/low/close available)
    if all(col in result.columns for col in ["high", "low", "close"]):
        spread_proxy = (result["high"] - result["low"]) / result["close"]
        result["spread_proxy"] = spread_proxy.astype("float64")
        logger.info("Added spread_proxy factor ((high - low) / close)")
    else:
        logger.warning(
            "High/low/close columns not available. Skipping spread_proxy calculation."
        )

    return result


# TODO: Implement add_intraday_noise_proxy() if intraday data aggregation is available
# This would require access to intraday price data (e.g., 1-minute bars) that has been
# aggregated to daily statistics. The function would compute proxies for intraday volatility
# and noise that aren't captured in daily OHLC data alone.
#
# def add_intraday_noise_proxy(
#     prices: pd.DataFrame,
#     intraday_stats_cols: list[str] | None = None,
#     group_col: str = "symbol",
#     timestamp_col: str = "timestamp",
# ) -> pd.DataFrame:
#     """
#     Add intraday noise/volatility proxies based on aggregated intraday statistics.
#
#     This function requires pre-aggregated intraday statistics (e.g., from 1-minute data
#     resampled to daily). Potential factors:
#     - Intraday realized volatility (separate from daily RV)
#     - Number of price changes per day
#     - Average bid-ask spread (if available)
#     - Tick-level volatility
#     """
#     pass
