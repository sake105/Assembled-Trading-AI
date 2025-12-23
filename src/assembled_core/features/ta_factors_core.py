"""Core TA/Price Factors module.

This module implements Phase A, Sprint A1 from the Advanced Analytics & Factor Labs roadmap.
It provides core technical analysis factors that build upon the existing TA features:

- Multi-Horizon Returns (1/3/6/12 months)
- Time-Series Trend Strength (Price vs. MA / ATR)
- Short-Term Reversal (1-3 days)

All factors are designed to work with the standard price data format:
- Columns: timestamp (UTC), symbol, close (required), optional: open, high, low, volume
- Sorted by symbol, then timestamp
- Panel format (multiple symbols over time)

Integration:
- Builds on existing ta_features.py functions (add_moving_averages, add_atr)
- Designed for factor research and ML feature engineering
- Compatible with backtest engine and EOD pipeline
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_core_ta_factors(
    prices: pd.DataFrame,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Build core TA/Price factors from price DataFrame.

    This function computes a comprehensive set of price-based factors:

    1. **Multi-Horizon Returns:**
       - returns_1m: 1-month forward returns (~21 trading days)
       - returns_3m: 3-month forward returns (~63 trading days)
       - returns_6m: 6-month forward returns (~126 trading days)
       - returns_12m: 12-month forward returns (~252 trading days)
       - momentum_12m_excl_1m: 12-month momentum excluding last month

    2. **Time-Series Trend Strength:**
       - trend_strength_20: (price - MA_20) / ATR_20
       - trend_strength_50: (price - MA_50) / ATR_20
       - trend_strength_200: (price - MA_200) / ATR_20

    3. **Short-Term Reversal:**
       - reversal_1d: z-score of 1-day returns
       - reversal_2d: z-score of 2-day returns
       - reversal_3d: z-score of 3-day returns

    Args:
        prices: DataFrame with price data
            Required columns: timestamp_col, group_col, price_col
            Optional columns: high, low, volume (for ATR and enhanced features)
        price_col: Column name for price data (default: "close")
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")

    Returns:
        DataFrame with original columns plus new factor columns:
        - returns_1m, returns_3m, returns_6m, returns_12m
        - momentum_12m_excl_1m
        - trend_strength_{lookback} for lookback in [20, 50, 200]
        - reversal_{horizon}d for horizon in [1, 2, 3]

        Sorted by group_col, then timestamp_col.
        Factors are computed per group (symbol).

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

    result = prices.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)

    # Sort by group and timestamp
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)

    # Compute factors in sequence (each builds on previous)

    # 1. Multi-Horizon Returns
    result = _add_multi_horizon_returns(
        result,
        price_col=price_col,
        group_col=group_col,
        timestamp_col=timestamp_col,
    )

    # 2. Time-Series Trend Strength
    result = _add_trend_strength_factors(
        result,
        price_col=price_col,
        group_col=group_col,
        timestamp_col=timestamp_col,
    )

    # 3. Short-Term Reversal
    result = _add_short_term_reversal(
        result,
        price_col=price_col,
        group_col=group_col,
    )

    logger.info(
        f"Built core TA factors for {result[group_col].nunique()} symbols, "
        f"{len(result)} rows. Added factor columns."
    )

    return result


def _add_multi_horizon_returns(
    df: pd.DataFrame,
    price_col: str,
    group_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """Add multi-horizon forward returns.

    Computes forward returns for 1, 3, 6, and 12 months.
    For daily data, assumes ~21 trading days per month.

    Also computes momentum_12m_excl_1m (12-month return excluding last month).
    """
    result = df.copy()

    # Forward returns (looking ahead)
    # For daily data: 1m ≈ 21 days, 3m ≈ 63 days, 6m ≈ 126 days, 12m ≈ 252 days
    horizons = {
        "returns_1m": 21,
        "returns_3m": 63,
        "returns_6m": 126,
        "returns_12m": 252,
    }

    # Group by symbol and compute forward returns
    grouped = result.groupby(group_col, group_keys=False)[price_col]

    for factor_name, periods in horizons.items():
        # Forward return: log(price[t+periods] / price[t])
        # Use shift(-periods) to look forward
        future_price = grouped.shift(-periods)
        current_price = result[price_col]

        # Log return
        log_return = np.log(future_price / current_price)
        result[factor_name] = log_return.astype("float64")

    # Momentum 12m excluding last month: 11-month return starting 1 month ahead
    # This is: log(price[t+12m] / price[t+1m])
    price_12m = grouped.shift(-252)
    price_1m = grouped.shift(-21)

    # Only compute where both are available
    mask = price_12m.notna() & price_1m.notna()
    result["momentum_12m_excl_1m"] = np.where(
        mask,
        np.log(price_12m / price_1m),
        np.nan,
    ).astype("float64")

    return result


def _add_trend_strength_factors(
    df: pd.DataFrame,
    price_col: str,
    group_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """Add trend strength factors: (price - MA) / ATR.

    Computes normalized trend strength using moving averages and ATR
    for different lookback windows (20, 50, 200 days).
    """
    from src.assembled_core.features.ta_features import add_moving_averages, add_atr

    result = df.copy()

    # Check if we have high/low for ATR (required for proper ATR)
    # Note: ATR function expects "high", "low", "close" as default column names
    has_ohlc = all(col in result.columns for col in ["high", "low", "close"])

    if not has_ohlc:
        logger.warning(
            "Missing high/low columns for ATR. Trend strength factors will use "
            "close-based volatility approximation instead."
        )
        # Fallback: use rolling std of returns as volatility proxy
        grouped_close = result.groupby(group_col, group_keys=False)[price_col]
        returns = grouped_close.pct_change()
        atr_proxy = (
            returns.groupby(result[group_col], group_keys=False)
            .rolling(window=20, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            * result[price_col]
        )
        atr_col = atr_proxy
    else:
        # Use proper ATR (reuse existing function)
        # Note: add_atr expects "symbol", "high", "low", "close" as column names
        # We need to temporarily rename columns if custom names are used
        if group_col != "symbol" or price_col != "close":
            temp_result = result.copy()
            if group_col != "symbol":
                temp_result = temp_result.rename(columns={group_col: "symbol"})
            if price_col != "close":
                temp_result = temp_result.rename(columns={price_col: "close"})
            temp_df = add_atr(temp_result, window=20)
            atr_col = temp_df["atr_20"].values  # Extract as array to avoid index issues
        else:
            temp_df = add_atr(result, window=20)
            atr_col = temp_df["atr_20"]

    # Compute moving averages for all lookback windows at once (reuse existing function)
    # Note: add_moving_averages expects "timestamp", "symbol", and price_col
    # We need to temporarily rename columns if custom names are used
    lookback_windows = [20, 50, 200]

    if timestamp_col != "timestamp" or group_col != "symbol":
        # Temporarily rename columns for add_moving_averages
        temp_result = result.copy()
        if timestamp_col != "timestamp":
            temp_result = temp_result.rename(columns={timestamp_col: "timestamp"})
        if group_col != "symbol":
            temp_result = temp_result.rename(columns={group_col: "symbol"})
        temp_df = add_moving_averages(
            temp_result, windows=lookback_windows, price_col=price_col
        )
        # Extract MA columns
        ma_cols = {f"ma_{lb}": temp_df[f"ma_{lb}"].values for lb in lookback_windows}
    else:
        temp_df = add_moving_averages(
            result, windows=lookback_windows, price_col=price_col
        )
        ma_cols = {f"ma_{lb}": temp_df[f"ma_{lb}"] for lb in lookback_windows}

    # Trend strength factors for different MA windows
    for lookback in lookback_windows:
        ma_col_name = f"ma_{lookback}"
        ma_values = ma_cols[ma_col_name]

        # Trend strength: (price - MA) / ATR
        price_ma_diff = (
            result[price_col].values - ma_values
            if isinstance(ma_values, np.ndarray)
            else result[price_col] - ma_values
        )
        atr_values = atr_col.values if hasattr(atr_col, "values") else atr_col
        trend_strength = price_ma_diff / atr_values

        # Handle division by zero
        trend_strength = np.where(atr_values > 1e-10, trend_strength, 0.0)

        result[f"trend_strength_{lookback}"] = trend_strength.astype("float64")

    return result


def _add_short_term_reversal(
    df: pd.DataFrame,
    price_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Add short-term reversal factors (z-scored returns over 1-3 days).

    Computes rolling z-scores of returns to capture mean-reversion patterns.
    """
    result = df.copy()

    # Compute returns per symbol
    grouped_price = result.groupby(group_col, group_keys=False)[price_col]
    grouped_price.pct_change()

    # Z-score window (typically 20-60 days for daily data)
    zscore_window = 60

    # Compute z-scores of returns for different horizons
    horizons = [1, 2, 3]

    for horizon in horizons:
        # Multi-day return (cumulative)
        multi_day_return = grouped_price.pct_change(periods=horizon)

        # Compute rolling mean and std per symbol
        rolling_mean = (
            multi_day_return.groupby(result[group_col], group_keys=False)
            .rolling(window=zscore_window, min_periods=10)
            .mean()
            .reset_index(level=0, drop=True)
        )

        rolling_std = (
            multi_day_return.groupby(result[group_col], group_keys=False)
            .rolling(window=zscore_window, min_periods=10)
            .std()
            .reset_index(level=0, drop=True)
        )

        # Z-score: (return - mean) / std
        zscore = (multi_day_return - rolling_mean) / rolling_std

        # Handle division by zero (when std is very small)
        zscore = np.where(rolling_std > 1e-10, zscore, 0.0)

        result[f"reversal_{horizon}d"] = zscore.astype("float64")

    return result
