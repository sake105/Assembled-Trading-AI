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
    log_price = np.log(temp[price_col].astype("float64").clip(lower=1e-10))

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


# ---------------------------------------------------------------------------
# Market Microstructure Indicators (Phase 3.3)
# ---------------------------------------------------------------------------


def add_amihud_illiquidity(
    prices: pd.DataFrame,
    windows: list | None = None,
    close_col: str = "close",
    volume_col: str = "volume",
    group_col: str = "symbol",
) -> pd.DataFrame:
    """Amihud (2002) illiquidity ratio: rolling mean of |return| / dollar_volume.

    Higher values indicate less liquid stocks.
    Output columns: amihud_illiq_{w}d for each window.
    """
    if windows is None:
        windows = [20, 60]
    result = prices.copy()
    if close_col not in result.columns or volume_col not in result.columns:
        logger.warning("[Amihud] Required columns missing — skipping")
        return result

    abs_ret = result.groupby(group_col)[close_col].transform(
        lambda x: x.pct_change().abs()
    )
    dollar_vol = (result[close_col] * result[volume_col]).replace(0, np.nan)
    result["_amihud_daily"] = (abs_ret / dollar_vol).astype("float64")

    for w in windows:
        col = f"amihud_illiq_{w}d"
        result[col] = (
            result.groupby(group_col)["_amihud_daily"]
            .transform(lambda x: x.rolling(w, min_periods=max(1, w // 4)).mean())
            .astype("float64")
        )

    result = result.drop(columns=["_amihud_daily"])
    return result


def add_roll_spread_estimate(
    prices: pd.DataFrame,
    windows: list | None = None,
    close_col: str = "close",
    group_col: str = "symbol",
) -> pd.DataFrame:
    """Roll (1984) implicit bid-ask spread: 2 * sqrt(max(-Cov(r_t, r_{t-1}), 0)).

    Output columns: roll_spread_{w}d for each window.
    """
    if windows is None:
        windows = [20]
    result = prices.copy()
    if close_col not in result.columns:
        return result

    ret = result.groupby(group_col)[close_col].transform(
        lambda x: x.pct_change().fillna(0.0)
    )
    result["_ret"] = ret

    for w in windows:
        col = f"roll_spread_{w}d"

        def _spread(series: pd.Series) -> pd.Series:
            ret_lag = series.shift(1)
            cov = series.rolling(w).cov(ret_lag)
            return (2 * ((-cov).clip(lower=0) ** 0.5)).astype("float64")

        result[col] = result.groupby(group_col)["_ret"].transform(_spread)

    result = result.drop(columns=["_ret"])
    return result


def add_kyle_lambda_proxy(
    prices: pd.DataFrame,
    windows: list | None = None,
    close_col: str = "close",
    volume_col: str = "volume",
    group_col: str = "symbol",
) -> pd.DataFrame:
    """Kyle lambda proxy: rolling mean of |return| / sqrt(dollar_volume).

    Higher = less liquid. Output columns: kyle_lambda_{w}d.
    """
    if windows is None:
        windows = [20]
    result = prices.copy()
    if close_col not in result.columns or volume_col not in result.columns:
        return result

    abs_ret = result.groupby(group_col)[close_col].transform(
        lambda x: x.pct_change().abs()
    )
    dollar_vol_sqrt = (result[close_col] * result[volume_col]).clip(lower=1e-9) ** 0.5
    result["_lambda_daily"] = (abs_ret / dollar_vol_sqrt).astype("float64")

    for w in windows:
        col = f"kyle_lambda_{w}d"
        result[col] = (
            result.groupby(group_col)["_lambda_daily"]
            .transform(lambda x: x.rolling(w, min_periods=max(1, w // 4)).mean())
            .astype("float64")
        )

    result = result.drop(columns=["_lambda_daily"])
    return result


def add_tick_rule_imbalance(
    prices: pd.DataFrame,
    windows: list | None = None,
    open_col: str = "open",
    close_col: str = "close",
    volume_col: str = "volume",
    group_col: str = "symbol",
) -> pd.DataFrame:
    """Tick-rule order flow imbalance: buy_vol / (buy_vol + sell_vol).

    close > open → buy volume; close < open → sell volume.
    Values > 0.5 = net buying pressure. Output columns: tick_imbalance_{w}d.
    """
    if windows is None:
        windows = [5, 20]
    result = prices.copy()
    if open_col not in result.columns or volume_col not in result.columns:
        return result

    direction = (result[close_col] - result[open_col]).clip(-1, 1)
    result["_buy_vol"] = result[volume_col] * (direction > 0).astype(float)
    result["_sell_vol"] = result[volume_col] * (direction < 0).astype(float)

    for w in windows:
        col = f"tick_imbalance_{w}d"

        def _imb(grp: pd.DataFrame) -> pd.Series:
            bv = grp["_buy_vol"].rolling(w, min_periods=1).sum()
            sv = grp["_sell_vol"].rolling(w, min_periods=1).sum()
            total = (bv + sv).replace(0, np.nan)
            return (bv / total).astype("float64")

        result[col] = (
            result.groupby(group_col, group_keys=False)
            .apply(_imb, include_groups=False)
            .reset_index(level=0, drop=True)
        )

    result = result.drop(columns=["_buy_vol", "_sell_vol"])
    return result


def add_abnormal_volume(
    prices: pd.DataFrame,
    windows: list | None = None,
    volume_col: str = "volume",
    group_col: str = "symbol",
) -> pd.DataFrame:
    """Abnormal volume ratio: current volume / rolling mean volume.

    Values > 2 indicate unusual activity. Output columns: abnormal_vol_{w}d.
    """
    if windows is None:
        windows = [20]
    result = prices.copy()
    if volume_col not in result.columns:
        return result

    for w in windows:
        col = f"abnormal_vol_{w}d"
        mean_vol = result.groupby(group_col)[volume_col].transform(
            lambda x: x.rolling(w, min_periods=max(1, w // 4)).mean()
        )
        result[col] = (result[volume_col] / mean_vol.replace(0, np.nan)).astype(
            "float64"
        )

    return result


def add_intraday_noise_proxy(
    prices: pd.DataFrame,
    intraday_stats_cols: list[str] | None = None,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add intraday noise/volatility proxies derived from OHLC data.

    Computes three proxies that capture intraday dispersion beyond daily close-to-close returns:

    - ``intraday_hl_ratio``: (high - low) / close — relative intraday range; higher = more noise
    - ``intraday_oc_return``: (close - open) / open — intraday directional drift
    - ``intraday_gk_vol``: Garman-Klass volatility estimator using OHLC; more efficient than
      close-to-close vol for measuring realised intraday variance

    If ``intraday_stats_cols`` is provided, those columns are passed through unchanged
    (assumed to be pre-aggregated statistics, e.g., realised vol from 1-minute bars).

    Args:
        prices: DataFrame with OHLC columns (``open``, ``high``, ``low``, ``close``).
            Must also contain ``group_col`` and ``timestamp_col``.
        intraday_stats_cols: Optional list of pre-aggregated intraday stats columns already
            present in ``prices`` to preserve. No transformation is applied to these.
        group_col: Symbol grouping column (default ``"symbol"``).
        timestamp_col: Timestamp column name (default ``"timestamp"``).

    Returns:
        Copy of ``prices`` with additional columns (if OHLC available):
        ``intraday_hl_ratio``, ``intraday_oc_return``, ``intraday_gk_vol``.
    """
    result = prices.copy()
    has_ohlc = all(c in result.columns for c in ("open", "high", "low", "close"))

    if has_ohlc:
        close = result["close"].replace(0, np.nan)
        open_ = result["open"].replace(0, np.nan)
        high = result["high"]
        low = result["low"]

        # Relative intraday range — higher means more intraday noise
        result["intraday_hl_ratio"] = ((high - low) / close).astype("float64")

        # Intraday directional drift (open → close)
        result["intraday_oc_return"] = ((close - open_) / open_).astype("float64")

        # Garman-Klass estimator: 0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2
        log_hl = np.log((high / low).clip(lower=1e-10))
        log_co = np.log((close / open_).clip(lower=1e-10))
        gk = 0.5 * log_hl**2 - (2.0 * np.log(2) - 1.0) * log_co**2
        result["intraday_gk_vol"] = np.sqrt(gk.clip(lower=0)).astype("float64")

    return result
