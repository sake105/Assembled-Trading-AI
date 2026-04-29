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
        temp_df = add_moving_averages(
            temp_df, windows=(ma_window,), price_col=price_col
        )
        ma_col = f"ma_{ma_window}"
        if ma_col in temp_df.columns:
            result_df[ma_col] = temp_df[ma_col].reindex(result_df.index)
    else:
        result_df = add_moving_averages(
            result_df, windows=(ma_window,), price_col=price_col
        )

    ma_col = f"ma_{ma_window}"
    if ma_col not in result_df.columns:
        raise ValueError(f"Failed to compute moving average column {ma_col}")

    # For each timestamp, compute fraction of symbols above MA
    breadth_data = []

    for timestamp in sorted(result_df[timestamp_col].unique()):
        timestamp_data = result_df[result_df[timestamp_col] == timestamp]

        # Filter rows where both price and MA are available
        valid_mask = timestamp_data[price_col].notna() & timestamp_data[ma_col].notna()
        valid_data = timestamp_data[valid_mask]

        if len(valid_data) == 0:
            continue

        # Count symbols above MA
        above_ma = (valid_data[price_col] > valid_data[ma_col]).sum()
        total_count = len(valid_data)
        fraction = above_ma / total_count if total_count > 0 else 0.0

        breadth_data.append(
            {
                timestamp_col: timestamp,
                f"fraction_above_ma_{ma_window}": fraction,
                "count_above_ma": above_ma,
                "count_total": total_count,
            }
        )

    breadth_df = pd.DataFrame(breadth_data)

    if breadth_df.empty:
        logger.warning("No market breadth data computed. Check input data.")
        return pd.DataFrame(
            columns=[
                timestamp_col,
                f"fraction_above_ma_{ma_window}",
                "count_above_ma",
                "count_total",
            ]
        )

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

        ad_data.append(
            {
                timestamp_col: timestamp,
                "advances": advances,
                "declines": declines,
                "net_advances": net_advances,
                "count_total": len(valid_returns),
            }
        )

    ad_df = pd.DataFrame(ad_data)

    if ad_df.empty:
        logger.warning("No advance/decline data computed. Check input data.")
        return pd.DataFrame(
            columns=[
                timestamp_col,
                "advances",
                "declines",
                "net_advances",
                "ad_line",
                "ad_line_normalized",
            ]
        )

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

        risk_data.append(
            {
                timestamp_col: timestamp,
                "risk_on_ratio": risk_on_ratio,
                "risk_off_ratio": risk_off_ratio,
                "risk_on_off_score": risk_on_off_score,
                "count_total": len(valid_returns),
            }
        )

    risk_df = pd.DataFrame(risk_data)

    if risk_df.empty:
        logger.warning("No risk-on/risk-off data computed. Check input data.")
        return pd.DataFrame(
            columns=[
                timestamp_col,
                "risk_on_ratio",
                "risk_off_ratio",
                "risk_on_off_score",
            ]
        )

    # Sort by timestamp
    risk_df = risk_df.sort_values(timestamp_col).reset_index(drop=True)

    if sector_col is not None:
        logger.info(
            "Sector-based risk-on/risk-off classification not yet implemented. Using simple ratio proxy."
        )

    logger.info(f"Computed Risk-On/Risk-Off indicator for {len(risk_df)} timestamps")

    return risk_df


# ---------------------------------------------------------------------------
# Advanced Breadth Indicators
# ---------------------------------------------------------------------------


def compute_mcclellan_oscillator(
    advance_decline_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    net_advances_col: str = "net_advances",
    fast_period: int = 19,
    slow_period: int = 39,
) -> pd.DataFrame:
    """Compute the McClellan Oscillator from a net-advances series.

    The McClellan Oscillator = 19-period EMA(net_advances) - 39-period EMA(net_advances).
    Positive values indicate more advancing than declining issues on a smoothed basis.

    Args:
        advance_decline_df: Output of compute_advance_decline_line() or any
            DataFrame with timestamp + net_advances columns.
        timestamp_col: Timestamp column name.
        net_advances_col: Net advances column (advances - declines).
        fast_period: Fast EMA period (default: 19).
        slow_period: Slow EMA period (default: 39).

    Returns:
        DataFrame with columns: timestamp, mcclellan_oscillator, ema_fast, ema_slow.
    """
    df = advance_decline_df.copy()
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    if net_advances_col not in df.columns:
        logger.warning("[Breadth] %s column not found in input", net_advances_col)
        return pd.DataFrame(columns=[timestamp_col, "mcclellan_oscillator"])

    na = df[net_advances_col].astype(float)
    ema_fast = na.ewm(span=fast_period, adjust=False).mean()
    ema_slow = na.ewm(span=slow_period, adjust=False).mean()
    oscillator = ema_fast - ema_slow

    result = pd.DataFrame({
        timestamp_col: df[timestamp_col],
        "mcclellan_oscillator": oscillator.values,
        "ema_fast": ema_fast.values,
        "ema_slow": ema_slow.values,
    })
    logger.info("[Breadth] McClellan Oscillator computed for %d rows", len(result))
    return result


def compute_mcclellan_summation_index(
    mcclellan_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    oscillator_col: str = "mcclellan_oscillator",
) -> pd.DataFrame:
    """Compute the McClellan Summation Index (cumulative sum of oscillator).

    The Summation Index is a longer-term market cycle indicator — positive
    and rising = broad bull market, negative and falling = broad bear market.

    Args:
        mcclellan_df: Output of compute_mcclellan_oscillator().
        timestamp_col: Timestamp column name.
        oscillator_col: Oscillator column name.

    Returns:
        DataFrame with columns: timestamp, mcclellan_summation_index.
    """
    df = mcclellan_df.copy().sort_values(timestamp_col).reset_index(drop=True)
    if oscillator_col not in df.columns:
        return pd.DataFrame(columns=[timestamp_col, "mcclellan_summation_index"])

    summation = df[oscillator_col].cumsum()
    result = pd.DataFrame({
        timestamp_col: df[timestamp_col],
        "mcclellan_summation_index": summation.values,
    })
    logger.info("[Breadth] McClellan Summation Index computed for %d rows", len(result))
    return result


def compute_zweig_breadth_thrust(
    advance_decline_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    advances_col: str = "advances",
    declines_col: str = "declines",
    window: int = 10,
) -> pd.DataFrame:
    """Compute the Zweig Breadth Thrust indicator.

    Zweig Breadth Thrust = 10-day EMA of (Advances / (Advances + Declines)).
    A "thrust" event occurs when the ratio moves from below 0.40 to above 0.615
    within 10 trading days — historically one of the strongest bull market signals.

    Args:
        advance_decline_df: DataFrame with advances and declines columns.
        timestamp_col: Timestamp column name.
        advances_col: Advancing issues column name.
        declines_col: Declining issues column name.
        window: EMA smoothing window (default: 10).

    Returns:
        DataFrame with columns: timestamp, zweig_bt_ratio, zweig_bt_ema, zweig_thrust_signal.
    """
    df = advance_decline_df.copy().sort_values(timestamp_col).reset_index(drop=True)
    required = [advances_col, declines_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("[Breadth] Zweig thrust: missing columns %s", missing)
        return pd.DataFrame(columns=[timestamp_col, "zweig_bt_ema"])

    adv = df[advances_col].astype(float)
    dec = df[declines_col].astype(float)
    total = adv + dec
    ratio = adv / total.replace(0, float("nan"))
    ema = ratio.ewm(span=window, adjust=False).mean()

    # Thrust signal: ema crosses from < 0.40 to > 0.615 within window days
    below_threshold = (ema < 0.40).astype(int)
    above_threshold = (ema > 0.615).astype(int)
    had_low_recently = below_threshold.rolling(window, min_periods=1).max().astype(int)
    thrust_signal = (above_threshold & had_low_recently).astype(float)

    result = pd.DataFrame({
        timestamp_col: df[timestamp_col],
        "zweig_bt_ratio": ratio.values,
        "zweig_bt_ema": ema.values,
        "zweig_thrust_signal": thrust_signal.values,
    })
    logger.info("[Breadth] Zweig Breadth Thrust computed for %d rows", len(result))
    return result


def compute_new_highs_minus_new_lows(
    prices: pd.DataFrame,
    lookback: int = 252,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    close_col: str = "close",
) -> pd.DataFrame:
    """Compute New Highs minus New Lows breadth indicator.

    For each date, counts symbols at a 52-week (252-day) high vs. low.
    Net reading = (new highs - new lows) / total symbols.

    Args:
        prices: Daily OHLCV panel.
        lookback: Rolling window to define highs/lows (default: 252 days = 52 weeks).
        timestamp_col: Timestamp column name.
        symbol_col: Symbol column name.
        close_col: Close price column name.

    Returns:
        DataFrame with columns: timestamp, new_highs, new_lows, nh_nl_net, nh_nl_ratio.
    """
    df = prices.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values([symbol_col, timestamp_col])

    def _at_high_low(grp: pd.DataFrame) -> pd.DataFrame:
        closes = grp.set_index(timestamp_col)[close_col].sort_index()
        roll_max = closes.rolling(lookback, min_periods=lookback // 4).max()
        roll_min = closes.rolling(lookback, min_periods=lookback // 4).min()
        at_high = (closes >= roll_max).astype(int)
        at_low = (closes <= roll_min).astype(int)
        return pd.DataFrame({
            timestamp_col: closes.index,
            symbol_col: grp[symbol_col].iloc[0],
            "at_52w_high": at_high.values,
            "at_52w_low": at_low.values,
        })

    pieces = [_at_high_low(g) for _, g in df.groupby(symbol_col)]
    if not pieces:
        return pd.DataFrame(columns=[timestamp_col, "new_highs", "new_lows"])

    combined = pd.concat(pieces, ignore_index=True)
    daily = (
        combined.groupby(timestamp_col)
        .agg(new_highs=("at_52w_high", "sum"), new_lows=("at_52w_low", "sum"), total=("at_52w_high", "count"))
        .reset_index()
    )
    daily["nh_nl_net"] = daily["new_highs"] - daily["new_lows"]
    daily["nh_nl_ratio"] = daily["nh_nl_net"] / daily["total"].replace(0, float("nan"))
    logger.info("[Breadth] New Highs/Lows computed for %d dates", len(daily))
    return daily.sort_values(timestamp_col).reset_index(drop=True)


def compute_arms_index(
    advance_decline_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    advances_col: str = "advances",
    declines_col: str = "declines",
    adv_volume_col: str = "advancing_volume",
    dec_volume_col: str = "declining_volume",
) -> pd.DataFrame:
    """Compute the Arms Index (TRIN — TRading INdex).

    TRIN = (Advances/Declines) / (Advancing_Volume/Declining_Volume).
    TRIN < 1.0 = bullish (more volume in advancing stocks), > 1.0 = bearish.
    Extreme readings (< 0.5 or > 2.0) signal potential reversal.

    Args:
        advance_decline_df: DataFrame with advances, declines, and volume breakdowns.
        timestamp_col: Timestamp column name.
        advances_col: Advancing issues column.
        declines_col: Declining issues column.
        adv_volume_col: Advancing volume column.
        dec_volume_col: Declining volume column.

    Returns:
        DataFrame with columns: timestamp, arms_index, arms_index_ma_10d.
        If volume columns are missing, returns approximate breadth-only proxy.
    """
    df = advance_decline_df.copy().sort_values(timestamp_col).reset_index(drop=True)

    adv = df[advances_col].astype(float) if advances_col in df.columns else None
    dec = df[declines_col].astype(float) if declines_col in df.columns else None

    if adv is None or dec is None:
        logger.warning("[Breadth] Arms Index: advances/declines columns missing")
        return pd.DataFrame(columns=[timestamp_col, "arms_index"])

    ad_ratio = adv / dec.replace(0, float("nan"))

    if adv_volume_col in df.columns and dec_volume_col in df.columns:
        adv_vol = df[adv_volume_col].astype(float)
        dec_vol = df[dec_volume_col].astype(float)
        vol_ratio = adv_vol / dec_vol.replace(0, float("nan"))
        arms = ad_ratio / vol_ratio.replace(0, float("nan"))
    else:
        # Approximate: use simple breadth ratio when volume breakdown unavailable
        logger.warning("[Breadth] Arms Index: volume columns missing — using breadth-only proxy")
        arms = ad_ratio

    arms_ma = arms.rolling(10, min_periods=5).mean()

    result = pd.DataFrame({
        timestamp_col: df[timestamp_col],
        "arms_index": arms.values,
        "arms_index_ma_10d": arms_ma.values,
    })
    logger.info("[Breadth] Arms Index (TRIN) computed for %d rows", len(result))
    return result


def hindenburg_omen(
    new_highs: pd.Series,
    new_lows: pd.Series,
    nyse_ma50: pd.Series,
    mcclellan_osc: pd.Series,
    total_issues: pd.Series | None = None,
    threshold_pct: float = 0.028,
) -> pd.Series:
    """Compute the Hindenburg Omen composite signal.

    Conditions (all must hold on same day):
    1. New 52w highs AND new 52w lows BOTH > threshold_pct of total NYSE issues
    2. NYSE (or index) is above its 50-day moving average
    3. McClellan Oscillator is negative
    4. New 52w highs ≤ 2 × new 52w lows

    Returns a boolean Series: True = omen triggered.

    Parameters
    ----------
    new_highs, new_lows:
        Daily new 52-week high/low counts (or fractions if total_issues is None).
    nyse_ma50:
        NYSE/SPX index value vs. its 50-day MA ratio (> 1 means above MA).
    mcclellan_osc:
        McClellan Oscillator values (negative = bearish breadth).
    total_issues:
        Total NYSE issues. If None, new_highs/new_lows are treated as fractions.
    threshold_pct:
        Default 2.8% as per Miekka original definition.
    """
    import numpy as np  # noqa: PLC0415

    if total_issues is not None:
        nh_pct = new_highs / total_issues.replace(0, np.nan)
        nl_pct = new_lows / total_issues.replace(0, np.nan)
    else:
        nh_pct = new_highs
        nl_pct = new_lows

    cond1 = (nh_pct > threshold_pct) & (nl_pct > threshold_pct)
    cond2 = nyse_ma50 > 1.0
    cond3 = mcclellan_osc < 0
    cond4 = new_highs <= 2 * new_lows

    omen = cond1 & cond2 & cond3 & cond4
    omen.name = "hindenburg_omen"
    logger.info("[Breadth] Hindenburg Omen computed: %d signals", int(omen.sum()))
    return omen


def compute_cbbi_composite(
    indicators: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Composite Bull-Bear Indicator (CBBI) — aggregate up to 10 breadth indicators.

    Parameters
    ----------
    indicators:
        Dict of name → pd.Series (all normalised 0-1 or z-scored).
    weights:
        Optional dict of name → weight. Defaults to equal weight.

    Returns
    -------
    pd.Series of composite score (0-100), indexed to intersection of all series.
    """
    import pandas as pd  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if not indicators:
        return pd.Series(dtype=float, name="cbbi")

    w = weights or {k: 1.0 / len(indicators) for k in indicators}
    aligned = pd.DataFrame(indicators).dropna()
    if aligned.empty:
        return pd.Series(dtype=float, name="cbbi")

    score = sum(aligned[k] * w.get(k, 1.0 / len(indicators)) for k in aligned.columns)
    normalised = (score - score.min()) / (score.max() - score.min() + 1e-9) * 100
    normalised.name = "cbbi"
    logger.info("[Breadth] CBBI composite computed for %d rows", len(normalised))
    return normalised
