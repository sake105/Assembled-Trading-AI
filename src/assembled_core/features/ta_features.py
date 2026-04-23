"""Technical analysis features module.

This module provides functions to compute technical indicators (SMA, ATR, RSI, log returns, etc.)
from price data. It extends the basic EMA functionality from pipeline.signals.

Zukünftige Integration:
- Nutzt pipeline.signals.compute_ema_signals als Basis für EMA
- Erweitert um weitere TA-Indikatoren (SMA, ATR, RSI, MACD, Bollinger Bands, etc.)
- Bietet Feature-Engineering-Pipeline für ML-Modelle
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_log_returns(
    df: pd.DataFrame,
    price_col: str = "close",
    out_col: str | None = None,
    use_namespace: bool = True,
) -> pd.DataFrame:
    """
    Füge logarithmische Returns pro Symbol hinzu.

    Erwartet:
    - Spalte 'symbol'
    - Spalte `price_col` (z.B. 'close')
    - Optional: 'timestamp' für zeitliche Sortierung

    Args:
        df: DataFrame mit Spalten: symbol, price_col
        price_col: Name der Preis-Spalte (default: "close")
        out_col: Optional output column name (default: "ta_log_return_v1" if use_namespace, else "log_return")
        use_namespace: If True, use namespaced feature name (default: True)

    Rückgabe:
    - DataFrame mit neuer Spalte (namespaced: "ta_log_return_v1" oder legacy: "log_return")
    """
    if "symbol" not in df.columns:
        raise KeyError("symbol")
    if price_col not in df.columns:
        raise KeyError(
            f"Price column '{price_col}' not found. Available columns: {list(df.columns)}"
        )

    # Use namespaced name by default (Sprint 5 / F2)
    if out_col is None:
        out_col = "ta_log_return_v1" if use_namespace else "log_return"

    result = df.copy()

    # Für stabile Berechnung nach Zeit sortieren
    sort_cols = ["symbol"]
    if "timestamp" in result.columns:
        sort_cols.append("timestamp")

    tmp = result.sort_values(sort_cols)

    # Log-Preis & Differenz pro Symbol
    log_price = np.log(tmp[price_col].astype("float64"))
    log_ret = log_price.groupby(tmp["symbol"]).diff()

    # Zurück in die ursprüngliche Index-Reihenfolge
    log_ret = log_ret.reindex(result.index)

    result[out_col] = log_ret.astype("float64")

    # Compatibility: also add legacy name if using namespace (deprecation)
    if (
        use_namespace
        and out_col == "ta_log_return_v1"
        and "log_return" not in result.columns
    ):
        result["log_return"] = result[out_col]

    return result


def add_moving_averages(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (20, 50, 200),
    price_col: str = "close",
    use_namespace: bool = True,
) -> pd.DataFrame:
    """Add Simple Moving Averages (SMA) to price DataFrame.

    Computes SMA for each window: SMA(window) = mean(price over window periods)

    Args:
        df: DataFrame with columns: timestamp, symbol, and price_col
        windows: Tuple of window sizes (default: (20, 50, 200))
        price_col: Column name for price data (default: "close")
        use_namespace: If True, use namespaced feature names (default: True)

    Returns:
        DataFrame with additional columns: ta_ma_{window}_v1 (or ma_{window} if use_namespace=False)
        Sorted by symbol, then timestamp

    Raises:
        KeyError: If required columns are missing
    """
    df = df.copy()

    # Ensure required columns
    if price_col not in df.columns:
        raise KeyError(
            f"Price column '{price_col}' not found. Available columns: {list(df.columns)}"
        )

    # Sort by symbol and timestamp
    if "symbol" in df.columns:
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute SMA for each window per symbol
    for window in windows:
        # Use namespaced name by default (Sprint 5 / F2)
        col_name = f"ta_ma_{window}_v1" if use_namespace else f"ma_{window}"
        if "symbol" in df.columns:
            df[col_name] = (
                df.groupby("symbol", group_keys=False)[price_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(drop=True)
            )
        else:
            df[col_name] = df[price_col].rolling(window=window, min_periods=1).mean()

        # Compatibility: also add legacy name if using namespace (deprecation)
        if (
            use_namespace
            and col_name.startswith("ta_")
            and f"ma_{window}" not in df.columns
        ):
            legacy_name = col_name.replace("ta_", "").replace("_v1", "")
            df[legacy_name] = df[col_name]

    return df


def add_atr(
    df: pd.DataFrame,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Füge Average True Range (ATR) pro Symbol hinzu.

    Erwartet:
    - Spalten: 'symbol', high_col, low_col, close_col
    - Optional: 'timestamp'

    Rückgabe:
    - DataFrame mit neuer Spalte f"atr_{window}"
    """
    required_cols = ["symbol", high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Tests erwarten KeyError mit "Missing required columns"
        raise KeyError(f"Missing required columns: {', '.join(missing_cols)}")

    result = df.copy()

    sort_cols = ["symbol"]
    if "timestamp" in result.columns:
        sort_cols.append("timestamp")

    tmp = result.sort_values(sort_cols)

    high = tmp[high_col].astype("float64")
    low = tmp[low_col].astype("float64")
    close = tmp[close_col].astype("float64")

    prev_close = close.groupby(tmp["symbol"]).shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # min_periods=1, damit nicht nur NaNs am Anfang sind
    atr = (
        true_range.groupby(tmp["symbol"])
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # zurück auf ursprünglichen Index
    atr = atr.reindex(result.index)

    # Use namespaced name by default (Sprint 5 / F2)
    col_name = f"ta_atr_{window}_v1"
    result[col_name] = atr.astype("float64")

    # Compatibility: also add legacy name (deprecation)
    legacy_name = f"atr_{window}"
    if legacy_name not in result.columns:
        result[legacy_name] = result[col_name]

    return result


def add_rsi(
    df: pd.DataFrame,
    window: int = 14,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Füge einen klassischen RSI (Wilder) pro Symbol hinzu.

    Erwartet:
    - Spalten: 'symbol', price_col
    - Optional: 'timestamp'

    Rückgabe:
    - DataFrame mit neuer Spalte f"rsi_{window}"
    """
    if "symbol" not in df.columns:
        raise KeyError("symbol")
    if price_col not in df.columns:
        raise KeyError(price_col)

    result = df.copy()

    sort_cols = ["symbol"]
    if "timestamp" in result.columns:
        sort_cols.append("timestamp")

    tmp = result.sort_values(sort_cols)

    close = tmp[price_col].astype("float64")

    delta = close.groupby(tmp["symbol"]).diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder-RSI klassisch mit gleitendem Mittel
    avg_gain = (
        gain.groupby(tmp["symbol"])
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    avg_loss = (
        loss.groupby(tmp["symbol"])
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    rs = avg_gain / avg_loss

    rsi = 100.0 - 100.0 / (1.0 + rs)

    rsi = rsi.reindex(result.index)

    # Use namespaced name by default (Sprint 5 / F2)
    col_name = f"ta_rsi_{window}_v1"
    result[col_name] = rsi.astype("float64")

    # Compatibility: also add legacy name (deprecation)
    legacy_name = f"rsi_{window}"
    if legacy_name not in result.columns:
        result[legacy_name] = result[col_name]

    return result


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add MACD (Moving Average Convergence Divergence) per symbol.

    Computes:
    - MACD line = EMA(fast) - EMA(slow)
    - Signal line = EMA(signal) of MACD line
    - Histogram = MACD - Signal

    Args:
        df: DataFrame with columns: symbol, price_col, timestamp
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal EMA period (default: 9)
        price_col: Price column name (default: "close")

    Returns:
        DataFrame with columns: ta_macd_v1, ta_macd_signal_v1, ta_macd_hist_v1
    """
    if "symbol" not in df.columns:
        raise KeyError("symbol")
    if price_col not in df.columns:
        raise KeyError(price_col)

    result = df.copy()
    sort_cols = ["symbol"]
    if "timestamp" in result.columns:
        sort_cols.append("timestamp")
    result = result.sort_values(sort_cols).reset_index(drop=True)

    close = result[price_col].astype("float64")

    ema_fast = close.groupby(result["symbol"]).transform(
        lambda x: x.ewm(span=fast, adjust=False).mean()
    )
    ema_slow = close.groupby(result["symbol"]).transform(
        lambda x: x.ewm(span=slow, adjust=False).mean()
    )

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.groupby(result["symbol"]).transform(
        lambda x: x.ewm(span=signal, adjust=False).mean()
    )
    histogram = macd_line - signal_line

    result["ta_macd_v1"] = macd_line.astype("float64")
    result["ta_macd_signal_v1"] = signal_line.astype("float64")
    result["ta_macd_hist_v1"] = histogram.astype("float64")

    return result


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add Bollinger Bands per symbol.

    Computes:
    - Middle band = SMA(window)
    - Upper band = Middle + num_std * StdDev(window)
    - Lower band = Middle - num_std * StdDev(window)
    - %B = (price - lower) / (upper - lower)
    - Bandwidth = (upper - lower) / middle

    Args:
        df: DataFrame with columns: symbol, price_col, timestamp
        window: SMA window (default: 20)
        num_std: Number of standard deviations (default: 2.0)
        price_col: Price column name (default: "close")

    Returns:
        DataFrame with columns: ta_bb_upper_v1, ta_bb_lower_v1, ta_bb_pctb_v1, ta_bb_bandwidth_v1
    """
    if "symbol" not in df.columns:
        raise KeyError("symbol")
    if price_col not in df.columns:
        raise KeyError(price_col)

    result = df.copy()
    sort_cols = ["symbol"]
    if "timestamp" in result.columns:
        sort_cols.append("timestamp")
    result = result.sort_values(sort_cols).reset_index(drop=True)

    close = result[price_col].astype("float64")

    sma = close.groupby(result["symbol"]).transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    std = close.groupby(result["symbol"]).transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )

    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = upper - lower

    # %B: position within bands (0 = lower, 1 = upper)
    pct_b = (close - lower) / band_width.replace(0, np.nan)
    # Bandwidth: width relative to middle
    bandwidth = band_width / sma.replace(0, np.nan)

    result["ta_bb_upper_v1"] = upper.astype("float64")
    result["ta_bb_lower_v1"] = lower.astype("float64")
    result["ta_bb_pctb_v1"] = pct_b.astype("float64")
    result["ta_bb_bandwidth_v1"] = bandwidth.astype("float64")

    return result


def add_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """Add Stochastic Oscillator (%K, %D) per symbol.

    Computes:
    - %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    - %D = SMA(%K, d_period)

    Args:
        df: DataFrame with columns: symbol, high_col, low_col, close_col, timestamp
        k_period: Lookback period for %K (default: 14)
        d_period: Smoothing period for %D (default: 3)

    Returns:
        DataFrame with columns: ta_stoch_k_v1, ta_stoch_d_v1
    """
    required = ["symbol", high_col, low_col, close_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    result = df.copy()
    sort_cols = ["symbol"]
    if "timestamp" in result.columns:
        sort_cols.append("timestamp")
    result = result.sort_values(sort_cols).reset_index(drop=True)

    high = result[high_col].astype("float64")
    low = result[low_col].astype("float64")
    close = result[close_col].astype("float64")

    highest_high = high.groupby(result["symbol"]).transform(
        lambda x: x.rolling(window=k_period, min_periods=1).max()
    )
    lowest_low = low.groupby(result["symbol"]).transform(
        lambda x: x.rolling(window=k_period, min_periods=1).min()
    )

    hl_range = highest_high - lowest_low
    pct_k = (close - lowest_low) / hl_range.replace(0, np.nan) * 100.0
    pct_d = pct_k.groupby(result["symbol"]).transform(
        lambda x: x.rolling(window=d_period, min_periods=1).mean()
    )

    result["ta_stoch_k_v1"] = pct_k.astype("float64")
    result["ta_stoch_d_v1"] = pct_d.astype("float64")

    return result


def add_adx(
    df: pd.DataFrame,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """Add Average Directional Index (ADX) per symbol.

    Measures trend strength (not direction). ADX > 25 suggests a strong trend.

    Args:
        df: DataFrame with columns: symbol, high_col, low_col, close_col, timestamp
        window: Smoothing period (default: 14)

    Returns:
        DataFrame with columns: ta_adx_v1, ta_plus_di_v1, ta_minus_di_v1
    """
    required = ["symbol", high_col, low_col, close_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    result = df.copy()
    sort_cols = ["symbol"]
    if "timestamp" in result.columns:
        sort_cols.append("timestamp")
    result = result.sort_values(sort_cols).reset_index(drop=True)

    high = result[high_col].astype("float64")
    low = result[low_col].astype("float64")
    close = result[close_col].astype("float64")

    prev_high = high.groupby(result["symbol"]).shift(1)
    prev_low = low.groupby(result["symbol"]).shift(1)
    prev_close = close.groupby(result["symbol"]).shift(1)

    # True Range
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    # Only keep the larger one
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    # Smoothed with EWM (Wilder's smoothing = EWM with alpha=1/window)
    alpha = 1.0 / window
    atr_smooth = tr.groupby(result["symbol"]).transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean()
    )
    plus_dm_smooth = plus_dm.groupby(result["symbol"]).transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean()
    )
    minus_dm_smooth = minus_dm.groupby(result["symbol"]).transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean()
    )

    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr_smooth.replace(0, np.nan)

    # DX and ADX
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = dx.groupby(result["symbol"]).transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean()
    )

    result["ta_adx_v1"] = adx.astype("float64")
    result["ta_plus_di_v1"] = plus_di.astype("float64")
    result["ta_minus_di_v1"] = minus_di.astype("float64")

    return result


def add_obv(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Add On-Balance Volume (OBV) per symbol.

    OBV accumulates volume based on price direction:
    - If close > prev_close: OBV += volume
    - If close < prev_close: OBV -= volume
    - If close == prev_close: OBV unchanged

    Args:
        df: DataFrame with columns: symbol, close_col, volume_col, timestamp
        close_col: Close price column (default: "close")
        volume_col: Volume column (default: "volume")

    Returns:
        DataFrame with column: ta_obv_v1
    """
    required = ["symbol", close_col, volume_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    result = df.copy()
    sort_cols = ["symbol"]
    if "timestamp" in result.columns:
        sort_cols.append("timestamp")
    result = result.sort_values(sort_cols).reset_index(drop=True)

    close = result[close_col].astype("float64")
    volume = result[volume_col].astype("float64")

    # Direction: +1 if up, -1 if down, 0 if unchanged
    # First bar of each symbol has NaN prev_close; treating that as "unchanged"
    # (direction=0) anchors OBV at 0 during warmup and silently bridges over
    # internal price gaps (corp actions, missing bars). Keep NaN so the cumsum
    # stays undefined until the symbol actually has a valid prior close, then
    # fill warmup positions to 0 at output time.
    prev_close = close.groupby(result["symbol"]).shift(1)
    direction = np.sign(close - prev_close)

    signed_volume = (direction * volume).where(direction.notna(), np.nan)

    obv = signed_volume.groupby(result["symbol"]).cumsum()

    # Fill warmup NaNs (first bar per symbol) to 0 per the OBV convention
    result["ta_obv_v1"] = obv.fillna(0.0).astype("float64")

    return result


# ---------------------------------------------------------------------------
# VWAP and Volume Profile
# ---------------------------------------------------------------------------


def add_vwap(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add daily cumulative VWAP and VWAP deviation.

    VWAP is computed as a cumulative value within each calendar day per symbol.
    For daily bars (one row per day), VWAP equals the typical price.
    Also adds ta_vwap_deviation_v1 = (close - VWAP) / VWAP.

    Args:
        df: DataFrame with OHLCV columns.
        close_col: Close price column.
        high_col: High price column.
        low_col: Low price column.
        volume_col: Volume column.
        symbol_col: Symbol column.
        timestamp_col: Timestamp column.

    Returns:
        DataFrame with added columns: ta_vwap_v1, ta_vwap_deviation_v1.
    """
    result = df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])

    required = [high_col, low_col, close_col, volume_col, symbol_col]
    missing = [c for c in required if c not in result.columns]
    if missing:
        logger.warning("[VWAP] Missing columns: %s — skipping", missing)
        return result

    result = result.sort_values([symbol_col, timestamp_col])
    typical_price = (result[high_col] + result[low_col] + result[close_col]) / 3.0
    pv = typical_price * result[volume_col]

    # Cumulative within each symbol × date group
    result["_date"] = result[timestamp_col].dt.date
    cum_pv = result.groupby([symbol_col, "_date"])["_pv"].transform("cumsum") if "_pv" in result.columns else None

    result["_pv"] = pv.values
    cum_pv = result.groupby([symbol_col, "_date"])["_pv"].cumsum()
    cum_vol = result.groupby([symbol_col, "_date"])[volume_col].cumsum()

    vwap = cum_pv / cum_vol.replace(0, float("nan"))
    result["ta_vwap_v1"] = vwap.values
    result["ta_vwap_deviation_v1"] = (result[close_col] - vwap) / vwap.replace(0, float("nan"))

    result = result.drop(columns=["_date", "_pv"], errors="ignore")
    return result


def add_vwap_bands(
    df: pd.DataFrame,
    std_multipliers: list[float] | None = None,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    window: int = 20,
) -> pd.DataFrame:
    """Add rolling VWAP bands (VWAP ± n * std).

    Uses a rolling window to compute VWAP and standard deviation of price
    deviations, then creates upper/lower bands at specified multipliers.
    These are mean-reversion reference levels.

    Args:
        df: DataFrame with OHLCV columns.
        std_multipliers: List of std multipliers for bands (default: [1.0, 2.0]).
        close_col: Close price column.
        high_col: High price column.
        low_col: Low price column.
        volume_col: Volume column.
        symbol_col: Symbol column.
        timestamp_col: Timestamp column.
        window: Rolling window for VWAP/std calculation (default: 20).

    Returns:
        DataFrame with ta_vwap_upper_{n}std_v1 and ta_vwap_lower_{n}std_v1 columns.
    """
    if std_multipliers is None:
        std_multipliers = [1.0, 2.0]

    result = df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])

    required = [high_col, low_col, close_col, volume_col, symbol_col]
    missing = [c for c in required if c not in result.columns]
    if missing:
        logger.warning("[VWAP Bands] Missing columns: %s — skipping", missing)
        return result

    result = result.sort_values([symbol_col, timestamp_col])

    def _vwap_bands_for_group(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        tp = (grp[high_col] + grp[low_col] + grp[close_col]) / 3.0
        pv_g = tp * grp[volume_col]

        cum_pv = pv_g.rolling(window, min_periods=1).sum()
        cum_vol = grp[volume_col].rolling(window, min_periods=1).sum()
        rolling_vwap = cum_pv / cum_vol.replace(0, float("nan"))

        # Rolling std of (typical_price - rolling_vwap)
        dev = tp - rolling_vwap
        rolling_std = dev.rolling(window, min_periods=max(2, window // 4)).std()

        for mult in std_multipliers:
            label = str(int(mult)) if mult == int(mult) else str(mult).replace(".", "_")
            grp[f"ta_vwap_upper_{label}std_v1"] = rolling_vwap + mult * rolling_std
            grp[f"ta_vwap_lower_{label}std_v1"] = rolling_vwap - mult * rolling_std
        grp["ta_vwap_rolling_v1"] = rolling_vwap
        return grp

    pieces = [_vwap_bands_for_group(g) for _, g in result.groupby(symbol_col, group_keys=False)]
    if not pieces:
        return result
    return pd.concat(pieces, ignore_index=True).sort_values([symbol_col, timestamp_col]).reset_index(drop=True)


def add_volume_weighted_momentum(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    close_col: str = "close",
    volume_col: str = "volume",
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add volume-weighted return momentum.

    Volume-weighted momentum = sum(return_i * volume_i) / sum(volume_i) over window.
    Higher volume during positive days → stronger signal than equal-weighted return.

    Args:
        df: DataFrame with close and volume columns.
        windows: List of look-back windows (default: [5, 20]).
        close_col: Close price column.
        volume_col: Volume column.
        symbol_col: Symbol column.
        timestamp_col: Timestamp column.

    Returns:
        DataFrame with ta_vol_weighted_mom_{w}d_v1 columns.
    """
    if windows is None:
        windows = [5, 20]

    result = df.copy()
    result = result.sort_values([symbol_col, timestamp_col])

    required = [close_col, volume_col, symbol_col]
    missing = [c for c in required if c not in result.columns]
    if missing:
        logger.warning("[VWM] Missing columns: %s — skipping", missing)
        return result

    ret = result.groupby(symbol_col)[close_col].pct_change()
    vol = result[volume_col].astype(float)

    for w in windows:
        ret_x_vol = ret * vol
        cum_rv = ret_x_vol.groupby(result[symbol_col]).transform(
            lambda s: s.rolling(w, min_periods=max(1, w // 4)).sum()
        )
        cum_v = vol.groupby(result[symbol_col]).transform(
            lambda s: s.rolling(w, min_periods=max(1, w // 4)).sum()
        )
        result[f"ta_vol_weighted_mom_{w}d_v1"] = cum_rv / cum_v.replace(0, float("nan"))

    return result


def add_all_features(
    df: pd.DataFrame,
    ma_windows: tuple[int, ...] = (20, 50, 200),
    atr_window: int = 14,
    rsi_window: int = 14,
    include_rsi: bool = True,
    include_macd: bool = True,
    include_bollinger: bool = True,
    include_stochastic: bool = True,
    include_adx: bool = True,
    include_obv: bool = True,
    use_namespace: bool = True,
) -> pd.DataFrame:
    """Add all technical analysis features to price DataFrame.

    Convenience function that adds log returns, moving averages, ATR, RSI,
    MACD, Bollinger Bands, Stochastic, ADX, and OBV.

    Args:
        df: DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        ma_windows: Tuple of SMA window sizes (default: (20, 50, 200))
        atr_window: ATR window size (default: 14)
        rsi_window: RSI window size (default: 14)
        include_rsi: Whether to include RSI (default: True)
        include_macd: Whether to include MACD (default: True)
        include_bollinger: Whether to include Bollinger Bands (default: True)
        include_stochastic: Whether to include Stochastic Oscillator (default: True)
        include_adx: Whether to include ADX (default: True)
        include_obv: Whether to include OBV (default: True)
        use_namespace: If True, use namespaced feature names (default: True)

    Returns:
        DataFrame with all features added
    """
    df = add_log_returns(df, use_namespace=use_namespace)
    df = add_moving_averages(df, windows=ma_windows, use_namespace=use_namespace)
    df = add_atr(df, window=atr_window)

    if include_rsi:
        df = add_rsi(df, window=rsi_window)

    if include_macd:
        df = add_macd(df)

    if include_bollinger:
        df = add_bollinger_bands(df)

    has_hlc = all(c in df.columns for c in ["high", "low", "close"])
    if include_stochastic and has_hlc:
        df = add_stochastic(df)

    if include_adx and has_hlc:
        df = add_adx(df)

    has_volume = "volume" in df.columns
    if include_obv and has_volume:
        df = add_obv(df)

    return df
