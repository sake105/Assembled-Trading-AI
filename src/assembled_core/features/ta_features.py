"""Technical analysis features module.

This module provides functions to compute technical indicators (SMA, ATR, RSI, log returns, etc.)
from price data. It extends the basic EMA functionality from pipeline.signals.

Zukünftige Integration:
- Nutzt pipeline.signals.compute_ema_signals als Basis für EMA
- Erweitert um weitere TA-Indikatoren (SMA, ATR, RSI, MACD, Bollinger Bands, etc.)
- Bietet Feature-Engineering-Pipeline für ML-Modelle
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_returns(
    df: pd.DataFrame,
    price_col: str = "close",
    out_col: str = "log_return",
) -> pd.DataFrame:
    """
    Füge logarithmische Returns pro Symbol hinzu.

    Erwartet:
    - Spalte 'symbol'
    - Spalte `price_col` (z.B. 'close')
    - Optional: 'timestamp' für zeitliche Sortierung

    Rückgabe:
    - DataFrame mit neuer Spalte `out_col`
    """
    if "symbol" not in df.columns:
        raise KeyError("symbol")
    if price_col not in df.columns:
        raise KeyError(
            f"Price column '{price_col}' not found. Available columns: {list(df.columns)}"
        )

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

    return result


def add_moving_averages(
    df: pd.DataFrame, windows: tuple[int, ...] = (20, 50, 200), price_col: str = "close"
) -> pd.DataFrame:
    """Add Simple Moving Averages (SMA) to price DataFrame.

    Computes SMA for each window: SMA(window) = mean(price over window periods)

    Args:
        df: DataFrame with columns: timestamp, symbol, and price_col
        windows: Tuple of window sizes (default: (20, 50, 200))
        price_col: Column name for price data (default: "close")

    Returns:
        DataFrame with additional columns: ma_{window} for each window
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
        col_name = f"ma_{window}"
        if "symbol" in df.columns:
            df[col_name] = (
                df.groupby("symbol", group_keys=False)[price_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(drop=True)
            )
        else:
            df[col_name] = df[price_col].rolling(window=window, min_periods=1).mean()

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

    result[f"atr_{window}"] = atr.astype("float64")

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

    result[f"rsi_{window}"] = rsi.astype("float64")

    return result


def add_all_features(
    df: pd.DataFrame,
    ma_windows: tuple[int, ...] = (20, 50, 200),
    atr_window: int = 14,
    rsi_window: int = 14,
    include_rsi: bool = True,
) -> pd.DataFrame:
    """Add all technical analysis features to price DataFrame.

    Convenience function that adds log returns, moving averages, ATR, and optionally RSI.

    Args:
        df: DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        ma_windows: Tuple of SMA window sizes (default: (20, 50, 200))
        atr_window: ATR window size (default: 14)
        rsi_window: RSI window size (default: 14)
        include_rsi: Whether to include RSI (default: True)

    Returns:
        DataFrame with all features added
        Columns added: log_return, ma_{window} for each window, atr_{atr_window}, rsi_{rsi_window} (if include_rsi)
    """
    df = add_log_returns(df)
    df = add_moving_averages(df, windows=ma_windows)
    df = add_atr(df, window=atr_window)

    if include_rsi:
        df = add_rsi(df, window=rsi_window)

    return df
