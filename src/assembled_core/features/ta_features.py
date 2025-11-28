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


def add_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add log returns to price DataFrame.
    
    Computes log returns: log(price_t / price_{t-1}) = log(price_t) - log(price_{t-1})
    
    Args:
        df: DataFrame with columns: timestamp, symbol, and price_col
        price_col: Column name for price data (default: "close")
    
    Returns:
        DataFrame with additional column: log_return
        Sorted by symbol, then timestamp
    
    Raises:
        KeyError: If required columns are missing
    """
    df = df.copy()
    
    # Ensure required columns
    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not found. Available columns: {list(df.columns)}")
    
    # Sort by symbol and timestamp
    if "symbol" in df.columns:
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Compute log returns per symbol
    if "symbol" in df.columns:
        df["log_return"] = (
            df.groupby("symbol", group_keys=False)[price_col]
            .apply(lambda x: np.log(x / x.shift(1)))
        )
    else:
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    
    return df


def add_moving_averages(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (20, 50, 200),
    price_col: str = "close"
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
        raise KeyError(f"Price column '{price_col}' not found. Available columns: {list(df.columns)}")
    
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


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add Average True Range (ATR) to price DataFrame.
    
    ATR measures volatility by computing the average of True Range over a window.
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    
    Args:
        df: DataFrame with columns: timestamp, symbol, high, low, close
        window: ATR window size (default: 14)
    
    Returns:
        DataFrame with additional column: atr_{window}
        Sorted by symbol, then timestamp
    
    Raises:
        KeyError: If required columns (high, low, close) are missing
    """
    df = df.copy()
    
    # Ensure required columns
    required = ["high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    
    # Sort by symbol and timestamp
    if "symbol" in df.columns:
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    def compute_atr_for_symbol(d: pd.DataFrame) -> pd.Series:
        """Compute ATR for a single symbol."""
        high = d["high"].values
        low = d["low"].values
        close = d["close"].values
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        # First TR should not use previous close
        tr2[0] = 0.0
        tr3[0] = 0.0
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR = rolling mean of True Range
        atr = pd.Series(true_range).rolling(window=window, min_periods=1).mean()
        
        return atr
    
    # Compute ATR per symbol
    if "symbol" in df.columns:
        df[f"atr_{window}"] = (
            df.groupby("symbol", group_keys=False)
            .apply(compute_atr_for_symbol, include_groups=False)
            .reset_index(drop=True)
        )
    else:
        df[f"atr_{window}"] = compute_atr_for_symbol(df)
    
    return df


def add_rsi(df: pd.DataFrame, window: int = 14, price_col: str = "close") -> pd.DataFrame:
    """Add Relative Strength Index (RSI) to price DataFrame.
    
    RSI measures momentum by comparing average gains to average losses over a window.
    RSI = 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss
    
    Args:
        df: DataFrame with columns: timestamp, symbol, and price_col
        window: RSI window size (default: 14)
        price_col: Column name for price data (default: "close")
    
    Returns:
        DataFrame with additional column: rsi_{window}
        Sorted by symbol, then timestamp
    
    Raises:
        KeyError: If required columns are missing
    """
    df = df.copy()
    
    # Ensure required columns
    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not found. Available columns: {list(df.columns)}")
    
    # Sort by symbol and timestamp
    if "symbol" in df.columns:
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    def compute_rsi_for_symbol(d: pd.DataFrame) -> pd.Series:
        """Compute RSI for a single symbol."""
        prices = d[price_col].values
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        # Average gains and losses (using exponential moving average)
        avg_gain = pd.Series(gains).ewm(alpha=1.0/window, adjust=False, min_periods=window).mean()
        avg_loss = pd.Series(losses).ewm(alpha=1.0/window, adjust=False, min_periods=window).mean()
        
        # RSI calculation
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # First value is NaN (no previous price)
        rsi = pd.concat([pd.Series([np.nan]), rsi]).reset_index(drop=True)
        
        return rsi
    
    # Compute RSI per symbol
    if "symbol" in df.columns:
        df[f"rsi_{window}"] = (
            df.groupby("symbol", group_keys=False)
            .apply(compute_rsi_for_symbol, include_groups=False)
            .reset_index(drop=True)
        )
    else:
        df[f"rsi_{window}"] = compute_rsi_for_symbol(df)
    
    return df


def add_all_features(
    df: pd.DataFrame,
    ma_windows: tuple[int, ...] = (20, 50, 200),
    atr_window: int = 14,
    rsi_window: int = 14,
    include_rsi: bool = True
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
