"""Trend-following signal rules module.

This module provides trend-following signal generation based on technical indicators.
It extends the basic EMA crossover functionality from pipeline.signals.

Zukünftige Integration:
- Nutzt pipeline.signals.compute_ema_signals als Basis für EMA-Crossover
- Erweitert um weitere Trend-Following-Regeln (Moving Average Crossovers, etc.)
- Bietet konfigurierbare Signal-Generierung
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.assembled_core.features.ta_features import add_moving_averages


def generate_trend_signals(
    df: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 50,
    volume_threshold: float | None = None,
    min_volume_multiplier: float = 1.0
) -> pd.DataFrame:
    """Generate trend-following signals based on moving average crossover.
    
    Signal logic:
    - LONG: ma_fast > ma_slow AND (volume > threshold OR no volume filter)
    - FLAT: otherwise
    
    Args:
        df: DataFrame with columns: timestamp, symbol, close, and optionally volume
            Should have moving averages already computed (ma_fast, ma_slow)
            If not present, they will be computed automatically
        ma_fast: Fast moving average window (default: 20)
        ma_slow: Slow moving average window (default: 50)
        volume_threshold: Optional volume threshold. If None, uses min_volume_multiplier * mean(volume)
        min_volume_multiplier: Multiplier for mean volume to compute threshold (default: 1.0)
    
    Returns:
        DataFrame with columns: timestamp, symbol, direction, score
        direction: "LONG" or "FLAT"
        score: Signal strength (0.0 to 1.0), based on MA spread and volume
        Sorted by symbol, then timestamp
    
    Raises:
        KeyError: If required columns are missing
    """
    df = df.copy()
    
    # Ensure required columns
    required = ["timestamp", "symbol", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    
    # Add moving averages if not present
    ma_fast_col = f"ma_{ma_fast}"
    ma_slow_col = f"ma_{ma_slow}"
    
    if ma_fast_col not in df.columns or ma_slow_col not in df.columns:
        df = add_moving_averages(df, windows=(ma_fast, ma_slow))
    
    # Sort by symbol and timestamp
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    # Compute volume threshold if needed
    if volume_threshold is None and "volume" in df.columns:
        # Use mean volume per symbol as baseline
        mean_volume = df.groupby("symbol")["volume"].mean()
        df["volume_threshold"] = df["symbol"].map(mean_volume) * min_volume_multiplier
    elif volume_threshold is not None:
        df["volume_threshold"] = volume_threshold
    else:
        # No volume filter
        df["volume_threshold"] = 0.0
    
    # Generate signals
    # LONG: ma_fast > ma_slow AND (volume > threshold OR no volume column)
    has_volume = "volume" in df.columns
    if has_volume:
        long_condition = (df[ma_fast_col] > df[ma_slow_col]) & (df["volume"] >= df["volume_threshold"])
    else:
        long_condition = df[ma_fast_col] > df[ma_slow_col]
    
    df["direction"] = np.where(long_condition, "LONG", "FLAT")
    
    # Compute signal score (0.0 to 1.0)
    # Score based on:
    # - MA spread: (ma_fast - ma_slow) / ma_slow (normalized)
    # - Volume strength: volume / threshold (if volume available)
    ma_spread = (df[ma_fast_col] - df[ma_slow_col]) / (df[ma_slow_col] + 1e-10)
    ma_score = pd.Series(ma_spread).clip(lower=0.0, upper=1.0)  # Normalize to [0, 1]
    
    if has_volume and (df["volume_threshold"] > 0).any():
        volume_score = (df["volume"] / (df["volume_threshold"] + 1e-10)).clip(lower=0.0, upper=1.0)
        df["score"] = (ma_score * 0.7 + volume_score * 0.3).fillna(0.0)
    else:
        df["score"] = ma_score.fillna(0.0)
    
    # Only set score for LONG signals, FLAT = 0.0
    df["score"] = np.where(df["direction"] == "LONG", df["score"], 0.0)
    
    # Select output columns
    result = df[["timestamp", "symbol", "direction", "score"]].copy()
    
    return result


def generate_trend_signals_from_prices(
    prices: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 50,
    volume_threshold: float | None = None,
    min_volume_multiplier: float = 1.0
) -> pd.DataFrame:
    """Generate trend signals directly from price DataFrame.
    
    Convenience function that computes moving averages and generates signals in one step.
    
    Args:
        prices: DataFrame with columns: timestamp, symbol, close, and optionally volume
        ma_fast: Fast moving average window (default: 20)
        ma_slow: Slow moving average window (default: 50)
        volume_threshold: Optional volume threshold. If None, uses min_volume_multiplier * mean(volume)
        min_volume_multiplier: Multiplier for mean volume to compute threshold (default: 1.0)
    
    Returns:
        DataFrame with columns: timestamp, symbol, direction, score
        direction: "LONG" or "FLAT"
        score: Signal strength (0.0 to 1.0)
    """
    return generate_trend_signals(
        prices,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        volume_threshold=volume_threshold,
        min_volume_multiplier=min_volume_multiplier
    )
