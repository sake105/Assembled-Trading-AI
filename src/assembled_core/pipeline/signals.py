# src/assembled_core/pipeline/signals.py
"""Signal generation for trading strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ema_signal_for_symbol(d: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Compute EMA crossover signal for a single symbol.
    
    Args:
        d: DataFrame with columns: timestamp, close (and optionally symbol)
        fast: Fast EMA period
        slow: Slow EMA period
    
    Returns:
        DataFrame with columns: timestamp, symbol, sig, price
        sig: -1 (SELL), 0 (neutral), +1 (BUY)
        price: Close price at timestamp
    """
    # Symbol aus Spalte oder Gruppenschlüssel ableiten
    sym = d["symbol"].iloc[0] if "symbol" in d.columns else d.name
    d = d.sort_values("timestamp")
    px = d["close"].astype("float64")

    ema_fast = px.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = px.ewm(span=slow, adjust=False, min_periods=slow).mean()
    sig = (ema_fast > ema_slow).astype(np.int8) - (ema_fast < ema_slow).astype(np.int8)

    return pd.DataFrame(
        {
            "timestamp": d["timestamp"].values,
            "symbol": np.full(len(d), sym, dtype=object),
            "sig": sig.values,
            "price": d["close"].values,
        }
    )


def compute_ema_signals(prices: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Compute EMA crossover signals for all symbols.
    
    Args:
        prices: DataFrame with columns: timestamp, symbol, close
        fast: Fast EMA period
        slow: Slow EMA period
    
    Returns:
        DataFrame with columns: timestamp, symbol, sig, price
        sig: -1 (SELL), 0 (neutral), +1 (BUY)
        Sorted by symbol, then timestamp
    """
    signals = (
        prices.groupby("symbol", group_keys=False)
        .apply(lambda d: compute_ema_signal_for_symbol(d, fast, slow), include_groups=False)
        .reset_index(drop=True)
    )
    return signals

