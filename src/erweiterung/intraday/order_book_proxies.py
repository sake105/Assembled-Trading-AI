"""Order-Book-Imbalance-Proxies aus OHLCV.

Wenn echter L1/L2-Order-Book nicht verfügbar: Proxies aus OHLCV.

Proxies
-------
1. **Close-Position-in-Range** (CPR): (close − low) / (high − low) ∈ [0, 1].
   Nahe 1 = strong buying-pressure, nahe 0 = selling-pressure.
2. **Volume-Weighted-Body**: (close − open) / range × volume.
3. **OBV — On-Balance-Volume** (Granville 1963): Σ sign(Δclose) × volume.
4. **Money-Flow-Index** (Wilder): typical-price-based oscillator.

Anwendung
---------
- Intraday-Direction-Prediction
- Confirmation/Divergence-Signale
- Pre-Trade Order-Book-Reconstruction (rough)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def close_position_in_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """CPR ∈ [0, 1]. Higher = more buying-pressure intraday."""
    rng = high - low
    return ((close - low) / rng.replace(0, np.nan)).clip(0, 1)


def volume_weighted_body(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Signed-Body × Volume — directional pressure-volume."""
    rng = high - low
    body_pct = (close - open_) / rng.replace(0, np.nan)
    return body_pct * volume


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV cumulated.

    On up-day add volume, on down-day subtract. Granville (1963).
    """
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume).fillna(0).cumsum()


def money_flow_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Money-Flow-Index — RSI mit Volume-Anteil.

    >80 = overbought, <20 = oversold (Wilder 1978).
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    up_mask = typical_price.diff() > 0
    pos_flow = money_flow.where(up_mask, 0)
    neg_flow = money_flow.where(~up_mask & (typical_price.diff() < 0), 0)
    pos_sum = pos_flow.rolling(window, min_periods=window // 2).sum()
    neg_sum = neg_flow.rolling(window, min_periods=window // 2).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def imbalance_composite(ohlcv: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Convenience: compute all proxies in one call.

    Args:
        ohlcv: DataFrame [date, open, high, low, close, volume].

    Returns:
        DataFrame with added proxy-columns.
    """
    df = ohlcv.copy()
    df["cpr"] = close_position_in_range(df["high"], df["low"], df["close"])
    df["vw_body"] = volume_weighted_body(
        df["open"], df["high"], df["low"], df["close"], df["volume"]
    )
    df["obv"] = on_balance_volume(df["close"], df["volume"])
    df["mfi"] = money_flow_index(
        df["high"], df["low"], df["close"], df["volume"], window
    )
    return df


__all__ = [
    "close_position_in_range",
    "volume_weighted_body",
    "on_balance_volume",
    "money_flow_index",
    "imbalance_composite",
]
