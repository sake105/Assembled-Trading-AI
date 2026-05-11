"""Stoikov Microprice (2018) — Fair-Value zwischen Bid und Ask.

Theorie
-------
Midprice (BB+BA)/2 ignoriert Imbalance — wenn Bid-Volumen 10× Ask-Volumen,
ist nächster Trade wahrscheinlich oben. **Microprice** ist:

    microprice = (BB × V_ask + BA × V_bid) / (V_bid + V_ask)

Note: V_ask gewichtet BB, V_bid gewichtet BA — counter-intuitive, aber
korrekt: hohe Bids ziehen Preis nach oben (nahe BA).

Vorteil
-------
- Reduziert Bid-Ask-Bounce
- Predicts next-trade-price besser als Mid
- Standard für Market-Making + HFT-Backtests

Reference
---------
Stoikov, S. (2018). The micro-price: A high-frequency estimator of future prices.
*Quantitative Finance* 18.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.orderbook.lob_state import LOBState


def microprice_from_lob(state: LOBState) -> float | None:
    """Compute microprice from current LOB-state."""
    bb = state.best_bid()
    ba = state.best_ask()
    if bb is None or ba is None:
        return None
    bid_p, bid_v = bb
    ask_p, ask_v = ba
    total = bid_v + ask_v
    if total <= 0:
        return None
    return (bid_p * ask_v + ask_p * bid_v) / total


def microprice_series(
    bids_p: pd.Series, bids_v: pd.Series, asks_p: pd.Series, asks_v: pd.Series
) -> pd.Series:
    """Vectorized microprice for time-series of top-of-book."""
    df = pd.concat(
        [
            bids_p.rename("bp"),
            bids_v.rename("bv"),
            asks_p.rename("ap"),
            asks_v.rename("av"),
        ],
        axis=1,
    ).dropna()
    total = df["bv"] + df["av"]
    mp = (df["bp"] * df["av"] + df["ap"] * df["bv"]) / total.replace(0, np.nan)
    return mp


def order_book_imbalance_signal(
    bids_v: pd.Series, asks_v: pd.Series, lookback: int = 60
) -> pd.Series:
    """OBI = (bid_v − ask_v) / (bid_v + ask_v) rolling-mean.

    Predicts short-horizon return-direction.
    """
    df = pd.concat([bids_v.rename("bv"), asks_v.rename("av")], axis=1).dropna()
    obi = (df["bv"] - df["av"]) / (df["bv"] + df["av"]).replace(0, np.nan)
    return obi.rolling(lookback, min_periods=lookback // 2).mean()


def microprice_to_mid_drift(
    bids_p: pd.Series,
    bids_v: pd.Series,
    asks_p: pd.Series,
    asks_v: pd.Series,
    horizon: int = 10,
) -> pd.Series:
    """Drift = next-h-mid − microprice — Predicts mid-reversion to microprice.

    Wenn microprice > mid: mid expected to rise toward microprice in next h.
    """
    mp = microprice_series(bids_p, bids_v, asks_p, asks_v)
    mid = (bids_p + asks_p) / 2
    future_mid = mid.shift(-horizon)
    return future_mid - mp


__all__ = [
    "microprice_from_lob",
    "microprice_series",
    "order_book_imbalance_signal",
    "microprice_to_mid_drift",
]
