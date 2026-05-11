"""Lee-Ready Trade-Sign-Classification (Lee/Ready 1991).

Theorie
-------
Bei reinen Trade-Prices (ohne Bid/Ask-Quotes) kann man Trade-Sign (buyer/seller-
initiated) klassifizieren via:

1. **Quote-Rule** (preferred): if price > midquote → buy; if < midquote → sell.
2. **Tick-Rule** (fallback when quote tied or unavailable):
   - up-tick (price > prev) → buy
   - down-tick → sell
   - zero-up-tick: last non-zero tick up → buy
   - zero-down-tick: last non-zero tick down → sell

Lee/Ready: combine quote-rule (primary) with tick-rule (when quote ambiguous).

Anwendung
---------
- Aufbau signed-volume-series für VPIN, Kyle-Lambda, Order-Flow-Imbalance
- Trade-Cost-Analysis (bid-ask-effective)

Reference
---------
Lee, C. & Ready, M. (1991). Inferring Trade Direction from Intraday Data.
*J. Finance* 46.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quote_rule_classify(
    trade_prices: pd.Series, bid: pd.Series, ask: pd.Series
) -> pd.Series:
    """Apply Quote-Rule when both bid and ask present.

    Returns:
        Series of +1 (buy) / -1 (sell) / 0 (ambiguous).
    """
    mid = (bid + ask) / 2
    out = pd.Series(0, index=trade_prices.index, dtype=int)
    out[trade_prices > mid] = 1
    out[trade_prices < mid] = -1
    # if trade_price == mid: leave 0 (use tick-rule fallback)
    return out


def tick_rule_classify(trade_prices: pd.Series) -> pd.Series:
    """Apply Tick-Rule (forward-fill for zero-ticks).

    Returns:
        Series of +1 / -1.
    """
    p = pd.Series(trade_prices).copy()
    diff = p.diff()
    sign = np.sign(diff).astype(float)
    # Replace 0 with NaN, then ffill → "zero-tick uses last directional sign"
    sign[sign == 0] = np.nan
    sign = sign.ffill().bfill().fillna(0)
    return sign.astype(int)


def lee_ready_classify(
    trade_prices: pd.Series,
    bid: pd.Series | None = None,
    ask: pd.Series | None = None,
) -> pd.Series:
    """Combined Lee-Ready classifier — Quote-Rule with Tick-Rule fallback.

    Args:
        trade_prices: Series.
        bid, ask: optional quote series. If None, pure tick-rule.

    Returns:
        Series of +1 / -1 (or 0 for very first trade if pure tick-rule).
    """
    p = pd.Series(trade_prices)
    if bid is not None and ask is not None:
        quote_sign = quote_rule_classify(p, bid, ask)
        tick_sign = tick_rule_classify(p)
        out = quote_sign.copy()
        ambiguous = out == 0
        out[ambiguous] = tick_sign[ambiguous]
        return out
    return tick_rule_classify(p)


def order_flow_imbalance(
    trade_prices: pd.Series,
    volumes: pd.Series,
    bid: pd.Series | None = None,
    ask: pd.Series | None = None,
) -> pd.Series:
    """Signed volume = trade-sign × volume.

    Aggregated → cumulative-order-flow.
    """
    sign = lee_ready_classify(trade_prices, bid, ask)
    return sign * pd.Series(volumes).reindex(sign.index)


def rolling_ofi_imbalance_ratio(
    signed_volume: pd.Series, total_volume: pd.Series, window: int = 60
) -> pd.Series:
    """Order-Flow-Imbalance-Ratio = sum(signed) / sum(|total|) im rolling window.

    Range [-1, 1]: positiv = buying pressure.
    """
    signed_sum = signed_volume.rolling(window, min_periods=window // 2).sum()
    total_sum = total_volume.rolling(window, min_periods=window // 2).sum().abs()
    return signed_sum / total_sum.replace(0, np.nan)


__all__ = [
    "quote_rule_classify",
    "tick_rule_classify",
    "lee_ready_classify",
    "order_flow_imbalance",
    "rolling_ofi_imbalance_ratio",
]
