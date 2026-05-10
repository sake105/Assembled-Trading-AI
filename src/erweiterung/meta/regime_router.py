"""Regime-Router — Online-Detection von Markt-Regimes + Strategy-Switching.

Theorie
-------
Markt hat Phasen: Trending, Mean-Reverting, High-Vol, Low-Vol.
Unterschiedliche Strategien performen in unterschiedlichen Regimes.

Regime-Detection-Methoden
--------------------------
1. **Volatility-Regime**: Realized Vol Quartile (Low/Mid/High).
2. **Trend-Regime**: ADX-Indikator + Slope von 200-Tages-MA.
3. **Hidden-Markov-Model** (extern; im Mainline-Code unter risk/regime_models).
4. **Crisis-Score** (siehe ``risk.correlation_breakdown``).

Composite-Regime
----------------
Wir bauen einen Composite aus Vol- und Trend-Regime → 4 Regimes:
- 0: Low-Vol Trending  (Pro-Cyclical: Long-only Momentum)
- 1: Low-Vol Range     (Mean-Reversion / Pairs)
- 2: High-Vol Trending (Carry-Trades, BL mit defensive Views)
- 3: High-Vol Crisis   (Cash, defensive ETFs, Short-Vol gefährlich)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_regime(
    returns: pd.Series,
    window: int = 21,
    percentiles: tuple[float, float] = (0.33, 0.67),
) -> pd.Series:
    """0 = low-vol, 1 = mid-vol, 2 = high-vol (rolling-quantile-basiert)."""
    rv = returns.rolling(window, min_periods=window // 2).std()
    # PIT shift
    rv = rv.shift(1)
    out = pd.Series(np.nan, index=rv.index)
    # use trailing quantile estimate
    for end in range(window * 4, len(rv)):
        sub = rv.iloc[: end + 1].dropna()
        if len(sub) < window * 4:
            continue
        q_lo, q_hi = sub.quantile([percentiles[0], percentiles[1]]).values
        v = rv.iloc[end]
        if pd.isna(v):
            continue
        if v < q_lo:
            out.iloc[end] = 0
        elif v < q_hi:
            out.iloc[end] = 1
        else:
            out.iloc[end] = 2
    return out


def trend_regime(prices: pd.Series, slow: int = 200, fast: int = 50) -> pd.Series:
    """0 = downtrend, 1 = sideways, 2 = uptrend."""
    sma_slow = prices.rolling(slow).mean()
    sma_fast = prices.rolling(fast).mean()
    slope = sma_slow.pct_change(20)
    out = pd.Series(1, index=prices.index, dtype=float)
    out[(sma_fast > sma_slow) & (slope > 0)] = 2
    out[(sma_fast < sma_slow) & (slope < 0)] = 0
    return out.shift(1)


def composite_regime(
    returns: pd.Series, prices: pd.Series, vol_window: int = 21
) -> pd.Series:
    """Composite Regime ∈ {0, 1, 2, 3}.

    0 = Low-Vol Trending (Long-only Momentum)
    1 = Low-Vol Range    (Mean-Reversion / Pairs)
    2 = High-Vol Trending (Carry / Defensive)
    3 = High-Vol Crisis   (Cash)
    """
    vr = vol_regime(returns, vol_window)
    tr = trend_regime(prices)
    out = pd.Series(np.nan, index=returns.index)
    for d in returns.index:
        v = vr.get(d)
        t = tr.get(d)
        if pd.isna(v) or pd.isna(t):
            continue
        v = int(v)
        t = int(t)
        if v <= 1 and t == 2:
            out.loc[d] = 0  # low-vol uptrend
        elif v <= 1 and t in (0, 1):
            out.loc[d] = 1  # low-vol range/down (cautious mean-rev)
        elif v == 2 and t == 2:
            out.loc[d] = 2  # high-vol uptrend (volatile but trending up)
        else:
            out.loc[d] = 3  # high-vol crisis
    return out


__all__ = ["vol_regime", "trend_regime", "composite_regime"]
