"""Liquidity & Spread-Proxies aus OHLCV (frei berechenbar).

Wenn keine Tick-Daten verfügbar sind, gibt es etablierte Proxies aus täglichen
OHLC-Daten:

1. **Amihud (2002) Illiquidity**: |return| / dollar_volume.
2. **Roll (1984) Spread**: 2·√(-cov(Δp_t, Δp_{t-1})).
3. **Corwin-Schultz (2012)**: aus Two-Day-High/Low.
4. **Hasbrouck (2009) Gibbs-Sampler-Spread**: nicht hier (zu aufwendig).
5. **VPIN**: Volume-bucket-basiert (siehe vpin.py).

Anwendung
---------
- Liquidity-Risk-Faktor (Pastor/Stambaugh 2003)
- Filter für Trade-Selection (illiquide raus)
- Adaptive Slippage-Kalibrierung
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def amihud_illiquidity(
    ohlcv_panel: pd.DataFrame,
    return_col: str = "return",
    volume_col: str = "dollar_volume",
    window: int = 21,
) -> pd.DataFrame:
    """Amihud Illiquidity Ratio.

    Args:
        ohlcv_panel: DataFrame [date, symbol, return, dollar_volume].
        return_col: name of return column.
        volume_col: name of dollar-volume column.
        window: rolling-window in Tagen.

    Returns:
        DataFrame [date, symbol, amihud] (höher = illiquider).
    """
    df = ohlcv_panel.copy().sort_values(["symbol", "date"])
    df["abs_ret"] = df[return_col].abs()
    df["illiq_daily"] = df["abs_ret"] / df[volume_col].replace(0, np.nan)
    df["illiq_daily"] = df["illiq_daily"].replace([np.inf, -np.inf], np.nan)
    grp = df.groupby("symbol", group_keys=False)
    df["amihud"] = grp["illiq_daily"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=window // 2).mean()
    )
    return df[["date", "symbol", "amihud"]]


def roll_spread_estimator(prices: pd.Series) -> float:
    """Roll (1984) effective spread = 2·√(−cov(Δp_t, Δp_{t-1}))."""
    p = pd.Series(prices).dropna()
    if len(p) < 30:
        return float("nan")
    dp = p.diff().dropna()
    cov = dp.cov(dp.shift(1).dropna())
    if cov >= 0:
        return float("nan")
    return float(2.0 * np.sqrt(-cov))


def corwin_schultz_spread(
    high: pd.Series, low: pd.Series, window: int = 1
) -> pd.Series:
    """Corwin/Schultz (2012) High-Low Spread Estimator.

    Args:
        high, low: daily H/L Series indexed by date.
        window: 1-day or 2-day version (Standard: 2).
    """
    h = pd.Series(high)
    lo = pd.Series(low)
    if h.empty or lo.empty:
        return pd.Series(dtype=float)

    beta = (np.log(h / lo) ** 2).rolling(2).sum()
    h2 = h.rolling(2).max()
    l2 = lo.rolling(2).min()
    gamma = np.log(h2 / l2) ** 2

    denom = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)


def kyle_lambda(
    returns: pd.Series, signed_volume: pd.Series, window: int = 60
) -> pd.Series:
    """Kyle's lambda: price impact per unit signed volume.

    Args:
        returns: return series.
        signed_volume: signed-volume series (positive = buy pressure).
        window: rolling-window for OLS.

    Returns:
        Series of rolling-OLS-slope estimates.
    """
    df = pd.concat([returns, signed_volume], axis=1).dropna()
    df.columns = ["r", "v"]
    out = pd.Series(np.nan, index=df.index)
    for end in range(window, len(df) + 1):
        sub = df.iloc[end - window : end]
        if sub["v"].std() == 0:
            continue
        cov = sub["r"].cov(sub["v"])
        var = sub["v"].var()
        out.iloc[end - 1] = cov / var
    return out


__all__ = [
    "amihud_illiquidity",
    "roll_spread_estimator",
    "corwin_schultz_spread",
    "kyle_lambda",
]
