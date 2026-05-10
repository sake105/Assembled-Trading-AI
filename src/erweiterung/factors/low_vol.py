"""Betting-Against-Beta + Low-Vol-Anomaly (Frazzini/Pedersen 2014).

Theorie
-------
**Low-Beta-Aktien outperformen risikoadjustiert** — gegen die CAPM-Vorhersage.
Erklärung (FP 2014): Leverage-Constraints zwingen Investoren in High-Beta-Aktien,
treiben deren Preise nach oben, senken Forward-Returns.

Strategie BAB
-------------
1. Schätze Beta_i je Asset (z. B. via Rolling 252-Day OLS gegen Market).
2. Rank-weighted Long Low-Beta, Short High-Beta — beide neutral mit Leverage,
   damit das Long-Bein Beta=1 hat (zusätzlich zum Short auf Hoch-Beta-Bein).

In voller Frazzini/Pedersen-Implementierung:
    BAB = (1/β_L) × (R_L − R_f) − (1/β_H) × (R_H − R_f)

Wobei β_L = Mean-Beta low-Bucket, β_H = Mean-Beta high-Bucket.
"""

from __future__ import annotations

import pandas as pd


def rolling_beta(
    asset_returns: pd.Series, market_returns: pd.Series, window: int = 252
) -> pd.Series:
    """Rolling-OLS-Beta von asset gegen market."""
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    df.columns = ["a", "m"]
    cov = df["a"].rolling(window).cov(df["m"])
    var = df["m"].rolling(window).var()
    return cov / var


def betting_against_beta(
    returns_panel: pd.DataFrame,
    market_returns: pd.Series,
    risk_free: float = 0.0,
    window: int = 252,
    n_buckets: int = 5,
) -> pd.Series:
    """Compute BAB-Faktor-Series.

    Args:
        returns_panel: DataFrame [date, symbol, return].
        market_returns: Series indexed by date.
        risk_free: daily risk-free.
        window: rolling-beta-window.
        n_buckets: rank-quantile (5 = top/bottom 20%).

    Returns:
        Series ``bab_return`` indexed by date.
    """
    if returns_panel.empty:
        return pd.Series(dtype=float)

    pivot = returns_panel.pivot_table(index="date", columns="symbol", values="return")
    betas = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
    for sym in pivot.columns:
        betas[sym] = rolling_beta(pivot[sym], market_returns, window=window)

    # Rank betas cross-sectionally each day
    rank = betas.rank(axis=1, pct=True)

    out_rows = []
    for d in pivot.index:
        if d not in rank.index:
            continue
        rk = rank.loc[d].dropna()
        if len(rk) < n_buckets * 2:
            continue
        low_thresh = 1.0 / n_buckets
        high_thresh = 1 - 1.0 / n_buckets
        low_syms = rk[rk <= low_thresh].index
        high_syms = rk[rk >= high_thresh].index
        if len(low_syms) == 0 or len(high_syms) == 0:
            continue
        # mean beta in each bucket
        beta_l = float(betas.loc[d, low_syms].mean())
        beta_h = float(betas.loc[d, high_syms].mean())
        if beta_l <= 0 or beta_h <= 0:
            continue
        # Bucket excess returns
        r_l = pivot.loc[d, low_syms].mean() - risk_free
        r_h = pivot.loc[d, high_syms].mean() - risk_free
        bab = (1.0 / beta_l) * r_l - (1.0 / beta_h) * r_h
        out_rows.append(
            {"date": d, "bab_return": bab, "beta_low": beta_l, "beta_high": beta_h}
        )
    return pd.DataFrame(out_rows).set_index("date")["bab_return"]


def low_vol_signal(returns_panel: pd.DataFrame, window: int = 60) -> pd.Series:
    """Negative idiosyncratic vol — niedrige Vola bekommt hohes Signal.

    Args:
        returns_panel: DataFrame [date, symbol, return].
        window: rolling-window in Tagen.

    Returns:
        Series indexed by (date, symbol), ``-vol`` (höher = besser).
    """
    df = returns_panel.copy().sort_values(["symbol", "date"])
    grp = df.groupby("symbol", group_keys=False)
    df["rolling_vol"] = grp["return"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=window // 2).std()
    )
    df["low_vol_signal"] = -df["rolling_vol"]
    return df.set_index(["date", "symbol"])["low_vol_signal"]


__all__ = ["rolling_beta", "betting_against_beta", "low_vol_signal"]
