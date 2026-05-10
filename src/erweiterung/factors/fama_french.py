"""Fama-French-Faktor-Konstruktion.

Faktoren
--------
- **MKT-RF**: Market excess return über Risk-Free.
- **SMB**: Small Minus Big — Long bottom-30%-cap, short top-30%-cap (Banz 1981; Fama/French 1993).
- **HML**: High Minus Low B/M — Long high-BM ("value"), short low-BM ("growth").
- **RMW**: Robust Minus Weak Profitability (Fama/French 2015).
- **CMA**: Conservative Minus Aggressive Investment.
- **MOM**: Carhart (1997) momentum (12-1).

Cross-Section-Mechanik
----------------------
Faktoren werden als equal-weighted long-short-Portfolios der jeweiligen 30/70-
Quantile gebaut.  Rebalancing typisch monatlich.
"""

from __future__ import annotations

import pandas as pd


def construct_long_short_factor(
    panel: pd.DataFrame,
    sort_col: str,
    return_col: str = "return_t1",
    by: str = "date",
    quantiles: tuple[float, float] = (0.3, 0.7),
    long_high: bool = True,
) -> pd.Series:
    """Konstruiere equal-weighted long-short-Faktor.

    Args:
        panel: DataFrame mit Spalten [date, symbol, sort_col, return_col].
        sort_col: Spalte zum Quantil-Sort (z. B. ``size`` oder ``book_to_market``).
        return_col: Forward-Return-Spalte (PIT-shift bereits angewandt).
        by: Datum-Spalte.
        quantiles: (low, high) Quantile.
        long_high: ``True`` => long high-quantile, short low-quantile (für value-style).

    Returns:
        Series ``factor_return`` indexed by date.
    """
    if panel.empty:
        return pd.Series(dtype=float)

    out_rows = []
    for d, g in panel.groupby(by):
        sub = g.dropna(subset=[sort_col, return_col])
        if len(sub) < 10:
            continue
        q_lo, q_hi = sub[sort_col].quantile(list(quantiles))
        long_pool = (
            sub[sub[sort_col] >= q_hi] if long_high else sub[sub[sort_col] <= q_lo]
        )
        short_pool = (
            sub[sub[sort_col] <= q_lo] if long_high else sub[sub[sort_col] >= q_hi]
        )
        if long_pool.empty or short_pool.empty:
            continue
        long_ret = long_pool[return_col].mean()
        short_ret = short_pool[return_col].mean()
        out_rows.append({by: d, "factor_return": long_ret - short_ret})

    return pd.DataFrame(out_rows).set_index(by)["factor_return"]


def smb_factor(panel: pd.DataFrame, return_col: str = "return_t1") -> pd.Series:
    """Small-Minus-Big: long small-cap (low size), short large-cap (high size)."""
    return construct_long_short_factor(
        panel, "market_cap", return_col=return_col, long_high=False
    )


def hml_factor(panel: pd.DataFrame, return_col: str = "return_t1") -> pd.Series:
    """High-Minus-Low: long high book-to-market, short low BTM."""
    return construct_long_short_factor(
        panel, "book_to_market", return_col=return_col, long_high=True
    )


def mom_factor(panel: pd.DataFrame, return_col: str = "return_t1") -> pd.Series:
    """Carhart Momentum (12-1)."""
    return construct_long_short_factor(
        panel, "momentum_12_1", return_col=return_col, long_high=True
    )


def rmw_factor(panel: pd.DataFrame, return_col: str = "return_t1") -> pd.Series:
    """Robust-Minus-Weak Profitability."""
    return construct_long_short_factor(
        panel, "profitability", return_col=return_col, long_high=True
    )


def cma_factor(panel: pd.DataFrame, return_col: str = "return_t1") -> pd.Series:
    """Conservative-Minus-Aggressive Investment."""
    return construct_long_short_factor(
        panel, "asset_growth", return_col=return_col, long_high=False
    )


def momentum_12_1(prices_panel: pd.DataFrame) -> pd.Series:
    """Akademische 12-1-Momentum-Definition.

    Args:
        prices_panel: DataFrame [date, symbol, close].

    Returns:
        Series indexed by (date, symbol) — return from t-12mo to t-1mo (skip last month).
    """
    df = prices_panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"])
    grp = df.groupby("symbol")["close"]
    df["close_lag_1m"] = grp.shift(21)
    df["close_lag_12m"] = grp.shift(252)
    df["momentum_12_1"] = df["close_lag_1m"] / df["close_lag_12m"] - 1
    return df.set_index(["date", "symbol"])["momentum_12_1"]


__all__ = [
    "construct_long_short_factor",
    "smb_factor",
    "hml_factor",
    "mom_factor",
    "rmw_factor",
    "cma_factor",
    "momentum_12_1",
]
