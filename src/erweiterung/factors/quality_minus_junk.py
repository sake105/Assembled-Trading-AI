"""Quality-Minus-Junk (QMJ) — Asness/Frazzini/Pedersen 2019.

Reference
---------
Asness, C., Frazzini, A. & Pedersen, L. (2019). Quality minus junk.
*Review of Accounting Studies* 24, 34-112.

Idee
----
"Quality" ist eine multidimensionale Eigenschaft eines Unternehmens:
1. **Profitability**:    Gross Profits / Assets, ROE, ROA
2. **Growth**:           5-year change in profitability metrics
3. **Safety**:           Low beta, low leverage, low ROE volatility
4. **Payout**:           Net equity issuance, total payout ratio

QMJ-Score = z(profitability) + z(growth) + z(safety) + z(payout).

Long high-Quality, short low-Quality liefert positive Sharpe in Equity-
Markets weltweit (~0.5-1.0 abhängig vom Markt).

Implementation
--------------
Hier ist eine **kompakte Quality-Score-Konstruktion** auf Basis von
Cross-Section-Z-Scores. Voraussetzung: Panel mit Spalten ``gross_profit``,
``total_assets``, ``net_income``, ``equity``, ``debt``, ``beta``,
``equity_issuance``, ``dividends``.

Wenn Daten fehlen: graceful degradation (Komponente skipt).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _z_per_date(df: pd.DataFrame, col: str, by: str = "date") -> pd.Series:
    """Cross-sectional z-score per date."""
    grp = df.groupby(by)[col]
    return (df[col] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)


def profitability_score(panel: pd.DataFrame) -> pd.Series:
    """Composite Profitability-Score = z(gross_profitability) + z(roe) + z(roa).

    Args:
        panel: DataFrame mit [date, symbol, gross_profit, total_assets,
            net_income, equity].

    Returns:
        Series indexed wie panel.index — Profitability-z-Score.
    """
    df = panel.copy()
    components = []
    if {"gross_profit", "total_assets"}.issubset(df.columns):
        df["gross_profitability"] = df["gross_profit"] / df["total_assets"].replace(
            0, np.nan
        )
        components.append(_z_per_date(df, "gross_profitability"))
    if {"net_income", "equity"}.issubset(df.columns):
        df["roe"] = df["net_income"] / df["equity"].replace(0, np.nan)
        components.append(_z_per_date(df, "roe"))
    if {"net_income", "total_assets"}.issubset(df.columns):
        df["roa"] = df["net_income"] / df["total_assets"].replace(0, np.nan)
        components.append(_z_per_date(df, "roa"))
    if not components:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(components, axis=1).mean(axis=1)


def safety_score(panel: pd.DataFrame) -> pd.Series:
    """Safety-Score = -z(beta) - z(debt/equity) - z(roe_vol).

    Höher = sicherer.
    """
    df = panel.copy()
    components = []
    if "beta" in df.columns:
        components.append(-_z_per_date(df, "beta"))
    if {"debt", "equity"}.issubset(df.columns):
        df["leverage"] = df["debt"] / df["equity"].replace(0, np.nan)
        components.append(-_z_per_date(df, "leverage"))
    if "roe_vol" in df.columns:
        components.append(-_z_per_date(df, "roe_vol"))
    if not components:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(components, axis=1).mean(axis=1)


def growth_score(panel: pd.DataFrame, lookback_years: int = 5) -> pd.Series:
    """Growth = z(change in profitability over lookback).

    Args:
        panel: DataFrame mit [date, symbol, gross_profitability].
        lookback_years: Standard 5 Jahre.

    Returns:
        Series of growth z-scores.
    """
    if "gross_profitability" not in panel.columns:
        return pd.Series(np.nan, index=panel.index)
    df = panel.copy().sort_values(["symbol", "date"])
    # 5-year growth = current - 5y-ago, both as fractions
    n_periods = lookback_years * 252  # daily approx; should be re-grained
    df["gp_lag"] = df.groupby("symbol")["gross_profitability"].shift(n_periods)
    df["gp_growth"] = df["gross_profitability"] - df["gp_lag"]
    return _z_per_date(df, "gp_growth")


def payout_score(panel: pd.DataFrame) -> pd.Series:
    """Payout = z(dividend_yield) - z(net_equity_issuance)."""
    df = panel.copy()
    components = []
    if "dividend_yield" in df.columns:
        components.append(_z_per_date(df, "dividend_yield"))
    if "net_equity_issuance" in df.columns:
        components.append(-_z_per_date(df, "net_equity_issuance"))
    if not components:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(components, axis=1).mean(axis=1)


def quality_score(
    panel: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Composite Quality-Score nach Asness/Frazzini/Pedersen.

    Args:
        panel: Panel mit Fundamentaldaten.
        weights: dict mit ``profitability``, ``growth``, ``safety``, ``payout``.
            Default: equal weights.

    Returns:
        Series of QMJ-z-Scores.
    """
    weights = weights or {
        "profitability": 0.25,
        "growth": 0.25,
        "safety": 0.25,
        "payout": 0.25,
    }
    components = {
        "profitability": profitability_score(panel),
        "growth": growth_score(panel),
        "safety": safety_score(panel),
        "payout": payout_score(panel),
    }
    valid = {k: v for k, v in components.items() if v.notna().any()}
    if not valid:
        return pd.Series(np.nan, index=panel.index)
    total_w = sum(weights[k] for k in valid)
    out = sum(weights[k] / total_w * v.fillna(0) for k, v in valid.items())
    return out


__all__ = [
    "profitability_score",
    "safety_score",
    "growth_score",
    "payout_score",
    "quality_score",
]
