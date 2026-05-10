"""Faktor-Neutralisierung: Sektor-, Multi-Faktor-, Beta-Neutralisierung.

Theorie
-------
Roher Signal kann von "unwanted exposures" (Sektor, Größe, Vola) dominiert sein.
Neutralisierung extrahiert die orthogonale Komponente.

Methoden
--------
1. **Demean per Group**: Sektor-neutral durch Sektor-Mittelwert-Abzug.
2. **Multi-Variate-Regression**: Residuum aus OLS gegen Sektor-Dummies + Beta + Size.
3. **Industry-Adjusted-Score**: rank within industry, then standardize.

PIT
---
Alle Operationen sind per (date, group) — sekretär-PIT-sicher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sector_demean(
    panel: pd.DataFrame,
    value_col: str,
    sector_col: str = "sector",
    date_col: str = "date",
) -> pd.Series:
    """Demean cross-sectionally innerhalb von (date, sector)."""
    if panel.empty:
        return pd.Series(dtype=float)
    grp = panel.groupby([date_col, sector_col])[value_col]
    return panel[value_col] - grp.transform("mean")


def industry_rank_normalize(
    panel: pd.DataFrame,
    value_col: str,
    sector_col: str = "sector",
    date_col: str = "date",
) -> pd.Series:
    """Rank within (date, sector) und konvertiere zu Z-Score-Skala."""
    if panel.empty:
        return pd.Series(dtype=float)
    grp = panel.groupby([date_col, sector_col])[value_col]
    pct = grp.rank(pct=True)
    # Map percentile to standard normal via Φ⁻¹(rank)
    eps = 1e-6
    pct_clipped = pct.clip(eps, 1 - eps)
    # erfinv approximation
    try:
        from scipy.special import erfinv  # type: ignore

        z = np.sqrt(2.0) * erfinv(2 * pct_clipped.values - 1)
    except ImportError:
        # Winitzki approximation
        a = 0.147
        x = 2 * pct_clipped.values - 1
        ln1m = np.log(np.clip(1 - x * x, 1e-10, None))
        first = 2 / (np.pi * a) + ln1m / 2
        ainv = np.sign(x) * np.sqrt(np.sqrt(first**2 - ln1m / a) - first)
        z = np.sqrt(2.0) * ainv
    return pd.Series(z, index=pct.index)


def regress_neutralize(
    panel: pd.DataFrame,
    value_col: str,
    feature_cols: list[str],
    date_col: str = "date",
) -> pd.Series:
    """Cross-sectional OLS-Residual: ``value_col ~ feature_cols`` per (date).

    Liefert das Residuum, das frei von linearen Beziehungen zu features ist.
    """
    if panel.empty:
        return pd.Series(dtype=float)
    out = pd.Series(np.nan, index=panel.index)
    for d, g in panel.groupby(date_col):
        sub = g.dropna(subset=[value_col] + feature_cols)
        if len(sub) < len(feature_cols) + 5:
            continue
        X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in feature_cols])
        y = sub[value_col].values
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = y - X @ beta
        out.loc[sub.index] = residual
    return out


__all__ = ["sector_demean", "industry_rank_normalize", "regress_neutralize"]
