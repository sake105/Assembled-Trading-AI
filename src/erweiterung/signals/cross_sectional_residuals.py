"""Cross-Sectional Residual Signals — Sector- und Beta-Neutralisierung.

DUPLIKAT-HINWEIS
================
Mainline hat ``src/assembled_core/features/residual_momentum.py`` (171 LoC) mit
``compute_residual_momentum`` und ``cross_sectional_residual_momentum``,
optional Fama-French-3-Faktor-basiert.

Diese Erweiterungs-Variante hat eine **andere API** (long-format Panel-DataFrame
statt wide-Format-Pivot) und liefert zusätzlich ``residual_reversal`` und
``residual_volatility`` als verwandte Signale. Komplementär nutzbar.

Theorie
-------
Naïve Momentum/Mean-Reversion-Signale werden in der Praxis von **Sektor- und
Marktbewegungen dominiert**: Wenn AAPL um 5 % steigt, kann das Apple-spezifisch
sein, oder einfach "Tech-Rally", oder "SPY +4 %".

Lösung: residualisiere die Renditen
    r_{i,t} = α_i + β_{mkt,i} * r_{mkt,t} + β_{sec,i} * r_{sec,t} + ε_{i,t}

und nutze ε_{i,t} (idiosynkratische Komponente) als Signalbasis.

Implementierung
---------------
- Rolling 60-Tage-OLS je (symbol, sector_etf).
- ``market_proxy`` default = 'SPY', ``sector_etf_map`` default für SP500-Sektoren.
- Rolling-Window vermeidet Look-ahead.

Resultat-Signale
----------------
1. ``residual_momentum`` : Σ(ε_{i,τ}) für τ in [t-21, t-1] (1-Monat-Residual-Mom).
2. ``residual_reversal`` : -ε_{i,t-1} (kurzfristiges Mean-Reversion auf Residual).
3. ``residual_volatility``: σ(ε_{i,τ}) — niedrige IV-Vol ist akademisch zu
   höheren Forward-Returns verbunden ("low-vol anomaly", Frazzini/Pedersen).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _rolling_ols_residuals(y: pd.Series, X: pd.DataFrame, window: int) -> pd.Series:
    """Rolling OLS y ~ X. Liefert Residual-Series in y-Index."""
    if len(y) < window:
        return pd.Series(np.nan, index=y.index)

    out = pd.Series(np.nan, index=y.index)
    yv = y.values
    Xv = X.values
    for end in range(window, len(y) + 1):
        start = end - window
        Xw = Xv[start:end]
        yw = yv[start:end]
        # Drop NaN rows
        mask = ~(np.isnan(yw) | np.isnan(Xw).any(axis=1))
        if mask.sum() < window // 2:
            continue
        Xc = np.column_stack([np.ones(mask.sum()), Xw[mask]])
        yc = yw[mask]
        try:
            beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        except np.linalg.LinAlgError:
            continue
        # Residuum für die letzte Beobachtung im Fenster:
        x_last = np.concatenate([[1.0], Xv[end - 1]])
        if not np.isfinite(x_last).all():
            continue
        pred = float(np.dot(beta, x_last))
        actual = float(y.iloc[end - 1])
        if np.isfinite(actual):
            out.iloc[end - 1] = actual - pred
    return out


def compute_residual_returns(
    returns_panel: pd.DataFrame,
    sector_map: dict[str, str],
    sector_etf_returns: dict[str, pd.Series],
    market_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Berechne idiosynkratische Residual-Returns je Symbol.

    Args:
        returns_panel: long DataFrame [date, symbol, return].
        sector_map: ``{symbol: sector_etf_ticker}``.
        sector_etf_returns: ``{etf_ticker: Series indexed by date}``.
        market_returns: Series indexed by date (e.g. SPY-Returns).
        window: Rolling-OLS-Fenster (Tage).

    Returns:
        DataFrame [date, symbol, residual_return].
    """
    if returns_panel.empty:
        return returns_panel.assign(residual_return=pd.Series(dtype=float))

    out_frames: list[pd.DataFrame] = []
    for sym, sym_df in returns_panel.groupby("symbol"):
        sec = sector_map.get(sym)
        if not sec or sec not in sector_etf_returns:
            continue
        df = sym_df.set_index("date")[["return"]].copy()
        df["mkt"] = market_returns.reindex(df.index)
        df["sec"] = sector_etf_returns[sec].reindex(df.index)
        df = df.dropna()
        if len(df) < window:
            continue
        eps = _rolling_ols_residuals(df["return"], df[["mkt", "sec"]], window=window)
        sub = pd.DataFrame(
            {
                "date": df.index,
                "symbol": sym,
                "residual_return": eps.values,
            }
        )
        out_frames.append(sub)
    if not out_frames:
        return pd.DataFrame(columns=["date", "symbol", "residual_return"])
    return pd.concat(out_frames, ignore_index=True)


def residual_momentum(
    residuals: pd.DataFrame, lookback: int = 21, skip: int = 1
) -> pd.DataFrame:
    """Sum der Residual-Returns über [t-lookback-skip, t-skip] — klassischer
    1-Monats-Residual-Momentum mit Skip-1-Day (gegen Reversal-Bias)."""
    if residuals.empty:
        return residuals.assign(residual_momentum=pd.Series(dtype=float))
    out = residuals.sort_values(["symbol", "date"]).copy()
    grp = out.groupby("symbol", group_keys=False)
    shifted = grp["residual_return"].shift(skip)
    out["residual_momentum"] = shifted.groupby(out["symbol"]).transform(
        lambda s: s.rolling(lookback).sum()
    )
    return out


def residual_reversal(residuals: pd.DataFrame) -> pd.DataFrame:
    """Kurzfristiges Mean-Reversion-Signal: ``-ε_{t-1}``."""
    out = residuals.sort_values(["symbol", "date"]).copy()
    out["residual_reversal"] = -out.groupby("symbol")["residual_return"].shift(1)
    return out


def residual_volatility(residuals: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Idiosynkratische Vola — niedrig = ``low-vol anomaly``."""
    out = residuals.sort_values(["symbol", "date"]).copy()
    grp = out.groupby("symbol", group_keys=False)
    out["residual_volatility"] = grp["residual_return"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=window // 2).std()
    )
    return out


def neutralize_cross_section(
    df: pd.DataFrame,
    value_col: str,
    sector_col: str = "sector",
    date_col: str = "date",
) -> pd.Series:
    """Demean a value cross-sectionally per (date, sector) — entfernt
    Sektor-Bias je Datum.

    Anschließendes Z-Scoring stellt sicher, dass das Signal eine
    Sektor-/Markt-neutrale Cross-Section darstellt.
    """
    if df.empty:
        return pd.Series(dtype=float)

    grp = df.groupby([date_col, sector_col])[value_col]
    demeaned = df[value_col] - grp.transform("mean")
    # Z-Score per Tag (demeaning is sector-relative, scale is per-day)
    by_date = df.groupby(date_col)
    sd = by_date[value_col].transform("std").replace(0, np.nan)
    return demeaned / sd


__all__ = [
    "compute_residual_returns",
    "residual_momentum",
    "residual_reversal",
    "residual_volatility",
    "neutralize_cross_section",
]
