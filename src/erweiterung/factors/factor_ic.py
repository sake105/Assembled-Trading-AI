"""Information-Coefficient + Alpha-Decay-Analyse.

Information-Coefficient (IC)
----------------------------
- **Pearson-IC**: ρ(forecast, future_return) — linear.
- **Rank-IC** (Spearman): rank-correlation, robuster gegen Outlier.
- **Top-K-Hit-Rate**: Anteil der "top K"-Vorhersagen, die positive Returns liefern.

IC ist die Standardmetrik für Faktor-/Signal-Bewertung. Akademisch:
- IR_strategy ≈ IC × √breadth  (Fundamental-Law-of-Active-Management, Grinold 1989)

Alpha-Decay
-----------
Wie schnell verliert das Signal seine Vorhersagekraft über den Forward-Horizont?
    decay_h = IC(signal_t, return_{t+1, t+h}) für h = 1, 5, 10, 21, 63 Tage
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pearson_ic(signal: pd.Series, future_return: pd.Series) -> float:
    """Pearson-Correlation zwischen signal und future-return."""
    df = pd.concat([signal, future_return], axis=1).dropna()
    if len(df) < 30:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def rank_ic(signal: pd.Series, future_return: pd.Series) -> float:
    """Spearman-Rank-Correlation."""
    df = pd.concat([signal, future_return], axis=1).dropna()
    if len(df) < 30:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman"))


def cross_sectional_ic(
    panel: pd.DataFrame,
    signal_col: str,
    return_col: str = "return_t1",
    method: str = "spearman",
) -> pd.Series:
    """Zeitreihe der täglichen Cross-Sectional-IC.

    Args:
        panel: DataFrame [date, symbol, signal_col, return_col].
        signal_col: Spalte des Signals.
        return_col: Forward-Return.
        method: 'pearson' | 'spearman'.

    Returns:
        Series indexed by date — IC-Wert je Tag.
    """
    if panel.empty:
        return pd.Series(dtype=float)
    out = []
    for d, g in panel.groupby("date"):
        sub = g.dropna(subset=[signal_col, return_col])
        if len(sub) < 10:
            continue
        ic = sub[signal_col].corr(sub[return_col], method=method)
        out.append({"date": d, "ic": ic})
    return pd.DataFrame(out).set_index("date")["ic"]


def ic_summary(ic_series: pd.Series) -> dict:
    """Ic mean + IR + sign rate."""
    s = ic_series.dropna()
    if s.empty:
        return {"error": "empty"}
    return {
        "ic_mean": float(s.mean()),
        "ic_std": float(s.std()),
        "ic_ir": (
            float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else float("nan")
        ),
        "sign_rate": float((s > 0).mean()),
        "n_obs": int(len(s)),
    }


def alpha_decay_curve(
    panel: pd.DataFrame,
    signal_col: str,
    prices_panel: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 21, 63),
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute IC at multiple forward horizons.

    Args:
        panel: DataFrame [date, symbol, signal_col].
        signal_col: signal column.
        prices_panel: DataFrame [date, symbol, close].
        horizons: list of forward-day-horizons.
        method: 'pearson' | 'spearman'.

    Returns:
        DataFrame [horizon, ic_mean, ic_ir, sign_rate, n_obs].
    """
    rows = []
    pivot_close = prices_panel.pivot_table(
        index="date", columns="symbol", values="close"
    )
    for h in horizons:
        fwd = pivot_close.shift(-h) / pivot_close - 1
        long_fwd = fwd.stack().reset_index().rename(columns={0: f"return_{h}d"})
        long_fwd.columns = ["date", "symbol", f"return_{h}d"]
        merged = panel.merge(long_fwd, on=["date", "symbol"], how="left")
        ic_ts = cross_sectional_ic(merged, signal_col, f"return_{h}d", method=method)
        summ = ic_summary(ic_ts)
        summ["horizon"] = h
        rows.append(summ)
    return pd.DataFrame(rows)


__all__ = [
    "pearson_ic",
    "rank_ic",
    "cross_sectional_ic",
    "ic_summary",
    "alpha_decay_curve",
]
