"""Sub-Period-Backtest-Analysis.

Split der Returns-Serie in **vordefinierte Epochen** und Berechnung pro Epoche
aller Standard-Metriken. So zeigt sich, ob eine Strategie in *jeder* Markt-Phase
funktioniert oder nur einen Zeitraum-spezifischen Effekt hat.

Standard-Epochen (US-Equity)
----------------------------
- Pre-2008      :  2003-01-01 ↔ 2008-09-14
- GFC           :  2008-09-15 ↔ 2009-06-30
- Post-GFC      :  2009-07-01 ↔ 2019-12-31
- COVID         :  2020-02-19 ↔ 2020-06-30
- Recovery      :  2020-07-01 ↔ 2021-12-31
- Inflation     :  2022-01-01 ↔ 2022-12-31
- Modern        :  2023-01-01 ↔ heute

Anwendung
---------
- Erkennt Strategien, die nur in Bull-Phasen funktionieren (klassischer Backtest-Trap).
- Liefert "Worst Sub-Period"-Sharpe als ehrlichere Strategy-Bewertung.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Epoch:
    name: str
    start: str
    end: str


STANDARD_EPOCHS_US_EQUITY: list[Epoch] = [
    Epoch("Pre_2008", "2003-01-01", "2008-09-14"),
    Epoch("GFC_2008_2009", "2008-09-15", "2009-06-30"),
    Epoch("Post_GFC", "2009-07-01", "2019-12-31"),
    Epoch("COVID_Crash", "2020-02-19", "2020-06-30"),
    Epoch("Recovery_2020_2021", "2020-07-01", "2021-12-31"),
    Epoch("Inflation_2022", "2022-01-01", "2022-12-31"),
    Epoch("Modern_2023_plus", "2023-01-01", "2030-12-31"),
]


def _to_utc_index(series: pd.Series) -> pd.Series:
    s = pd.Series(series).copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


def sub_period_metrics(
    returns: pd.Series, epochs: list[Epoch] | None = None, annual_factor: float = 252
) -> pd.DataFrame:
    """Compute per-epoch metrics.

    Args:
        returns: pd.Series of daily returns.
        epochs: list of Epoch. Default = STANDARD_EPOCHS_US_EQUITY.
        annual_factor: 252 für Daily.

    Returns:
        DataFrame [epoch, n_obs, mean_daily, ann_return, ann_vol, sharpe,
        sortino, max_dd, win_rate].
    """
    epochs = epochs or STANDARD_EPOCHS_US_EQUITY
    r = _to_utc_index(returns).dropna()
    rows = []
    for epoch in epochs:
        start = pd.Timestamp(epoch.start, tz="UTC")
        end = pd.Timestamp(epoch.end, tz="UTC")
        sub = r.loc[start:end]
        if len(sub) < 5:
            rows.append(
                {
                    "epoch": epoch.name,
                    "n_obs": int(len(sub)),
                    "ann_return": np.nan,
                    "ann_vol": np.nan,
                    "sharpe": np.nan,
                    "sortino": np.nan,
                    "max_dd": np.nan,
                    "win_rate": np.nan,
                }
            )
            continue
        eq = (1 + sub).cumprod()
        max_dd = float((eq / eq.cummax() - 1).min())
        mean_d = float(sub.mean())
        std_d = float(sub.std(ddof=0))
        ann_ret = float(eq.iloc[-1] ** (annual_factor / len(sub)) - 1)
        ann_vol = std_d * np.sqrt(annual_factor)
        sharpe = (
            (mean_d / std_d * np.sqrt(annual_factor)) if std_d > 0 else float("nan")
        )
        downside = sub[sub < 0]
        if len(downside) > 1:
            dd_std = float(np.sqrt((downside**2).mean()))
            sortino = (
                mean_d / dd_std * np.sqrt(annual_factor) if dd_std > 0 else float("nan")
            )
        else:
            sortino = float("nan")
        win = float((sub > 0).mean())
        rows.append(
            {
                "epoch": epoch.name,
                "n_obs": int(len(sub)),
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_dd": max_dd,
                "win_rate": win,
            }
        )
    return pd.DataFrame(rows)


def worst_period_sharpe(returns: pd.Series, epochs: list[Epoch] | None = None) -> dict:
    """Identify worst-epoch Sharpe — ehrliche pessimistische Strategy-Bewertung."""
    df = sub_period_metrics(returns, epochs)
    valid = df.dropna(subset=["sharpe"])
    if valid.empty:
        return {"error": "no valid epochs"}
    worst = valid.loc[valid["sharpe"].idxmin()]
    return {
        "epoch": str(worst["epoch"]),
        "sharpe": float(worst["sharpe"]),
        "ann_return": float(worst["ann_return"]),
        "max_dd": float(worst["max_dd"]),
        "n_obs": int(worst["n_obs"]),
    }


def consistency_score(returns: pd.Series, epochs: list[Epoch] | None = None) -> float:
    """Stability across epochs: 1 − std(sharpe-across-epochs) / mean(sharpe).

    Höher = stabil. Werte ≤ 0 deuten auf periodisch-instabile Strategy.
    """
    df = sub_period_metrics(returns, epochs)
    valid = df.dropna(subset=["sharpe"])
    if len(valid) < 2:
        return float("nan")
    mu = float(valid["sharpe"].mean())
    sd = float(valid["sharpe"].std(ddof=0))
    if abs(mu) < 1e-9:
        return float("nan")
    return 1.0 - sd / abs(mu)


__all__ = [
    "Epoch",
    "STANDARD_EPOCHS_US_EQUITY",
    "sub_period_metrics",
    "worst_period_sharpe",
    "consistency_score",
]
