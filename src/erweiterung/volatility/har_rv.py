"""HAR-RV — Heterogeneous Autoregressive Realized Volatility (Corsi 2009).

Theorie
-------
Realized Volatility hat lange-Memory-Strukturen mit fraktionaler Integration.
HAR-RV approximiert das **ohne** Long-Memory-Schätzung durch drei Lags:
    RV_t = β_0 + β_d·RV_{t-1} + β_w·RV^w_{t-1} + β_m·RV^m_{t-1} + ε_t

mit
- RV^w = mean(RV) der letzten 5 Tage (Wochen-Komponente)
- RV^m = mean(RV) der letzten 22 Tage (Monats-Komponente)

Ist eines der besten Vola-Forecasting-Modelle (siehe Andersen/Bollerslev/Diebold 2007).

Erweiterung HAR-RV-J
--------------------
Trennt diffusive Komponente und Jump-Komponente:
    RV_t = β_0 + β_d·C_{t-1} + β_w·C^w + β_m·C^m + γ_d·J_{t-1} + ...
mit C = continuous bipower variation, J = jump component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HARFit:
    beta0: float
    beta_d: float
    beta_w: float
    beta_m: float
    in_sample_r2: float
    n_obs: int


def fit_har_rv(rv: pd.Series) -> HARFit:
    """Fit HAR-RV via OLS.

    Args:
        rv: Realized Vola Series (e.g. annualized daily, indexed by date).

    Returns:
        ``HARFit`` mit OLS-Koeffizienten + In-Sample R².
    """
    s = pd.Series(rv).dropna()
    if len(s) < 60:
        raise ValueError("need >= 60 RV observations")
    rvw = s.rolling(5, min_periods=5).mean()
    rvm = s.rolling(22, min_periods=22).mean()
    df = pd.DataFrame(
        {"rv": s, "d": s.shift(1), "w": rvw.shift(1), "m": rvm.shift(1)}
    ).dropna()
    if len(df) < 30:
        raise ValueError("not enough data after shifts")
    X = np.column_stack([np.ones(len(df)), df["d"], df["w"], df["m"]])
    y = df["rv"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return HARFit(
        beta0=float(beta[0]),
        beta_d=float(beta[1]),
        beta_w=float(beta[2]),
        beta_m=float(beta[3]),
        in_sample_r2=r2,
        n_obs=len(df),
    )


def har_forecast(fit: HARFit, last_rv_history: pd.Series) -> float:
    """Ein-Schritt-Vorausschätzung des nächsten RV-Wertes."""
    s = pd.Series(last_rv_history).dropna()
    if len(s) < 22:
        return float("nan")
    return float(
        fit.beta0
        + fit.beta_d * s.iloc[-1]
        + fit.beta_w * s.iloc[-5:].mean()
        + fit.beta_m * s.iloc[-22:].mean()
    )


__all__ = ["HARFit", "fit_har_rv", "har_forecast"]
