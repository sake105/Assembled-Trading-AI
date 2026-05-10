"""GARCH-Familie für Vola-Forecasting.

DUPLIKAT-HINWEIS
================
Mainline hat ``src/assembled_core/risk/garch_vol.py`` (187 LoC), GJR-GARCH(1,1)-
spezifisch via ``arch``-Lib mit Vol-Targeting-Sizing. Für Production die mainline.

Diese Erweiterungs-Variante ist allgemeiner (GARCH/EGARCH/GJR-GARCH) mit
NumPy-only-MLE-Fallback ohne arch-Abhängigkeit.

Modelle
-------
- **GARCH(1,1)**: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}.  Engle (1982) + Bollerslev (1986).
- **EGARCH**:    log(σ²_t) = ω + α·g(z_{t-1}) + β·log(σ²_{t-1}).  Asymmetrie via
                 Sign-Term g(z) = θ z + γ(|z| − E|z|).  Nelson (1991).
- **GJR-GARCH**: σ²_t = ω + α·ε²_{t-1} + γ·1[ε_{t-1}<0]·ε²_{t-1} + β·σ²_{t-1}.
                 Glosten/Jagannathan/Runkle (1993).

Implementation: bevorzugt via ``arch``-Library (Sheppard) — installierte Industrie-
Standard-Implementierung.  Fallback auf NumPy-MLE falls nicht verfügbar.

Anwendung
---------
- Vola-Forecast für Vol-Targeting / Position-Sizing
- Conditional VaR über Student-t Innovations
- Vola-Regime-Indikator (Persistenz α + β)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GarchFit:
    model: str
    params: dict[str, float]
    conditional_vol: pd.Series
    log_likelihood: float
    aic: float
    bic: float
    persistence: float  # α + β (close to 1 = high persistence)


def _arch_available() -> bool:
    try:
        import arch  # noqa: F401

        return True
    except ImportError:
        return False


def fit_garch(
    returns: pd.Series,
    model: Literal["GARCH", "EGARCH", "GJR-GARCH"] = "GARCH",
    p: int = 1,
    q: int = 1,
    dist: Literal["normal", "studentst", "ged"] = "studentst",
) -> GarchFit:
    """Fit a GARCH-family model.

    Args:
        returns: pct-returns series (decimal).
        model: 'GARCH', 'EGARCH', 'GJR-GARCH'.
        p, q: AR/MA orders.
        dist: error distribution.

    Returns:
        ``GarchFit`` with conditional vol series + parameter estimates.
    """
    r = returns.dropna() * 100  # arch package expects pct in *percent*
    if len(r) < 50:
        raise ValueError("need >= 50 observations")

    if not _arch_available():
        return _fallback_garch_mle(r / 100)

    from arch import arch_model

    if model == "GARCH":
        am = arch_model(r, vol="GARCH", p=p, q=q, dist=dist)
    elif model == "EGARCH":
        am = arch_model(r, vol="EGARCH", p=p, q=q, dist=dist)
    elif model == "GJR-GARCH":
        am = arch_model(r, vol="GARCH", p=p, o=1, q=q, dist=dist)
    else:
        raise ValueError(f"unknown model: {model}")

    res = am.fit(disp="off", show_warning=False)
    params = dict(res.params)
    cond_vol = (
        pd.Series(res.conditional_volatility, index=r.index) / 100
    )  # back to decimal
    persist = float(params.get("alpha[1]", 0.0)) + float(params.get("beta[1]", 0.0))

    return GarchFit(
        model=model,
        params=params,
        conditional_vol=cond_vol,
        log_likelihood=float(res.loglikelihood),
        aic=float(res.aic),
        bic=float(res.bic),
        persistence=persist,
    )


def _fallback_garch_mle(returns: pd.Series) -> GarchFit:
    """Minimaler GARCH(1,1)-MLE-Fallback ohne arch-Library.

    Optimization via simple grid + scipy.optimize.minimize falls verfügbar.
    """
    r = returns.dropna().values
    var0 = float(np.var(r))

    def garch_var_path(omega: float, alpha: float, beta: float) -> np.ndarray:
        var = np.zeros(len(r))
        var[0] = var0
        for t in range(1, len(r)):
            var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        return var

    def neg_log_lik(theta: np.ndarray) -> float:
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        var = garch_var_path(omega, alpha, beta)
        ll = -0.5 * (np.log(2 * np.pi) + np.log(var) + r**2 / var).sum()
        return -ll

    try:
        from scipy.optimize import minimize  # type: ignore

        res = minimize(
            neg_log_lik,
            x0=np.array([0.05 * var0, 0.1, 0.85]),
            method="L-BFGS-B",
            bounds=[(1e-9, None), (0, 0.999), (0, 0.999)],
        )
        omega, alpha, beta = res.x
        ll = -res.fun
    except ImportError:
        # Crude grid search
        best_ll = -np.inf
        omega = alpha = beta = 0.0
        for a in np.linspace(0.05, 0.2, 4):
            for b in np.linspace(0.7, 0.95, 5):
                if a + b >= 1:
                    continue
                w = (1 - a - b) * var0
                ll = -neg_log_lik(np.array([w, a, b]))
                if ll > best_ll:
                    best_ll = ll
                    omega, alpha, beta = w, a, b
        ll = best_ll

    var_path = garch_var_path(omega, alpha, beta)
    n = len(r)
    aic = 2 * 3 - 2 * ll
    bic = np.log(n) * 3 - 2 * ll

    return GarchFit(
        model="GARCH(fallback)",
        params={"omega": omega, "alpha[1]": alpha, "beta[1]": beta},
        conditional_vol=pd.Series(np.sqrt(var_path), index=returns.dropna().index),
        log_likelihood=float(ll),
        aic=float(aic),
        bic=float(bic),
        persistence=alpha + beta,
    )


def garch_forecast(fit: GarchFit, horizon: int = 5) -> np.ndarray:
    """Multi-step ahead vol forecast (annualized via √252).

    GARCH long-run variance: σ²_∞ = ω / (1 − α − β).
    h-step forecast: σ²_h = σ²_∞ + (α+β)^h × (σ²_t − σ²_∞).
    """
    omega = fit.params.get("omega", 0.0)
    alpha = fit.params.get("alpha[1]", 0.0)
    beta = fit.params.get("beta[1]", 0.0)
    last_var = float(fit.conditional_vol.iloc[-1] ** 2)
    if alpha + beta >= 1:
        # Non-stationary; just return last vol
        return np.full(horizon, np.sqrt(last_var)) * np.sqrt(252)
    long_run = omega / (1 - alpha - beta) if omega > 0 else last_var
    out = np.zeros(horizon)
    for h in range(1, horizon + 1):
        var_h = long_run + (alpha + beta) ** h * (last_var - long_run)
        out[h - 1] = np.sqrt(max(var_h, 0)) * np.sqrt(252)
    return out


__all__ = ["GarchFit", "fit_garch", "garch_forecast"]
