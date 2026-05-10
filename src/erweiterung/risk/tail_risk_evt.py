"""Extreme-Value-Theory (EVT) für Tail-Risk-Modeling.

Theorie
-------
Klassische Verteilungen (Gauss, Student-t) unterschätzen extreme Verluste
("fat tails"). EVT modelliert nur den Tail durch eine **Generalized Pareto
Distribution** (GPD):
    P(X − u > x | X > u) = (1 + ξ x / β)^{-1/ξ}

Pickands-Theorem: Für hinreichend hohes Threshold u ist diese Approximation
asymptotisch exakt — unabhängig von der ursprünglichen Verteilung.

Output
------
- ``var_evt(α)``: VaR aus der gefitteten GPD.
- ``cvar_evt(α)``: CVaR (= Expected Shortfall) aus GPD.
- ``tail_index ξ``: ξ > 0 = heavy-tailed (typisch für Aktien-Tails).

Method-of-Moments (statt MLE)
------------------------------
Hier Method-of-Moments für Robustheit ohne SciPy-Abhängigkeit:
    β̂ = mean(excess) * (1 + (mean(excess)/std(excess))²) / 2
    ξ̂ = (1 - (mean(excess)/std(excess))²) / 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GPDFit:
    threshold: float
    xi: float  # shape parameter (tail index)
    beta: float  # scale parameter
    n_excess: int


def fit_gpd(losses: np.ndarray, threshold_quantile: float = 0.95) -> GPDFit:
    """Fit Generalized Pareto Distribution to losses above threshold.

    Args:
        losses: Array von Verlusten (positiv = Verlust). Negative Returns
            müssen vorher negiert werden!
        threshold_quantile: 0.95 = top 5% losses sind Tail.

    Returns:
        GPDFit mit (xi, beta, threshold).
    """
    if len(losses) < 50:
        raise ValueError("need >= 50 samples for stable GPD fit")
    u = float(np.quantile(losses, threshold_quantile))
    excess = losses[losses > u] - u
    if len(excess) < 10:
        raise ValueError("too few exceedances; lower threshold")
    m = float(excess.mean())
    s = float(excess.std(ddof=0))
    if s == 0:
        return GPDFit(threshold=u, xi=0.0, beta=m, n_excess=len(excess))
    cv2 = (m / s) ** 2
    xi = (1 - cv2) / 2
    beta = m * (1 + cv2) / 2
    return GPDFit(threshold=u, xi=xi, beta=beta, n_excess=len(excess))


def var_evt(fit: GPDFit, n_total: int, alpha: float = 0.99) -> float:
    """VaR aus der GPD-Approximation.

    Formel: VaR_α = u + (β/ξ) * ((n/k * (1-α))^(-ξ) - 1) für ξ ≠ 0.
    """
    if fit.n_excess == 0:
        return float("nan")
    p_exceed = fit.n_excess / n_total
    target = (1 - alpha) / p_exceed
    if abs(fit.xi) < 1e-9:
        return fit.threshold + fit.beta * (-np.log(target))
    return fit.threshold + (fit.beta / fit.xi) * (target ** (-fit.xi) - 1)


def cvar_evt(fit: GPDFit, n_total: int, alpha: float = 0.99) -> float:
    """CVaR (Expected Shortfall) aus GPD.

    CVaR_α = (VaR_α + β - ξ u) / (1 - ξ), für ξ < 1.
    """
    if fit.xi >= 1:
        return float("inf")
    var = var_evt(fit, n_total, alpha)
    return (var + fit.beta - fit.xi * fit.threshold) / (1 - fit.xi)


def estimate_tail_metrics(
    returns: pd.Series, alpha: float = 0.99, threshold_quantile: float = 0.95
) -> dict:
    """Komplettes Tail-Risk-Diagnostic für eine Returns-Series.

    Returns:
        Dict mit ``var_hist``, ``cvar_hist``, ``var_evt``, ``cvar_evt``, ``xi``, ``beta``.
    """
    losses = -returns.dropna().values
    if len(losses) < 50:
        return {"error": "too few obs"}
    var_hist = float(np.quantile(losses, alpha))
    tail_hist = losses[losses >= var_hist]
    cvar_hist = float(tail_hist.mean()) if len(tail_hist) > 0 else float("nan")
    fit = fit_gpd(losses, threshold_quantile=threshold_quantile)
    return {
        "var_hist": var_hist,
        "cvar_hist": cvar_hist,
        "var_evt": var_evt(fit, len(losses), alpha=alpha),
        "cvar_evt": cvar_evt(fit, len(losses), alpha=alpha),
        "xi": fit.xi,
        "beta": fit.beta,
        "threshold": fit.threshold,
        "n_excess": fit.n_excess,
    }


__all__ = ["GPDFit", "fit_gpd", "var_evt", "cvar_evt", "estimate_tail_metrics"]
