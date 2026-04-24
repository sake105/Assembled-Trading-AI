"""Extreme Value Theory (EVT): Peaks-Over-Threshold for tail risk.

Historical VaR at 99% confidence relies on the worst 1% of days (~2-3
observations per year).  At 99.9% there is essentially no data.  EVT
extrapolates mathematically correctly using the Pickands-Balkema-de Haan
theorem: exceedances over a high threshold follow a Generalized Pareto
Distribution (GPD).

Outputs: evt_var_99, evt_var_999, evt_cvar_99 — more precise tail-risk
estimates than the historical percentile method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy.stats import genpareto  # type: ignore[import-untyped]

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class EVTResult:
    """Result of EVT Peaks-Over-Threshold analysis."""

    threshold: float  # u: threshold for exceedances
    n_exceedances: int  # number of observations above threshold
    n_total: int  # total observations
    shape_xi: float  # GPD shape parameter (xi > 0 = fat tails)
    scale_sigma: float  # GPD scale parameter
    var_95: float  # 95% VaR (loss, positive number)
    var_99: float  # 99% VaR
    var_999: float  # 99.9% VaR
    cvar_95: float  # 95% CVaR / Expected Shortfall
    cvar_99: float  # 99% CVaR
    return_period_100y: float  # expected loss that occurs once in 100 years


def fit_evt_pot(
    returns: pd.Series | np.ndarray,
    *,
    threshold_quantile: float = 0.95,
    min_exceedances: int = 20,
) -> EVTResult | None:
    """Fit GPD to loss exceedances via Peaks-Over-Threshold.

    Args:
        returns: Daily returns (can be positive or negative).
            Losses are computed as ``-returns`` so that positive values
            represent losses.
        threshold_quantile: Quantile of the loss distribution to use as
            threshold (default 0.95 = top 5% of losses).
        min_exceedances: Minimum exceedances required for a reliable fit.

    Returns:
        :class:`EVTResult` or ``None`` if scipy is unavailable or fit fails.
    """
    if not SCIPY_AVAILABLE:
        logger.debug("[EVT] scipy not installed — skipping")
        return None

    # Convert to numpy losses (positive = bad)
    if isinstance(returns, pd.Series):
        losses = -returns.dropna().values
    else:
        losses = -returns[~np.isnan(returns)]

    n_total = len(losses)
    if n_total < 100:
        logger.debug("[EVT] insufficient data (%d < 100)", n_total)
        return None

    # Threshold
    u = float(np.quantile(losses, threshold_quantile))
    if u <= 0:
        # Threshold at or below zero means most returns are positive (normal)
        # Shift to use absolute losses
        u = float(np.quantile(np.abs(losses), threshold_quantile))
        if u <= 0:
            return None

    # Exceedances
    exceedances = losses[losses > u] - u
    n_exc = len(exceedances)

    if n_exc < min_exceedances:
        logger.debug(
            "[EVT] only %d exceedances above threshold %.4f (need %d)",
            n_exc, u, min_exceedances,
        )
        return None

    # Fit GPD via MLE
    try:
        # scipy.stats.genpareto parametrization: c = shape (xi), scale = sigma
        c, _loc, scale = genpareto.fit(exceedances, floc=0)
    except Exception as exc:
        logger.debug("[EVT] GPD fit failed: %s", exc)
        return None

    xi = float(c)
    sigma = float(scale)

    if sigma <= 0:
        return None

    # Exceedance probability
    p_exceed = n_exc / n_total

    # VaR and CVaR computation via GPD quantile function
    def _gpd_var(p_level: float) -> float:
        """VaR at confidence level p_level (e.g. 0.99)."""
        # Probability of exceeding VaR: 1 - p_level
        # VaR = u + (sigma / xi) * ((n/N_u * (1-p))^(-xi) - 1)  for xi != 0
        p_tail = 1.0 - p_level
        if abs(xi) < 1e-10:
            # Exponential tail (xi → 0)
            return u + sigma * np.log(p_exceed / p_tail)
        else:
            return u + (sigma / xi) * ((p_exceed / p_tail) ** xi - 1.0)

    def _gpd_cvar(p_level: float) -> float:
        """CVaR (Expected Shortfall) at confidence level p_level."""
        var = _gpd_var(p_level)
        if xi >= 1.0:
            return float("inf")  # infinite mean for xi >= 1
        return (var + sigma - xi * u) / (1.0 - xi)

    var_95 = _gpd_var(0.95)
    var_99 = _gpd_var(0.99)
    var_999 = _gpd_var(0.999)
    cvar_95 = _gpd_cvar(0.95)
    cvar_99 = _gpd_cvar(0.99)

    # Return period: expected loss that occurs once in 100 years (~25,200 trading days)
    return_period_p = 1.0 / (100 * 252)  # probability per day
    return_period_100y = _gpd_var(1.0 - return_period_p)

    return EVTResult(
        threshold=round(u, 6),
        n_exceedances=n_exc,
        n_total=n_total,
        shape_xi=round(xi, 4),
        scale_sigma=round(sigma, 6),
        var_95=round(max(0.0, var_95), 6),
        var_99=round(max(0.0, var_99), 6),
        var_999=round(max(0.0, var_999), 6),
        cvar_95=round(max(0.0, cvar_95), 6),
        cvar_99=round(max(0.0, cvar_99), 6),
        return_period_100y=round(max(0.0, return_period_100y), 6),
    )


def compute_evt_risk_metrics(
    returns: pd.Series | np.ndarray,
    *,
    threshold_quantile: float = 0.95,
) -> dict[str, float]:
    """Convenience wrapper returning a flat dict of EVT risk metrics.

    Returns conservative zeros if EVT fit fails (no false safety).
    """
    result = fit_evt_pot(returns, threshold_quantile=threshold_quantile)
    if result is None:
        return {
            "evt_var_95": 0.0,
            "evt_var_99": 0.0,
            "evt_var_999": 0.0,
            "evt_cvar_95": 0.0,
            "evt_cvar_99": 0.0,
            "evt_shape_xi": 0.0,
            "evt_return_period_100y": 0.0,
        }
    return {
        "evt_var_95": result.var_95,
        "evt_var_99": result.var_99,
        "evt_var_999": result.var_999,
        "evt_cvar_95": result.cvar_95,
        "evt_cvar_99": result.cvar_99,
        "evt_shape_xi": result.shape_xi,
        "evt_return_period_100y": result.return_period_100y,
    }


__all__ = [
    "EVTResult",
    "compute_evt_risk_metrics",
    "fit_evt_pot",
]
