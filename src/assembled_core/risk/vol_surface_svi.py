"""SVI (Stochastic Volatility Inspired) implied-volatility surface model.

Gatheral (2004) SVI parametrization for the total implied variance smile:

    w(k) = a + b * (rho*(k-m) + sqrt((k-m)**2 + sigma**2))

where
    k     = log(K/F)   log-moneyness
    w     = sigma_imp**2 * T   total implied variance
    a     vertical shift (overall variance level)
    b     wing slope — b >= 0
    rho   skew / asymmetry, -1 < rho < 1
    m     horizontal shift (ATM offset)
    sigma ATM smoothness parameter, sigma > 0

Butterfly no-arbitrage constraint (Lee 2005):
    g(k) = (1 - k*d_w/(2*w))**2 - (d_w**2/4)*(1/w + 1/4) + d2_w/2 >= 0
    for all k.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

try:
    from scipy import optimize as sp_opt

    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------


@dataclass
class SVIParams:
    """Fitted SVI parameters for one expiry slice."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    expiry_T: float  # time to expiry in years
    fit_rmse: float = 0.0  # root-mean-squared error of the fit (total var units)

    def is_valid(self) -> bool:
        """True when SVI constraints are satisfied."""
        return (
            self.b >= 0
            and -1.0 < self.rho < 1.0
            and self.sigma > 0
            and (self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2)) >= -1e-8
        )


# ---------------------------------------------------------------------------
# SVI evaluation
# ---------------------------------------------------------------------------


def svi_total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """Evaluate w(k) = total implied variance for log-moneyness array k."""
    d = k - params.m
    return cast(
        np.ndarray,
        params.a + params.b * (params.rho * d + np.sqrt(d**2 + params.sigma**2)),
    )


def svi_implied_vol(
    log_moneyness: np.ndarray,
    params: SVIParams,
) -> np.ndarray:
    """Return implied volatility (annualised) from SVI total variance.

    sigma_imp = sqrt(w(k) / T)
    """
    T = max(params.expiry_T, 1e-8)
    w = np.maximum(svi_total_variance(log_moneyness, params), 0.0)
    return cast(np.ndarray, np.sqrt(w / T))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_svi(
    log_moneyness: np.ndarray,
    market_total_var: np.ndarray,
    expiry_T: float,
    *,
    a0: float | None = None,
    method: str = "L-BFGS-B",
) -> SVIParams | None:
    """Fit SVI parameters to observed (k, w) smile data.

    Args:
        log_moneyness: Array of log(K/F) values.
        market_total_var: Corresponding total implied variances (sigma_imp**2 * T).
        expiry_T: Time to expiry in years.
        a0: Initial guess for *a*; defaults to median(market_total_var).
        method: scipy minimisation method.

    Returns:
        Fitted :class:`SVIParams` or ``None`` when scipy unavailable / fit fails.
    """
    if not _SCIPY:
        log.warning("[SVI] scipy not available — cannot fit SVI surface")
        return None

    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(market_total_var, dtype=float)
    mask = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[mask], w[mask]
    if len(k) < 5:
        log.warning("[SVI] Too few valid observations (%d) for fitting", len(k))
        return None

    a_init = float(np.median(w)) if a0 is None else a0

    # Initial guess: [a, b, rho, m, sigma]
    x0 = [a_init, 0.1, -0.3, 0.0, 0.1]

    # Bounds: a unconstrained, b>=0, -1<rho<1, m unconstrained, sigma>0
    bounds = [
        (None, None),  # a
        (1e-6, None),  # b
        (-0.999, 0.999),  # rho
        (None, None),  # m
        (1e-6, None),  # sigma
    ]

    def objective(x: np.ndarray) -> float:
        a, b, rho, m, sigma = x
        p = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, expiry_T=expiry_T)
        w_hat = svi_total_variance(k, p)
        # Extra penalty for constraint violation
        penalty = float(max(0.0, -(a + b * sigma * np.sqrt(1 - rho**2)))) * 1e4
        return float(np.mean((w_hat - w) ** 2)) + penalty

    try:
        res = sp_opt.minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        a, b, rho, m, sigma = res.x
        params = SVIParams(
            a=float(a),
            b=float(b),
            rho=float(rho),
            m=float(m),
            sigma=float(sigma),
            expiry_T=expiry_T,
            fit_rmse=float(np.sqrt(res.fun)),
        )
        if not params.is_valid():
            log.warning("[SVI] Fitted parameters violate constraints: %s", params)
        return params
    except Exception as exc:
        log.warning("[SVI] Fitting failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Arbitrage checks
# ---------------------------------------------------------------------------


def _svi_derivatives(k: np.ndarray, params: SVIParams) -> tuple[np.ndarray, np.ndarray]:
    """Return (dw/dk, d²w/dk²) evaluated at k."""
    d = k - params.m
    sq = np.sqrt(d**2 + params.sigma**2)
    dw = params.b * (params.rho + d / sq)
    d2w = params.b * params.sigma**2 / sq**3
    return dw, d2w


def butterfly_arbitrage_free(
    params: SVIParams,
    k_grid: np.ndarray | None = None,
) -> dict[str, object]:
    """Check Lee (2005) butterfly no-arbitrage condition.

    g(k) = (1 - k*dw/(2*w))² - (dw²/4)*(1/w + 1/4) + d²w/2 >= 0

    Returns dict with ``arbitrage_free`` (bool), ``min_g``, ``n_violations``.
    """
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 200)

    w = svi_total_variance(k_grid, params)
    dw, d2w = _svi_derivatives(k_grid, params)
    w = np.maximum(w, 1e-12)
    g = (
        (1.0 - k_grid * dw / (2.0 * w)) ** 2
        - (dw**2 / 4.0) * (1.0 / w + 0.25)
        + d2w / 2.0
    )

    n_violations = int(np.sum(g < -1e-6))
    return {
        "arbitrage_free": n_violations == 0,
        "min_g": float(np.min(g)),
        "n_violations": n_violations,
    }


def surface_summary(params: SVIParams) -> dict[str, float]:
    """Return smile diagnostics at ATM and ±1σ wings."""
    k_atm = 0.0
    w_atm = float(svi_total_variance(np.array([k_atm]), params)[0])
    iv_atm = float(svi_implied_vol(np.array([k_atm]), params)[0])
    dw, _ = _svi_derivatives(np.array([k_atm]), params)
    skew = float(dw[0])  # dw/dk at ATM ≈ vol skew slope

    k_wing = np.array([-0.25, 0.25])
    iv_wings = svi_implied_vol(k_wing, params)
    put_wing_iv = float(iv_wings[0])
    call_wing_iv = float(iv_wings[1])

    return {
        "atm_total_var": round(w_atm, 6),
        "atm_iv": round(iv_atm, 4),
        "skew_dw_dk": round(skew, 4),
        "put_wing_iv": round(put_wing_iv, 4),
        "call_wing_iv": round(call_wing_iv, 4),
        "expiry_T": params.expiry_T,
        "fit_rmse": params.fit_rmse,
    }


__all__ = [
    "SVIParams",
    "fit_svi",
    "svi_implied_vol",
    "svi_total_variance",
    "butterfly_arbitrage_free",
    "surface_summary",
]
