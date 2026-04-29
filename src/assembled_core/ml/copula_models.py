"""Copula models for tail dependence analysis.

Linear correlation understates joint tail risk.  In normal markets pairwise
correlation may be 0.3, but during crises tail dependence can be much
stronger.  Copula models capture this non-linear dependence structure.

Implements:
  - Clayton copula (lower tail dependence — joint crash risk)
  - Gumbel copula (upper tail dependence — joint rally)
  - Gaussian copula (symmetric, no tail dependence — baseline)

Uses scipy for marginal CDF transforms and MLE fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy import optimize as sp_opt
    from scipy import stats as sp_stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class CopulaResult:
    """Result of copula fitting for an asset pair."""

    symbol_a: str
    symbol_b: str
    best_copula: str  # "clayton", "gumbel", or "gaussian"
    theta: float  # copula parameter
    lower_tail_dep: float  # lambda_L (Clayton)
    upper_tail_dep: float  # lambda_U (Gumbel)
    log_likelihood: float
    n_obs: int


def _rank_transform(x: np.ndarray) -> np.ndarray:
    """Transform data to pseudo-uniform [0,1] via empirical CDF (rank transform)."""
    n = len(x)
    ranks = sp_stats.rankdata(x)
    # Scale to (0, 1) open interval to avoid boundary issues
    return ranks / (n + 1)


def _clayton_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    """Log-density of Clayton copula."""
    if theta <= 0:
        return -np.inf
    _n = len(u)
    # c(u,v) = (1+theta) * (u*v)^{-(1+theta)} * (u^{-theta} + v^{-theta} - 1)^{-(2+1/theta)}
    eps = 1e-10
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    term1 = np.log(1 + theta)
    term2 = -(1 + theta) * (np.log(u) + np.log(v))
    inner = u ** (-theta) + v ** (-theta) - 1.0
    inner = np.maximum(inner, eps)
    term3 = -(2 + 1.0 / theta) * np.log(inner)

    return float(np.sum(term1 + term2 + term3))


def _gumbel_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    """Log-density of Gumbel copula (approximate via finite differences)."""
    if theta < 1:
        return -np.inf
    eps = 1e-10
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    # Gumbel CDF: C(u,v) = exp(-((- ln u)^theta + (- ln v)^theta)^(1/theta))
    lu = (-np.log(u)) ** theta
    lv = (-np.log(v)) ** theta
    A = (lu + lv) ** (1.0 / theta)

    # Log-density via mixed partial derivative (computed analytically)
    # c(u,v) = C(u,v) * (1/(u*v)) * A^{theta-2} * ((- ln u)(- ln v))^{theta-1}
    #          * (1 + (theta-1)/A)
    _C_uv = np.exp(-A)
    log_C = -A

    log_neg_ln_u = np.log(np.maximum(-np.log(u), eps))
    log_neg_ln_v = np.log(np.maximum(-np.log(v), eps))

    log_density = (
        log_C
        - np.log(u) - np.log(v)
        + (theta - 2) * np.log(np.maximum(A, eps))
        + (theta - 1) * (log_neg_ln_u + log_neg_ln_v)
        + np.log(np.maximum(1 + (theta - 1) / np.maximum(A, eps), eps))
    )

    total = float(np.sum(log_density))
    if not np.isfinite(total):
        return -np.inf
    return total


def _gaussian_logpdf(u: np.ndarray, v: np.ndarray, rho: float) -> float:
    """Log-density of Gaussian copula."""
    if abs(rho) >= 1:
        return -np.inf
    eps = 1e-10
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    x = sp_stats.norm.ppf(u)
    y = sp_stats.norm.ppf(v)

    # c(u,v) = (1/sqrt(1-rho²)) * exp(-(rho²(x²+y²) - 2*rho*x*y) / (2*(1-rho²)))
    r2 = rho ** 2
    denom = 1 - r2
    log_density = (
        -0.5 * np.log(denom)
        - (r2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * denom)
    )
    return float(np.sum(log_density))


def fit_copula_pair(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    symbol_a: str = "A",
    symbol_b: str = "B",
) -> CopulaResult | None:
    """Fit Clayton, Gumbel and Gaussian copulas to a pair of return series.

    Selects the best model by log-likelihood.

    Args:
        returns_a: Daily returns for asset A.
        returns_b: Daily returns for asset B.
        symbol_a: Label for asset A.
        symbol_b: Label for asset B.

    Returns:
        :class:`CopulaResult` or ``None`` on failure.
    """
    if not SCIPY_AVAILABLE:
        return None

    # Align and clean
    mask = np.isfinite(returns_a) & np.isfinite(returns_b)
    ra = returns_a[mask]
    rb = returns_b[mask]

    if len(ra) < 50:
        return None

    # Rank transform to pseudo-uniform
    u = _rank_transform(ra)
    v = _rank_transform(rb)

    results: list[tuple[str, float, float, float, float]] = []
    # (name, theta, lower_tail, upper_tail, loglik)

    # --- Clayton ---
    try:
        def neg_ll_clayton(theta: float) -> float:
            return -_clayton_logpdf(u, v, theta)

        res_c = sp_opt.minimize_scalar(neg_ll_clayton, bounds=(0.01, 20.0), method="bounded")
        if res_c.success:
            theta_c = float(res_c.x)
            ll_c = -float(res_c.fun)
            lambda_L = 2.0 ** (-1.0 / theta_c) if theta_c > 0 else 0.0
            results.append(("clayton", theta_c, lambda_L, 0.0, ll_c))
    except Exception as exc:
        logger.warning("[CopulaModels] Clayton fit failed: %s", exc)

    # --- Gumbel ---
    try:
        def neg_ll_gumbel(theta: float) -> float:
            return -_gumbel_logpdf(u, v, theta)

        res_g = sp_opt.minimize_scalar(neg_ll_gumbel, bounds=(1.01, 20.0), method="bounded")
        if res_g.success:
            theta_g = float(res_g.x)
            ll_g = -float(res_g.fun)
            lambda_U = 2.0 - 2.0 ** (1.0 / theta_g) if theta_g > 1 else 0.0
            results.append(("gumbel", theta_g, 0.0, lambda_U, ll_g))
    except Exception as exc:
        logger.warning("[CopulaModels] Gumbel fit failed: %s", exc)

    # --- Gaussian ---
    try:
        def neg_ll_gauss(rho: float) -> float:
            return -_gaussian_logpdf(u, v, rho)

        res_n = sp_opt.minimize_scalar(neg_ll_gauss, bounds=(-0.99, 0.99), method="bounded")
        if res_n.success:
            rho = float(res_n.x)
            ll_n = -float(res_n.fun)
            results.append(("gaussian", rho, 0.0, 0.0, ll_n))
    except Exception as exc:
        logger.warning("[CopulaModels] Gaussian fit failed: %s", exc)

    if not results:
        return None

    # Select best by log-likelihood
    best = max(results, key=lambda x: x[4])
    name, theta, lower_td, upper_td, loglik = best

    return CopulaResult(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        best_copula=name,
        theta=round(theta, 4),
        lower_tail_dep=round(lower_td, 4),
        upper_tail_dep=round(upper_td, 4),
        log_likelihood=round(loglik, 2),
        n_obs=len(ra),
    )


def compute_portfolio_tail_risk(
    returns_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute portfolio-level tail dependence metrics.

    Fits Clayton copulas to all pairs and returns the average lower-tail
    dependence as a portfolio joint-crash-risk score.

    Args:
        returns_df: Wide-format DataFrame (dates × symbols) of returns.
        weights: Optional symbol → weight mapping for weighted averaging.

    Returns:
        Dict with ``avg_lower_tail_dep``, ``max_lower_tail_dep``,
        ``n_pairs``, ``most_dependent_pair``.
    """
    symbols = list(returns_df.columns)
    if len(symbols) < 2:
        return {"avg_lower_tail_dep": 0.0, "max_lower_tail_dep": 0.0, "n_pairs": 0, "most_dependent_pair": ""}

    tail_deps: list[float] = []
    max_td = 0.0
    max_pair = ""

    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1:]:
            ra = returns_df[sym_a].values
            rb = returns_df[sym_b].values
            result = fit_copula_pair(ra, rb, sym_a, sym_b)
            if result is not None:
                td = result.lower_tail_dep
                tail_deps.append(td)
                if td > max_td:
                    max_td = td
                    max_pair = f"{sym_a}/{sym_b}"

    if not tail_deps:
        return {"avg_lower_tail_dep": 0.0, "max_lower_tail_dep": 0.0, "n_pairs": 0, "most_dependent_pair": ""}

    return {
        "avg_lower_tail_dep": round(float(np.mean(tail_deps)), 4),
        "max_lower_tail_dep": round(max_td, 4),
        "n_pairs": len(tail_deps),
        "most_dependent_pair": max_pair,
    }


# ---------------------------------------------------------------------------
# D-vine copula (3-asset)
# ---------------------------------------------------------------------------
# A D-vine for d=3 variables (X1, X2, X3) uses a 2-layer tree:
#   Tree 1: C12 (X1, X2),  C23 (X2, X3)
#   Tree 2: C13|2 (F(X1|X2), F(X3|X2))   — conditioned on X2
#
# Log-likelihood = loglik(C12) + loglik(C23) + loglik(C13|2)
# h-functions   = conditional CDFs needed to compute Tree-2 pseudo-obs.
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc_dvine


@_dc_dvine
class DVineResult:
    """Fitted D-vine copula for a triple of assets."""
    symbols: tuple[str, str, str]
    copula_12: str      # "clayton" | "gumbel" | "gaussian"
    theta_12: float
    copula_23: str
    theta_23: float
    copula_13_2: str   # C(1,3|2) — conditional copula
    theta_13_2: float
    log_likelihood: float
    n_obs: int


# -- h-functions (conditional CDFs) -----------------------------------------

def _h_clayton(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """h(u|v; Clayton) = P(U1 <= u | U2 = v)."""
    eps = 1e-8
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    # h(u|v) = (u^{-theta} + v^{-theta} - 1)^{-(1+1/theta)} * v^{-(theta+1)}
    inner = np.maximum(u ** (-theta) + v ** (-theta) - 1.0, eps)
    return inner ** (-(1.0 + 1.0 / theta)) * v ** (-(theta + 1.0))


def _h_gumbel(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """h(u|v; Gumbel) via numerical finite difference (width=1e-5)."""
    eps = 1e-5
    v_lo = np.clip(v - eps, 1e-8, 1 - 1e-8)
    v_hi = np.clip(v + eps, 1e-8, 1 - 1e-8)

    def _gumbel_cdf(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        lu = (-np.log(np.clip(uu, 1e-8, 1 - 1e-8))) ** theta
        lv = (-np.log(np.clip(vv, 1e-8, 1 - 1e-8))) ** theta
        return np.exp(-((lu + lv) ** (1.0 / theta)))

    return (_gumbel_cdf(u, v_hi) - _gumbel_cdf(u, v_lo)) / (2.0 * eps)


def _h_gaussian(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    """h(u|v; Gaussian) = Phi((Phi^{-1}(u) - rho*Phi^{-1}(v)) / sqrt(1-rho^2))."""
    if not SCIPY_AVAILABLE:
        return np.full_like(u, 0.5)
    eps = 1e-8
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    xu = sp_stats.norm.ppf(u)
    xv = sp_stats.norm.ppf(v)
    denom = max(np.sqrt(1.0 - rho ** 2), 1e-8)
    return sp_stats.norm.cdf((xu - rho * xv) / denom)


def _h(u: np.ndarray, v: np.ndarray, copula: str, theta: float) -> np.ndarray:
    """Dispatch h-function by copula name."""
    if copula == "clayton":
        return np.clip(_h_clayton(u, v, theta), 1e-8, 1 - 1e-8)
    if copula == "gumbel":
        return np.clip(_h_gumbel(u, v, theta), 1e-8, 1 - 1e-8)
    # gaussian
    return np.clip(_h_gaussian(u, v, float(theta)), 1e-8, 1 - 1e-8)


def _logpdf(u: np.ndarray, v: np.ndarray, copula: str, theta: float) -> float:
    """Dispatch bivariate copula log-likelihood."""
    if copula == "clayton":
        return _clayton_logpdf(u, v, theta)
    if copula == "gumbel":
        return _gumbel_logpdf(u, v, theta)
    return _gaussian_logpdf(u, v, theta)


def _fit_pair(
    u: np.ndarray, v: np.ndarray
) -> tuple[str, float, float]:
    """Return (best_copula, theta, loglik) for a pseudo-uniform pair."""
    if not SCIPY_AVAILABLE:
        return ("gaussian", 0.0, 0.0)

    results = []
    for name, bounds, neg_ll in [
        ("clayton", (0.01, 20.0),  lambda t: -_clayton_logpdf(u, v, t)),
        ("gumbel",  (1.01, 20.0),  lambda t: -_gumbel_logpdf(u, v, t)),
        ("gaussian", (-0.99, 0.99), lambda t: -_gaussian_logpdf(u, v, t)),
    ]:
        try:
            res = sp_opt.minimize_scalar(neg_ll, bounds=bounds, method="bounded")
            if res.success:
                results.append((name, float(res.x), -float(res.fun)))
        except Exception:
            pass

    if not results:
        return ("gaussian", 0.0, -np.inf)
    best = max(results, key=lambda x: x[2])
    return best


def fit_dvine_trio(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    returns_c: np.ndarray,
    symbol_a: str = "A",
    symbol_b: str = "B",
    symbol_c: str = "C",
) -> DVineResult | None:
    """Fit a D-vine copula to three return series.

    The vine structure is:
        Tree 1: C(A,B) and C(B,C)
        Tree 2: C(A,C | B)  — conditional copula

    Args:
        returns_a/b/c: Daily returns for assets A, B, C.
        symbol_a/b/c: Labels.

    Returns:
        :class:`DVineResult` or ``None`` when fewer than 50 observations or
        scipy unavailable.
    """
    if not SCIPY_AVAILABLE:
        return None

    # Align on finite observations
    mask = np.isfinite(returns_a) & np.isfinite(returns_b) & np.isfinite(returns_c)
    ra, rb, rc = returns_a[mask], returns_b[mask], returns_c[mask]
    if len(ra) < 50:
        return None

    ua = _rank_transform(ra)
    ub = _rank_transform(rb)
    uc = _rank_transform(rc)

    # -- Tree 1 -----------------------------------------------------------
    cop12, theta12, ll12 = _fit_pair(ua, ub)
    cop23, theta23, ll23 = _fit_pair(ub, uc)

    # -- Conditional pseudo-observations for Tree 2 -----------------------
    # v1_2 = h(ua | ub; C12)  i.e. F(A | B)
    # v3_2 = h(uc | ub; C23)  i.e. F(C | B)
    v1_2 = _h(ua, ub, cop12, theta12)
    v3_2 = _h(uc, ub, cop23, theta23)

    # -- Tree 2 -----------------------------------------------------------
    cop132, theta132, ll132 = _fit_pair(v1_2, v3_2)

    total_ll = ll12 + ll23 + ll132

    return DVineResult(
        symbols=(symbol_a, symbol_b, symbol_c),
        copula_12=cop12,
        theta_12=round(theta12, 4),
        copula_23=cop23,
        theta_23=round(theta23, 4),
        copula_13_2=cop132,
        theta_13_2=round(theta132, 4),
        log_likelihood=round(total_ll, 2),
        n_obs=int(len(ra)),
    )


__all__ = [
    "CopulaResult",
    "DVineResult",
    "compute_portfolio_tail_risk",
    "fit_copula_pair",
    "fit_dvine_trio",
]
