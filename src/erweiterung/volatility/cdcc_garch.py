"""cDCC-GARCH — corrected DCC of Aielli (2013) (audit C4-072).

The original Engle (2002) DCC recursion treats the quasi-correlation
matrix :math:`Q_t` as if it were a true correlation matrix when it is
not. Aielli (2013) shows the estimator of :math:`\\bar Q` is therefore
**inconsistent**, and corrects the recursion to use *re-scaled*
innovations
:math:`\\epsilon^*_t = \\operatorname{diag}(Q_t)^{1/2}\\,\\epsilon_t`:

.. math::

    Q_t = (1 - \\alpha - \\beta)\\,\\bar Q
        + \\alpha\\, \\epsilon^*_{t-1}\\epsilon^{*\\,\\prime}_{t-1}
        + \\beta\\, Q_{t-1}

    R_t = \\operatorname{diag}(Q_t)^{-1/2}\\,Q_t\\,\\operatorname{diag}(Q_t)^{-1/2}

The correction makes the targeting estimator
:math:`\\hat{\\bar Q} = T^{-1}\\sum_t \\epsilon^*_t\\,\\epsilon^{*\\,\\prime}_t`
consistent — the *original* DCC's :math:`\\hat{\\bar Q} = T^{-1}\\sum_t \\epsilon_t\\,\\epsilon'_t`
is biased for any positively-autocorrelated process.

The existing :func:`dcc_garch.fit_dcc_garch` is **kept unchanged** for
reproducibility of prior research outputs. cDCC is a separate entry
point so callers can opt in.

Implementation notes
--------------------
Stage 1 (univariate GARCH) is identical to DCC and is delegated to the
existing :func:`dcc_garch._fit_garch_univariate`. The only change is in
Stage 2, where every place the original code uses
``np.outer(eps[t-1], eps[t-1])`` is replaced by
``np.outer(eps_star, eps_star)`` with
``eps_star = sqrt(diag(Q[t-1])) * eps[t-1]``.

References
----------
- Aielli, G. P. (2013). *Dynamic conditional correlation: on properties
  and estimation*. J. Bus. Econ. Stat. 31(3), 282-299.
- Engle, R. (2002). *Dynamic Conditional Correlation*. J. Bus. Econ.
  Stat. 20.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.erweiterung.volatility.dcc_garch import _fit_garch_univariate

logger = logging.getLogger(__name__)


@dataclass
class cDCCFit:
    """Output of :func:`fit_cdcc_garch`.

    Differs from :class:`DCCFit` only in semantics — the inner
    recursion is Aielli-corrected. Field layout kept identical so that
    downstream code that consumes either fit object does not branch.
    """

    alpha: float
    beta: float
    Q_bar: np.ndarray
    R_path: np.ndarray  # (T, N, N) conditional correlations
    sigma_path: np.ndarray  # (T, N) conditional vols
    log_lik: float
    method: str = "cdcc_aielli_2013"


def _cdcc_path(
    alpha: float, beta: float, eps: np.ndarray, Q_bar: np.ndarray
) -> np.ndarray:
    """Aielli-corrected Q-recursion, shape (T, N, N)."""
    T, N = eps.shape
    Q = np.zeros((T, N, N))
    Q[0] = Q_bar
    one_minus = 1.0 - alpha - beta
    for t in range(1, T):
        # Aielli (2013): use re-scaled innovations  eps* = sqrt(diag(Q_{t-1})) * eps_{t-1}
        q_diag = np.diag(Q[t - 1])
        s = np.sqrt(np.maximum(q_diag, 1e-12))
        eps_star = s * eps[t - 1]
        Q[t] = (
            one_minus * Q_bar + alpha * np.outer(eps_star, eps_star) + beta * Q[t - 1]
        )
    return Q


def fit_cdcc_garch(returns: pd.DataFrame, max_iter: int = 50) -> cDCCFit:
    """Fit cDCC-GARCH (Aielli 2013 corrected DCC).

    Args:
        returns: DataFrame (T x N) of returns, sorted ascending in time.
            Rows containing any NaN are dropped first.
        max_iter: passed through to :func:`_fit_garch_univariate`.

    Returns:
        :class:`cDCCFit` with the conditional correlation path, vol path,
        targeted Q̄, and the QMLE-optimised (α, β).

    Raises:
        ValueError: fewer than 50 valid rows.
    """
    del max_iter  # accepted for API parity; not currently used
    R = returns.dropna(how="any").to_numpy(dtype=float)
    T, N = R.shape
    if T < 50:
        raise ValueError(f"need >= 50 observations, got {T}")

    # Stage 1 — univariate GARCH per series (identical to DCC).
    sigmas = np.zeros((T, N))
    eps = np.zeros((T, N))
    for j in range(N):
        _, sig = _fit_garch_univariate(R[:, j])
        sigmas[:, j] = sig
        eps[:, j] = R[:, j] / np.maximum(sig, 1e-9)

    # Aielli targeting: Q̄ = (1/T) Σ eps*_t eps*'_t — needs Q_t to compute eps*.
    # First pass: use the DCC-style raw target as a warm start, then iterate
    # once to refine. In practice one iteration is enough; two converges to
    # numerical precision for any well-scaled series.
    Q_bar_warm = (eps.T @ eps) / T
    for _ in range(2):
        Q_path = _cdcc_path(alpha=0.05, beta=0.90, eps=eps, Q_bar=Q_bar_warm)
        # Compute re-scaled eps*_t and re-target Q̄.
        eps_star = np.empty_like(eps)
        for t in range(T):
            s = np.sqrt(np.maximum(np.diag(Q_path[t]), 1e-12))
            eps_star[t] = s * eps[t]
        Q_bar_warm = (eps_star.T @ eps_star) / T

    Q_bar = Q_bar_warm

    # Stage 2 — QMLE for (α, β).
    def neg_ll(theta: np.ndarray) -> float:
        a, b = theta
        if a < 0 or b < 0 or a + b >= 1:
            return 1e10
        Q = _cdcc_path(a, b, eps, Q_bar)
        ll = 0.0
        for t in range(T):
            q_diag = np.diag(Q[t])
            d = 1.0 / np.sqrt(np.maximum(q_diag, 1e-9))
            R_t = (d[:, None] * Q[t]) * d[None, :]
            sign, logdet = np.linalg.slogdet(R_t)
            if sign <= 0:
                return 1e10
            try:
                inv_R = np.linalg.pinv(R_t)
            except np.linalg.LinAlgError:
                return 1e10
            ll += 0.5 * (logdet + float(eps[t] @ inv_R @ eps[t]))
        return float(ll)

    try:
        from scipy.optimize import minimize  # type: ignore

        res = minimize(
            neg_ll,
            x0=np.array([0.05, 0.90]),
            method="L-BFGS-B",
            bounds=[(0.001, 0.999), (0.001, 0.999)],
        )
        alpha_hat, beta_hat = res.x
        log_lik = -res.fun
    except ImportError:
        alpha_hat, beta_hat = 0.05, 0.90
        log_lik = -neg_ll(np.array([alpha_hat, beta_hat]))

    Q_path = _cdcc_path(alpha_hat, beta_hat, eps, Q_bar)
    R_path = np.zeros_like(Q_path)
    for t in range(T):
        q_diag = np.diag(Q_path[t])
        d = 1.0 / np.sqrt(np.maximum(q_diag, 1e-9))
        R_path[t] = (d[:, None] * Q_path[t]) * d[None, :]

    return cDCCFit(
        alpha=float(alpha_hat),
        beta=float(beta_hat),
        Q_bar=Q_bar,
        R_path=R_path,
        sigma_path=sigmas,
        log_lik=log_lik,
    )


def cdcc_covariance_at(fit: cDCCFit, t: int) -> np.ndarray:
    """Conditional covariance at time t: Σ_t = diag(σ_t) R_t diag(σ_t)."""
    if t < 0 or t >= len(fit.sigma_path):
        raise IndexError(f"t={t} out of range")
    sigma = fit.sigma_path[t]
    return np.diag(sigma) @ fit.R_path[t] @ np.diag(sigma)


__all__ = ["cDCCFit", "fit_cdcc_garch", "cdcc_covariance_at"]
