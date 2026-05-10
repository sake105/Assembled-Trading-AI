"""Bayesian Linear Regression with Conjugate Normal-Inverse-Gamma Prior.

Theorie
-------
Für y = Xβ + ε mit ε ~ N(0, σ²I) und konjugiertem Prior
- β | σ² ~ N(β_0, σ² Λ_0⁻¹)
- σ² ~ InverseGamma(a_0, b_0)

ist der Posterior in geschlossener Form:
- β | y, σ² ~ N(μ_n, σ² Λ_n⁻¹) wobei Λ_n = X'X + Λ_0
- σ² | y ~ InverseGamma(a_n, b_n)

Vorteil
-------
- Volle Posterior-Verteilung ohne MCMC.
- Quantifizierte Parameter-Unsicherheit für Sizing-Decisions.

Anwendung
---------
- Faktor-Modell-Schätzung mit Confidence-Intervals
- Predictive-Distribution für ein neues x_new
- Bayesian Sharpe-Posterior
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BayesianRegressionResult:
    mu_n: np.ndarray  # posterior mean of β
    Lambda_n: np.ndarray  # posterior precision
    a_n: float  # posterior IG shape
    b_n: float  # posterior IG rate
    sigma2_post_mean: float
    n: int
    p: int


def fit_bayesian_linear(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray | None = None,
    prior_precision: np.ndarray | None = None,
    a_0: float = 1.0,
    b_0: float = 1.0,
) -> BayesianRegressionResult:
    """Bayesian Linear Regression with NIG prior.

    Args:
        X: design matrix (n, p).
        y: response (n,).
        prior_mean: β_0 (default zeros).
        prior_precision: Λ_0 (default tiny ridge).
        a_0, b_0: IG hyperparameters.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if prior_mean is None:
        prior_mean = np.zeros(p)
    if prior_precision is None:
        prior_precision = 1e-6 * np.eye(p)

    Lambda_n = X.T @ X + prior_precision
    mu_n = np.linalg.solve(Lambda_n, X.T @ y + prior_precision @ prior_mean)
    a_n = a_0 + n / 2
    resid = y - X @ mu_n
    b_n = b_0 + 0.5 * (
        float(resid @ resid)
        + float((mu_n - prior_mean) @ prior_precision @ (mu_n - prior_mean))
    )
    sigma2_post_mean = b_n / (a_n - 1) if a_n > 1 else b_n

    return BayesianRegressionResult(
        mu_n=mu_n,
        Lambda_n=Lambda_n,
        a_n=float(a_n),
        b_n=float(b_n),
        sigma2_post_mean=float(sigma2_post_mean),
        n=n,
        p=p,
    )


def predictive_distribution(
    fit: BayesianRegressionResult, x_new: np.ndarray
) -> tuple[float, float]:
    """Posterior predictive: y_new | x_new ~ Student-t(df=2*a_n).

    Returns:
        (mean, variance) of the predictive distribution.
    """
    cov_post = np.linalg.pinv(fit.Lambda_n) * fit.sigma2_post_mean
    mu = float(x_new @ fit.mu_n)
    var = float(fit.sigma2_post_mean * (1 + x_new @ cov_post @ x_new))
    return mu, var


def sharpe_posterior_samples(
    returns: pd.Series, n_samples: int = 5000, seed: int = 42
) -> np.ndarray:
    """Posterior samples of the *true* Sharpe ratio given observed returns.

    Assumes Normal returns with conjugate Normal-Inverse-Gamma prior.
    Returns posterior draws of μ/σ × √annual.
    """
    r = pd.Series(returns).dropna().values
    n = len(r)
    if n < 30:
        return np.full(n_samples, np.nan)
    mu_hat = float(r.mean())
    s2_hat = float(r.var(ddof=1))
    rng = np.random.default_rng(seed)
    # Use posterior under flat priors:
    # σ² ~ scaled inverse chi-squared (n-1, s²)
    # μ | σ² ~ N(mu_hat, σ²/n)
    chi = rng.chisquare(df=n - 1, size=n_samples)
    sigma2 = s2_hat * (n - 1) / chi
    mu = rng.normal(mu_hat, np.sqrt(sigma2 / n))
    sharpe = mu / np.sqrt(sigma2) * np.sqrt(252)
    return sharpe


def credible_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Equal-tail credible interval."""
    a = (1 - level) / 2
    lo, hi = np.quantile(samples, [a, 1 - a])
    return float(lo), float(hi)


__all__ = [
    "BayesianRegressionResult",
    "fit_bayesian_linear",
    "predictive_distribution",
    "sharpe_posterior_samples",
    "credible_interval",
]
