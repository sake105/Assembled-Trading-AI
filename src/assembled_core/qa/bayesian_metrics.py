"""Bayesian performance metrics for strategy evaluation.

Provides:
  - Bayesian Sharpe ratio posterior (PyMC-based when available, analytic fallback)
  - Hierarchical strategy comparison with partial pooling

When PyMC is not installed, the module falls back to analytic / simulation-based
estimates so downstream code never hard-fails.

References:
  - Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier"
  - Bayes Sharpe: Fernandez-Perez et al. (2019)
  - Hierarchical partial pooling: Gelman et al. "ARM" ch. 12
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pymc as pm  # type: ignore[import-not-found]  # noqa: F401

    _PYMC_AVAILABLE = True
except ImportError:
    _PYMC_AVAILABLE = False


@dataclass
class SharpePosterior:
    """Posterior summary for a single strategy's Sharpe ratio."""

    strategy: str
    mean: float  # posterior mean Sharpe
    std: float  # posterior std
    hdi_lower: float  # 94% HDI lower bound
    hdi_upper: float  # 94% HDI upper bound
    p_positive: float  # P(Sharpe > 0)
    n_obs: int
    backend: str  # "pymc" or "analytic"


@dataclass
class StrategyComparison:
    """Result of hierarchical comparison across strategies."""

    strategies: list[str]
    posteriors: list[SharpePosterior]
    # P(strategy_i best) for each strategy
    p_best: dict[str, float]
    # Pooled population mean Sharpe (partial-pooling estimate)
    population_mean: float
    population_std: float
    backend: str


def _analytic_sharpe_posterior(
    returns: np.ndarray,
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
) -> SharpePosterior:
    """Analytic approximation of Bayesian Sharpe posterior.

    Uses a conjugate-like Normal-Normal update for the mean, with observed
    variance treated as known. This under-estimates posterior uncertainty
    but is close for N > 50.

    The posterior is:
        mu | data ~ N(mu_post, sigma_post^2)
        Sharpe_post = mu_post / sigma_obs * sqrt(252)
    """
    n = len(returns)
    if n < 2:
        return SharpePosterior(
            strategy="",
            mean=0.0,
            std=1.0,
            hdi_lower=-2.0,
            hdi_upper=2.0,
            p_positive=0.5,
            n_obs=n,
            backend="analytic",
        )

    obs_mean = float(np.mean(returns))
    obs_std = float(np.std(returns, ddof=1))
    obs_var = obs_std**2

    # Prior precision
    prior_prec = 1.0 / (prior_std**2)
    data_prec = n / max(obs_var, 1e-12)

    # Posterior precision and mean
    post_prec = prior_prec + data_prec
    post_var = 1.0 / post_prec
    post_mean_daily = (prior_prec * prior_mean + data_prec * obs_mean) / post_prec

    # Sharpe scaling
    sharpe_mean = post_mean_daily / max(obs_std, 1e-9) * math.sqrt(252)
    # Posterior std for Sharpe (delta method approximation)
    sharpe_std = math.sqrt(post_var) / max(obs_std, 1e-9) * math.sqrt(252)

    # 94% HDI (approximately ±1.88 std for Normal)
    hdi_lo = sharpe_mean - 1.88 * sharpe_std
    hdi_hi = sharpe_mean + 1.88 * sharpe_std

    # P(Sharpe > 0)
    from scipy.stats import norm

    p_pos = float(norm.sf(0.0, loc=sharpe_mean, scale=max(sharpe_std, 1e-9)))

    return SharpePosterior(
        strategy="",
        mean=round(sharpe_mean, 4),
        std=round(sharpe_std, 4),
        hdi_lower=round(hdi_lo, 4),
        hdi_upper=round(hdi_hi, 4),
        p_positive=round(p_pos, 4),
        n_obs=n,
        backend="analytic",
    )


def bayesian_sharpe_posterior(
    returns: list[float] | np.ndarray,
    strategy: str = "strategy",
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    n_samples: int = 2000,
    use_pymc: bool = True,
) -> SharpePosterior:
    """Compute Bayesian posterior for a strategy's annualised Sharpe ratio.

    Uses a Student-t likelihood (robust to non-normality / fat tails) with
    PyMC MCMC when available, analytic fallback otherwise.

    Args:
        returns: Daily return series (decimal, e.g. 0.01 for +1%).
        strategy: Strategy name for labelling.
        prior_mean: Prior mean for daily return (default 0).
        prior_std: Prior std for mean hyperparameter (default 1%).
        n_samples: MCMC samples per chain (PyMC only).
        use_pymc: If False, always use analytic approximation.

    Returns:
        SharpePosterior with posterior summary.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]

    if _PYMC_AVAILABLE and use_pymc and len(arr) >= 30:
        result = _pymc_sharpe_posterior(arr, strategy, prior_mean, prior_std, n_samples)
    else:
        result = _analytic_sharpe_posterior(arr, prior_mean, prior_std)

    result.strategy = strategy
    return result


def _pymc_sharpe_posterior(
    arr: np.ndarray,
    strategy: str,
    prior_mean: float,
    prior_std: float,
    n_samples: int,
) -> SharpePosterior:
    """PyMC-based Bayesian Sharpe using Student-t likelihood."""
    import pymc as pm

    obs_std = float(np.std(arr, ddof=1))
    with pm.Model():
        mu = pm.Normal("mu", mu=prior_mean, sigma=prior_std)
        sigma = pm.HalfNormal("sigma", sigma=obs_std * 2)
        nu = pm.Exponential("nu", lam=1.0 / 30.0)  # heavy tails ~ Student-t
        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=arr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pm.sample(
                n_samples,
                tune=500,
                chains=2,
                progressbar=False,
                return_inferencedata=True,
                target_accept=0.90,
                random_seed=42,
            )

    mu_samples = trace.posterior["mu"].values.flatten()
    sig_samples = trace.posterior["sigma"].values.flatten()
    sharpe_samples = mu_samples / np.maximum(sig_samples, 1e-9) * math.sqrt(252)

    hdi = pm.hdi(sharpe_samples, hdi_prob=0.94)
    p_pos = float(np.mean(sharpe_samples > 0))

    return SharpePosterior(
        strategy=strategy,
        mean=round(float(np.mean(sharpe_samples)), 4),
        std=round(float(np.std(sharpe_samples)), 4),
        hdi_lower=round(float(hdi[0]), 4),
        hdi_upper=round(float(hdi[1]), 4),
        p_positive=round(p_pos, 4),
        n_obs=len(arr),
        backend="pymc",
    )


def hierarchical_strategy_comparison(
    strategy_returns: dict[str, list[float] | np.ndarray],
    n_samples: int = 2000,
    use_pymc: bool = True,
) -> StrategyComparison:
    """Compare strategies via hierarchical Bayesian model with partial pooling.

    Each strategy's Sharpe is modelled as drawn from a shared population
    distribution. This shrinks estimates toward the group mean, reducing
    over-fitting on short return histories.

    Args:
        strategy_returns: Dict mapping strategy name → daily return list.
        n_samples: MCMC samples (PyMC path).
        use_pymc: If False, runs independent analytic posteriors (no pooling).

    Returns:
        StrategyComparison with per-strategy posteriors and P(best).
    """
    names = list(strategy_returns.keys())
    arrays = [np.asarray(v, dtype=float) for v in strategy_returns.values()]
    arrays = [a[np.isfinite(a)] for a in arrays]

    if _PYMC_AVAILABLE and use_pymc and all(len(a) >= 20 for a in arrays):
        return _pymc_hierarchical_comparison(names, arrays, n_samples)

    # Analytic fallback: independent posteriors, no partial pooling
    posteriors: list[SharpePosterior] = []
    for name, arr in zip(names, arrays):
        p = _analytic_sharpe_posterior(arr)
        p.strategy = name
        posteriors.append(p)

    # Monte Carlo estimate of P(best) from analytic posteriors
    n_mc = 5000
    rng = np.random.default_rng(0)
    sharpe_mc = np.column_stack(
        [rng.normal(p.mean, max(p.std, 1e-6), n_mc) for p in posteriors]
    )
    best_idx = np.argmax(sharpe_mc, axis=1)
    p_best = {name: float(np.mean(best_idx == i)) for i, name in enumerate(names)}

    pop_means = [p.mean for p in posteriors]
    pop_mean = float(np.mean(pop_means)) if pop_means else 0.0
    pop_std = float(np.std(pop_means)) if len(pop_means) > 1 else 0.0

    return StrategyComparison(
        strategies=names,
        posteriors=posteriors,
        p_best=p_best,
        population_mean=round(pop_mean, 4),
        population_std=round(pop_std, 4),
        backend="analytic",
    )


def _pymc_hierarchical_comparison(
    names: list[str],
    arrays: list[np.ndarray],
    n_samples: int,
) -> StrategyComparison:
    """PyMC hierarchical model for strategy comparison."""
    import pymc as pm

    obs_stds = [max(float(np.std(a, ddof=1)), 1e-4) for a in arrays]
    n_strats = len(names)

    with pm.Model():
        # Hyperpriors (population distribution)
        mu_pop = pm.Normal("mu_pop", mu=0.0, sigma=0.01)
        sigma_pop = pm.HalfNormal("sigma_pop", sigma=0.005)

        # Per-strategy means drawn from population
        mu_strat = pm.Normal("mu_strat", mu=mu_pop, sigma=sigma_pop, shape=n_strats)

        # Per-strategy observation noise (half-normal around observed std)
        sigma_strat = pm.HalfNormal(
            "sigma_strat", sigma=np.array(obs_stds), shape=n_strats
        )

        # Observations
        for i, arr in enumerate(arrays):
            pm.Normal(f"obs_{i}", mu=mu_strat[i], sigma=sigma_strat[i], observed=arr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pm.sample(
                n_samples,
                tune=500,
                chains=2,
                progressbar=False,
                return_inferencedata=True,
                target_accept=0.90,
                random_seed=42,
            )

    mu_samples = trace.posterior["mu_strat"].values.reshape(-1, n_strats)
    sig_samples = trace.posterior["sigma_strat"].values.reshape(-1, n_strats)
    sharpe_samples = mu_samples / np.maximum(sig_samples, 1e-9) * math.sqrt(252)

    posteriors: list[SharpePosterior] = []
    for i, name in enumerate(names):
        s = sharpe_samples[:, i]
        hdi = pm.hdi(s, hdi_prob=0.94)
        posteriors.append(
            SharpePosterior(
                strategy=name,
                mean=round(float(np.mean(s)), 4),
                std=round(float(np.std(s)), 4),
                hdi_lower=round(float(hdi[0]), 4),
                hdi_upper=round(float(hdi[1]), 4),
                p_positive=round(float(np.mean(s > 0)), 4),
                n_obs=len(arrays[i]),
                backend="pymc",
            )
        )

    # P(strategy i is best)
    best_idx = np.argmax(sharpe_samples, axis=1)
    p_best = {
        name: round(float(np.mean(best_idx == i)), 4) for i, name in enumerate(names)
    }

    mu_pop_samples = trace.posterior["mu_pop"].values.flatten()
    _sharpe_pop_mean = (
        float(np.mean(mu_pop_samples))
        / max(float(np.mean(sig_samples)), 1e-9)
        * math.sqrt(252)
    )

    return StrategyComparison(
        strategies=names,
        posteriors=posteriors,
        p_best=p_best,
        population_mean=round(float(np.mean([p.mean for p in posteriors])), 4),
        population_std=round(float(np.std([p.mean for p in posteriors])), 4),
        backend="pymc",
    )
