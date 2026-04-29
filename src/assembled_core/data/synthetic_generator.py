"""Synthetic Data Generator (Plan 10.10).

Generates synthetic price data based on historical crisis templates for stress testing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Crisis templates: (name, duration_days, mean_daily_return, vol_daily)
CRISIS_TEMPLATES: dict[str, tuple[int, float, float]] = {
    "2008_gfc": (252, -0.0020, 0.035),
    "2020_covid": (30, -0.0080, 0.050),
    "2000_dotcom": (504, -0.0008, 0.020),
    "1987_crash": (5, -0.0400, 0.080),
    "2022_rate_shock": (180, -0.0010, 0.018),
}


def generate_crisis_returns(
    template: str = "2008_gfc",
    n_assets: int = 10,
    correlation: float = 0.7,
    seed: int = 42,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Generate synthetic crisis return series.

    Args:
        template: Crisis template name.
        n_assets: Number of assets to simulate.
        correlation: Pairwise correlation during crisis.
        seed: Random seed.
        scale: Magnitude multiplier (2.0 = twice as severe).

    Returns:
        DataFrame of daily returns (rows=days, columns=asset names).
    """
    np.random.seed(seed)

    if template not in CRISIS_TEMPLATES:
        raise ValueError(f"Unknown template: {template}. Available: {list(CRISIS_TEMPLATES.keys())}")

    n_days, mean_ret, vol = CRISIS_TEMPLATES[template]
    mean_ret *= scale
    vol *= scale

    # Correlated multivariate normal
    corr_matrix = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    cov = np.outer(np.full(n_assets, vol), np.full(n_assets, vol)) * corr_matrix

    try:
        L = np.linalg.cholesky(cov + np.eye(n_assets) * 1e-10)
    except np.linalg.LinAlgError:
        L = np.eye(n_assets) * vol

    z = np.random.standard_normal((n_days, n_assets))
    returns = mean_ret + z @ L.T

    columns = [f"ASSET_{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=columns)


def generate_normal_returns(
    n_days: int = 252,
    n_assets: int = 10,
    mean_annual: float = 0.08,
    vol_annual: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic normal-market return series.

    Args:
        n_days: Number of trading days.
        n_assets: Number of assets.
        mean_annual: Annualized mean return.
        vol_annual: Annualized volatility.
        seed: Random seed.

    Returns:
        DataFrame of daily returns.
    """
    np.random.seed(seed)
    daily_mean = mean_annual / 252
    daily_vol = vol_annual / np.sqrt(252)

    returns = np.random.normal(daily_mean, daily_vol, (n_days, n_assets))
    columns = [f"ASSET_{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=columns)


def generate_garch_returns(
    n_days: int = 252,
    n_assets: int = 5,
    omega: float = 1e-6,
    alpha: float = 0.09,
    beta: float = 0.90,
    mean_annual: float = 0.06,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate GARCH(1,1) return series with volatility clustering.

    Conditional variance: h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}

    Args:
        n_days: Number of trading days.
        n_assets: Number of independent GARCH series.
        omega: Long-run variance intercept.
        alpha: ARCH coefficient (shock persistence).
        beta: GARCH coefficient (variance persistence).
        mean_annual: Annualised drift applied to every asset.
        seed: Random seed.

    Returns:
        DataFrame of daily returns (rows=days, columns=asset names).
    """
    rng = np.random.default_rng(seed)
    daily_mean = mean_annual / 252
    columns = [f"ASSET_{i}" for i in range(n_assets)]
    all_returns: list[np.ndarray] = []

    for _ in range(n_assets):
        h = np.empty(n_days)
        r = np.empty(n_days)
        h[0] = omega / max(1 - alpha - beta, 1e-8)   # unconditional variance
        eps_prev = 0.0
        for t in range(n_days):
            if t > 0:
                h[t] = omega + alpha * eps_prev ** 2 + beta * h[t - 1]
            eps = rng.standard_normal() * np.sqrt(max(h[t], 1e-12))
            eps_prev = eps
            r[t] = daily_mean + eps
        all_returns.append(r)

    return pd.DataFrame(np.column_stack(all_returns), columns=columns)


def generate_jump_diffusion_returns(
    n_days: int = 252,
    n_assets: int = 5,
    mu_annual: float = 0.06,
    sigma_annual: float = 0.18,
    jump_intensity: float = 5.0,       # expected jumps per year
    jump_mean: float = -0.02,          # mean jump size (log return)
    jump_std: float = 0.03,            # std of jump size
    seed: int = 42,
) -> pd.DataFrame:
    """Merton (1976) jump-diffusion return series.

    r_t = mu_d + sigma_d * Z_t + sum_{j=1}^{N_t} J_j

    where N_t ~ Poisson(lambda/252) per day and J_j ~ N(jump_mean, jump_std).

    Args:
        n_days: Number of trading days.
        n_assets: Number of independent series.
        mu_annual: Annualised drift (adjusted for jump risk premium internally).
        sigma_annual: Annualised diffusion volatility.
        jump_intensity: Expected number of jumps per year (lambda).
        jump_mean: Mean log-return of a single jump.
        jump_std: Std dev of a single jump.
        seed: Random seed.

    Returns:
        DataFrame of daily returns.
    """
    rng = np.random.default_rng(seed)
    daily_mu = mu_annual / 252
    daily_sigma = sigma_annual / np.sqrt(252)
    daily_lambda = jump_intensity / 252

    # Drift correction: subtract expected jump contribution
    drift_adj = daily_mu - daily_lambda * (np.exp(jump_mean + 0.5 * jump_std ** 2) - 1)

    columns = [f"ASSET_{i}" for i in range(n_assets)]
    all_returns: list[np.ndarray] = []

    for _ in range(n_assets):
        diffusion = rng.normal(drift_adj, daily_sigma, n_days)
        n_jumps = rng.poisson(daily_lambda, n_days)
        jump_component = np.array([
            np.sum(rng.normal(jump_mean, jump_std, int(n))) if n > 0 else 0.0
            for n in n_jumps
        ])
        all_returns.append(diffusion + jump_component)

    return pd.DataFrame(np.column_stack(all_returns), columns=columns)


def generate_regime_switching_returns(
    n_days: int = 504,
    n_assets: int = 5,
    bull_mu_annual: float = 0.12,
    bull_vol_annual: float = 0.12,
    bear_mu_annual: float = -0.15,
    bear_vol_annual: float = 0.30,
    p_bull_to_bear: float = 0.02,      # daily transition probability
    p_bear_to_bull: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Two-state (bull/bear) Markov regime-switching return series.

    State transitions follow a first-order Markov chain with daily
    transition probabilities *p_bull_to_bear* and *p_bear_to_bull*.

    Args:
        n_days: Number of trading days.
        n_assets: Number of correlated assets (same regime state shared).
        bull_mu_annual: Annualised drift in bull state.
        bull_vol_annual: Annualised volatility in bull state.
        bear_mu_annual: Annualised drift in bear state.
        bear_vol_annual: Annualised volatility in bear state.
        p_bull_to_bear: Daily probability of switching from bull to bear.
        p_bear_to_bull: Daily probability of switching from bear to bull.
        seed: Random seed.

    Returns:
        Tuple of (returns DataFrame, regime array) where regime=0 is bull,
        regime=1 is bear.
    """
    rng = np.random.default_rng(seed)

    bull_mu_d = bull_mu_annual / 252
    bull_sig_d = bull_vol_annual / np.sqrt(252)
    bear_mu_d = bear_mu_annual / 252
    bear_sig_d = bear_vol_annual / np.sqrt(252)

    # Simulate regime path
    regime = np.zeros(n_days, dtype=int)
    # Start in bull (0) if stationary distribution favours it
    pi_bear = p_bull_to_bear / (p_bull_to_bear + p_bear_to_bull)
    regime[0] = int(rng.random() < pi_bear)

    for t in range(1, n_days):
        if regime[t - 1] == 0:
            regime[t] = int(rng.random() < p_bull_to_bear)
        else:
            regime[t] = int(rng.random() >= p_bear_to_bull)

    # Generate returns for each asset under the shared regime
    columns = [f"ASSET_{i}" for i in range(n_assets)]
    all_returns = np.empty((n_days, n_assets))

    for t in range(n_days):
        if regime[t] == 0:
            all_returns[t] = rng.normal(bull_mu_d, bull_sig_d, n_assets)
        else:
            all_returns[t] = rng.normal(bear_mu_d, bear_sig_d, n_assets)

    return pd.DataFrame(all_returns, columns=columns), regime


__all__ = [
    "CRISIS_TEMPLATES",
    "generate_crisis_returns",
    "generate_normal_returns",
    "generate_garch_returns",
    "generate_jump_diffusion_returns",
    "generate_regime_switching_returns",
]
