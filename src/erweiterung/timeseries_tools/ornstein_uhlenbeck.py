"""Ornstein-Uhlenbeck Process Estimation für Mean-Reversion-Strategien.

Model
-----
    dX_t = θ(μ - X_t) dt + σ dW_t

OU ist die kontinuierliche Form von AR(1):
    X_{t+1} = X_t + θ Δt (μ - X_t) + σ √Δt ε_t

Estimation
----------
MLE oder OLS auf der diskretisierten Form:
    X_{t+1} - X_t = a + b X_t + ε_t
    => θ = -b/Δt,  μ = -a/b,  σ = std(ε) / √Δt

Half-Life of Mean-Reversion: ln(2)/θ. Niedrige Half-Life = schnelle Mean-Reversion.

Anwendung
---------
- Pairs-Trading Spread-Modellierung
- Bewertung der Mean-Reversion-Stärke einer Series
- Trade-Entry/Exit-Timing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OUFit:
    theta: float  # mean-reversion speed
    mu: float  # long-run mean
    sigma: float  # diffusion
    half_life: float
    n_obs: int
    r_squared: float


def fit_ornstein_uhlenbeck(series: pd.Series, dt: float = 1.0) -> OUFit:
    """OLS-Fit für OU-Process auf einer Time-Series.

    Args:
        series: 1-D pandas Series.
        dt: time step (1.0 = daily; 1/252 if you want annualized θ).

    Returns:
        ``OUFit`` mit Parameter + Half-Life.
    """
    s = pd.Series(series).dropna().values.astype(float)
    if len(s) < 30:
        raise ValueError("need >= 30 observations")
    # Δx = a + b x_lag + ε
    x = s[:-1]
    dx = s[1:] - s[:-1]
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, dx, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    if b >= 0:
        # Not mean-reverting
        return OUFit(
            theta=0.0,
            mu=float("nan"),
            sigma=float(np.std(dx)),
            half_life=float("inf"),
            n_obs=len(s),
            r_squared=0.0,
        )

    theta = -b / dt
    mu = -a / b
    pred = X @ beta
    resid = dx - pred
    sigma_eps = float(np.std(resid, ddof=2))
    sigma = sigma_eps / np.sqrt(dt)
    half_life = np.log(2) / theta if theta > 0 else float("inf")

    ss_tot = float(((dx - dx.mean()) ** 2).sum())
    ss_res = float((resid**2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return OUFit(
        theta=float(theta),
        mu=float(mu),
        sigma=float(sigma),
        half_life=float(half_life),
        n_obs=len(s),
        r_squared=float(r2),
    )


def ou_simulate(
    fit: OUFit,
    n_steps: int,
    x0: float | None = None,
    dt: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate OU-path with given params."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps)
    x[0] = x0 if x0 is not None else fit.mu
    for t in range(1, n_steps):
        x[t] = (
            x[t - 1]
            + fit.theta * (fit.mu - x[t - 1]) * dt
            + fit.sigma * np.sqrt(dt) * rng.standard_normal()
        )
    return x


def is_mean_reverting(
    series: pd.Series, p_threshold: float = 0.05, half_life_max: float = 60
) -> bool:
    """Quick test ob Series mean-reverting ist.

    Kriterien:
    1. ADF-Test p < threshold
    2. OU-Fit liefert finite half-life < max
    """
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore

        adf = adfuller(series.dropna().values, regression="c", autolag="AIC")
        if adf[1] > p_threshold:
            return False
    except ImportError:
        pass

    try:
        fit = fit_ornstein_uhlenbeck(series)
        if np.isfinite(fit.half_life) and fit.half_life < half_life_max:
            return True
    except ValueError:
        return False
    return False


__all__ = ["OUFit", "fit_ornstein_uhlenbeck", "ou_simulate", "is_mean_reverting"]
