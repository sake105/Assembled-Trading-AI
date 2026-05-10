"""Michaud Resampled Efficient Frontier (Michaud 1998).

Theorie
-------
Markowitz-MVO ist instabil: kleine Schätzfehler in μ und Σ -> riesige
Gewichtsänderungen. Michaud's Resampling adressiert das durch Bootstrap:

1. Bootstrap-Sample aus historischen Returns ziehen.
2. Auf jedem Bootstrap-Sample MVO lösen → eine Gewichts-Trajektorie.
3. Resampled Efficient Frontier = **Mittelwert** über alle Bootstrap-MVO-Lösungen.

Resultat ist signifikant glatter und out-of-sample stabiler als rohe MVO.

Reference
---------
- Michaud, R. (1998). *Efficient Asset Management*. HBS Press.
- Michaud, R. & Michaud, R. (2008). Estimation Error and Portfolio Optimization:
  A Resampling Solution. *Journal Of Investment Management* 6.

Implementation
--------------
- Pure NumPy + scipy.optimize (linprog) — keine externen Portfolio-Libs.
- Optional: Constraint long-only, Constraint max-per-asset, Constraint target-vol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class REFConfig:
    n_bootstrap: int = 200
    n_frontier_points: int = 30  # number of return-target points along frontier
    long_only: bool = True
    max_weight: float = 0.25
    seed: int = 42


def _solve_mvo_target_return(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float,
    long_only: bool,
    max_weight: float,
) -> np.ndarray | None:
    """Quadratic programming for min variance s.t. target return.

    Uses scipy.optimize.minimize-SLSQP.
    """
    try:
        from scipy.optimize import minimize  # type: ignore
    except ImportError:
        return None
    n = len(mu)

    def obj(w):
        return float(w @ cov @ w)

    def obj_grad(w):
        return 2 * cov @ w

    cons = [
        {"type": "eq", "fun": lambda w: float(w.sum() - 1)},
        {"type": "eq", "fun": lambda w: float(w @ mu - target_return)},
    ]
    bounds = [(0 if long_only else -max_weight, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n
    res = minimize(
        obj, x0, jac=obj_grad, method="SLSQP", bounds=bounds, constraints=cons
    )
    if not res.success:
        return None
    return res.x


def resampled_efficient_frontier(
    returns: pd.DataFrame, config: REFConfig | None = None
) -> dict:
    """Michaud Resampled Efficient Frontier.

    Args:
        returns: DataFrame (T × N) of asset returns.
        config: REFConfig.

    Returns:
        Dict mit ``frontier_weights`` (DataFrame, n_frontier_points × N) und
        ``mean_returns`` und ``std_returns`` der frontier-portfolios.
    """
    cfg = config or REFConfig()
    rng = np.random.default_rng(cfg.seed)
    R = returns.dropna(how="any").values
    T, N = R.shape
    if T < 30:
        raise ValueError("need >= 30 historical observations")

    # Build target-return grid based on full-sample MVO range
    mu_full = R.mean(axis=0)
    target_returns = np.linspace(mu_full.min(), mu_full.max(), cfg.n_frontier_points)

    accumulated_weights = np.zeros((cfg.n_frontier_points, N))
    n_success = np.zeros(cfg.n_frontier_points, dtype=int)
    cov_full = np.cov(R, rowvar=False)

    for b in range(cfg.n_bootstrap):
        idx = rng.integers(0, T, size=T)
        sample = R[idx]
        mu = sample.mean(axis=0)
        cov = np.cov(sample, rowvar=False)
        # Regularize cov
        cov = cov + 1e-8 * np.eye(N)
        for k, tgt in enumerate(target_returns):
            w = _solve_mvo_target_return(mu, cov, tgt, cfg.long_only, cfg.max_weight)
            if w is not None:
                accumulated_weights[k] += w
                n_success[k] += 1

    # Average weights per frontier point (only where we had success)
    mean_w = np.zeros_like(accumulated_weights)
    for k in range(cfg.n_frontier_points):
        if n_success[k] > 0:
            mean_w[k] = accumulated_weights[k] / n_success[k]
        else:
            mean_w[k] = np.ones(N) / N
        # Renormalize to sum=1
        s = mean_w[k].sum()
        if s != 0:
            mean_w[k] /= s

    out_df = pd.DataFrame(mean_w, columns=returns.columns)
    out_df["target_return"] = target_returns
    out_df["mean_return"] = mean_w @ mu_full
    out_df["volatility"] = np.array([float(np.sqrt(w @ cov_full @ w)) for w in mean_w])
    out_df["sharpe_naive"] = out_df["mean_return"] / out_df["volatility"].replace(
        0, np.nan
    )
    return {
        "frontier_df": out_df,
        "n_bootstrap_used": int(np.median(n_success)),
        "n_assets": N,
    }


def select_target_vol_portfolio(
    frontier_df: pd.DataFrame, target_vol: float
) -> pd.Series:
    """Wähle das Portfolio aus der REF-Frontier, das target-vol am nächsten kommt.

    Returns:
        Series of weights for selected portfolio.
    """
    asset_cols = [
        c
        for c in frontier_df.columns
        if c not in ("target_return", "mean_return", "volatility", "sharpe_naive")
    ]
    diff = (frontier_df["volatility"] - target_vol).abs()
    idx = diff.idxmin()
    return frontier_df.loc[idx, asset_cols]


__all__ = [
    "REFConfig",
    "resampled_efficient_frontier",
    "select_target_vol_portfolio",
]
