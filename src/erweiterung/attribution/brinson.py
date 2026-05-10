"""Brinson Performance Attribution (Brinson-Hood-Beebower 1986).

Decomposition
-------------
Portfolio active return = AllocationEffect + SelectionEffect + InteractionEffect

mit  AllocationEffect_i = (w_p_i - w_b_i) × (r_b_i - r_b_total)
     SelectionEffect_i  = w_b_i × (r_p_i - r_b_i)
     InteractionEffect_i = (w_p_i - w_b_i) × (r_p_i - r_b_i)

Σ_i über alle Sektoren/Gruppen.

Reference
---------
- Brinson, G., Hood, R. & Beebower, G. (1986). Determinants of Portfolio
  Performance. *FAJ* 42.
- Brinson, G., Singer, B. & Beebower, G. (1991). Determinants of Portfolio
  Performance II.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BrinsonResult:
    allocation_effect: dict[str, float]
    selection_effect: dict[str, float]
    interaction_effect: dict[str, float]
    total_active_return: float
    portfolio_return: float
    benchmark_return: float


def brinson_attribution(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
) -> BrinsonResult:
    """Single-period Brinson-Hood-Beebower Attribution.

    Args:
        portfolio_weights: dict {group: w_p_i}
        benchmark_weights: dict {group: w_b_i}
        portfolio_returns: dict {group: r_p_i}
        benchmark_returns: dict {group: r_b_i}

    Returns:
        BrinsonResult with effects per group.
    """
    groups = set(portfolio_weights) | set(benchmark_weights)
    r_b_total = sum(
        benchmark_weights.get(g, 0) * benchmark_returns.get(g, 0) for g in groups
    )
    r_p_total = sum(
        portfolio_weights.get(g, 0) * portfolio_returns.get(g, 0) for g in groups
    )

    allocation = {}
    selection = {}
    interaction = {}
    for g in groups:
        w_p = portfolio_weights.get(g, 0)
        w_b = benchmark_weights.get(g, 0)
        r_p = portfolio_returns.get(g, 0)
        r_b = benchmark_returns.get(g, 0)
        allocation[g] = float((w_p - w_b) * (r_b - r_b_total))
        selection[g] = float(w_b * (r_p - r_b))
        interaction[g] = float((w_p - w_b) * (r_p - r_b))

    return BrinsonResult(
        allocation_effect=allocation,
        selection_effect=selection,
        interaction_effect=interaction,
        total_active_return=float(r_p_total - r_b_total),
        portfolio_return=float(r_p_total),
        benchmark_return=float(r_b_total),
    )


def factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free: float = 0.0,
) -> dict:
    """Multi-Factor Performance Attribution via OLS.

    Args:
        portfolio_returns: Series.
        factor_returns: DataFrame (T × K), columns = factor names (e.g., MKT, SMB, HML, MOM).
        risk_free: optional rf rate.

    Returns:
        Dict with ``alpha``, ``factor_loadings`` (Series), ``r_squared``, ``t_alpha``.
    """
    df = pd.concat([portfolio_returns.rename("y"), factor_returns], axis=1).dropna()
    y = df["y"].values - risk_free
    X = df.drop(columns=["y"]).values
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    alpha = float(beta[0])
    loadings = beta[1:]
    resid = y - Xb @ beta
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # t-stat alpha
    se_resid = np.sqrt(ss_res / max(len(y) - len(beta), 1))
    XX_inv = np.linalg.pinv(Xb.T @ Xb)
    se_alpha = float(np.sqrt(XX_inv[0, 0]) * se_resid)
    t_alpha = alpha / se_alpha if se_alpha > 0 else float("nan")
    return {
        "alpha": alpha,
        "alpha_annualized": alpha * 252,
        "factor_loadings": pd.Series(loadings, index=factor_returns.columns),
        "r_squared": float(r2),
        "t_alpha": float(t_alpha),
        "se_alpha": se_alpha,
    }


def pnl_decomposition(
    portfolio_weights_history: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """PnL per Asset über Zeit.

    Args:
        portfolio_weights_history: DataFrame (T × N) — Gewichte je Tag.
        asset_returns: DataFrame (T × N) — Returns je Tag.

    Returns:
        DataFrame (T × N) — PnL contribution = w_lag * return.
    """
    w_lag = portfolio_weights_history.shift(1).fillna(0)
    # align columns
    common = w_lag.columns.intersection(asset_returns.columns)
    pnl = w_lag[common] * asset_returns[common]
    return pnl


__all__ = [
    "BrinsonResult",
    "brinson_attribution",
    "factor_attribution",
    "pnl_decomposition",
]
