"""Active-Risk-Analytics: Tracking-Error, Active-Share, Information-Ratio.

Definitions
-----------
- **Tracking Error**: σ(r_portfolio − r_benchmark) annualized
- **Active Share** (Cremers/Petajisto 2009): 0.5 × Σ |w_p − w_b| ∈ [0, 1]
- **Information Ratio**: mean(r_p - r_b) / σ(r_p - r_b)
- **Active Risk Decomposition**: variance of active returns split by factor exposure
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    annual_factor: float = 252,
) -> float:
    """Annualized tracking error."""
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(df) < 30:
        return float("nan")
    active = df.iloc[:, 0] - df.iloc[:, 1]
    return float(active.std(ddof=0) * np.sqrt(annual_factor))


def active_share(portfolio_weights: pd.Series, benchmark_weights: pd.Series) -> float:
    """Cremers/Petajisto Active Share ∈ [0, 1].

    0 = identisch zum Benchmark, 1 = komplett unabhängig.
    """
    common = portfolio_weights.index.union(benchmark_weights.index)
    p = portfolio_weights.reindex(common).fillna(0)
    b = benchmark_weights.reindex(common).fillna(0)
    return float(0.5 * (p - b).abs().sum())


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    annual_factor: float = 252,
) -> float:
    """IR = mean(active return) / std(active return) × √ann_factor."""
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(df) < 30:
        return float("nan")
    active = df.iloc[:, 0] - df.iloc[:, 1]
    if active.std(ddof=0) == 0:
        return float("nan")
    return float(active.mean() / active.std(ddof=0) * np.sqrt(annual_factor))


def active_risk_decomposition(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> dict:
    """Decompose active risk into systematic (factor-explained) + specific.

    Returns:
        dict mit total_active_risk, systematic_risk, specific_risk + R².
    """
    df = pd.concat(
        [portfolio_returns.rename("p"), benchmark_returns.rename("b"), factor_returns],
        axis=1,
    ).dropna()
    if len(df) < 30:
        return {"error": "too few obs"}
    active = df["p"] - df["b"]
    X = df[factor_returns.columns].values
    y = active.values
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ beta
    resid = y - pred
    total_var = float(np.var(y, ddof=0))
    sys_var = float(np.var(pred, ddof=0))
    spec_var = float(np.var(resid, ddof=0))
    return {
        "total_active_risk_ann": float(np.sqrt(total_var * 252)),
        "systematic_risk_ann": float(np.sqrt(sys_var * 252)),
        "specific_risk_ann": float(np.sqrt(spec_var * 252)),
        "factor_share": sys_var / total_var if total_var > 0 else 0.0,
        "factor_loadings": pd.Series(beta[1:], index=factor_returns.columns),
        "alpha_ann": float(beta[0] * 252),
    }


def turnover_ratio(weights_history: pd.DataFrame) -> pd.Series:
    """Daily turnover = Σ |w_t - w_{t-1}| / 2."""
    diff = weights_history.diff().abs().sum(axis=1) / 2
    return diff.fillna(0)


def concentration_metrics(weights: pd.Series) -> dict:
    """Portfolio-Concentration: HHI + Effective-N + Top-K-Share."""
    w = weights.abs()
    w_norm = w / w.sum() if w.sum() > 0 else w
    hhi = float((w_norm**2).sum())
    eff_n = 1.0 / hhi if hhi > 0 else 0.0
    top5 = float(w_norm.sort_values(ascending=False).head(5).sum())
    return {
        "hhi": hhi,
        "effective_n": eff_n,
        "top5_share": top5,
        "max_weight": float(w_norm.max()) if len(w_norm) > 0 else 0.0,
    }


__all__ = [
    "tracking_error",
    "active_share",
    "information_ratio",
    "active_risk_decomposition",
    "turnover_ratio",
    "concentration_metrics",
]
