"""Performance-Metriken mit korrigierten Standard-Errors.

Inhalt
------
- Sharpe Ratio + IID-corrected SE
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- Recovery Time
- Tail-Ratio
- Win/Loss Rate
- Profit Factor
- Information Ratio (vs benchmark)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, annual_factor: float = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=0) * np.sqrt(annual_factor))


def sortino_ratio(
    returns: pd.Series, mar: float = 0.0, annual_factor: float = 252
) -> float:
    r = returns.dropna()
    downside = r[r < mar]
    if len(downside) < 2:
        return float("nan")
    dd_std = float(np.sqrt((downside**2).mean()))
    if dd_std == 0:
        return float("nan")
    return float((r.mean() - mar) / dd_std * np.sqrt(annual_factor))


def max_drawdown(
    equity: pd.Series,
) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    if equity.empty:
        return (0.0, None, None)
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    if dd.empty or pd.isna(dd.min()):
        return (0.0, None, None)
    trough = dd.idxmin()
    peak_before = cummax.loc[:trough].idxmax()
    return float(dd.min()), peak_before, trough


def calmar_ratio(returns: pd.Series, annual_factor: float = 252) -> float:
    eq = (1 + returns).cumprod()
    mdd, _, _ = max_drawdown(eq)
    if mdd >= 0:
        return float("nan")
    cagr = (
        float(eq.iloc[-1] ** (annual_factor / len(returns)) - 1)
        if len(returns) > 0
        else float("nan")
    )
    return cagr / abs(mdd)


def tail_ratio(returns: pd.Series, alpha: float = 0.05) -> float:
    """|q(α)| / |q(1-α)| — Asymmetrie zwischen Tails."""
    r = returns.dropna()
    if len(r) < 30:
        return float("nan")
    pos = float(r.quantile(1 - alpha))
    neg = float(r.quantile(alpha))
    if neg == 0:
        return float("nan")
    return pos / abs(neg)


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, annual_factor: float = 252
) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["r", "b"]
    excess = aligned["r"] - aligned["b"]
    if excess.std(ddof=0) == 0:
        return float("nan")
    return float(excess.mean() / excess.std(ddof=0) * np.sqrt(annual_factor))


def all_metrics(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    annual_factor: float = 252,
) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"error": "empty returns"}
    eq = (1 + r).cumprod()
    mdd, peak, trough = max_drawdown(eq)
    out = {
        "n_obs": len(r),
        "annualized_return": float(eq.iloc[-1] ** (annual_factor / len(r)) - 1),
        "annualized_vol": float(r.std(ddof=0) * np.sqrt(annual_factor)),
        "sharpe": sharpe_ratio(r, annual_factor),
        "sortino": sortino_ratio(r, annual_factor=annual_factor),
        "max_drawdown": mdd,
        "calmar": calmar_ratio(r, annual_factor),
        "tail_ratio": tail_ratio(r),
        "profit_factor": profit_factor(r),
        "skew": float(r.skew()),
        "kurt": float(r.kurt()),  # excess
        "win_rate": float((r > 0).mean()),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
    }
    if benchmark is not None:
        out["information_ratio"] = information_ratio(r, benchmark, annual_factor)
    return out


__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "tail_ratio",
    "profit_factor",
    "information_ratio",
    "all_metrics",
]
