"""Erweiterte Performance- und Risk-Metriken.

Inhalt
------
- ``omega_ratio``         : Keating/Shadwick (2002)
- ``treynor_ratio``       : Excess Return / Beta
- ``jensens_alpha``       : CAPM regression alpha
- ``modigliani_rap``      : M² (Risk-Adjusted Performance)
- ``ulcer_index``         : Martin (1989)
- ``pain_index``          : Mean Drawdown
- ``burke_ratio``         : SQRT-sum of squared drawdowns
- ``stutzer_index``       : Stutzer (2000) — info-theoretic
- ``upside_potential_ratio``: Sortino-like with upside
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega-Ratio (Keating/Shadwick 2002): area above threshold / area below."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    above = (r - threshold).clip(lower=0).sum()
    below = (threshold - r).clip(lower=0).sum()
    if below == 0:
        return float("inf") if above > 0 else float("nan")
    return float(above / below)


def treynor_ratio(returns: pd.Series, market: pd.Series, rf: float = 0.0) -> float:
    """Treynor: (mean(r) - rf) / β."""
    df = pd.concat([returns, market], axis=1).dropna()
    df.columns = ["r", "m"]
    if len(df) < 30 or df["m"].var(ddof=0) == 0:
        return float("nan")
    beta = float(df["r"].cov(df["m"]) / df["m"].var(ddof=0))
    if beta == 0:
        return float("nan")
    return float((df["r"].mean() - rf) / beta)


def jensens_alpha(
    returns: pd.Series, market: pd.Series, rf: float = 0.0, annual_factor: float = 252
) -> float:
    """Jensen's α from CAPM regression (annualized)."""
    df = pd.concat([returns, market], axis=1).dropna()
    df.columns = ["r", "m"]
    if len(df) < 30:
        return float("nan")
    excess_r = df["r"] - rf
    excess_m = df["m"] - rf
    if excess_m.var(ddof=0) == 0:
        return float("nan")
    beta = float(excess_r.cov(excess_m) / excess_m.var(ddof=0))
    alpha_per_period = float(excess_r.mean() - beta * excess_m.mean())
    return alpha_per_period * annual_factor


def modigliani_rap(returns: pd.Series, benchmark: pd.Series, rf: float = 0.0) -> float:
    """M² Risk-Adjusted Performance."""
    r = pd.Series(returns).dropna()
    b = pd.Series(benchmark).reindex(r.index).fillna(0)
    if r.empty:
        return float("nan")
    sharpe_p = (r.mean() - rf) / r.std(ddof=0) if r.std(ddof=0) > 0 else float("nan")
    sigma_b = b.std(ddof=0)
    if pd.isna(sharpe_p) or pd.isna(sigma_b):
        return float("nan")
    return float(sharpe_p * sigma_b + rf)


def ulcer_index(equity: pd.Series) -> float:
    """Martin (1989) Ulcer Index: SQRT-mean of squared drawdown percentages."""
    eq = pd.Series(equity).dropna()
    if eq.empty:
        return float("nan")
    cummax = eq.cummax()
    dd_pct = ((eq / cummax) - 1) * 100
    return float(np.sqrt((dd_pct**2).mean()))


def pain_index(equity: pd.Series) -> float:
    """Mean Drawdown — simpler than Ulcer."""
    eq = pd.Series(equity).dropna()
    if eq.empty:
        return float("nan")
    cummax = eq.cummax()
    dd = (eq / cummax) - 1  # negative
    return float(dd.abs().mean())


def burke_ratio(returns: pd.Series, equity: pd.Series, rf: float = 0.0) -> float:
    """Burke Ratio: (mean(r) - rf) / sqrt(sum(D²)) where D are individual drawdowns."""
    r = pd.Series(returns).dropna()
    eq = pd.Series(equity).reindex(r.index).dropna()
    if eq.empty:
        return float("nan")
    cummax = eq.cummax()
    dd = ((eq / cummax) - 1).abs()
    # Pick discrete drawdown bottoms (where cummax stops increasing)
    bottoms = dd[dd > 0]
    if bottoms.empty:
        return float("inf")
    sqrt_sum = float(np.sqrt((bottoms**2).sum()))
    if sqrt_sum == 0:
        return float("nan")
    return float((r.mean() - rf) / sqrt_sum)


def stutzer_index(
    returns: pd.Series, threshold: float = 0.0, annual_factor: float = 252
) -> float:
    """Stutzer (2000) information-theoretic performance index.

    Approximates rate of decay of probability that excess return < 0.
    """
    r = pd.Series(returns).dropna() - threshold
    if len(r) < 30:
        return float("nan")
    # Maximize over θ: I(θ) = log(E[exp(-θ * r)])
    thetas = np.linspace(0.01, 5.0, 50)
    info_rates = [float(np.log(np.mean(np.exp(-t * r.values)))) for t in thetas]
    min_idx = int(np.argmin(info_rates))
    decay = -info_rates[min_idx]
    return float(np.sign(r.mean()) * np.sqrt(2 * abs(decay) * annual_factor))


def upside_potential_ratio(returns: pd.Series, mar: float = 0.0) -> float:
    """Sortino-like: upside-mean / downside-deviation."""
    r = pd.Series(returns).dropna()
    above = (r - mar).clip(lower=0)
    below_sq = ((mar - r).clip(lower=0)) ** 2
    if below_sq.mean() == 0:
        return float("inf") if above.mean() > 0 else float("nan")
    return float(above.mean() / np.sqrt(below_sq.mean()))


def comprehensive_metrics(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    rf_daily: float = 0.0,
    annual_factor: float = 252,
) -> dict:
    """Compute all advanced metrics in one call."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return {"error": "empty"}
    eq = (1 + r).cumprod()
    out: dict = {
        "omega_ratio_0": omega_ratio(r, 0.0),
        "ulcer_index": ulcer_index(eq),
        "pain_index": pain_index(eq),
        "burke_ratio": burke_ratio(r, eq, rf_daily),
        "stutzer_index": stutzer_index(
            r, threshold=rf_daily, annual_factor=annual_factor
        ),
        "upside_potential": upside_potential_ratio(r, mar=rf_daily),
    }
    if benchmark is not None:
        out["treynor_ratio"] = treynor_ratio(r, benchmark, rf_daily)
        out["jensens_alpha_ann"] = jensens_alpha(r, benchmark, rf_daily, annual_factor)
        out["modigliani_rap_ann"] = (
            modigliani_rap(r, benchmark, rf_daily) * annual_factor
        )
    return out


__all__ = [
    "omega_ratio",
    "treynor_ratio",
    "jensens_alpha",
    "modigliani_rap",
    "ulcer_index",
    "pain_index",
    "burke_ratio",
    "stutzer_index",
    "upside_potential_ratio",
    "comprehensive_metrics",
]
