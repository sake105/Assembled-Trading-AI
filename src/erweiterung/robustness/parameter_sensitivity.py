"""Parameter-Sensitivity-Analysis.

Idee
----
Eine Strategy mit unrealistischer Param-Optimum (z. B. lookback=42 best
und Sharpe stürzt bei lookback=40 oder 44 ab) ist overfit. Stabile Strategien
zeigen **monoton oder smooth** abhängig vom Parameter.

Methodik
--------
1. Sweep über mehrere Param-Werte → Sharpe-Liste.
2. Stability-Score = 1 − std(sharpe) / |mean(sharpe)|.
3. Best/Worst-Parameter-Sharpe + Smoothness via finite-difference.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def parameter_sweep(
    backtest_fn: Callable[[float | int], pd.Series],
    param_values: list,
    annual_factor: float = 252,
) -> pd.DataFrame:
    """Run backtest for each param-value, return per-param-metrics.

    Args:
        backtest_fn: callable(param) -> pd.Series of returns.
        param_values: list of param-values to test.
        annual_factor: 252 etc.

    Returns:
        DataFrame [param, sharpe, ann_return, max_dd, n_obs].
    """
    rows = []
    for v in param_values:
        try:
            r = backtest_fn(v)
        except Exception:  # noqa: BLE001
            continue
        r = pd.Series(r).dropna()
        if r.empty:
            continue
        mean = float(r.mean())
        std = float(r.std(ddof=0))
        sharpe = mean / std * np.sqrt(annual_factor) if std > 0 else float("nan")
        eq = (1 + r).cumprod()
        ann_ret = (
            float(eq.iloc[-1] ** (annual_factor / len(r)) - 1)
            if len(r) > 0
            else float("nan")
        )
        max_dd = float((eq / eq.cummax() - 1).min())
        rows.append(
            {
                "param": v,
                "sharpe": sharpe,
                "ann_return": ann_ret,
                "max_dd": max_dd,
                "n_obs": len(r),
            }
        )
    return pd.DataFrame(rows)


def stability_score(sweep_df: pd.DataFrame, metric: str = "sharpe") -> float:
    """1 − std(metric) / |mean(metric)|. Höher = stabiler."""
    s = sweep_df[metric].dropna()
    if len(s) < 2 or abs(s.mean()) < 1e-9:
        return float("nan")
    return float(1.0 - s.std(ddof=0) / abs(s.mean()))


def smoothness_score(sweep_df: pd.DataFrame, metric: str = "sharpe") -> float:
    """Finite-difference roughness = mean(|Δmetric|) / |mean(metric)|.

    Niedrig = smooth. Werte > 1 = high roughness = wahrscheinlich overfit.
    """
    s = sweep_df[metric].dropna()
    if len(s) < 3 or abs(s.mean()) < 1e-9:
        return float("nan")
    diffs = s.diff().abs().dropna()
    return float(diffs.mean() / abs(s.mean()))


def best_robust_parameter(
    sweep_df: pd.DataFrame, metric: str = "sharpe", neighbor_weight: float = 0.5
) -> dict:
    """Find param that maximizes metric + neighbors-average (robust optimum).

    Adjusted-metric_i = metric_i + neighbor_weight × (avg(metric_{i-1}, metric_{i+1}))
    """
    df = sweep_df.sort_values("param").reset_index(drop=True)
    if len(df) < 3:
        if df.empty:
            return {"error": "empty"}
        idx = df[metric].idxmax()
        return {
            "param": df.loc[idx, "param"],
            "metric": float(df.loc[idx, metric]),
            "adjusted_metric": float(df.loc[idx, metric]),
        }
    df["neighbor_avg"] = df[metric].rolling(3, center=True, min_periods=1).mean()
    df["adjusted"] = df[metric] + neighbor_weight * df["neighbor_avg"]
    idx = df["adjusted"].idxmax()
    return {
        "param": df.loc[idx, "param"],
        "metric": float(df.loc[idx, metric]),
        "adjusted_metric": float(df.loc[idx, "adjusted"]),
        "stability_neighbors": float(df.loc[idx, "neighbor_avg"]),
    }


__all__ = [
    "parameter_sweep",
    "stability_score",
    "smoothness_score",
    "best_robust_parameter",
]
