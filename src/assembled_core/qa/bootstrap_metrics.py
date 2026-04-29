"""Bootstrap confidence intervals for performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sharpe(arr: np.ndarray) -> float:
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(252))


def _sortino(arr: np.ndarray) -> float:
    downside = arr[arr < 0]
    if len(downside) == 0:
        return float("inf")
    dstd = downside.std(ddof=1)
    if dstd == 0:
        return float("inf")
    return float(arr.mean() / dstd * np.sqrt(252))


def _max_drawdown(arr: np.ndarray) -> float:
    cumret = np.cumprod(1 + arr)
    running_max = np.maximum.accumulate(cumret)
    dd = (cumret - running_max) / running_max
    return float(dd.min())


def compute_sharpe_with_ci(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    arr = returns.dropna().values
    rng = np.random.default_rng(seed)
    samples = [
        _sharpe(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ]
    lo = (1 - ci) / 2 * 100
    hi = 100 - lo
    return {
        "sharpe": _sharpe(arr),
        "sharpe_ci_lower": float(np.percentile(samples, lo)),
        "sharpe_ci_upper": float(np.percentile(samples, hi)),
        "sharpe_p_value": float((np.array(samples) <= 0).mean()),
        "n_obs": len(arr),
        "n_bootstrap": n_bootstrap,
    }


def compute_sortino_with_ci(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    arr = returns.dropna().values
    rng = np.random.default_rng(seed)
    samples = [
        _sortino(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ]
    finite = [s for s in samples if np.isfinite(s)]
    lo = (1 - ci) / 2 * 100
    hi = 100 - lo
    return {
        "sortino": _sortino(arr),
        "sortino_ci_lower": float(np.percentile(finite, lo)) if finite else float("nan"),
        "sortino_ci_upper": float(np.percentile(finite, hi)) if finite else float("nan"),
        "n_obs": len(arr),
        "n_bootstrap": n_bootstrap,
    }


def compute_max_drawdown_with_ci(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    arr = returns.dropna().values
    rng = np.random.default_rng(seed)
    samples = [
        _max_drawdown(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ]
    lo = (1 - ci) / 2 * 100
    hi = 100 - lo
    return {
        "max_drawdown": _max_drawdown(arr),
        "max_drawdown_ci_lower": float(np.percentile(samples, lo)),
        "max_drawdown_ci_upper": float(np.percentile(samples, hi)),
        "n_obs": len(arr),
        "n_bootstrap": n_bootstrap,
    }


def compute_all_with_ci(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Compute Sharpe, Sortino, and max_drawdown with bootstrap CIs in one call."""
    result: dict[str, float] = {}
    result.update(compute_sharpe_with_ci(returns, n_bootstrap, ci, seed))
    result.update(compute_sortino_with_ci(returns, n_bootstrap, ci, seed))
    result.update(compute_max_drawdown_with_ci(returns, n_bootstrap, ci, seed))
    return result
