"""Calmar-Bootstrap-Test — robuster als Sharpe-Bootstrap für MDD-Verbesserer.

Idee
----
Standard-Reality-Check / Hansen-SPA basieren auf Sharpe-Differenz, was fast
nichts misst, wenn der Edge primär im MDD liegt (wie bei Vol-Targeting).

Calmar-Ratio (AnnRet / |MDD|) erfasst beides: Return und Tail-Risk. Ein
Block-Bootstrap der Calmar-Ratio mit Stationary-Bootstrap (Politis-Romano)
ist die methodisch saubere Test-Statistik dafür.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _calmar(ret: pd.Series) -> float:
    """Calmar-Ratio einer Return-Series."""
    if ret.empty or ret.std() == 0:
        return -np.inf
    eq = (1 + ret).cumprod()
    ann_ret = eq.iloc[-1] ** (252 / len(ret)) - 1
    dd = (eq / eq.cummax() - 1).min()
    return ann_ret / abs(dd) if dd != 0 else -np.inf


def _stationary_bootstrap_indices(
    n: int, avg_block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano Stationary-Bootstrap: random-Block-Size mit Mean=avg_block_size."""
    p = 1.0 / avg_block_size
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        # geometrische Block-Länge
        length = rng.geometric(p)
        for k in range(length):
            if i >= n:
                break
            idx[i] = (start + k) % n
            i += 1
    return idx


def calmar_diff_bootstrap(
    a: pd.Series,
    b: pd.Series,
    n_bootstrap: int = 2000,
    avg_block_size: int = 20,
    seed: int = 42,
) -> dict:
    """Block-Bootstrap der Calmar-Differenz a − b mit Stationary-Bootstrap.

    Args:
        a, b: zwei Strategie-Return-Series, gleicher Index.
        n_bootstrap: Anzahl Bootstrap-Samples.
        avg_block_size: erwartete Block-Länge (Politis-Romano).
        seed: Random-Seed.

    Returns:
        dict mit mean_diff, ci_low_2.5, ci_high_97.5, p(diff>0), observed_diff.
    """
    rng = np.random.default_rng(seed)
    aligned = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if len(aligned) < 50:
        return {"error": "insufficient overlap"}

    a_arr = aligned["a"].to_numpy()
    b_arr = aligned["b"].to_numpy()
    n = len(aligned)

    observed_diff = _calmar(aligned["a"]) - _calmar(aligned["b"])

    diffs = []
    for _ in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n, avg_block_size, rng)
        sa = pd.Series(a_arr[idx])
        sb = pd.Series(b_arr[idx])
        d = _calmar(sa) - _calmar(sb)
        if np.isfinite(d):
            diffs.append(d)
    if not diffs:
        return {"error": "no finite bootstrap samples"}
    diffs = np.array(diffs)
    return {
        "observed_diff": float(observed_diff),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "ci_low_2.5": float(np.percentile(diffs, 2.5)),
        "ci_high_97.5": float(np.percentile(diffs, 97.5)),
        "p_value_one_sided_greater": float((diffs <= 0).mean()),
        "n_bootstrap": int(len(diffs)),
    }


__all__ = ["calmar_diff_bootstrap"]
