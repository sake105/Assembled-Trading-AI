"""Maximum-Entropy Bootstrap (Vinod 2004, 2006).

Theorie
-------
Klassischer Bootstrap zerstört Autokorrelation. Block-Bootstrap respektiert
Lokalstruktur, aber bricht Blöcke harsh. **MaxEnt-Bootstrap** generiert
synthetic time-series die:
1. **Stationarity-preserving** (mean ≈ original mean)
2. **Local-rank-preserving** (Reihenfolge der lokalen Variation erhalten)
3. **Maximum-Entropy** unter diesen Constraints

Algorithmus
-----------
1. Sortiere data → erhalte sortierte Werte z_(1) ≤ ... ≤ z_(n).
2. Konstruiere ZULÄSSIGE intervalle: z_(i) ist zwischen empirical CDF (i-1)/(2n) und (i+1)/(2n).
3. Generiere uniform-distributed U_i ∼ U(0,1).
4. Map U_i via empirische CDF zu sortierten Quantilen.
5. Sortiere die generierten Werte gemäß der ORIGINAL-Reihenfolge (rank-preserving).

Anwendung
---------
- Confidence-Intervals für Backtest-Statistiken (Sharpe, Max-DD, etc.)
- Robust gegen Volatility-Clustering + Non-i.i.d.

Reference
---------
Vinod, H. (2004). Ranking Mutual Funds Using Unconventional Utility Theory and
Stochastic Dominance. *J. Empirical Finance* 11.
Vinod, H. (2006). Maximum entropy ensembles for time series inference.
*J. Asian Economics* 17.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def maxent_bootstrap_sample(
    series: pd.Series, seed: int = 42, trim_quantile: float = 0.01
) -> pd.Series:
    """One MaxEnt-Bootstrap sample of length len(series).

    Args:
        series: input series.
        seed: RNG seed.
        trim_quantile: pad-fraction for tail-handling.

    Returns:
        Series with same index/length, max-entropy-bootstrap values.
    """
    s = pd.Series(series).dropna()
    n = len(s)
    if n < 30:
        raise ValueError("need >= 30 obs")
    rng = np.random.default_rng(seed)

    # 1. Sort original
    order = np.argsort(s.values)
    sorted_vals = s.values[order]

    # 2. Build intervals around each sorted value (use midpoints)
    # m_i = midpoint between z_(i) and z_(i+1)
    midpoints = (sorted_vals[:-1] + sorted_vals[1:]) / 2
    # Boundaries: z_min - σ*trim_q, midpoints, z_max + σ*trim_q
    sigma = float(s.std())
    pad = trim_quantile * sigma
    z_low = sorted_vals[0] - pad
    z_high = sorted_vals[-1] + pad
    intervals_lo = np.concatenate([[z_low], midpoints])
    intervals_hi = np.concatenate([midpoints, [z_high]])

    # 3. Draw uniform [0, 1], map to interval positions
    u = rng.uniform(size=n)
    quantiles_idx = (u * n).astype(int).clip(0, n - 1)
    # Within-interval position
    within_pos = u * n - quantiles_idx
    sampled_sorted = intervals_lo[quantiles_idx] + within_pos * (
        intervals_hi[quantiles_idx] - intervals_lo[quantiles_idx]
    )

    # 4. Place values back in original rank-order
    sampled = np.zeros(n)
    for rank, orig_idx in enumerate(order):
        sampled[orig_idx] = sampled_sorted[rank]

    return pd.Series(sampled, index=s.index)


def maxent_bootstrap_ensemble(
    series: pd.Series, n_samples: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Generate n_samples MaxEnt-Bootstrap-Pfade.

    Returns:
        DataFrame (n_obs, n_samples) — each column = one bootstrap.
    """
    out = {}
    for i in range(n_samples):
        out[i] = maxent_bootstrap_sample(series, seed=seed + i)
    return pd.DataFrame(out)


def bootstrap_confidence_interval(
    series: pd.Series,
    statistic: callable,
    n_samples: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """CI für eine Statistik via MaxEnt-Bootstrap.

    Args:
        series: input.
        statistic: callable(series) -> scalar.
        n_samples: bootstrap reps.
        confidence: e.g. 0.95.
        seed: RNG.

    Returns:
        dict mit ``point_estimate``, ``ci_low``, ``ci_high``, ``se``.
    """
    point = float(statistic(series))
    bootstrap_stats = []
    for i in range(n_samples):
        try:
            sample = maxent_bootstrap_sample(series, seed=seed + i)
            bootstrap_stats.append(float(statistic(sample)))
        except Exception:  # noqa: BLE001
            continue
    if not bootstrap_stats:
        return {"point_estimate": point, "error": "no successful samples"}
    arr = np.array(bootstrap_stats)
    a = (1 - confidence) / 2
    return {
        "point_estimate": point,
        "ci_low": float(np.quantile(arr, a)),
        "ci_high": float(np.quantile(arr, 1 - a)),
        "se": float(np.std(arr, ddof=1)),
        "n_samples_used": int(len(arr)),
    }


__all__ = [
    "maxent_bootstrap_sample",
    "maxent_bootstrap_ensemble",
    "bootstrap_confidence_interval",
]
