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
    """One MaxEnt-Bootstrap sample of length len(series) (Vinod 2004).

    Algorithmus (Vinod 2004 §2):
    ---------------------------
    1. Sortiere die n original-Werte: ``x_(1) ≤ x_(2) ≤ ... ≤ x_(n)``.
    2. Bestimme intermediate-points ``z_t``:
       - ``z_0 = x_(1) − pad`` (linker Rand)
       - ``z_t = (x_(t) + x_(t+1)) / 2`` für t = 1..n-1 (Midpoints)
       - ``z_n = x_(n) + pad`` (rechter Rand)
       → n disjunkte Intervalle ``[z_{t-1}, z_t]`` für t = 1..n.
    3. Für jeden Sample-Schritt: ziehe ``u ~ U(0, 1)`` und mappe via
       **empirischer Inverse-CDF**: Intervall-Index ``k = ⌊u·n⌋``, dann sample
       uniform aus ``[z_k, z_{k+1}]`` (= Maximum-Entropy unter der Konstraint,
       dass Intervall-Position-Wahrscheinlichkeit 1/n ist).
    4. Sortiere die gezogenen Werte und ordne sie gemäß **Original-Ranks** an
       (Rank-Preservation = Auto-Korrelations-Erhaltung).

    Args:
        series: input series.
        seed: RNG seed.
        trim_quantile: pad-fraction für tail-handling (in σ-Einheiten).

    Returns:
        Series with same index/length, max-entropy-bootstrap values.
    """
    s = pd.Series(series).dropna()
    n = len(s)
    if n < 30:
        raise ValueError("need >= 30 obs")
    rng = np.random.default_rng(seed)

    # Step 1: Sort
    order = np.argsort(s.values)
    sorted_vals = s.values[order]

    # Step 2: Midpoints + Boundaries → n Intervalle
    midpoints = (sorted_vals[:-1] + sorted_vals[1:]) / 2
    sigma = float(s.std())
    pad = trim_quantile * sigma
    z = np.concatenate([[sorted_vals[0] - pad], midpoints, [sorted_vals[-1] + pad]])
    # z hat n+1 Elemente; Intervall t = [z[t], z[t+1]] für t = 0..n-1.

    # Step 3: empirische Inverse-CDF Inversion
    # u ∈ [0, 1) → Intervall-Index k = ⌊u·n⌋ ∈ {0, .., n-1}
    u = rng.uniform(size=n)
    k = np.floor(u * n).astype(int).clip(0, n - 1)
    within = u * n - k  # uniform innerhalb gewähltem Intervall
    sampled_sorted = z[k] + within * (z[k + 1] - z[k])
    # Vinod's MaxEnt-Garantie: jeder Sample ist uniform in einem zufällig
    # gewählten Intervall — entspricht maximaler Entropie unter Empirical-CDF.

    # Step 4: Rang-erhaltend zurückmappen
    # sampled_sorted ist NICHT sortiert — wir sortieren erst, dann verteilen
    # gemäß original-rank-ordnung.
    sampled_sorted.sort()
    sampled = np.empty(n)
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
