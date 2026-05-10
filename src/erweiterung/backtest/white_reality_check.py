"""White's Reality Check + Hansen's SPA-Test (Multi-Strategy-Bias-Korrektur).

Theorie
-------
Wenn man N Strategien testet und die beste hat Sharpe X, ist das **nicht**
direkt mit einer einzeln getesteten Strategie vergleichbar. Die Maximum-
Statistik unter H₀ hat eine andere Verteilung als die einzelne.

White (2000) "Reality Check"
----------------------------
Test-Statistik: V̄ = max_k mean(d_k_t) wobei d_k_t = perf_k_t − perf_benchmark_t.
Critical-Value via stationary-bootstrap-Resampling unter H₀: alle d̄_k = 0.

Hansen (2005) SPA-Test
----------------------
Verbesserung: Studentisierte Statistik
    T^SPA = max_k √n · max(d̄_k, 0) / σ̂_k
und re-zentrierte Bootstrap-Verteilung (entfernt schlechte Strategien).
SPA hat höhere Power als Reality Check.

Anwendung
---------
- Bei N = 1000 Strategien gegen S&P-Benchmark testen.
- Beste Strategie hat Sharpe 2.0; SPA-p-value sagt, ob das ein Artefakt ist.

Referenzen
----------
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica* 68(5).
- Hansen, P. (2005). A Test for Superior Predictive Ability. *J Bus & Econ Stat*.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def stationary_bootstrap_indices(
    n: int, expected_block_length: float = 5.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Politis-Romano (1994) Stationary Bootstrap.

    Block-Längen sind geometrisch verteilt mit Mean = ``expected_block_length``.
    Resultierende Sequenz ist stationär.
    """
    rng = rng or np.random.default_rng()
    p = 1.0 / expected_block_length
    out = np.empty(n, dtype=int)
    out[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            out[i] = rng.integers(0, n)
        else:
            out[i] = (out[i - 1] + 1) % n
    return out


def whites_reality_check(
    excess_returns: pd.DataFrame,
    n_bootstrap: int = 1000,
    expected_block_length: float = 5.0,
    seed: int = 42,
) -> dict:
    """White's Reality Check.

    Args:
        excess_returns: DataFrame (T × K) mit excess returns je Strategie über
            Benchmark. Index = date, columns = strategy_name.
        n_bootstrap: Anzahl Bootstrap-Wiederholungen.
        expected_block_length: für stationary bootstrap.
        seed: RNG-Seed.

    Returns:
        Dict mit
        - ``best_strategy``: Spaltenname mit höchstem Mean.
        - ``observed_stat``: max_k √n d̄_k.
        - ``p_value``: Anteil bootstrap-Statistiken >= observed (eindeutig).

    Interpretation
    --------------
    p_value < 0.05 ⇒ Mindestens eine Strategie schlägt Benchmark signifikant
    auch nach Multi-Test-Korrektur.
    """
    if excess_returns.empty or excess_returns.shape[1] < 1:
        return {"error": "need at least one strategy"}
    rng = np.random.default_rng(seed)

    n, K = excess_returns.shape
    means = excess_returns.mean()
    observed = float(np.sqrt(n) * means.max())

    boot_stats = np.empty(n_bootstrap)
    R = excess_returns.values
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n, expected_block_length, rng)
        sample = R[idx]
        # Re-center under H0: d̄_k = 0
        boot_means = sample.mean(axis=0) - means.values
        boot_stats[b] = np.sqrt(n) * boot_means.max()

    p = float((boot_stats >= observed).mean())
    return {
        "best_strategy": str(means.idxmax()),
        "observed_stat": observed,
        "p_value": p,
        "n_bootstrap": n_bootstrap,
    }


def hansen_spa_test(
    excess_returns: pd.DataFrame,
    n_bootstrap: int = 1000,
    expected_block_length: float = 5.0,
    seed: int = 42,
) -> dict:
    """Hansen's SPA-Test (Superior Predictive Ability).

    Verbesserter Reality-Check: Bootstrap-Distribution wird re-zentriert, um
    nur "binding" Strategien (mean > -threshold) zu zählen. Höhere Power.
    """
    if excess_returns.empty or excess_returns.shape[1] < 1:
        return {"error": "need at least one strategy"}
    rng = np.random.default_rng(seed)
    n, K = excess_returns.shape
    means = excess_returns.mean()
    stds = excess_returns.std(ddof=0).replace(0, np.nan).fillna(1)
    observed = float(np.sqrt(n) * (means / stds).max())

    # SPA-c re-centering threshold
    threshold = -np.sqrt(2 * np.log(np.log(n)) / n)
    centered_means = np.where(means.values > threshold, means.values, 0.0)

    boot_stats = np.empty(n_bootstrap)
    R = excess_returns.values
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n, expected_block_length, rng)
        sample = R[idx]
        boot_means = sample.mean(axis=0) - centered_means
        boot_t = np.sqrt(n) * boot_means / stds.values
        boot_stats[b] = boot_t.max()

    p = float((boot_stats >= observed).mean())
    return {
        "best_strategy": str((means / stds).idxmax()),
        "observed_t_stat": observed,
        "p_value": p,
        "n_bootstrap": n_bootstrap,
    }


__all__ = [
    "stationary_bootstrap_indices",
    "whites_reality_check",
    "hansen_spa_test",
]
