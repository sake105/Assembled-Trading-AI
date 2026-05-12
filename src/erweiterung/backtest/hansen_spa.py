"""Hansen (2005) Superior-Predictive-Ability Test — studentized variant (audit C4-066).

The existing :mod:`erweiterung.backtest.white_reality_check.hansen_spa_test`
re-centers the bootstrap distribution using a threshold on the **raw**
mean, which is only correct when σ_k = 1. Hansen's published recipe
applies the threshold to the **studentized** statistic, i.e.
:math:`\\sqrt{n}\\,\\hat\\mu_k/\\hat\\sigma_k > -\\sqrt{2 \\log \\log n}`.

This module ships the studentized recentering as a separate function so:

* existing research outputs that referenced :func:`hansen_spa_test`
  remain reproducible,
* new code can opt into the studentized variant via
  :func:`hansen_spa_test_studentized`,
* the test-suite documents both behaviors and the size of the divergence.

The bootstrap engine itself is the same Politis-Romano (1994) stationary
bootstrap used by :func:`whites_reality_check`.

References
----------
- Hansen, P. R. (2005). *A Test for Superior Predictive Ability*,
  J. Bus. & Econ. Stat. 23(4), 365–380, equation (10) for SPA-c.
- Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap*,
  JASA 89(428), 1303–1313.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.erweiterung.backtest.white_reality_check import (
    stationary_bootstrap_indices,
)


logger = logging.getLogger(__name__)


Recentering = Literal["c", "l", "u"]
"""SPA recentering variant — c (consistent, default), l (lower), u (upper)."""


def hansen_spa_test_studentized(
    excess_returns: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    expected_block_length: float = 5.0,
    recentering: Recentering = "c",
    seed: int = 42,
) -> dict:
    """Hansen 2005 SPA test with **studentized** recentering threshold.

    Args:
        excess_returns: T × K DataFrame of per-period excess returns
            (strategy minus benchmark). Columns are strategy names.
        n_bootstrap: number of stationary-bootstrap replications.
        expected_block_length: geometric-block mean (Politis-Romano).
        recentering: which variant of the recentering test to use:

            * ``"c"`` — *consistent* (default, Hansen eq. 10): drops
              strategies whose studentized statistic falls below
              :math:`-\\sqrt{2 \\log \\log n}`.
            * ``"l"`` — *lower* bound: ignores poor strategies entirely
              (most powerful but anti-conservative).
            * ``"u"`` — *upper* bound: centers all strategies at their
              observed mean (matches the raw "Reality Check" recipe;
              least powerful).
        seed: RNG seed for reproducibility.

    Returns:
        A dict with ``best_strategy``, ``observed_t_stat``,
        ``p_value``, ``n_bootstrap``, and ``recentering``.

    Raises:
        ValueError: if the input DataFrame is empty.
    """
    if excess_returns.empty or excess_returns.shape[1] < 1:
        raise ValueError("need at least one strategy with one observation")
    if recentering not in ("c", "l", "u"):
        raise ValueError(f"recentering must be one of c/l/u, got {recentering!r}")

    rng = np.random.default_rng(seed)
    n, K = excess_returns.shape
    means = excess_returns.mean().to_numpy(dtype=float)
    stds = excess_returns.std(ddof=0).to_numpy(dtype=float)
    # Avoid division by zero for degenerate strategies (e.g. cash-only).
    stds_safe = np.where(stds > 0, stds, 1.0)
    t_stats = np.sqrt(n) * means / stds_safe
    observed = float(t_stats.max())

    # Studentized threshold from Hansen eq. (10).
    log_log_n = float(np.log(max(np.log(n), 1.0)))
    spa_threshold = -np.sqrt(2.0 * log_log_n)

    if recentering == "c":
        centered_means = np.where(t_stats > spa_threshold, means, 0.0)
    elif recentering == "l":
        # SPA-l: zero out all strategies (most powerful upper bound on H0).
        centered_means = np.zeros_like(means)
    else:  # "u"
        # SPA-u: center each at its own mean (matches Reality Check).
        centered_means = means.copy()

    R = excess_returns.to_numpy(dtype=float)
    boot_stats = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n, expected_block_length, rng)
        sample = R[idx]
        boot_means = sample.mean(axis=0) - centered_means
        boot_t = np.sqrt(n) * boot_means / stds_safe
        boot_stats[b] = float(boot_t.max())

    p_value = float((boot_stats >= observed).mean())
    best_idx = int(np.argmax(t_stats))
    return {
        "best_strategy": str(excess_returns.columns[best_idx]),
        "observed_t_stat": observed,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "recentering": recentering,
    }


__all__ = ["hansen_spa_test_studentized", "Recentering"]
