# src/assembled_core/qa/spa_test.py
"""Hansen Superior Predictive Ability (SPA) test wrapper (audit C2-022).

Wraps ``arch.bootstrap.SPA`` so the rest of the codebase has a single
contract for "does my candidate strategy beat the entire benchmark set
at the level of statistical significance, accounting for multiple
testing?".

Hansen (2005) corrects the well-known White (2000) Reality-Check bias
that downward-biases the test statistic when many strategies are
compared. The Hansen-SPA p-value is the audit's preferred number for
promotion-gate decisions involving a benchmark **set** (as opposed to a
single benchmark, where PSR / DSR suffice).

Usage::

    p_lower, p_consistent, p_upper = spa_p_values(
        candidate_returns, benchmark_returns_matrix
    )
    # p_consistent < 0.01 is the audit's "promote" threshold.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def spa_p_values(
    candidate_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.DataFrame | np.ndarray,
    *,
    block_size: int | None = None,
    reps: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Hansen-SPA lower / consistent / upper p-values.

    Args:
        candidate_returns: 1-D returns of the strategy under test.
        benchmark_returns: 2-D array (T x M) of M benchmark strategies'
            returns aligned with the candidate. The SPA test asks
            whether the candidate beats the best benchmark.
        block_size: stationary-bootstrap mean block length. Defaults to
            ceil(T^(1/3)).
        reps: bootstrap iterations.
        seed: RNG seed for arch.

    Returns:
        Dict with ``p_lower``, ``p_consistent``, ``p_upper`` and ``n_obs``,
        ``n_benchmarks``. The audit treats ``p_consistent`` as the
        reportable number; ``p_lower`` / ``p_upper`` bracket the
        nuisance-parameter uncertainty.

        Returns NaN p-values if ``arch`` is not importable, so callers
        can degrade gracefully.
    """
    cand = pd.Series(candidate_returns).dropna().to_numpy(dtype=float)
    if isinstance(benchmark_returns, pd.DataFrame):
        bench_df = benchmark_returns.dropna()
        bench_arr = bench_df.to_numpy(dtype=float)
    else:
        bench_arr = np.asarray(benchmark_returns, dtype=float)
        if bench_arr.ndim == 1:
            bench_arr = bench_arr.reshape(-1, 1)
    # Align lengths conservatively — use the trailing T common observations.
    T = min(len(cand), len(bench_arr))
    if T < 30:
        return {
            "p_lower": float("nan"),
            "p_consistent": float("nan"),
            "p_upper": float("nan"),
            "n_obs": int(T),
            "n_benchmarks": int(bench_arr.shape[1]),
            "error": "too few common observations (need >= 30)",
        }
    cand = cand[-T:]
    bench_arr = bench_arr[-T:]

    try:
        from arch.bootstrap import SPA  # type: ignore
    except ImportError:
        logger.warning("[spa_test] arch not installed — returning NaN p-values")
        return {
            "p_lower": float("nan"),
            "p_consistent": float("nan"),
            "p_upper": float("nan"),
            "n_obs": int(T),
            "n_benchmarks": int(bench_arr.shape[1]),
            "error": "arch not installed",
        }

    if block_size is None:
        block_size = max(2, int(np.ceil(T ** (1.0 / 3.0))))

    spa = SPA(cand, bench_arr, reps=reps, block_size=block_size, seed=seed)
    spa.compute()
    return {
        "p_lower": float(spa.pvalues["lower"]),
        "p_consistent": float(spa.pvalues["consistent"]),
        "p_upper": float(spa.pvalues["upper"]),
        "n_obs": int(T),
        "n_benchmarks": int(bench_arr.shape[1]),
        "block_size": int(block_size),
    }


__all__ = ["spa_p_values"]
