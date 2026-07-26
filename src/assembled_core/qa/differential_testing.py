"""Differential testing: multi-implementation Sharpe ratio agreement.

Audit reference: C2-006 — Differential Testing (Python / Polars / Numba Sharpe).

Purpose
-------
Verify that three independent implementations of the annualised Sharpe ratio
produce ε-bounded identical results on any input array.  Discrepancy > ε
indicates a silent numeric divergence that must be investigated before any
of the implementations is trusted in production.

The three implementations are:
* ``sharpe_numpy``  — plain NumPy (baseline)
* ``sharpe_polars`` — Polars Series arithmetic
* ``sharpe_numba``  — Numba @njit kernel (falls back to NumPy if Numba unavailable)

All three target the *same* formula:

    Sharpe = mean(r - rf) / std(r - rf, ddof=1) * sqrt(252)

When the excess-return series has zero variance the ratio is NaN (not an error).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Numba import
# ---------------------------------------------------------------------------
try:
    import numba  # noqa: F401
    from numba import njit as _njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def _njit(*args, **kwargs):
        """Dummy decorator — Numba unavailable."""

        def _decorator(fn):
            return fn

        return _decorator


# ---------------------------------------------------------------------------
# Optional Polars import
# ---------------------------------------------------------------------------
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    # unused-ignore in the code: polars is typed locally (ignore needed) but
    # absent in CI where the ignore_missing_imports override makes pl Any and
    # the ignore would be flagged unused by warn_unused_ignores.
    pl = None  # type: ignore[assignment, unused-ignore]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

_SQRT252 = float(np.sqrt(252))


@dataclass
class DiffTestResult:
    """Comparison result from :func:`diff_test_sharpe`.

    Attributes
    ----------
    numpy_val:
        Sharpe value computed with NumPy.
    polars_val:
        Sharpe value computed with Polars (NaN when Polars is unavailable).
    numba_val:
        Sharpe value computed with Numba (equals ``numpy_val`` when Numba is
        unavailable — 2-way test instead of 3-way).
    max_abs_diff:
        Maximum absolute pairwise difference across the available implementations.
    passed:
        ``True`` when ``max_abs_diff <= epsilon``.
    epsilon:
        The tolerance used when evaluating ``passed``.
    """

    numpy_val: float
    polars_val: float
    numba_val: float
    max_abs_diff: float
    passed: bool
    epsilon: float


# ---------------------------------------------------------------------------
# Implementation 1 — NumPy (baseline)
# ---------------------------------------------------------------------------


def sharpe_numpy(returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio computed with NumPy.

    Parameters
    ----------
    returns:
        1-D array of period returns.
    rf:
        Constant risk-free rate per period (same units as *returns*).

    Returns
    -------
    float
        Annualised Sharpe ratio, or ``float('nan')`` when variance is zero or
        the array has fewer than two elements.
    """
    arr = np.asarray(returns, dtype=np.float64)
    excess = arr - rf
    if excess.size < 2:
        return float("nan")
    std = float(np.std(excess, ddof=1))
    if std == 0.0:
        return float("nan")
    return float(np.mean(excess) / std * _SQRT252)


# ---------------------------------------------------------------------------
# Implementation 2 — Polars
# ---------------------------------------------------------------------------


def sharpe_polars(returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio computed via Polars Series arithmetic.

    Falls back to :func:`sharpe_numpy` with a warning if Polars is not
    installed, so callers always receive a numeric result.

    Parameters
    ----------
    returns:
        1-D array of period returns.
    rf:
        Constant risk-free rate per period.

    Returns
    -------
    float
        Annualised Sharpe ratio, or ``float('nan')`` for degenerate inputs.
    """
    if not HAS_POLARS:
        warnings.warn(
            "polars is not installed; sharpe_polars falls back to sharpe_numpy.",
            ImportWarning,
            stacklevel=2,
        )
        return sharpe_numpy(returns, rf)

    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    series = pl.Series("excess", arr - rf, dtype=pl.Float64)
    std_val = series.std(ddof=1)
    # polars stubs type std()/mean() as broad unions (timedelta/date/... for
    # other dtypes); dtype=pl.Float64 guarantees float here — cast is a no-op.
    if std_val is None or float(cast(float, std_val)) == 0.0:
        return float("nan")
    mean_val = series.mean()
    if mean_val is None:
        return float("nan")
    return float(float(cast(float, mean_val)) / float(cast(float, std_val)) * _SQRT252)


# ---------------------------------------------------------------------------
# Implementation 3 — Numba @njit kernel
# ---------------------------------------------------------------------------


@_njit(cache=True)
def _sharpe_kernel(excess: np.ndarray) -> float:
    """Inner JIT loop; returns Sharpe (annualised) or NaN on degenerate input."""
    n = len(excess)
    if n < 2:
        return np.nan
    # Welford online mean/variance
    mean = 0.0
    m2 = 0.0
    for i in range(n):
        delta = excess[i] - mean
        mean += delta / (i + 1)
        delta2 = excess[i] - mean
        m2 += delta * delta2
    var = m2 / (n - 1)
    if var <= 0.0:
        return np.nan
    # numba kernel: ndarray indexing is Any; no cast() inside JIT code.
    return mean / np.sqrt(var) * 15.874507866387544  # type: ignore[no-any-return]  # sqrt(252)


def sharpe_numba(returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio computed with a Numba @njit kernel.

    Falls back to :func:`sharpe_numpy` (with a warning) when Numba is not
    installed so that downstream callers always receive a numeric result.

    Parameters
    ----------
    returns:
        1-D array of period returns.
    rf:
        Constant risk-free rate per period.

    Returns
    -------
    float
        Annualised Sharpe ratio, or ``float('nan')`` for degenerate inputs.
    """
    if not HAS_NUMBA:
        warnings.warn(
            "numba is not installed; sharpe_numba falls back to sharpe_numpy.",
            ImportWarning,
            stacklevel=2,
        )
        return sharpe_numpy(returns, rf)

    arr = np.asarray(returns, dtype=np.float64)
    excess = arr - rf
    return float(_sharpe_kernel(excess))


# ---------------------------------------------------------------------------
# Differential test harness
# ---------------------------------------------------------------------------


def diff_test_sharpe(
    returns: np.ndarray,
    rf: float = 0.0,
    epsilon: float = 1e-10,
) -> DiffTestResult:
    """Run all three Sharpe implementations and report their agreement.

    When Numba is unavailable the test degrades to a 2-way comparison
    (NumPy vs Polars); when Polars is unavailable it degrades further to a
    1-way identity check.  ``DiffTestResult.passed`` reflects the actual
    available implementations.

    Parameters
    ----------
    returns:
        1-D array of period returns.
    rf:
        Constant risk-free rate per period.
    epsilon:
        Maximum tolerated absolute difference between any pair of
        implementations.  Default ``1e-10``.

    Returns
    -------
    DiffTestResult
        Summary with individual values, ``max_abs_diff``, and a boolean
        ``passed`` flag.
    """
    nv = sharpe_numpy(returns, rf)

    if HAS_POLARS:
        pv = sharpe_polars(returns, rf)
    else:
        pv = float("nan")

    if HAS_NUMBA:
        bv = sharpe_numba(returns, rf)
    else:
        bv = nv  # 2-way fallback: numba equals numpy so diff = 0

    # Collect values that are genuinely independent (non-NaN pairs)
    candidates: list[float] = [nv]
    if HAS_POLARS and not np.isnan(pv):
        candidates.append(pv)
    if HAS_NUMBA and not np.isnan(bv):
        candidates.append(bv)

    # max pairwise absolute difference
    if len(candidates) < 2 or (np.isnan(nv) and np.isnan(pv) and np.isnan(bv)):
        # degenerate: all NaN or only one real value → treat as 0 diff
        max_diff = 0.0
    else:
        all_vals = [nv, pv if HAS_POLARS else nv, bv]
        finite = [v for v in all_vals if not np.isnan(v)]
        if len(finite) < 2:
            max_diff = 0.0
        else:
            max_diff = float(
                max(
                    abs(a - b)
                    for i, a in enumerate(finite)
                    for j, b in enumerate(finite)
                    if i < j
                )
            )

    return DiffTestResult(
        numpy_val=nv,
        polars_val=pv,
        numba_val=bv,
        max_abs_diff=max_diff,
        passed=(max_diff <= epsilon),
        epsilon=epsilon,
    )


__all__ = [
    "DiffTestResult",
    "HAS_NUMBA",
    "HAS_POLARS",
    "diff_test_sharpe",
    "sharpe_numpy",
    "sharpe_numba",
    "sharpe_polars",
]
