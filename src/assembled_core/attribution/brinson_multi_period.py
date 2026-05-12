"""Multi-period Brinson attribution via Cariño linking (audit C4-077).

The existing :class:`brinson_hood.BrinsonAttribution` sums per-period
allocation / selection / interaction contributions. That is **wrong**
for multi-period analysis because returns compound multiplicatively:
the sum of single-period active returns does not equal the
compounded multi-period active return. The discrepancy is called the
*linking* or *residual* problem.

Cariño (1999) introduced a logarithmic linking coefficient that
ensures the linked contributions exactly reconcile to the geometric
active return:

.. math::

    k_t = \\frac{\\frac{\\ln(1+r_t) - \\ln(1+b_t)}{r_t - b_t}}
              {\\frac{\\ln(1+R_p) - \\ln(1+R_b)}{R_p - R_b}}

with :math:`r_t / b_t` per-period portfolio / benchmark returns and
:math:`R_p / R_b` the compounded multi-period returns. The linked
contribution for period ``t`` is :math:`k_t` times the single-period
contribution; summing over ``t`` reproduces the full multi-period
attribution decomposition with zero residual.

When :math:`r_t \\approx b_t` the formula is indeterminate; we use the
analytic L'Hôpital limit
:math:`k_t = (1/(1+r_t)) / ((R_p - R_b) / (\\ln(1+R_p) - \\ln(1+R_b)))`
to stay numerically well-behaved.

References
----------
- Cariño, D. R. (1999). *Combining Attribution Effects Over Time*.
  Journal of Performance Measurement 3(4).
- Frongello, A. (2002). *Linking Single Period Attribution Results*.
  Journal of Performance Measurement 6(3) — cumulative-wealth variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_LINK_EPS = 1e-12


def _safe_log1p_ratio(r: np.ndarray) -> np.ndarray:
    """Compute ``log1p(r) / r`` element-wise, returning 1.0 when ``r → 0``.

    This is the natural-limit value (L'Hôpital) and matches the
    behavior required by the Cariño formula's denominator.
    """
    out = np.ones_like(r, dtype=float)
    mask = np.abs(r) > _LINK_EPS
    out[mask] = np.log1p(r[mask]) / r[mask]
    return out


def carino_link_coefficients(
    port_period_returns: pd.Series,
    bench_period_returns: pd.Series,
) -> pd.Series:
    """Return the per-period Cariño linking coefficients k_t."""
    r = port_period_returns.to_numpy(dtype=float)
    b = bench_period_returns.to_numpy(dtype=float)
    if r.shape != b.shape:
        raise ValueError("port and bench return series must have the same length")

    # Multi-period compounded returns.
    R_p = float(np.prod(1.0 + r) - 1.0)
    R_b = float(np.prod(1.0 + b) - 1.0)

    # Denominator: ((ln(1+R_p) - ln(1+R_b)) / (R_p - R_b)).
    if abs(R_p - R_b) <= _LINK_EPS:
        # Degenerate case: portfolio matches benchmark exactly.
        denom = 1.0 / (1.0 + R_p)
    else:
        denom = (np.log1p(R_p) - np.log1p(R_b)) / (R_p - R_b)

    # Numerator per period.
    diff = r - b
    log_r = np.log1p(r)
    log_b = np.log1p(b)
    numer = np.empty_like(r, dtype=float)
    safe = np.abs(diff) > _LINK_EPS
    numer[safe] = (log_r[safe] - log_b[safe]) / diff[safe]
    # L'Hôpital limit at r==b: derivative of log1p(x) at x=r is 1/(1+r).
    numer[~safe] = 1.0 / (1.0 + r[~safe])

    k = numer / denom
    return pd.Series(k, index=port_period_returns.index, name="carino_k")


def link_multi_period_attribution(
    single_period_attribution: pd.DataFrame,
    port_period_returns: pd.Series,
    bench_period_returns: pd.Series,
) -> pd.DataFrame:
    """Multiply each per-period attribution column by the Cariño coefficient.

    Args:
        single_period_attribution: DataFrame indexed by period with
            columns ``allocation``, ``selection``, ``interaction`` (the
            output of :meth:`BrinsonAttribution.attribute`).
        port_period_returns: per-period portfolio returns aligned with
            the attribution DataFrame.
        bench_period_returns: per-period benchmark returns aligned with
            the attribution DataFrame.

    Returns:
        A new DataFrame with the same columns, each scaled by the
        per-period Cariño coefficient. The column sums now reconcile
        exactly to the multi-period compounded active return.
    """
    if not isinstance(single_period_attribution.index, pd.Index):
        raise TypeError("single_period_attribution must be a DataFrame")
    k = carino_link_coefficients(port_period_returns, bench_period_returns)
    linked = single_period_attribution.mul(k, axis=0)
    return linked


def reconciliation_residual(
    linked_attribution: pd.DataFrame,
    port_period_returns: pd.Series,
    bench_period_returns: pd.Series,
    *,
    column: str = "active_total",
) -> float:
    """Diagnostic: sum-of-linked minus geometric active return.

    Should be near zero (within ~1e-10) for any well-formed input.
    Larger residual signals input mis-alignment or numerical issue.
    """
    R_p = float(np.prod(1.0 + port_period_returns.to_numpy(dtype=float)) - 1.0)
    R_b = float(np.prod(1.0 + bench_period_returns.to_numpy(dtype=float)) - 1.0)
    target_active = R_p - R_b
    summed = float(linked_attribution[column].sum())
    return summed - target_active


__all__ = [
    "carino_link_coefficients",
    "link_multi_period_attribution",
    "reconciliation_residual",
]
