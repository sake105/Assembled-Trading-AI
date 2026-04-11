"""Empirical tail-dependence diagnostic (C8 sidecar — pre-wiring).

This module is the *diagnostic half* of C8 (Copula Tail-Dependence).  It
computes empirical lower-tail dependence coefficients for every pair of
return series and summarises them into a single portfolio-level score plus
a qualitative regime classification.

Scope and boundaries:
  - Sidecar only. Pure pandas/numpy. No scipy dependency.
  - Does NOT touch ``risk/correlation_guard.py``. The integration of this
    diagnostic into the correlation guard (tighter cluster limits when the
    tail regime is "high") is a deliberate follow-up commit.
  - Complementary to ``ml/copula_models.py`` (which fits parametric
    copulas via scipy MLE). This module uses the direct empirical
    estimator ``P(U_j < alpha | U_i < alpha)`` so it works without
    optional dependencies.

Formula (lower-tail dependence, empirical):

    lambda_L(i, j) = P(U_j < alpha | U_i < alpha)
                   = # { u_i < alpha AND u_j < alpha } / # { u_i < alpha }

where ``u_i`` are the empirical CDF ranks of column ``i`` in ``(0, 1]``
obtained via ``rank(pct=True)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_ROWS = 30


def compute_empirical_tail_dependence(
    returns_panel: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute pairwise empirical lower-tail dependence coefficients.

    Args:
        returns_panel: Wide-format returns (rows = dates, columns = symbols).
        alpha: Tail probability threshold. Must lie in the open interval
            ``(0, 0.5)``.

    Returns:
        Square ``pd.DataFrame`` indexed by symbol with diagonals set to
        ``1.0``. ``result.loc[i, j]`` is the empirical estimate of
        ``P(U_j < alpha | U_i < alpha)``.

    Raises:
        ValueError: If fewer than 30 rows, fewer than 2 symbols, or
            ``alpha`` is not strictly between 0 and 0.5.
    """
    if not isinstance(returns_panel, pd.DataFrame):
        raise ValueError("returns_panel must be a pandas DataFrame")
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    if returns_panel.shape[1] < 2:
        raise ValueError(
            f"need at least 2 symbols, got {returns_panel.shape[1]}"
        )
    if returns_panel.shape[0] < _MIN_ROWS:
        raise ValueError(
            f"need at least {_MIN_ROWS} rows, got {returns_panel.shape[0]}"
        )

    symbols = list(returns_panel.columns)
    # Empirical CDF ranks in (0, 1].  rank(pct=True) yields rank/n so the
    # smallest observation gets 1/n and the largest gets 1.0.
    u = returns_panel.rank(pct=True, method="average")

    # Boolean mask of tail exceedances per column.
    tail = (u < alpha).to_numpy()
    n_syms = len(symbols)
    out = np.zeros((n_syms, n_syms), dtype=float)

    for i in range(n_syms):
        tail_i = tail[:, i]
        denom = tail_i.sum()
        for j in range(n_syms):
            if i == j:
                out[i, j] = 1.0
                continue
            if denom == 0:
                # No tail exceedances for column i — dependence undefined,
                # report 0.0 as a conservative "no evidence" value.
                out[i, j] = 0.0
                continue
            joint = np.logical_and(tail_i, tail[:, j]).sum()
            out[i, j] = float(joint) / float(denom)

    return pd.DataFrame(out, index=symbols, columns=symbols)


def compute_portfolio_tail_dependence_score(
    tail_dep_df: pd.DataFrame,
) -> float:
    """Average off-diagonal lower-tail dependence.

    Args:
        tail_dep_df: Output of :func:`compute_empirical_tail_dependence`.

    Returns:
        Mean of all off-diagonal entries. Range ``[0, 1]``. Higher values
        indicate more synchronised crashes across the panel.
    """
    if tail_dep_df.shape[0] != tail_dep_df.shape[1]:
        raise ValueError("tail_dep_df must be square")
    n = tail_dep_df.shape[0]
    if n < 2:
        return 0.0
    arr = tail_dep_df.to_numpy()
    mask = ~np.eye(n, dtype=bool)
    off_diag = arr[mask]
    return float(np.mean(off_diag))


def classify_tail_regime(score: float) -> str:
    """Map a portfolio tail-dependence score to a qualitative regime.

    Thresholds:
      - ``score < 0.15``: ``"low"`` (diversified tails)
      - ``0.15 <= score < 0.35``: ``"medium"``
      - ``score >= 0.35``: ``"high"`` (tail-synchronised; a downstream
        correlation guard should tighten cluster limits)
    """
    if score < 0.15:
        return "low"
    if score < 0.35:
        return "medium"
    return "high"


__all__ = [
    "compute_empirical_tail_dependence",
    "compute_portfolio_tail_dependence_score",
    "classify_tail_regime",
]
