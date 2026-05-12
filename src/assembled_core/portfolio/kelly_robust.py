"""Robust-Kelly sizing (audit C2-065).

This module implements two well-known fixes to the textbook Kelly
fraction that address its single biggest practical failure mode —
plug-in Kelly with sample-estimated drift/volatility is **biased
upward** and routinely produces 100%+ leverage even when the true
edge is fragile.

The two adjustments here, used in combination:

1. **Browne-Whitt (1996) estimation-adjusted Kelly.**
   When :math:`\\mu, \\sigma^2` are estimated from T samples, the
   plug-in Kelly fraction :math:`f_{plug} = \\mu / \\sigma^2`
   over-states the optimal bet. Browne & Whitt derived the
   shrinkage factor :math:`T / (T + d)` (where :math:`d \\approx 2`
   for the one-parameter mean estimation case) which converges to
   1 as T grows but provides material protection at the typical
   T ≈ 250 trading-day horizons we use in research.

   See: Browne S., Whitt W. (1996), *Portfolio choice and the
   Bayesian Kelly criterion*, Adv. Appl. Probab. 28, p. 1145-76.

2. **Fractional Kelly (half-Kelly).**
   Multiply the result by 0.5 (configurable). This is the standard
   risk-of-ruin trade: half-Kelly retains ~75% of the long-run
   growth while halving the drawdown variance. Recommended by both
   Thorp (2006 reprint) and MacLean-Thorp-Ziemba (2010).

Combined formula:

.. math::

    f^* = \\min\\Big(
        \\frac{\\mu}{\\sigma^2}
        \\cdot \\frac{T}{T+d}
        \\cdot k_{frac}
        ,\\, f_{max}
    \\Big)

where :math:`k_{frac} \\in (0,1]` is the fractional-Kelly multiplier
(default 0.5) and :math:`f_{max}` is the hard upper bound from the
project's leverage policy (default 0.25).

If ``mu <= 0`` the result is 0 — no positive edge, no position. The
caller is responsible for handling short positions separately by
inverting sign on ``mu``.

This is a *pure-math* helper. It does not read or write state and
makes no assumption about how the upstream caller estimated
``mu``/``sigma2``/``T``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_FRACTIONAL_KELLY = 0.5
DEFAULT_MAX_FRACTION = 0.25
DEFAULT_BROWNE_WHITT_D = 2.0
MIN_VARIANCE = 1e-12


@dataclass(frozen=True)
class RobustKellyResult:
    """Output of :func:`robust_kelly_fraction`.

    All fields are dimensionless and useful for audit logs.
    """

    raw_kelly: float
    estimation_shrinkage: float
    fractional_kelly_multiplier: float
    capped_fraction: float
    binding_constraint: str  # one of: "max_fraction", "zero_edge", "none"


def robust_kelly_fraction(
    *,
    mu: float,
    sigma2: float,
    n_samples: int,
    fractional_kelly: float = DEFAULT_FRACTIONAL_KELLY,
    max_fraction: float = DEFAULT_MAX_FRACTION,
    browne_whitt_d: float = DEFAULT_BROWNE_WHITT_D,
) -> RobustKellyResult:
    """Compute a robust Kelly position fraction.

    Args:
        mu: estimated drift / expected return per period.
        sigma2: estimated variance per period; must be > 0.
        n_samples: number of independent samples used to estimate
            ``mu``; the Browne-Whitt shrinkage = T / (T + d).
        fractional_kelly: half-Kelly multiplier (default 0.5).
        max_fraction: hard upper bound on the result.
        browne_whitt_d: dimensionality term in the shrinkage factor
            (default 2 — matches the one-mean / one-variance case
            common in single-asset research).

    Returns:
        A :class:`RobustKellyResult` with the final ``capped_fraction``
        and full audit-trail of intermediate quantities.

    Raises:
        ValueError: if any input is malformed (negative variance,
            non-positive sample count, etc.).
    """
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be > 0, got {sigma2}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if not (0.0 < fractional_kelly <= 1.0):
        raise ValueError(f"fractional_kelly must be in (0, 1], got {fractional_kelly}")
    if max_fraction < 0.0:
        raise ValueError(f"max_fraction must be >= 0, got {max_fraction}")
    if browne_whitt_d < 0.0:
        raise ValueError(f"browne_whitt_d must be >= 0, got {browne_whitt_d}")

    sigma2_safe = max(sigma2, MIN_VARIANCE)
    raw = mu / sigma2_safe

    if mu <= 0.0:
        return RobustKellyResult(
            raw_kelly=raw,
            estimation_shrinkage=float("nan"),
            fractional_kelly_multiplier=fractional_kelly,
            capped_fraction=0.0,
            binding_constraint="zero_edge",
        )

    shrinkage = n_samples / (n_samples + browne_whitt_d)
    adjusted = raw * shrinkage * fractional_kelly

    if adjusted > max_fraction:
        return RobustKellyResult(
            raw_kelly=raw,
            estimation_shrinkage=shrinkage,
            fractional_kelly_multiplier=fractional_kelly,
            capped_fraction=max_fraction,
            binding_constraint="max_fraction",
        )

    return RobustKellyResult(
        raw_kelly=raw,
        estimation_shrinkage=shrinkage,
        fractional_kelly_multiplier=fractional_kelly,
        capped_fraction=float(adjusted),
        binding_constraint="none",
    )


def robust_kelly_from_returns(
    returns: np.ndarray,
    *,
    fractional_kelly: float = DEFAULT_FRACTIONAL_KELLY,
    max_fraction: float = DEFAULT_MAX_FRACTION,
) -> RobustKellyResult:
    """Convenience wrapper — compute robust Kelly from a return series.

    Uses sample-mean and sample-variance with Bessel correction.
    """
    arr = np.asarray(returns, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError("need at least 2 finite return observations")
    mu_hat = float(np.mean(arr))
    sigma2_hat = float(np.var(arr, ddof=1))
    return robust_kelly_fraction(
        mu=mu_hat,
        sigma2=sigma2_hat,
        n_samples=int(arr.size),
        fractional_kelly=fractional_kelly,
        max_fraction=max_fraction,
    )


__all__ = [
    "robust_kelly_fraction",
    "robust_kelly_from_returns",
    "RobustKellyResult",
    "DEFAULT_FRACTIONAL_KELLY",
    "DEFAULT_MAX_FRACTION",
    "DEFAULT_BROWNE_WHITT_D",
]
