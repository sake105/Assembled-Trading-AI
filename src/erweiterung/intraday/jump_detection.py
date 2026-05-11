"""Lee-Mykland Jump-Test (2008).

Theorie
-------
Standard-Asset-Pricing-Modelle nehmen Brownian-Path an — kontinuierliche
Trajektorie. **Jumps** (discontinuous moves) sind statistisch unterscheidbar:

Lee-Mykland-Test-Statistik:
    L_i = r_i / √(σ̂_i² × Δ)

mit σ̂_i² = bipower variation in trailing window (robust gegen jumps).

Unter Null (kein jump) sind L_i ~ Standard-Normal. Wir flaggen jumps wenn
|L_i| > critical value mit Bonferroni-Korrektur über n tests.

Reference
---------
Lee, S. & Mykland, P. (2008). Jumps in Financial Markets: A New Nonparametric
Test and Jump Dynamics. *RFS* 21.

Anwendung
---------
- Jump-Trades (mean-reversion oder continuation)
- Risk-Management (separat von continuous-volatility)
- Earnings-Drift-Validation
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class JumpDetection:
    jump_indices: np.ndarray  # positions where jumps detected
    test_statistics: np.ndarray  # L_i values
    critical_value: float
    n_jumps: int


def bipower_variation(returns: np.ndarray) -> float:
    """Bipower-Variation BV = (π/2) Σ |r_i| × |r_{i+1}|.

    Robust gegen Jumps — schätzt nur continuous-quadratic-variation.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    return float(np.pi / 2 * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))


def lee_mykland_test(
    returns: np.ndarray,
    window: int = 270,  # ~10 trading days at 5-min freq
    alpha: float = 0.01,
) -> JumpDetection:
    """Lee-Mykland Jump-Test.

    Args:
        returns: high-freq returns.
        window: trailing bipower-window for σ̂ estimation.
        alpha: significance-level (per-test, Bonferroni-applied).

    Returns:
        JumpDetection.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < window + 2:
        raise ValueError(f"need > {window} returns")

    # Local σ̂² via bipower-variation in trailing window
    sigma2 = np.zeros(n)
    for i in range(window, n):
        bv = bipower_variation(r[i - window : i])
        sigma2[i] = bv / (window - 1) if window > 1 else float("nan")

    # Standardized returns
    L = r / np.sqrt(np.maximum(sigma2, 1e-15))

    # Critical value: under Null, max |L_i| in n tests has Gumbel distribution.
    # Critical value approximation: c = √(2 ln n) − (ln(π) + ln(ln(n))) / (2 √(2 ln n))
    # at level α: x_α = c + (-ln(-ln(1-α))) / √(2 ln n)
    cn = math.sqrt(2 * math.log(n)) - (math.log(math.pi) + math.log(math.log(n))) / (
        2 * math.sqrt(2 * math.log(n))
    )
    x_alpha = cn + (-math.log(-math.log(1 - alpha))) / math.sqrt(2 * math.log(n))

    valid = np.arange(window, n)
    L_valid = L[valid]
    jump_idx_local = np.where(np.abs(L_valid) > x_alpha)[0]
    jump_indices = valid[jump_idx_local]

    return JumpDetection(
        jump_indices=jump_indices,
        test_statistics=L,
        critical_value=x_alpha,
        n_jumps=int(len(jump_indices)),
    )


def jump_intensity(
    returns: np.ndarray, window: int = 270, alpha: float = 0.01
) -> float:
    """Rate of jumps per observation (= n_jumps / n_obs)."""
    result = lee_mykland_test(returns, window=window, alpha=alpha)
    return result.n_jumps / max(len(returns) - window, 1)


def split_continuous_jump_variance(
    returns: np.ndarray, window: int = 270, alpha: float = 0.01
) -> dict:
    """Decompose Realized-Variance = Continuous-Variance + Jump-Variance.

    Returns:
        dict mit total_rv, continuous_rv (BV), jump_rv.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    total_rv = float(np.sum(r**2))
    bv = bipower_variation(r)
    jump_rv = max(total_rv - bv, 0.0)
    return {
        "total_rv": total_rv,
        "continuous_rv": bv,
        "jump_rv": jump_rv,
        "jump_share": jump_rv / total_rv if total_rv > 0 else 0.0,
    }


__all__ = [
    "JumpDetection",
    "bipower_variation",
    "lee_mykland_test",
    "jump_intensity",
    "split_continuous_jump_variance",
]
