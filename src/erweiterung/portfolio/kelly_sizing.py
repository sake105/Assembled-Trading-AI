"""Kelly-Criterion Sizing — Fractional Kelly mit Confidence-Discount.

Theorie
-------
Kelly-Optimization (1956) maximiert log-Wachstumsrate:
    f* = μ / σ²  (für continuous returns)

Probleme von "Full-Kelly":
- Bei Schätzfehler in μ extrem volatil.
- 50% Drawdown erwartet bei Full-Kelly!

**Fractional Kelly** (typisch f = 0.25 oder 0.5 of Kelly) ist Branchenstandard.
Wir koppeln die Fraction an eine Konfidenz-Schätzung (z. B. Conformal-Width).

Multi-Asset-Kelly
-----------------
f* = Σ⁻¹ μ (vector form). Bei Estimation-Errors verstärkt sich Instabilität.
Wir wenden zusätzlich Cap pro Asset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fractional_kelly_single(
    expected_return: float,
    variance: float,
    fraction: float = 0.25,
    max_size: float = 1.0,
) -> float:
    """Single-Asset fractional-Kelly: ``f * (μ / σ²)``."""
    if variance <= 0:
        return 0.0
    f_full = expected_return / variance
    return float(np.clip(fraction * f_full, -max_size, max_size))


def confidence_discounted_kelly(
    expected_return: float,
    variance: float,
    confidence: float,
    base_fraction: float = 0.5,
    max_size: float = 1.0,
) -> float:
    """Kelly-Fraction skaliert mit Konfidenz [0, 1].

    Args:
        confidence: 1.0 = volle Sicherheit, 0.0 = keine Sicherheit.
        base_fraction: Basis-Kelly-Fraction bei voller Konfidenz.

    f = base_fraction * confidence * (μ / σ²)
    """
    if variance <= 0 or confidence <= 0:
        return 0.0
    f_kelly = expected_return / variance
    f = base_fraction * confidence * f_kelly
    return float(np.clip(f, -max_size, max_size))


def multi_asset_kelly(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    fraction: float = 0.25,
    max_per_asset: float = 0.20,
    long_only: bool = False,
) -> pd.Series:
    """Multi-Asset Kelly: ``f* = fraction * Σ⁻¹ μ`` mit Caps.

    Args:
        expected_returns: pro Asset.
        cov: Cov-Matrix.
        fraction: globale Kelly-Fraction.
        max_per_asset: Asset-Cap.
        long_only: Wenn True, Negative -> 0.

    Returns:
        Series of weights (summe ≤ 1; Rest = Cash).
    """
    cov_inv = np.linalg.pinv(cov.values)
    f = fraction * (cov_inv @ expected_returns.values)
    f = np.clip(f, -max_per_asset, max_per_asset)
    if long_only:
        f = np.clip(f, 0, max_per_asset)
    # Hard cap on total leverage
    total = np.abs(f).sum()
    if total > 1.0:
        f = f / total
    return pd.Series(f, index=expected_returns.index)


__all__ = [
    "fractional_kelly_single",
    "confidence_discounted_kelly",
    "multi_asset_kelly",
]
