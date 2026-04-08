"""Crowded-trade and factor-concentration detection (V20).

Monitors:
- Factor concentration: how concentrated is the portfolio on single factors?
- Momentum crowding: warns when portfolio is heavily tilted to momentum
- Provides automatic weight reduction when crowding score exceeds threshold

Reference: Quant Quake 2007 — crowded factor positions unwound simultaneously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class CrowdingResult:
    """Result of crowding detection analysis."""

    factor_concentration: dict[str, float]
    # Factor name -> concentration score (0-1, higher = more concentrated)
    hhi_score: float  # Herfindahl-Hirschman Index of factor exposures
    dominant_factor: str | None
    dominant_factor_share: float  # Fraction of portfolio explained by dominant factor
    momentum_crowding_score: float  # 0-1 specific momentum crowding
    is_crowded: bool
    recommended_reduction: dict[str, float]  # Symbol -> reduction fraction
    warning_message: str | None


def compute_factor_concentration(
    weights: dict[str, float],
    factor_exposures: pd.DataFrame,
    factor_cols: list[str] | None = None,
) -> dict[str, float]:
    """Compute per-factor concentration of the portfolio.

    For each factor, concentration = |sum(w_i * exposure_i)| / sum(|w_i| * |exposure_i|).
    Values near 1.0 mean the portfolio is heavily tilted in one direction on that factor.

    Args:
        weights: Symbol -> weight.
        factor_exposures: DataFrame with symbol index and factor columns.
        factor_cols: Factor columns to analyze. If None, uses all numeric columns.

    Returns:
        Factor name -> concentration score (0-1).
    """
    if factor_exposures.empty or not weights:
        return {}

    if factor_cols is None:
        factor_cols = [c for c in factor_exposures.columns if factor_exposures[c].dtype in ("float64", "float32", "int64")]

    symbols = list(weights.keys())
    w = np.array([weights.get(s, 0.0) for s in symbols])

    concentrations = {}
    for fc in factor_cols:
        if fc not in factor_exposures.columns:
            continue
        exp = np.array([
            float(factor_exposures.loc[s, fc]) if s in factor_exposures.index else 0.0
            for s in symbols
        ])
        weighted_exp = w * exp
        numerator = abs(weighted_exp.sum())
        denominator = np.sum(np.abs(w) * np.abs(exp))
        if denominator > 1e-10:
            concentrations[fc] = round(float(numerator / denominator), 4)
        else:
            concentrations[fc] = 0.0

    return concentrations


def compute_hhi(weights: dict[str, float]) -> float:
    """Compute Herfindahl-Hirschman Index of portfolio weights.

    HHI = sum(w_i^2) where w_i are normalized weights.
    Range: 1/N (perfectly diversified) to 1.0 (single position).
    """
    if not weights:
        return 0.0
    w = np.array(list(weights.values()))
    total = np.abs(w).sum()
    if total < 1e-10:
        return 0.0
    w_norm = w / total
    return float(np.sum(w_norm ** 2))


def detect_crowding(
    weights: dict[str, float],
    factor_exposures: pd.DataFrame | None = None,
    factor_cols: list[str] | None = None,
    momentum_col: str | None = None,
    crowding_threshold: float = 0.70,
    hhi_threshold: float = 0.15,
) -> CrowdingResult:
    """Run full crowding detection analysis.

    Args:
        weights: Symbol -> portfolio weight.
        factor_exposures: Factor exposure matrix (symbol as index).
        factor_cols: Factor columns to analyze.
        momentum_col: Column name for momentum exposure (for specific check).
        crowding_threshold: Factor concentration above this triggers warning.
        hhi_threshold: HHI above this triggers warning.

    Returns:
        CrowdingResult with analysis and recommendations.
    """
    # Factor concentration
    concentrations: dict[str, float] = {}
    if factor_exposures is not None and not factor_exposures.empty:
        concentrations = compute_factor_concentration(weights, factor_exposures, factor_cols)

    # HHI
    hhi = compute_hhi(weights)

    # Dominant factor
    dominant = None
    dominant_share = 0.0
    if concentrations:
        dominant = max(concentrations, key=concentrations.get)
        dominant_share = concentrations[dominant]

    # Momentum-specific crowding
    mom_score = 0.0
    if momentum_col and momentum_col in concentrations:
        mom_score = concentrations[momentum_col]

    # Crowding check
    is_crowded = (
        dominant_share > crowding_threshold
        or hhi > hhi_threshold
        or mom_score > crowding_threshold
    )

    # Recommend reductions for crowded positions
    reductions: dict[str, float] = {}
    warning = None

    if is_crowded and factor_exposures is not None and dominant:
        warning_parts = []
        if dominant_share > crowding_threshold:
            warning_parts.append(f"factor '{dominant}' concentration={dominant_share:.2f}")
        if hhi > hhi_threshold:
            warning_parts.append(f"HHI={hhi:.3f}")
        if mom_score > crowding_threshold:
            warning_parts.append(f"momentum crowding={mom_score:.2f}")
        warning = "CROWDING WARNING: " + ", ".join(warning_parts)

        # Reduce positions most exposed to the dominant factor
        if dominant in factor_exposures.columns:
            for sym in weights:
                if sym in factor_exposures.index:
                    exp = abs(float(factor_exposures.loc[sym, dominant]))
                    if exp > 0.5:  # High exposure to crowded factor
                        reductions[sym] = min(0.50, exp * 0.3)  # Up to 50% reduction

    if warning:
        _log.warning(warning)

    return CrowdingResult(
        factor_concentration=concentrations,
        hhi_score=round(hhi, 4),
        dominant_factor=dominant,
        dominant_factor_share=dominant_share,
        momentum_crowding_score=round(mom_score, 4),
        is_crowded=is_crowded,
        recommended_reduction=reductions,
        warning_message=warning,
    )


def apply_crowding_reduction(
    target_weights: dict[str, float],
    crowding_result: CrowdingResult,
) -> dict[str, float]:
    """Apply crowding-based weight reductions.

    Args:
        target_weights: Symbol -> target weight.
        crowding_result: Result from detect_crowding().

    Returns:
        Adjusted weights with crowding reductions applied.
    """
    if not crowding_result.is_crowded:
        return dict(target_weights)

    adjusted = dict(target_weights)
    for sym, reduction in crowding_result.recommended_reduction.items():
        if sym in adjusted:
            adjusted[sym] *= (1.0 - reduction)

    return adjusted


__all__ = [
    "CrowdingResult",
    "compute_factor_concentration",
    "compute_hhi",
    "detect_crowding",
    "apply_crowding_reduction",
]
