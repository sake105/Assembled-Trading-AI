"""Structural Cycles Overlay (Plan 4.10).

Long-term macro-cycle scoring based on Dalio Big Cycle / Fourth Turning framework:
- Debt/GDP trend
- Wealth inequality proxy
- Institutional trust proxy
- Great power rivalry intensity

Output is a risk multiplier, not a trading signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StructuralCycleScore:
    """Composite structural risk score."""
    debt_gdp_score: float  # 0-1 (higher = more debt)
    inequality_score: float  # 0-1
    trust_score: float  # 0-1 (higher = less trust)
    rivalry_score: float  # 0-1
    composite: float  # weighted average
    risk_multiplier: float  # 1.0 = normal, >1 = more conservative


def compute_structural_cycle_score(
    debt_gdp_pct: float = 120.0,
    gini_index: float = 0.40,
    trust_index: float = 0.30,
    rivalry_index: float = 0.60,
    weights: tuple[float, float, float, float] = (0.30, 0.20, 0.20, 0.30),
) -> StructuralCycleScore:
    """Compute structural cycle risk score.

    Args:
        debt_gdp_pct: Government debt as % of GDP.
        gini_index: Gini coefficient (0-1).
        trust_index: Institutional trust (0=no trust, 1=full trust).
        rivalry_index: Great power rivalry intensity (0-1).
        weights: Relative weights for components.

    Returns:
        StructuralCycleScore with risk multiplier.
    """
    # Normalize to 0-1 scores
    debt_score = min(1.0, max(0.0, (debt_gdp_pct - 60) / 140))  # 60%=0, 200%=1
    ineq_score = min(1.0, max(0.0, (gini_index - 0.25) / 0.35))  # 0.25=0, 0.60=1
    trust_score = 1.0 - min(1.0, max(0.0, trust_index))  # invert: low trust = high risk
    rivalry_score = min(1.0, max(0.0, rivalry_index))

    w = weights
    composite = (
        w[0] * debt_score + w[1] * ineq_score
        + w[2] * trust_score + w[3] * rivalry_score
    )
    composite = round(composite, 4)

    # Risk multiplier: 1.0 at composite=0.3 (normal), up to 1.5 at composite=0.8
    if composite <= 0.3:
        multiplier = 1.0
    elif composite >= 0.8:
        multiplier = 1.5
    else:
        multiplier = 1.0 + (composite - 0.3) / 0.5 * 0.5

    return StructuralCycleScore(
        debt_gdp_score=round(debt_score, 4),
        inequality_score=round(ineq_score, 4),
        trust_score=round(trust_score, 4),
        rivalry_score=round(rivalry_score, 4),
        composite=composite,
        risk_multiplier=round(multiplier, 4),
    )


__all__ = ["StructuralCycleScore", "compute_structural_cycle_score"]
