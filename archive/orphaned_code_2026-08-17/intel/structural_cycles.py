"""Structural cycle risk scoring for long-horizon geopolitical analysis.

Estimates systemic fragility based on debt burden, inequality,
institutional trust, and great-power rivalry indices.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StructuralCycleResult:
    composite: float
    risk_multiplier: float
    debt_score: float
    inequality_score: float
    trust_score: float
    rivalry_score: float


def compute_structural_cycle_score(
    *,
    debt_gdp_pct: float,
    gini_index: float,
    trust_index: float,
    rivalry_index: float,
) -> StructuralCycleResult:
    """Compute a composite structural cycle risk score.

    Args:
        debt_gdp_pct: Government debt as % of GDP (e.g. 80 → 80%).
        gini_index: Income inequality 0–1 (higher = more unequal).
        trust_index: Institutional trust 0–1 (higher = more trust).
        rivalry_index: Great-power rivalry intensity 0–1.

    Returns:
        StructuralCycleResult with composite score and risk_multiplier >= 1.0.
    """
    debt_score = min(debt_gdp_pct / 100.0, 2.0)  # normalise; cap at 2x
    inequality_score = gini_index  # 0–1
    trust_score = 1.0 - trust_index  # invert (low trust = high risk)
    rivalry_score = rivalry_index  # 0–1

    composite = (
        0.35 * debt_score
        + 0.25 * inequality_score
        + 0.25 * trust_score
        + 0.15 * rivalry_score
    )
    risk_multiplier = max(1.0, 1.0 + composite)

    return StructuralCycleResult(
        composite=round(composite, 6),
        risk_multiplier=round(risk_multiplier, 6),
        debt_score=round(debt_score, 6),
        inequality_score=round(inequality_score, 6),
        trust_score=round(trust_score, 6),
        rivalry_score=round(rivalry_score, 6),
    )
