"""Scenario Trees with weighted expected impact.

For each geopolitical trigger, models 3-4 probabilistic scenarios:

- **BASE** (P~0.50): Moderate escalation, typical market impact.
- **ESCALATION** (P~0.25): Significant escalation, strong impact.
- **DE_ESCALATION** (P~0.20): Tension relief, positive impact.
- **BLACK_SWAN** (P~0.05): Extreme escalation, catastrophic impact.

Key outputs:
- ``expected_impact``: Probability-weighted average impact.
- ``tail_impact``: Impact of BLACK_SWAN scenario (worst case).
- ``impact_skew``: Ratio of tail-to-expected impact (>3 → fat tail risk).

Pre-trade integration: if ``expected_impact × portfolio_exposure > risk_budget``
then reduce position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """A single scenario within a scenario tree."""

    name: str
    probability: float  # [0, 1]
    impact_pct: float  # expected market impact in percentage (negative = loss)
    duration_days: int  # expected duration of impact
    confidence: float = 0.5  # confidence in this scenario's probability [0, 1]
    description: str = ""


@dataclass
class ScenarioTree:
    """Collection of scenarios for a trigger event."""

    trigger_type: str
    trigger_description: str
    scenarios: list[Scenario] = field(default_factory=list)
    timestamp: str = ""

    @property
    def expected_impact(self) -> float:
        """Probability-weighted expected impact."""
        return sum(s.probability * s.impact_pct for s in self.scenarios)

    @property
    def tail_impact(self) -> float:
        """Worst-case scenario impact (most negative)."""
        if not self.scenarios:
            return 0.0
        return min(s.impact_pct for s in self.scenarios)

    @property
    def impact_skew(self) -> float:
        """Ratio of tail to expected impact.  >3 indicates fat-tail risk."""
        ei = self.expected_impact
        if abs(ei) < 1e-6:
            return 0.0
        return abs(self.tail_impact / ei)

    @property
    def is_valid(self) -> bool:
        """Check that probabilities sum to ~1."""
        total = sum(s.probability for s in self.scenarios)
        return abs(total - 1.0) < 0.05

    def confidence_weighted_impact(self) -> float:
        """Expected impact weighted by scenario confidence."""
        return sum(
            s.probability * s.impact_pct * s.confidence
            for s in self.scenarios
        )

    def to_dict(self) -> dict:
        return {
            "trigger_type": self.trigger_type,
            "trigger_description": self.trigger_description,
            "expected_impact_pct": round(self.expected_impact, 4),
            "tail_impact_pct": round(self.tail_impact, 4),
            "impact_skew": round(self.impact_skew, 2),
            "confidence_weighted_impact": round(self.confidence_weighted_impact(), 4),
            "n_scenarios": len(self.scenarios),
            "scenarios": [
                {
                    "name": s.name,
                    "probability": s.probability,
                    "impact_pct": s.impact_pct,
                    "duration_days": s.duration_days,
                    "confidence": s.confidence,
                }
                for s in self.scenarios
            ],
        }


# ── Default scenario templates ────────────────────────────────────────

def _make_default_scenarios(
    base_impact: float,
    escalation_multiplier: float = 2.5,
    deesc_fraction: float = 0.3,
    black_swan_multiplier: float = 5.0,
) -> list[Scenario]:
    """Generate standard 4-scenario template.

    Args:
        base_impact: Expected impact in BASE scenario (negative = loss).
        escalation_multiplier: How much worse ESCALATION is vs BASE.
        deesc_fraction: How much recovery DE_ESCALATION brings (fraction of base).
        black_swan_multiplier: How much worse BLACK_SWAN is vs BASE.

    Returns:
        List of 4 Scenarios with probabilities summing to 1.0.
    """
    return [
        Scenario("BASE", 0.50, base_impact, 10, 0.7, "Moderate escalation, typical response"),
        Scenario("ESCALATION", 0.25, base_impact * escalation_multiplier, 20, 0.5, "Significant escalation"),
        Scenario("DE_ESCALATION", 0.20, base_impact * deesc_fraction, 5, 0.6, "Tensions ease, partial recovery"),
        Scenario("BLACK_SWAN", 0.05, base_impact * black_swan_multiplier, 40, 0.3, "Extreme tail event"),
    ]


# Template library: trigger_type → (base_impact, esc_mult, deesc_frac, bs_mult)
TRIGGER_SCENARIO_TEMPLATES: dict[str, tuple[float, float, float, float]] = {
    "SANCTIONS_NEW": (-3.0, 2.5, 0.3, 6.0),
    "SANCTIONS_ESCALATION": (-4.0, 2.0, 0.25, 5.0),
    "MILITARY_BUILDUP": (-2.5, 3.0, 0.4, 8.0),
    "MILITARY_CONFLICT": (-6.0, 2.0, 0.2, 5.0),
    "TRADE_WAR": (-2.0, 2.5, 0.35, 5.0),
    "NUCLEAR_THREAT": (-5.0, 2.5, 0.2, 10.0),
    "BANKING_CRISIS": (-5.0, 2.0, 0.3, 6.0),
    "SOVEREIGN_DEFAULT": (-4.0, 2.5, 0.2, 7.0),
    "REGIME_CHANGE_RISK": (-3.0, 2.5, 0.3, 6.0),
    "ENERGY_DISRUPTION": (-3.5, 2.0, 0.35, 5.0),
    "SUPPLY_SHOCK": (-3.0, 2.5, 0.3, 6.0),
    "CREDIT_DOWNGRADE": (-2.0, 2.5, 0.3, 5.0),
    "RATE_SURPRISE": (-2.5, 2.0, 0.3, 4.0),
    "CYBER_ATTACK": (-2.0, 3.0, 0.4, 7.0),
    "PANDEMIC_ESCALATION": (-4.0, 2.0, 0.25, 6.0),
}


def build_scenario_tree(
    trigger_type: str,
    description: str = "",
    *,
    escalation_probability: float | None = None,
    custom_base_impact: float | None = None,
) -> ScenarioTree:
    """Build a scenario tree for a given trigger type.

    Uses template library for known triggers, falls back to moderate
    defaults for unknown types.

    Args:
        trigger_type: Type of geopolitical trigger.
        description: Free-text description of the specific event.
        escalation_probability: Override escalation probability from
            external model (e.g., ``escalation_model.py``).
        custom_base_impact: Override base impact percentage.

    Returns:
        ScenarioTree with 4 scenarios.
    """
    template = TRIGGER_SCENARIO_TEMPLATES.get(
        trigger_type,
        (-2.0, 2.5, 0.3, 5.0),  # generic default
    )
    base_impact, esc_mult, deesc_frac, bs_mult = template

    if custom_base_impact is not None:
        base_impact = custom_base_impact

    scenarios = _make_default_scenarios(base_impact, esc_mult, deesc_frac, bs_mult)

    # Override probabilities if external escalation model provides them
    if escalation_probability is not None:
        p_esc = min(max(escalation_probability, 0.05), 0.80)
        p_bs = max(0.02, p_esc * 0.15)  # black swan scales with escalation
        p_deesc = max(0.05, (1.0 - p_esc - p_bs) * 0.3)
        p_base = 1.0 - p_esc - p_deesc - p_bs

        scenarios[0].probability = round(p_base, 3)
        scenarios[1].probability = round(p_esc, 3)
        scenarios[2].probability = round(p_deesc, 3)
        scenarios[3].probability = round(p_bs, 3)

    tree = ScenarioTree(
        trigger_type=trigger_type,
        trigger_description=description,
        scenarios=scenarios,
    )

    logger.debug(
        "[Scenario] %s: E[impact]=%.2f%%, tail=%.2f%%, skew=%.1f",
        trigger_type, tree.expected_impact, tree.tail_impact, tree.impact_skew,
    )

    return tree


def evaluate_portfolio_scenario_risk(
    scenario_trees: list[ScenarioTree],
    portfolio_exposure: float,
    risk_budget_pct: float = 5.0,
) -> dict[str, float | bool | str]:
    """Evaluate aggregate scenario risk for the portfolio.

    Args:
        scenario_trees: Active scenario trees from current triggers.
        portfolio_exposure: Current gross portfolio exposure (0-1+).
        risk_budget_pct: Maximum acceptable expected loss (%).

    Returns:
        Dict with ``total_expected_impact``, ``total_tail_impact``,
        ``max_skew``, ``exposure_adjusted_risk``, ``within_budget``,
        ``worst_trigger``.
    """
    if not scenario_trees:
        return {
            "total_expected_impact": 0.0,
            "total_tail_impact": 0.0,
            "max_skew": 0.0,
            "exposure_adjusted_risk": 0.0,
            "within_budget": True,
            "worst_trigger": "",
        }

    total_ei = sum(t.expected_impact for t in scenario_trees)
    total_tail = sum(t.tail_impact for t in scenario_trees)
    max_skew = max(t.impact_skew for t in scenario_trees)

    # Find worst trigger by expected impact
    worst = min(scenario_trees, key=lambda t: t.expected_impact)

    exposure_risk = abs(total_ei) * portfolio_exposure

    return {
        "total_expected_impact": round(total_ei, 4),
        "total_tail_impact": round(total_tail, 4),
        "max_skew": round(max_skew, 2),
        "exposure_adjusted_risk": round(exposure_risk, 4),
        "within_budget": exposure_risk <= risk_budget_pct,
        "worst_trigger": worst.trigger_type,
    }


__all__ = [
    "Scenario",
    "ScenarioTree",
    "TRIGGER_SCENARIO_TEMPLATES",
    "build_scenario_tree",
    "evaluate_portfolio_scenario_risk",
]
