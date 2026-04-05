"""Feedback loop modeling for shock propagation.

Geopolitical/macro shocks often trigger cascading chains of effects.
Oil shock → inflation → rate hike → recession → risk-off → gold rally.
This module defines known feedback loops and propagates them through
shock sequences with amplification and convergence checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FeedbackLoop:
    """A causal chain of shock types that self-reinforce."""

    loop_id: str
    name: str
    chain: list[str]                # Ordered list of shock types in the chain
    amplification: float = 1.2     # How much each step amplifies the previous
    typical_duration_months: int = 6
    historical_examples: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)  # Conditions for loop to activate


FEEDBACK_LOOPS: list[FeedbackLoop] = [
    FeedbackLoop(
        loop_id="OIL_INFLATION_RECESSION",
        name="Oil Shock → Stagflation → Recession",
        chain=["oil_supply_risk", "energy_price_spike", "inflation_spike",
               "rate_shock", "global_risk_off", "fiscal_shock"],
        amplification=1.30,
        typical_duration_months=6,
        historical_examples=["1973 Oil Crisis", "1979 Iranian Revolution", "2022 Ukraine"],
    ),
    FeedbackLoop(
        loop_id="SANCTIONS_ISOLATION",
        name="Sanctions → Isolation → Currency Crisis → Fiscal",
        chain=["sanctions_exposure", "banking_isolation", "capital_flight",
               "currency_crisis", "fiscal_shock"],
        amplification=1.40,
        typical_duration_months=12,
        historical_examples=["Russia 2022", "Iran 2012+", "Venezuela 2019"],
    ),
    FeedbackLoop(
        loop_id="TECH_DECOUPLING_INNOVATION",
        name="Tech Restriction → Chip Shortage → Innovation Gap → Decoupling",
        chain=["tech_restriction_shock", "semiconductor_supply_risk",
               "chip_shortage", "innovation_gap", "hegemonic_decoupling"],
        amplification=1.20,
        typical_duration_months=24,
        historical_examples=["US-China Tech War 2018+", "CHIPS Act 2022"],
    ),
    FeedbackLoop(
        loop_id="CLIMATE_FOOD_INSTABILITY",
        name="Climate → Crop Failure → Food Crisis → Political Instability",
        chain=["climate_disruption", "crop_failure", "food_supply_risk",
               "fiscal_shock"],
        amplification=1.50,
        typical_duration_months=3,
        historical_examples=["2010 Russian Drought/Arab Spring", "2022 Grain Crisis"],
    ),
    FeedbackLoop(
        loop_id="CURRENCY_CAPITAL_FLIGHT",
        name="Currency Weakness → Capital Flight → Reserve Depletion → Default",
        chain=["currency_crisis", "capital_flight", "reserve_depletion",
               "sovereign_default"],
        amplification=1.60,
        typical_duration_months=6,
        historical_examples=["Argentina 2001", "Asian Crisis 1997", "Turkey 2021"],
    ),
    FeedbackLoop(
        loop_id="SHIPPING_INSURANCE_INFLATION",
        name="Shipping Disruption → Insurance Surge → Supply Chain → Inflation",
        chain=["shipping_lane_disruption", "insurance_cost_risk",
               "shipping_cost_risk", "supply_chain_break", "inflation_spike"],
        amplification=1.25,
        typical_duration_months=3,
        historical_examples=["Suez Ever Given 2021", "Red Sea Crisis 2023"],
    ),
    FeedbackLoop(
        loop_id="MILITARY_ESCALATION_ENERGY",
        name="Military Escalation → Energy Disruption → Global Risk-Off",
        chain=["military_loss_surge", "supply_line_threat",
               "oil_supply_risk", "global_risk_off"],
        amplification=1.35,
        typical_duration_months=2,
        historical_examples=["Gulf War 1991", "Iraq War 2003", "Ukraine 2022"],
    ),
    FeedbackLoop(
        loop_id="CYBER_FINANCIAL_SYSTEMIC",
        name="Cyber Attack → Financial System → Credit Freeze → Risk-Off",
        chain=["data_breach_systemic", "financial_system_stress",
               "banking_isolation", "global_risk_off"],
        amplification=1.45,
        typical_duration_months=1,
        historical_examples=["WannaCry 2017", "NotPetya 2017", "SolarWinds 2020"],
    ),
    FeedbackLoop(
        loop_id="RATE_TAPER_EM_CRISIS",
        name="Rate Shock → Taper → EM Capital Flight → EM Currency Crisis",
        chain=["rate_shock", "taper_shock", "policy_divergence",
               "capital_flight", "currency_crisis"],
        amplification=1.30,
        typical_duration_months=9,
        historical_examples=["Taper Tantrum 2013", "EM Crisis 2018"],
    ),
    FeedbackLoop(
        loop_id="RARE_EARTH_SEMI_DEPENDENCY",
        name="Rare Earth Shock → Semiconductor Crunch → Tech Supply Crisis",
        chain=["rare_earth_supply_risk", "semiconductor_supply_risk",
               "chip_shortage", "tech_restriction_shock"],
        amplification=1.35,
        typical_duration_months=12,
        historical_examples=["China Rare Earth Restrictions 2010",
                              "TSMC Supply Crunch 2021-2022"],
    ),
]

# Lookup by loop_id
_LOOP_REGISTRY: dict[str, FeedbackLoop] = {loop.loop_id: loop for loop in FEEDBACK_LOOPS}


def get_feedback_loop(loop_id: str) -> FeedbackLoop | None:
    """Return a feedback loop by ID."""
    return _LOOP_REGISTRY.get(loop_id)


def detect_active_feedback_loops(
    current_shocks: list[str],
) -> list[FeedbackLoop]:
    """Identify which feedback loops are potentially active given current shocks.

    A loop is active if at least the first two shocks in its chain are present.

    Returns list of active loops sorted by chain coverage (descending).
    """
    active = []
    for loop in FEEDBACK_LOOPS:
        present = sum(1 for shock in loop.chain if shock in current_shocks)
        if present >= 2:
            active.append((loop, present))

    return [loop for loop, _ in sorted(active, key=lambda x: x[1], reverse=True)]


def propagate_with_feedback(
    active_shocks: dict[str, float],
    loops: list[FeedbackLoop] | None = None,
    max_iterations: int = 3,
) -> dict[str, float]:
    """Propagate shocks through feedback loops.

    Starting from active_shocks {shock_type: magnitude}, follows feedback
    loop chains to identify downstream shocks that would be triggered.

    Args:
        active_shocks: Initial shock magnitudes {shock_type: magnitude}
        loops: Feedback loops to check (default: all FEEDBACK_LOOPS)
        max_iterations: Maximum propagation iterations (prevents infinite loops)

    Returns:
        Updated dict of all shocks (initial + triggered) with magnitudes
    """
    if loops is None:
        loops = FEEDBACK_LOOPS

    result = dict(active_shocks)
    converged = False
    iteration = 0

    while not converged and iteration < max_iterations:
        iteration += 1
        new_shocks: dict[str, float] = {}

        for loop in loops:
            triggered_by_chain = False

            # Find the latest shock in the chain that is active
            latest_chain_idx = -1
            for i, shock in enumerate(loop.chain):
                if shock in result:
                    latest_chain_idx = i

            if latest_chain_idx < 0:
                continue  # No shocks in this chain

            # Propagate to next shocks in chain
            current_mag = result[loop.chain[latest_chain_idx]]
            for next_shock in loop.chain[latest_chain_idx + 1:]:
                next_mag = current_mag * loop.amplification * 0.7  # Decay
                if next_mag < 0.05:
                    break  # Too weak to propagate further
                if next_mag > new_shocks.get(next_shock, 0):
                    new_shocks[next_shock] = next_mag
                    triggered_by_chain = True
                current_mag = next_mag

            if triggered_by_chain:
                logger.debug(
                    "[FeedbackLoops] Loop %s triggered %d downstream shocks (iter %d)",
                    loop.loop_id, len(new_shocks), iteration
                )

        # Check convergence: no new shocks or no magnitude increase
        if not new_shocks:
            converged = True
        else:
            changed = False
            for shock, mag in new_shocks.items():
                if mag > result.get(shock, 0) + 0.01:
                    result[shock] = mag
                    changed = True
            if not changed:
                converged = True

    if not converged:
        logger.warning(
            "[FeedbackLoops] Did not converge in %d iterations — forcing stop",
            max_iterations
        )

    return result


def compute_loop_activation_probability(
    loop: FeedbackLoop,
    current_shocks: list[str],
) -> float:
    """Estimate probability that a feedback loop will fully activate.

    Based on what fraction of the chain's precursor shocks are already active.
    """
    if not loop.chain:
        return 0.0

    # First half of chain as precursors
    precursors = loop.chain[:max(1, len(loop.chain) // 2)]
    active_precursors = sum(1 for s in precursors if s in current_shocks)
    coverage = active_precursors / len(precursors)

    # Scale: 0 precursors = 5% base, all precursors = 70%
    return 0.05 + coverage * 0.65
