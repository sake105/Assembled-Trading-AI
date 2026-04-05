"""Shock correlation and compound magnitude modeling.

When multiple shocks occur simultaneously, their combined effect is often
non-linear — oil shock + war = more than the sum of parts. This module
models pairwise amplification/dampening between shock types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import DependencySignal, ShockType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shock Correlation Matrix
# Positive value = amplification (compound effect larger than sum)
# Negative value = dampening (shocks partially cancel)
# 1.0 = neutral (no interaction)
# ---------------------------------------------------------------------------

SHOCK_CORRELATION: dict[tuple[str, str], float] = {
    # Energy + Shipping = severe supply disruption (Hormuz + Suez both blocked)
    ("oil_supply_risk", "shipping_lane_disruption"): 1.55,
    ("oil_supply_risk", "shipping_cost_risk"): 1.40,
    ("oil_supply_risk", "insurance_cost_risk"): 1.30,

    # War + Energy = major market shock
    ("oil_supply_risk", "global_risk_off"): 1.35,
    ("energy_price_spike", "global_risk_off"): 1.30,

    # Currency + Capital Flight = vicious circle
    ("currency_crisis", "capital_flight"): 1.60,
    ("currency_crisis", "reserve_depletion"): 1.50,
    ("reserve_depletion", "capital_flight"): 1.45,

    # Sanctions + Banking = financial isolation multiplier
    ("sanctions_exposure", "banking_isolation"): 1.65,
    ("sanctions_exposure", "secondary_sanctions_risk"): 1.40,
    ("banking_isolation", "currency_crisis"): 1.35,

    # Semiconductor + Tech decoupling = innovation gap acceleration
    ("semiconductor_supply_risk", "hegemonic_decoupling"): 1.50,
    ("semiconductor_supply_risk", "tech_restriction_shock"): 1.45,
    ("rare_earth_supply_risk", "semiconductor_supply_risk"): 1.40,
    ("rare_earth_supply_risk", "hegemonic_decoupling"): 1.35,

    # Inflation + Rate shock = debt servicing crisis
    ("inflation_spike", "rate_shock"): 1.35,
    ("rate_shock", "fiscal_shock"): 1.30,
    ("rate_shock", "currency_crisis"): 1.25,

    # Food + Climate = instability multiplier
    ("food_supply_risk", "climate_disruption"): 1.55,
    ("crop_failure", "food_supply_risk"): 1.40,
    ("climate_disruption", "port_closure"): 1.30,
    ("climate_disruption", "supply_chain_break"): 1.35,

    # Cyber + Military = hybrid warfare
    ("cyber_risk", "global_risk_off"): 1.20,
    ("data_breach_systemic", "financial_system_stress"): 1.50,
    ("financial_system_stress", "global_risk_off"): 1.40,
    ("logistics_visibility_loss", "shipping_cost_risk"): 1.30,

    # Sovereign default + Currency = cascading EM crisis
    ("sovereign_default", "currency_crisis"): 1.60,
    ("fiscal_shock", "sovereign_default"): 1.45,
    ("fiscal_shock", "currency_crisis"): 1.30,

    # Policy divergence + Taper shock = EM outflows
    ("policy_divergence", "taper_shock"): 1.35,
    ("taper_shock", "capital_flight"): 1.40,
    ("taper_shock", "currency_crisis"): 1.30,

    # Nuclear escalation = everything worse
    ("nuclear_escalation_risk", "global_risk_off"): 2.0,
    ("nuclear_escalation_risk", "oil_supply_risk"): 1.80,

    # Dampening effects (partial cancellation)
    ("defense_demand_surge", "global_risk_off"): 0.85,  # Defense buffers some panic
    ("oil_supply_risk", "renewable_energy_supply_risk"): 0.90,  # Partial substitution
}


@dataclass
class ShockCluster:
    """A cluster of simultaneously active shocks."""

    cluster_id: str
    shock_types: list[str]
    compound_magnitude: float
    correlation_factor: float
    dominant_shock: str
    affected_sectors: list[str] = field(default_factory=list)


def get_correlation_factor(shock_a: str, shock_b: str) -> float:
    """Return the correlation factor between two shock types.

    Returns 1.0 if no known correlation.
    """
    key = (shock_a, shock_b)
    rev = (shock_b, shock_a)
    return SHOCK_CORRELATION.get(key, SHOCK_CORRELATION.get(rev, 1.0))


def compute_compound_magnitude(
    shocks: list[tuple[str, float]],
) -> float:
    """Compute the compound magnitude of multiple simultaneous shocks.

    Args:
        shocks: List of (shock_type, magnitude) tuples

    Returns:
        Compound magnitude accounting for pairwise correlations
    """
    if not shocks:
        return 0.0
    if len(shocks) == 1:
        return shocks[0][1]

    # Start with the largest shock
    shock_types = [s[0] for s in shocks]
    magnitudes = [s[1] for s in shocks]
    dominant_idx = magnitudes.index(max(magnitudes))
    compound = magnitudes[dominant_idx]

    # Add each additional shock with correlation adjustment
    for i, (stype, mag) in enumerate(shocks):
        if i == dominant_idx:
            continue
        # Find max correlation with any already-counted shock
        max_corr = max(
            get_correlation_factor(stype, shock_types[j])
            for j in range(len(shocks)) if j != i
        )
        # Contribution = magnitude * correlation factor (discounted for non-dominant shocks)
        contribution = mag * max_corr * 0.5  # 50% of secondary shock adds
        compound += contribution

    logger.debug(
        "[ShockCorrelation] Compound magnitude: %.3f from %d shocks",
        compound, len(shocks)
    )
    return compound


def detect_simultaneous_shocks(
    signals: list,
    time_window_hours: int = 24,
) -> list[ShockCluster]:
    """Detect clusters of simultaneous shocks within a time window.

    Args:
        signals: List of DependencySignal or similar objects with shock_type attribute
        time_window_hours: Window for considering shocks simultaneous

    Returns:
        List of ShockCluster objects
    """
    if not signals:
        return []

    # Group by shock types present
    shock_by_type: dict[str, float] = {}
    for sig in signals:
        stype = str(getattr(sig, "shock_type", getattr(sig, "trigger_type", "unknown")))
        severity = float(getattr(sig, "severity", 1))
        magnitude = severity / 3.0  # Normalize 0-3 to 0-1
        shock_by_type[stype] = max(shock_by_type.get(stype, 0), magnitude)

    if len(shock_by_type) < 2:
        return []

    shock_list = list(shock_by_type.items())
    compound = compute_compound_magnitude(shock_list)
    dominant = max(shock_by_type, key=lambda k: shock_by_type[k])

    cluster = ShockCluster(
        cluster_id=f"cluster_{len(shock_list)}shocks",
        shock_types=list(shock_by_type.keys()),
        compound_magnitude=compound,
        correlation_factor=compound / sum(shock_by_type.values()),
        dominant_shock=dominant,
    )
    return [cluster]


def compute_systemic_risk_score(active_shocks: dict[str, float]) -> float:
    """Compute global systemic risk from a dict of {shock_type: magnitude}.

    Returns 0.0 (normal) to 1.0 (systemic crisis).
    """
    if not active_shocks:
        return 0.0

    shocks = list(active_shocks.items())
    compound = compute_compound_magnitude(shocks)

    # Systemic risk threshold: compound magnitude > 1.5 = systemic
    # Normalize to 0-1
    return min(compound / 2.0, 1.0)
