"""Military/geopolitical escalation ladder modeling.

Tracks active conflicts, escalation probabilities, and market impacts
for different escalation levels. Provides structured risk assessment
for geopolitical conflict scenarios.
"""

from __future__ import annotations

import logging
from typing import Any

from .models import ConflictState, EscalationLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Escalation Ladder (0-10)
# ---------------------------------------------------------------------------

ESCALATION_LADDER: list[EscalationLevel] = [
    EscalationLevel(0, "Peace", "Normal diplomatic relations, no active conflict",
                    market_impact_multiplier=1.0, expected_duration_days=365),
    EscalationLevel(1, "Diplomatic Tension", "Ambassador recall, public threats, sanctions",
                    market_impact_multiplier=1.05, expected_duration_days=90),
    EscalationLevel(2, "Economic Warfare", "Comprehensive sanctions, tariffs, embargoes",
                    market_impact_multiplier=1.15, expected_duration_days=180),
    EscalationLevel(3, "Proxy Conflict", "Proxy war, weapons delivery to allies",
                    market_impact_multiplier=1.30, expected_duration_days=365),
    EscalationLevel(4, "Limited Military Action", "Airstrikes, naval blockade, limited operations",
                    market_impact_multiplier=1.50, expected_duration_days=90),
    EscalationLevel(5, "Regional War", "Ground offensive, full air campaign",
                    market_impact_multiplier=1.80, expected_duration_days=180),
    EscalationLevel(6, "Great Power Intervention", "Direct involvement of a major power",
                    market_impact_multiplier=2.20, expected_duration_days=120),
    EscalationLevel(7, "Multi-Theater War", "Simultaneous conflict on multiple fronts",
                    market_impact_multiplier=2.80, expected_duration_days=90),
    EscalationLevel(8, "Total Conventional War", "Full mobilization, strategic bombing",
                    market_impact_multiplier=3.50, expected_duration_days=60),
    EscalationLevel(9, "Nuclear Escalation", "Tactical nuclear use or credible nuclear threat",
                    market_impact_multiplier=5.00, expected_duration_days=14),
    EscalationLevel(10, "Strategic Nuclear Exchange", "Strategic nuclear weapons deployed",
                    market_impact_multiplier=10.0, expected_duration_days=7),
]

# ---------------------------------------------------------------------------
# Active Conflict States
# ---------------------------------------------------------------------------

ACTIVE_CONFLICTS: dict[str, ConflictState] = {
    "UKRAINE_RUSSIA": ConflictState(
        conflict_id="UKRAINE_RUSSIA",
        parties=["RUSSIA", "UKRAINE", "NATO"],  # NATO indirect
        current_level=5,  # Regional war
        escalation_probability={
            6: 0.12,  # NATO direct intervention
            7: 0.04,
            8: 0.01,
            9: 0.003,
        },
        affected_commodities=["WHEAT", "OIL", "NATURAL_GAS", "NEON_GAS",
                               "PALLADIUM", "IRON_ORE", "URANIUM"],
        affected_sectors=["ENERGY", "DEFENSE", "AGRICULTURE", "SEMIS",
                           "SHIPPING", "MINING"],
    ),
    "TAIWAN_TENSION": ConflictState(
        conflict_id="TAIWAN_TENSION",
        parties=["CHINA", "TAIWAN", "US", "JAPAN"],
        current_level=2,  # Economic warfare / tech war
        escalation_probability={
            3: 0.20,  # Proxy/blockade exercises
            4: 0.08,  # Naval blockade
            5: 0.03,  # Invasion attempt
            6: 0.01,  # US direct military response
            9: 0.001,
        },
        affected_commodities=["SEMICONDUCTORS", "RARE_EARTHS", "NEON_GAS",
                               "COPPER", "LITHIUM"],
        affected_sectors=["SEMIS", "TECH", "SHIPPING", "DEFENSE", "AUTO",
                           "CONSUMER"],
    ),
    "SOUTH_CHINA_SEA": ConflictState(
        conflict_id="SOUTH_CHINA_SEA",
        parties=["CHINA", "US", "PHILIPPINES", "VIETNAM", "MALAYSIA"],
        current_level=2,  # Territorial disputes, economic pressure
        escalation_probability={
            3: 0.15,
            4: 0.05,
            5: 0.01,
        },
        affected_commodities=["OIL", "LNG", "NATURAL_GAS", "FISH"],
        affected_sectors=["ENERGY", "SHIPPING", "DEFENSE"],
    ),
    "MIDDLE_EAST": ConflictState(
        conflict_id="MIDDLE_EAST",
        parties=["IRAN", "ISRAEL", "US", "HAMAS", "HEZBOLLAH", "HOUTHIS"],
        current_level=4,  # Limited military action (Iran proxies, Israel-Gaza)
        escalation_probability={
            5: 0.20,  # Direct Iran-Israel war
            6: 0.08,  # US direct military intervention
            7: 0.02,
            9: 0.005,  # Nuclear threat (Israeli or Iranian)
        },
        affected_commodities=["OIL", "LNG", "NATURAL_GAS"],
        affected_sectors=["ENERGY", "SHIPPING", "DEFENSE", "FINANCE"],
    ),
    "HORN_OF_AFRICA": ConflictState(
        conflict_id="HORN_OF_AFRICA",
        parties=["HOUTHIS", "US", "UK", "SAUDI_ARABIA"],
        current_level=3,  # Proxy conflict — Houthi Red Sea attacks
        escalation_probability={
            4: 0.25,  # Ground operations in Yemen
            5: 0.08,
        },
        affected_commodities=["OIL", "NATURAL_GAS"],
        affected_sectors=["SHIPPING", "ENERGY", "INSURANCE"],
    ),
    "SAHEL": ConflictState(
        conflict_id="SAHEL",
        parties=["FRANCE", "MALI", "BURKINA_FASO", "NIGER", "RUSSIA_WAGNER"],
        current_level=3,  # Proxy conflicts, coups
        escalation_probability={
            4: 0.10,
            5: 0.03,
        },
        affected_commodities=["GOLD", "URANIUM", "COBALT"],
        affected_sectors=["MINING", "DEFENSE", "ENERGY"],
    ),
}

# Sector impact per escalation level relative to current level
# Format: {level_delta: {sector: additional_impact 0-1}}
_SECTOR_IMPACT_BY_DOMAIN: dict[str, dict[str, float]] = {
    "UKRAINE_RUSSIA": {
        "ENERGY": 0.80, "DEFENSE": -0.60, "AGRICULTURE": 0.50,
        "SEMIS": 0.30, "FINANCE": 0.40,
    },
    "TAIWAN_TENSION": {
        "SEMIS": 0.95, "TECH": 0.75, "SHIPPING": 0.55,
        "DEFENSE": -0.55, "AUTO": 0.50, "CONSUMER": 0.40,
    },
    "MIDDLE_EAST": {
        "ENERGY": 0.75, "SHIPPING": 0.60, "DEFENSE": -0.50,
        "FINANCE": 0.35,
    },
    "HORN_OF_AFRICA": {
        "SHIPPING": 0.80, "ENERGY": 0.45, "CONSUMER": 0.30,
    },
    "SOUTH_CHINA_SEA": {
        "ENERGY": 0.50, "SHIPPING": 0.65, "DEFENSE": -0.40,
    },
    "SAHEL": {
        "MINING": 0.60, "ENERGY": 0.30, "DEFENSE": -0.30,
    },
}

# What triggers escalation in each conflict
_ESCALATION_TRIGGERS: dict[str, list[str]] = {
    "UKRAINE_RUSSIA": [
        "NATO_TROOPS_DEPLOYMENT", "RUSSIAN_USE_OF_NUCLEAR_WEAPON",
        "UKRAINE_JOINS_NATO", "ATTACK_ON_NATO_TERRITORY",
        "RUSSIAN_OFFENSIVE_KYIV", "WESTERN_F16_SHOOT_DOWN",
    ],
    "TAIWAN_TENSION": [
        "TAIWAN_INDEPENDENCE_DECLARATION", "US_CARRIER_STRAIT_TRANSIT",
        "PLA_LIVE_FIRE_EXERCISE", "CHIP_EXPORT_BAN_TOTAL",
        "TAIWAN_US_DEFENSE_TREATY", "PLA_ISLAND_SEIZURE",
    ],
    "MIDDLE_EAST": [
        "IRAN_NUCLEAR_BREAKOUT", "ISRAEL_AIR_STRIKE_IRAN",
        "US_CARRIER_ATTACK", "HORMUZ_MINING",
        "HEZBOLLAH_FULL_ATTACK", "IRAN_PROXY_US_BASE_ATTACK",
    ],
    "SOUTH_CHINA_SEA": [
        "CHINA_PHILIPPINE_VESSEL_SINKING", "US_FREEDOM_OF_NAVIGATION_INCIDENT",
        "CHINA_ISLAND_MILITARY_BUILDUP", "RESOURCE_EXTRACTION_CONFRONTATION",
    ],
    "HORN_OF_AFRICA": [
        "HOUTHI_WARSHIP_ATTACK", "US_ESCALATED_STRIKES",
        "IRAN_DIRECT_INVOLVEMENT", "SAUDI_GROUND_OPERATION",
    ],
    "SAHEL": [
        "FRENCH_FORCES_EXPULSION", "RUSSIAN_MILITARY_BASE",
        "URANIUM_MINE_SEIZURE", "REGIONAL_COUP_WAVE",
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_escalation_level(level: int) -> EscalationLevel:
    """Return the escalation level descriptor."""
    level = max(0, min(10, level))
    return ESCALATION_LADDER[level]


def get_conflict(conflict_id: str) -> ConflictState | None:
    """Return an active conflict by ID."""
    return ACTIVE_CONFLICTS.get(conflict_id)


def compute_escalation_probability(
    conflict: ConflictState,
    trigger_type: str | None = None,
) -> dict[int, float]:
    """Compute revised escalation probabilities after a trigger event.

    If trigger_type matches a known escalation trigger for this conflict,
    probabilities are amplified upward.
    """
    base_probs = dict(conflict.escalation_probability)

    if trigger_type is not None:
        triggers = _ESCALATION_TRIGGERS.get(conflict.conflict_id, [])
        if trigger_type in triggers:
            # Amplify all probabilities by 1.5x
            base_probs = {level: min(p * 1.5, 0.99) for level, p in base_probs.items()}
            logger.info(
                "[Escalation] Trigger %s in %s → probabilities amplified",
                trigger_type, conflict.conflict_id
            )

    return base_probs


def compute_market_impact_by_level(
    conflict: ConflictState,
    level: int,
) -> dict[str, float]:
    """Compute sector market impact at a given escalation level.

    Returns {sector: impact_score} where positive = negative market impact
    (prices drop) and negative = positive (prices rise, e.g., defense).
    Impact is scaled by the escalation level's market_impact_multiplier.
    """
    esc_level = get_escalation_level(level)
    multiplier = esc_level.market_impact_multiplier
    base_impacts = _SECTOR_IMPACT_BY_DOMAIN.get(conflict.conflict_id, {})

    # Scale by how much above current level
    delta = max(0, level - conflict.current_level)
    scaling = 1.0 + delta * 0.15  # +15% per escalation step

    return {
        sector: min(abs(impact) * multiplier * scaling, 1.0) * (1 if impact > 0 else -1)
        for sector, impact in base_impacts.items()
    }


def identify_escalation_triggers(conflict_id: str) -> list[str]:
    """Return list of events that could escalate this conflict."""
    return _ESCALATION_TRIGGERS.get(conflict_id, [])


def compute_contagion_risk(
    conflict_id: str,
    other_conflicts: dict[str, ConflictState] | None = None,
) -> dict[str, float]:
    """Estimate risk of conflict spreading to other regions.

    Returns {region_or_conflict: contagion_score 0-1}.
    """
    conflict = ACTIVE_CONFLICTS.get(conflict_id)
    if conflict is None:
        return {}

    contagion: dict[str, float] = {}
    level_factor = conflict.current_level / 10.0

    # Define contagion paths
    _contagion_map: dict[str, dict[str, float]] = {
        "UKRAINE_RUSSIA": {
            "MIDDLE_EAST": 0.15,       # Russia-Iran cooperation
            "TAIWAN_TENSION": 0.10,    # Emboldened revisionism
            "SAHEL": 0.20,             # Wagner group
        },
        "TAIWAN_TENSION": {
            "SOUTH_CHINA_SEA": 0.60,   # Direct linkage
            "HORN_OF_AFRICA": 0.05,
        },
        "MIDDLE_EAST": {
            "HORN_OF_AFRICA": 0.45,    # Iran-Houthi link
            "SOUTH_CHINA_SEA": 0.05,
        },
        "HORN_OF_AFRICA": {
            "MIDDLE_EAST": 0.35,
        },
    }

    paths = _contagion_map.get(conflict_id, {})
    for target, base_risk in paths.items():
        contagion[target] = min(base_risk * (1 + level_factor), 1.0)

    return dict(sorted(contagion.items(), key=lambda x: x[1], reverse=True))


def get_all_active_conflicts() -> list[ConflictState]:
    """Return all currently tracked conflict states."""
    return list(ACTIVE_CONFLICTS.values())


def compute_global_conflict_risk() -> float:
    """Compute aggregate global conflict risk as a single score (0-1).

    Weights each conflict by its current escalation level and probability of
    further escalation.
    """
    total_risk = 0.0
    weights = {
        "UKRAINE_RUSSIA": 0.25,
        "TAIWAN_TENSION": 0.30,
        "MIDDLE_EAST": 0.25,
        "HORN_OF_AFRICA": 0.10,
        "SOUTH_CHINA_SEA": 0.05,
        "SAHEL": 0.05,
    }

    for conflict_id, weight in weights.items():
        conflict = ACTIVE_CONFLICTS.get(conflict_id)
        if conflict is None:
            continue
        level_score = conflict.current_level / 10.0
        # Expected escalation contribution
        esc_score = sum(
            p * (level - conflict.current_level) / 10.0
            for level, p in conflict.escalation_probability.items()
            if level > conflict.current_level
        )
        total_risk += weight * (level_score + esc_score)

    return min(total_risk, 1.0)
