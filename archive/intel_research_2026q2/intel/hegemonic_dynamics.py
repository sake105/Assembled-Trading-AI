"""Hegemonic dynamics and strategic decoupling modeling.

Models geopolitical decoupling between major powers, strategic pivots,
alliance stress, and the market implications of hegemonic shifts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dependency_graph import DependencyGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decoupling Scenario
# ---------------------------------------------------------------------------


@dataclass
class DecouplingScenario:
    """Models the decoupling trajectory between two powers in specific domains."""

    scenario_id: str
    nations: tuple[str, str]
    domains: list[str]                          # tech, finance, energy, military, trade
    current_coupling_score: float               # 0=fully decoupled, 1=fully coupled
    projected_decoupling_rate: float            # annual rate (e.g., 0.08 = -8%/year)
    acceleration_triggers: list[str]
    market_sectors_affected: dict[str, float]  # sector → impact (-1 hurt, +1 benefit)
    probability_full_decoupling_5y: float       # 0-1


DECOUPLING_SCENARIOS: dict[str, DecouplingScenario] = {
    "US_CHINA_TECH": DecouplingScenario(
        scenario_id="US_CHINA_TECH",
        nations=("US", "CHINA"),
        domains=["semiconductors", "ai", "quantum", "space", "telecom"],
        current_coupling_score=0.45,
        projected_decoupling_rate=0.08,
        acceleration_triggers=["TAIWAN_INVASION", "CHIP_EXPORT_BAN_TOTAL",
                                 "CHINA_AI_MILITARY_USE", "TikTok_ban_expansion"],
        market_sectors_affected={
            "SEMIS": -0.35, "TECH": -0.20, "AUTO": -0.15,
            "DEFENSE": +0.25, "RENEWABLE_ENERGY": -0.10,
        },
        probability_full_decoupling_5y=0.20,
    ),
    "US_CHINA_FINANCE": DecouplingScenario(
        scenario_id="US_CHINA_FINANCE",
        nations=("US", "CHINA"),
        domains=["capital_markets", "banking", "investment", "dollar_dominance"],
        current_coupling_score=0.55,
        projected_decoupling_rate=0.04,
        acceleration_triggers=["CHINA_TAIWAN_SANCTIONS", "ADR_DELISTING_FULL",
                                 "HONG_KONG_FINANCIAL_CENTER_LOSS"],
        market_sectors_affected={
            "FINANCE": -0.30, "TECH": -0.15, "CONSUMER": -0.10,
        },
        probability_full_decoupling_5y=0.12,
    ),
    "US_CHINA_ENERGY": DecouplingScenario(
        scenario_id="US_CHINA_ENERGY",
        nations=("US", "CHINA"),
        domains=["solar", "ev_batteries", "critical_minerals", "lng"],
        current_coupling_score=0.60,
        projected_decoupling_rate=0.05,
        acceleration_triggers=["CHINA_CRITICAL_MINERAL_EXPORT_BAN",
                                 "US_IRA_EXCLUSION_CHINA", "EV_TARIFFS_50PCT"],
        market_sectors_affected={
            "RENEWABLE_ENERGY": -0.25, "AUTO": -0.20, "MINING": +0.15,
        },
        probability_full_decoupling_5y=0.15,
    ),
    "RUSSIA_EUROPE_ENERGY": DecouplingScenario(
        scenario_id="RUSSIA_EUROPE_ENERGY",
        nations=("RUSSIA", "EU"),
        domains=["natural_gas", "oil", "coal", "nuclear"],
        current_coupling_score=0.30,  # Already mostly decoupled since 2022
        projected_decoupling_rate=0.10,  # Completing the decoupling
        acceleration_triggers=["NORD_STREAM_TOTAL_CLOSURE", "UKRAINE_PIPELINE_STOP",
                                 "EU_FULL_ENERGY_EMBARGO"],
        market_sectors_affected={
            "ENERGY": -0.40, "AUTO": -0.20, "CONSUMER": -0.25,
            "DEFENSE": +0.15,
        },
        probability_full_decoupling_5y=0.70,  # Already well advanced
    ),
    "CHINA_TAIWAN_SUPPLY": DecouplingScenario(
        scenario_id="CHINA_TAIWAN_SUPPLY",
        nations=("CHINA", "TAIWAN"),
        domains=["semiconductors", "manufacturing", "investment"],
        current_coupling_score=0.65,  # Deep economic ties despite tension
        projected_decoupling_rate=0.06,
        acceleration_triggers=["PLA_EXERCISES_BLOCKADE", "TSMC_CHINA_EXIT",
                                 "TAIWAN_INDEPENDENCE_DECLARATION"],
        market_sectors_affected={
            "SEMIS": -0.90, "TECH": -0.70, "AUTO": -0.55,
        },
        probability_full_decoupling_5y=0.08,
    ),
}

# Strategic pivot templates: who is pivoting from whom to whom
STRATEGIC_PIVOTS: dict[str, dict[str, Any]] = {
    "US_PIVOT_INDIA": {
        "from": "CHINA", "to": "INDIA",
        "domains": ["manufacturing", "tech_supply_chain", "defense"],
        "timeline_years": 5,
        "current_progress": 0.25,  # 25% done
        "market_impact": {
            "INDIA_EQUITIES": +0.35, "CHINA_EQUITIES": -0.25,
            "SEMIS": +0.10, "DEFENSE": +0.15,
        },
    },
    "US_PIVOT_MEXICO": {
        "from": "CHINA", "to": "MEXICO",
        "domains": ["manufacturing", "auto", "electronics"],
        "timeline_years": 3,
        "current_progress": 0.40,
        "market_impact": {
            "MEXICO_EQUITIES": +0.30, "AUTO": +0.10, "CONSUMER": +0.10,
        },
    },
    "EU_PIVOT_DIVERSIFICATION": {
        "from": "RUSSIA", "to": ["NORWAY", "US", "QATAR", "ALGERIA"],
        "domains": ["energy"],
        "timeline_years": 5,
        "current_progress": 0.55,
        "market_impact": {
            "LNG": +0.20, "ENERGY": +0.15, "GERMANY_EQUITIES": -0.10,
        },
    },
    "CHINA_PIVOT_BRICS": {
        "from": "US_ALLIES", "to": ["RUSSIA", "INDIA", "BRAZIL", "GULF"],
        "domains": ["trade", "finance", "tech"],
        "timeline_years": 10,
        "current_progress": 0.20,
        "market_impact": {
            "USD": -0.10, "CNY": +0.05, "EM_EQUITIES": +0.10,
        },
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_decoupling_score(
    nation_a: str,
    nation_b: str,
    graph: "DependencyGraph | None" = None,
) -> float:
    """Compute current decoupling score between two nations.

    Returns 0.0 (fully coupled) to 1.0 (fully decoupled).
    """
    # Check direct scenario
    for scenario in DECOUPLING_SCENARIOS.values():
        if set(scenario.nations) == {nation_a, nation_b}:
            return 1.0 - scenario.current_coupling_score

    # Graph-based estimation
    if graph is not None:
        from .models import EdgeType
        # Count friendly vs hostile edges
        n1 = graph.get_node(nation_a)
        n2 = graph.get_node(nation_b)
        if n1 and n2:
            friendly = [EdgeType.ALLY_OF, EdgeType.INVESTS_IN, EdgeType.IMPORTS_FROM]
            hostile = [EdgeType.RIVAL_OF, EdgeType.SANCTIONS_TARGET, EdgeType.SANCTIONED_BY]
            friendly_edges = len(graph.get_neighbors(nation_a, edge_types=friendly))
            hostile_edges = len(graph.get_neighbors(nation_a, edge_types=hostile))
            if friendly_edges + hostile_edges > 0:
                return hostile_edges / (friendly_edges + hostile_edges)

    # Default based on known rivalries
    known_rivalries = {
        frozenset({"US", "RUSSIA"}): 0.85,
        frozenset({"US", "CHINA"}): 0.55,
        frozenset({"RUSSIA", "UKRAINE"}): 0.95,
        frozenset({"ISRAEL", "IRAN"}): 0.90,
        frozenset({"INDIA", "PAKISTAN"}): 0.80,
        frozenset({"TAIWAN", "CHINA"}): 0.60,
    }
    return known_rivalries.get(frozenset({nation_a, nation_b}), 0.1)


def model_strategic_pivot(
    from_nation: str,
    to_nation: str | list[str],
    domains: list[str],
) -> dict[str, Any]:
    """Model the market implications of a strategic pivot.

    Returns estimated market impacts and transition costs.
    """
    to_nations = [to_nation] if isinstance(to_nation, str) else to_nation

    # Check if this matches a known pivot
    for pivot_id, pivot in STRATEGIC_PIVOTS.items():
        pivot_to = pivot["to"] if isinstance(pivot["to"], list) else [pivot["to"]]
        if (pivot["from"] == from_nation and
                any(t in to_nations for t in pivot_to) and
                any(d in pivot["domains"] for d in domains)):
            return {
                "pivot_id": pivot_id,
                "progress": pivot["current_progress"],
                "timeline_years": pivot["timeline_years"],
                "market_impact": pivot["market_impact"],
                "status": "known_pivot",
            }

    # Generic pivot estimation
    return {
        "pivot_id": f"PIVOT_{from_nation}_TO_{'_'.join(to_nations[:2])}",
        "progress": 0.0,
        "timeline_years": 5,
        "market_impact": {
            f"{n}_EQUITIES": +0.20 for n in to_nations[:3]
        },
        "status": "estimated",
    }


def compute_alliance_stress(
    alliance: str,
    trigger_type: str | None = None,
) -> float:
    """Estimate stress level within an alliance (0=cohesive, 1=fracturing).

    Args:
        alliance: Alliance ID (NATO, BRICS, EU, etc.)
        trigger_type: Optional event that increases stress
    """
    base_stress = {
        "NATO": 0.25,         # Hungary friction, Turkey issues
        "EU": 0.30,           # Hungary, Poland rule-of-law disputes
        "BRICS": 0.20,        # Diverse interests, India-China rivalry
        "AUKUS": 0.05,        # New, aligned
        "BELT_AND_ROAD": 0.40, # Debt trap concerns, defections
        "QUAD": 0.15,          # India hedging
        "SCO": 0.35,           # India-China rivalry within
        "ASEAN": 0.25,         # Non-alignment tradition
    }.get(alliance, 0.30)

    # Trigger amplification
    stress_triggers = {
        "NATO": ["TRUMP_WITHDRAWAL_THREAT", "ARTICLE5_INVOCATION",
                  "TURKEY_RUSSIA_DEAL", "NUCLEAR_WEAPONS_DEBATE"],
        "EU": ["FRANCE_GERMANY_SPLIT", "ORBAN_VETO",
                "EURO_BOND_CRISIS", "SCHENGEN_SUSPENSION"],
        "BRICS": ["INDIA_CHINA_BORDER_CLASH", "DOLLAR_DOMINANCE_CHALLENGE"],
    }.get(alliance, [])

    if trigger_type and trigger_type in stress_triggers:
        base_stress = min(base_stress * 1.5, 1.0)

    return base_stress


def simulate_bloc_formation(
    nations_a: list[str],
    nations_b: list[str],
) -> dict[str, Any]:
    """Model market impact of two blocs forming (e.g., US+allies vs China+Russia).

    Returns expected market bifurcation impacts.
    """
    bloc_size_a = len(nations_a)
    bloc_size_b = len(nations_b)
    total = bloc_size_a + bloc_size_b

    return {
        "bloc_a": nations_a,
        "bloc_b": nations_b,
        "trade_fragmentation_pct": min((bloc_size_a * bloc_size_b) / total**2 * 0.4, 0.30),
        "global_gdp_loss_est_pct": min(total * 0.005, 0.05),
        "beneficiaries": ["DEFENSE", "DOMESTIC_SEMIS"],
        "losers": ["GLOBAL_TRADE", "SHIPPING", "CONSUMER_GOODS"],
        "currency_impact": {
            "USD": +0.10 if "US" in nations_a else -0.05,
            "CNY": +0.05 if "CHINA" in nations_b else -0.05,
        },
    }


def estimate_decoupling_acceleration(
    trigger_event: str,
) -> float:
    """Estimate how much a trigger event accelerates decoupling.

    Returns additional annual decoupling rate (e.g., 0.15 = 15% faster).
    """
    high_acceleration = {
        "TAIWAN_INVASION", "NUCLEAR_USE", "FULL_CHIP_BAN",
        "ADR_DELISTING_ALL", "SWIFT_CHINA_EXCLUSION",
    }
    medium_acceleration = {
        "NEW_EXPORT_CONTROL", "ENTITY_LISTING_MAJOR",
        "TRADE_WAR_ESCALATION", "ALLIANCE_SHIFT",
    }
    low_acceleration = {
        "DIPLOMATIC_CRISIS", "TARIFF_INCREASE",
        "TECHNOLOGY_GAP_WIDENING",
    }

    if trigger_event in high_acceleration:
        return 0.20
    elif trigger_event in medium_acceleration:
        return 0.08
    elif trigger_event in low_acceleration:
        return 0.03
    return 0.01
