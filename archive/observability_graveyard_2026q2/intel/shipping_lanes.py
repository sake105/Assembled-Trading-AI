"""Shipping lane dependency modeling for geopolitical risk analysis.

Models major maritime shipping lanes, their chokepoint dependencies,
and the economic impact of disruptions on nations and sectors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .models import ShippingLane

if TYPE_CHECKING:
    from .dependency_graph import DependencyGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shipping Lane Database
# ---------------------------------------------------------------------------

LANES_DATABASE: dict[str, ShippingLane] = {
    "PERSIAN_GULF_ASIA": ShippingLane(
        lane_id="PERSIAN_GULF_ASIA",
        name="Persian Gulf → Asia (via Hormuz + Malacca)",
        chokepoints=["HORMUZ", "MALACCA"],
        daily_traffic_value_bn=5.5,
        oil_transit_mbpd=17.0,      # ~21% of global oil supply
        lng_transit_bcm_year=140.0,
        container_teu_year=0.0,
        alternative_route="CAPE_ROUTE",
        reroute_cost_multiplier=2.8,
        reroute_time_days=14.0,
        insurance_surge_pct=300.0,  # Houthi crisis reference
        nations_dependent=["CHINA", "JAPAN", "SOUTH_KOREA", "INDIA", "TAIWAN",
                           "THAILAND", "INDONESIA", "VIETNAM"],
    ),
    "ASIA_EUROPE": ShippingLane(
        lane_id="ASIA_EUROPE",
        name="Asia → Europe (via Malacca + Suez)",
        chokepoints=["MALACCA", "SUEZ", "BAB_EL_MANDEB"],
        daily_traffic_value_bn=9.0,  # ~12-15% of global trade value
        oil_transit_mbpd=4.0,
        lng_transit_bcm_year=60.0,
        container_teu_year=19_000_000,  # ~19M TEU/year
        alternative_route="CAPE_ROUTE",
        reroute_cost_multiplier=1.7,
        reroute_time_days=10.0,
        insurance_surge_pct=200.0,
        nations_dependent=["CHINA", "JAPAN", "SOUTH_KOREA", "TAIWAN", "INDIA",
                           "GERMANY", "FRANCE", "UK", "NETHERLANDS", "ITALY"],
    ),
    "ASIA_NAMERICA": ShippingLane(
        lane_id="ASIA_NAMERICA",
        name="Asia → North America (Transpacific)",
        chokepoints=["MALACCA", "PANAMA"],
        daily_traffic_value_bn=7.0,
        oil_transit_mbpd=0.5,
        lng_transit_bcm_year=20.0,
        container_teu_year=25_000_000,
        alternative_route=None,  # Too far for Cape alternative
        reroute_cost_multiplier=1.5,  # Panama bypass possible
        reroute_time_days=7.0,
        insurance_surge_pct=50.0,
        nations_dependent=["US", "CANADA", "MEXICO", "CHINA", "JAPAN",
                           "SOUTH_KOREA", "TAIWAN", "VIETNAM"],
    ),
    "EUROPE_NAMERICA": ShippingLane(
        lane_id="EUROPE_NAMERICA",
        name="Europe ↔ North America (Transatlantic)",
        chokepoints=["DANISH_STRAITS"],
        daily_traffic_value_bn=4.0,
        oil_transit_mbpd=2.0,
        lng_transit_bcm_year=50.0,
        container_teu_year=8_000_000,
        alternative_route=None,
        reroute_cost_multiplier=1.1,
        reroute_time_days=2.0,
        insurance_surge_pct=20.0,
        nations_dependent=["US", "CANADA", "UK", "GERMANY", "FRANCE",
                           "NETHERLANDS", "BELGIUM", "SPAIN"],
    ),
    "SOUTH_CHINA_SEA": ShippingLane(
        lane_id="SOUTH_CHINA_SEA",
        name="South China Sea (Regional Asia Trade)",
        chokepoints=["MALACCA", "TAIWAN_STRAIT", "LOMBOK_STRAIT"],
        daily_traffic_value_bn=3.5,
        oil_transit_mbpd=16.0,  # Overlaps with Persian Gulf lane
        lng_transit_bcm_year=90.0,
        container_teu_year=30_000_000,
        alternative_route="LOMBOK_STRAIT",
        reroute_cost_multiplier=1.3,
        reroute_time_days=3.0,
        insurance_surge_pct=150.0,
        nations_dependent=["CHINA", "JAPAN", "SOUTH_KOREA", "TAIWAN",
                           "VIETNAM", "INDONESIA", "THAILAND", "PHILIPPINES"],
    ),
    "CAPE_ROUTE": ShippingLane(
        lane_id="CAPE_ROUTE",
        name="Cape of Good Hope Route (Africa Circumnavigation)",
        chokepoints=["CAPE_GOOD_HOPE"],
        daily_traffic_value_bn=2.0,
        oil_transit_mbpd=6.0,
        lng_transit_bcm_year=30.0,
        container_teu_year=5_000_000,
        alternative_route=None,
        reroute_cost_multiplier=1.0,  # This IS the alternative
        reroute_time_days=0.0,
        insurance_surge_pct=15.0,
        nations_dependent=["SOUTH_AFRICA", "BRAZIL", "NIGERIA"],
    ),
    "MIDDLE_EAST_EUROPE": ShippingLane(
        lane_id="MIDDLE_EAST_EUROPE",
        name="Middle East → Europe (via Suez or Pipeline)",
        chokepoints=["SUEZ", "BAB_EL_MANDEB", "HORMUZ"],
        daily_traffic_value_bn=3.0,
        oil_transit_mbpd=7.0,
        lng_transit_bcm_year=45.0,
        container_teu_year=3_000_000,
        alternative_route="CAPE_ROUTE",
        reroute_cost_multiplier=2.0,
        reroute_time_days=12.0,
        insurance_surge_pct=250.0,
        nations_dependent=["GERMANY", "FRANCE", "ITALY", "SPAIN", "GREECE",
                           "TURKEY", "EGYPT", "SAUDI_ARABIA", "UAE", "QATAR"],
    ),
    "ARCTIC_ROUTE": ShippingLane(
        lane_id="ARCTIC_ROUTE",
        name="Northern Sea Route (Arctic, Seasonal)",
        chokepoints=["DANISH_STRAITS"],
        daily_traffic_value_bn=0.3,   # Still emerging
        oil_transit_mbpd=0.2,
        lng_transit_bcm_year=15.0,
        container_teu_year=500_000,
        alternative_route="ASIA_EUROPE",
        reroute_cost_multiplier=1.0,
        reroute_time_days=0.0,
        insurance_surge_pct=30.0,
        nations_dependent=["RUSSIA", "CHINA", "JAPAN", "SOUTH_KOREA", "NORWAY"],
    ),
}

# Chokepoint → lanes that pass through it
_CHOKEPOINT_TO_LANES: dict[str, list[str]] = {}
for _lane in LANES_DATABASE.values():
    for _cp in _lane.chokepoints:
        _CHOKEPOINT_TO_LANES.setdefault(_cp, []).append(_lane.lane_id)

# World trade share per chokepoint (approximate)
CHOKEPOINT_WORLD_TRADE_SHARE: dict[str, float] = {
    "HORMUZ": 0.21,         # 21% of global oil, ~10% of total trade
    "SUEZ": 0.12,           # 12-15% of global trade
    "MALACCA": 0.25,        # 25% of global maritime trade
    "TAIWAN_STRAIT": 0.22,  # 22% of global maritime trade
    "PANAMA": 0.06,         # 6% of global trade
    "BAB_EL_MANDEB": 0.10,  # 10% of global trade (Red Sea gateway)
    "DANISH_STRAITS": 0.04,
    "CAPE_GOOD_HOPE": 0.08,
    "LOMBOK_STRAIT": 0.05,
    "TURKISH_STRAITS": 0.03,
}

# Sector impact per chokepoint disruption: {sector: impact_score 0-1}
CHOKEPOINT_SECTOR_IMPACT: dict[str, dict[str, float]] = {
    "HORMUZ": {
        "ENERGY": 0.90, "SHIPPING": 0.70, "AUTO": 0.40,
        "PHARMA": 0.30, "TECH": 0.25, "FINANCE": 0.50,
        "AGRICULTURE": 0.20, "SEMIS": 0.30, "DEFENSE": -0.40,  # benefits
    },
    "SUEZ": {
        "SHIPPING": 0.80, "ENERGY": 0.40, "CONSUMER": 0.35,
        "TECH": 0.30, "PHARMA": 0.25, "AUTO": 0.35,
        "AGRICULTURE": 0.25, "FINANCE": 0.30, "DEFENSE": -0.20,
    },
    "MALACCA": {
        "ENERGY": 0.60, "SHIPPING": 0.75, "SEMIS": 0.70,
        "TECH": 0.50, "AUTO": 0.45, "CONSUMER": 0.40,
        "AGRICULTURE": 0.20, "FINANCE": 0.35,
    },
    "TAIWAN_STRAIT": {
        "SEMIS": 0.95, "TECH": 0.75, "AUTO": 0.60,
        "SHIPPING": 0.65, "CONSUMER": 0.50, "DEFENSE": -0.50,
        "FINANCE": 0.45,
    },
    "PANAMA": {
        "SHIPPING": 0.40, "CONSUMER": 0.25, "AGRICULTURE": 0.30,
        "ENERGY": 0.15, "AUTO": 0.20,
    },
    "BAB_EL_MANDEB": {
        "ENERGY": 0.45, "SHIPPING": 0.65, "CONSUMER": 0.30,
        "AUTO": 0.25, "PHARMA": 0.20,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_lane(lane_id: str) -> ShippingLane | None:
    """Return a shipping lane by ID."""
    return LANES_DATABASE.get(lane_id)


def get_lanes_through_chokepoint(chokepoint_id: str) -> list[ShippingLane]:
    """Return all lanes that pass through a given chokepoint."""
    ids = _CHOKEPOINT_TO_LANES.get(chokepoint_id, [])
    return [LANES_DATABASE[lid] for lid in ids if lid in LANES_DATABASE]


def compute_shipping_disruption_impact(
    chokepoint_id: str,
    graph: "DependencyGraph | None" = None,
) -> dict[str, float]:
    """Compute economic disruption impact of a chokepoint blockade.

    Returns a dict of {node_id_or_sector: impact_score 0-1}, combining
    the chokepoint's world-trade share with sector-specific sensitivities
    and (if graph provided) downstream cascade effects.
    """
    base_trade_share = CHOKEPOINT_WORLD_TRADE_SHARE.get(chokepoint_id, 0.05)
    sector_impacts = CHOKEPOINT_SECTOR_IMPACT.get(chokepoint_id, {})

    impact: dict[str, float] = {}

    # Sector impacts
    for sector, raw_impact in sector_impacts.items():
        impact[sector] = float(raw_impact) * base_trade_share * 3.0  # amplify by share
        impact[sector] = max(-1.0, min(1.0, impact[sector]))

    # Nation impacts from lanes
    lanes = get_lanes_through_chokepoint(chokepoint_id)
    for lane in lanes:
        for nation in lane.nations_dependent:
            reroute_penalty = (lane.reroute_cost_multiplier - 1.0) * base_trade_share
            impact[nation] = max(impact.get(nation, 0.0), min(reroute_penalty, 1.0))

    # Graph cascade if available
    if graph is not None:
        node = graph.get_node(chokepoint_id)
        if node is not None:
            cascade = graph.get_cascade_impact(chokepoint_id, max_hops=3)
            for nid, cascade_score in cascade.items():
                impact[nid] = max(impact.get(nid, 0.0), cascade_score * base_trade_share * 2)

    logger.debug(
        "[ShippingLanes] %s disruption: %d entities impacted, base_share=%.2f",
        chokepoint_id, len(impact), base_trade_share
    )
    return impact


def estimate_reroute_penalty(
    lane_id: str,
    disrupted_chokepoint: str,
) -> dict[str, float]:
    """Estimate the economic penalty of rerouting a lane around a disrupted chokepoint.

    Returns dict with cost_multiplier, time_days, insurance_surge_pct.
    """
    lane = LANES_DATABASE.get(lane_id)
    if lane is None or disrupted_chokepoint not in lane.chokepoints:
        return {"cost_multiplier": 1.0, "time_days": 0.0, "insurance_surge_pct": 0.0}

    return {
        "cost_multiplier": lane.reroute_cost_multiplier,
        "time_days": lane.reroute_time_days,
        "insurance_surge_pct": lane.insurance_surge_pct,
        "alternative_route": lane.alternative_route,
        "has_alternative": lane.alternative_route is not None,
    }


def compute_global_trade_exposure(disrupted_chokepoints: list[str]) -> float:
    """Estimate the fraction of global trade affected by simultaneous chokepoint disruptions.

    Uses set-union logic to avoid double-counting overlapping lanes.
    Returns 0.0 to 1.0.
    """
    if not disrupted_chokepoints:
        return 0.0

    affected_lanes: set[str] = set()
    for cp in disrupted_chokepoints:
        for lane_id in _CHOKEPOINT_TO_LANES.get(cp, []):
            affected_lanes.add(lane_id)

    # Sum daily traffic of affected lanes, capped at total world trade
    total_affected_value = sum(
        LANES_DATABASE[lid].daily_traffic_value_bn
        for lid in affected_lanes
        if lid in LANES_DATABASE
    )
    # Global daily maritime trade ~$15 trillion/year ≈ $41B/day
    GLOBAL_DAILY_TRADE_BN = 41.0
    return min(total_affected_value / GLOBAL_DAILY_TRADE_BN, 1.0)


def identify_nations_at_risk(chokepoint_id: str) -> list[tuple[str, float]]:
    """Return nations sorted by exposure to a chokepoint disruption, descending."""
    lanes = get_lanes_through_chokepoint(chokepoint_id)
    base_share = CHOKEPOINT_WORLD_TRADE_SHARE.get(chokepoint_id, 0.05)

    nation_scores: dict[str, float] = {}
    for lane in lanes:
        reroute_penalty = (lane.reroute_cost_multiplier - 1.0)
        for nation in lane.nations_dependent:
            score = reroute_penalty * base_share * 5.0
            nation_scores[nation] = max(nation_scores.get(nation, 0.0), min(score, 1.0))

    return sorted(nation_scores.items(), key=lambda x: x[1], reverse=True)


def simulate_simultaneous_disruption(chokepoints: list[str]) -> dict:
    """Simulate the combined impact of multiple simultaneous chokepoint disruptions.

    Returns summary with global_trade_exposure, most_impacted_sectors,
    most_impacted_nations, and estimated_oil_disruption_mbpd.
    """
    global_exposure = compute_global_trade_exposure(chokepoints)

    # Aggregate sector impacts
    combined_sectors: dict[str, float] = {}
    for cp in chokepoints:
        for sector, impact in CHOKEPOINT_SECTOR_IMPACT.get(cp, {}).items():
            combined_sectors[sector] = max(combined_sectors.get(sector, 0.0),
                                           abs(float(impact)))

    # Aggregate nation risks
    combined_nations: dict[str, float] = {}
    for cp in chokepoints:
        for nation, score in identify_nations_at_risk(cp):
            combined_nations[nation] = max(combined_nations.get(nation, 0.0), score)

    # Oil disruption
    affected_lanes: set[str] = set()
    for cp in chokepoints:
        for lid in _CHOKEPOINT_TO_LANES.get(cp, []):
            affected_lanes.add(lid)
    # Deduplicate oil (Hormuz and SCS overlap)
    oil_disrupted_mbpd = max(
        (LANES_DATABASE[lid].oil_transit_mbpd for lid in affected_lanes if lid in LANES_DATABASE),
        default=0.0,
    )

    top_sectors = sorted(combined_sectors.items(), key=lambda x: x[1], reverse=True)[:5]
    top_nations = sorted(combined_nations.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "chokepoints_disrupted": chokepoints,
        "global_trade_exposure": global_exposure,
        "oil_disruption_mbpd": oil_disrupted_mbpd,
        "most_impacted_sectors": top_sectors,
        "most_impacted_nations": top_nations,
        "severity_label": (
            "CATASTROPHIC" if global_exposure > 0.30 else
            "SEVERE" if global_exposure > 0.15 else
            "SIGNIFICANT" if global_exposure > 0.08 else
            "MODERATE"
        ),
    }
