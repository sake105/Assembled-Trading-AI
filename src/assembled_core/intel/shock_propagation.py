"""Shock propagation through the geopolitical dependency graph."""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from datetime import datetime, timedelta, timezone

from .dependency_graph import DependencyGraph
from .models import (
    DependencySignal,
    GeoTrigger,
    NodeType,
    ShockTransmission,
    ShockType,
    TransmissionHop,
    TriggerType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime-aware magnitude multipliers
# ---------------------------------------------------------------------------

REGIME_MAGNITUDE_MULTIPLIER: dict[str, float] = {
    "bull": 0.70,
    "reflation": 1.00,
    "sideways": 1.00,
    "bear": 1.10,
    "crisis": 1.30,
}

# Regime-specific confidence thresholds (lower in crisis = more sensitive)
REGIME_MIN_CONFIDENCE: dict[str, float] = {
    "bull": 0.20,
    "reflation": 0.18,
    "sideways": 0.15,
    "bear": 0.12,
    "crisis": 0.10,
}

# ---------------------------------------------------------------------------
# Trigger → Shock mapping (expanded)
# ---------------------------------------------------------------------------

TRIGGER_TO_SHOCKS: dict[TriggerType, list[ShockType]] = {
    TriggerType.CHOKEPOINT_STRESS: [
        ShockType.OIL_SUPPLY_RISK,
        ShockType.SHIPPING_COST_RISK,
        ShockType.INSURANCE_COST_RISK,
        ShockType.SHIPPING_LANE_DISRUPTION,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.WAR_ESCALATION: [
        ShockType.DEFENSE_DEMAND_SURGE,
        ShockType.GLOBAL_RISK_OFF,
        ShockType.OIL_SUPPLY_RISK,
        ShockType.MILITARY_LOSS_SURGE,
    ],
    TriggerType.ENERGY_SUPPLY_RISK: [
        ShockType.ENERGY_PRICE_SPIKE,
        ShockType.OIL_SUPPLY_RISK,
        ShockType.LNG_SUPPLY_RISK,
    ],
    TriggerType.SANCTIONS_ESCALATION: [
        ShockType.SANCTIONS_EXPOSURE,
        ShockType.BANKING_ISOLATION,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.CYBER_ESCALATION: [
        ShockType.CYBER_RISK,
        ShockType.DATA_BREACH_SYSTEMIC,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.SHIPPING_DISRUPTION: [
        ShockType.SHIPPING_COST_RISK,
        ShockType.INSURANCE_COST_RISK,
        ShockType.SHIPPING_LANE_DISRUPTION,
        ShockType.SUPPLY_CHAIN_BREAK,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.COUP_RISK: [
        ShockType.GLOBAL_RISK_OFF,
        ShockType.OIL_SUPPLY_RISK,
        ShockType.CAPITAL_FLIGHT,
    ],
    TriggerType.POLICY_SHIFT: [
        ShockType.GLOBAL_RISK_OFF,
        ShockType.SANCTIONS_EXPOSURE,
        ShockType.POLICY_DIVERGENCE,
    ],
    # New expanded triggers
    TriggerType.TRADE_WAR_ESCALATION: [
        ShockType.HEGEMONIC_DECOUPLING,
        ShockType.SEMICONDUCTOR_SUPPLY_RISK,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.STRAIT_BLOCKADE: [
        ShockType.OIL_SUPPLY_RISK,
        ShockType.SHIPPING_LANE_DISRUPTION,
        ShockType.INSURANCE_COST_RISK,
        ShockType.SHIPPING_COST_RISK,
    ],
    TriggerType.MILITARY_BUILDUP: [
        ShockType.DEFENSE_DEMAND_SURGE,
        ShockType.GLOBAL_RISK_OFF,
        ShockType.OIL_SUPPLY_RISK,
    ],
    TriggerType.NUCLEAR_THREAT: [
        ShockType.NUCLEAR_ESCALATION_RISK,
        ShockType.GLOBAL_RISK_OFF,
        ShockType.DEFENSE_DEMAND_SURGE,
    ],
    TriggerType.PEG_STRESS: [
        ShockType.CURRENCY_CRISIS,
        ShockType.CAPITAL_FLIGHT,
        ShockType.RESERVE_DEPLETION,
    ],
    TriggerType.RATE_SURPRISE: [
        ShockType.RATE_SHOCK,
        ShockType.TAPER_SHOCK,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.FISCAL_CLIFF: [
        ShockType.FISCAL_SHOCK,
        ShockType.GLOBAL_RISK_OFF,
        ShockType.SOVEREIGN_DEFAULT,
    ],
    TriggerType.CREDIT_DOWNGRADE: [
        ShockType.FISCAL_SHOCK,
        ShockType.CAPITAL_FLIGHT,
        ShockType.CURRENCY_CRISIS,
    ],
    TriggerType.NEW_EXPORT_CONTROL: [
        ShockType.TECH_RESTRICTION_SHOCK,
        ShockType.SEMICONDUCTOR_SUPPLY_RISK,
        ShockType.CHIP_SHORTAGE,
    ],
    TriggerType.ENTITY_LISTING: [
        ShockType.TECH_RESTRICTION_SHOCK,
        ShockType.SANCTIONS_EXPOSURE,
        ShockType.SECONDARY_SANCTIONS_RISK,
    ],
    TriggerType.ZERO_DAY_DISCLOSURE: [
        ShockType.CYBER_RISK,
        ShockType.DATA_BREACH_SYSTEMIC,
        ShockType.FINANCIAL_SYSTEM_STRESS,
    ],
    TriggerType.SEVERE_WEATHER_ALERT: [
        ShockType.CLIMATE_DISRUPTION,
        ShockType.PORT_CLOSURE,
        ShockType.CROP_FAILURE,
        ShockType.SUPPLY_CHAIN_BREAK,
    ],
    TriggerType.TERRITORIAL_ESCALATION: [
        ShockType.GLOBAL_RISK_OFF,
        ShockType.DEFENSE_DEMAND_SURGE,
        ShockType.OIL_SUPPLY_RISK,
    ],
    TriggerType.PROXY_WAR_EXPANSION: [
        ShockType.DEFENSE_DEMAND_SURGE,
        ShockType.GLOBAL_RISK_OFF,
        ShockType.OIL_SUPPLY_RISK,
        ShockType.SUPPLY_LINE_THREAT,
    ],
    TriggerType.HEGEMONIC_CHALLENGE: [
        ShockType.HEGEMONIC_DECOUPLING,
        ShockType.ALLIANCE_SHIFT,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.RESOURCE_NATIONALIZATION: [
        ShockType.RARE_EARTH_SUPPLY_RISK,
        ShockType.OIL_SUPPLY_RISK,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.RESERVE_DRAIN: [
        ShockType.RESERVE_DEPLETION,
        ShockType.CURRENCY_CRISIS,
        ShockType.CAPITAL_FLIGHT,
    ],
    TriggerType.LOGISTICS_DISRUPTION: [
        ShockType.SUPPLY_CHAIN_BREAK,
        ShockType.SHIPPING_COST_RISK,
        ShockType.INFLATION_SPIKE,
    ],
}

# ---------------------------------------------------------------------------
# Shock → entry-node mapping (expanded)
# ---------------------------------------------------------------------------

SHOCK_TO_ORIGIN_NODES: dict[ShockType, list[str]] = {
    ShockType.OIL_SUPPLY_RISK: ["HORMUZ", "OIL", "SAUDI_ARABIA"],
    ShockType.SHIPPING_COST_RISK: ["SUEZ", "HORMUZ", "MALACCA"],
    ShockType.INSURANCE_COST_RISK: ["SUEZ", "HORMUZ", "BAB_EL_MANDEB"],
    ShockType.GLOBAL_RISK_OFF: ["US_EQUITIES", "GOLD"],
    ShockType.ENERGY_PRICE_SPIKE: ["OIL", "LNG", "ENERGY"],
    ShockType.DEFENSE_DEMAND_SURGE: ["UKRAINE_CONFLICT", "DEFENSE"],
    ShockType.SANCTIONS_EXPOSURE: ["RUSSIA", "IRAN"],
    ShockType.CYBER_RISK: ["CYBER"],
    # New shocks
    ShockType.RARE_EARTH_SUPPLY_RISK: ["RARE_EARTHS", "CHINA"],
    ShockType.SEMICONDUCTOR_SUPPLY_RISK: ["SEMICONDUCTORS", "TSMC_FABRICATION", "TAIWAN"],
    ShockType.FOOD_SUPPLY_RISK: ["WHEAT", "CORN", "UKRAINE_CONFLICT"],
    ShockType.LITHIUM_SUPPLY_RISK: ["LITHIUM", "EV_BATTERY_CHAIN"],
    ShockType.LNG_SUPPLY_RISK: ["LNG", "RUSSIA", "QATAR"],
    ShockType.CURRENCY_CRISIS: ["TRY", "ARS", "RUB"],
    ShockType.RESERVE_DEPLETION: ["TURKEY", "ARGENTINA"],
    ShockType.CAPITAL_FLIGHT: ["RUSSIA", "TURKEY"],
    ShockType.FISCAL_SHOCK: ["US_DEBT_CEILING", "EU_FISCAL_RULES"],
    ShockType.INFLATION_SPIKE: ["OIL", "WHEAT", "LNG"],
    ShockType.RATE_SHOCK: ["FED", "ECB"],
    ShockType.POLICY_DIVERGENCE: ["FED", "BOJ"],
    ShockType.TAPER_SHOCK: ["FED", "ECB"],
    ShockType.SOVEREIGN_DEFAULT: ["ARGENTINA", "US_DEBT_CEILING"],
    ShockType.HEGEMONIC_DECOUPLING: ["CHINA", "US"],
    ShockType.SHIPPING_LANE_DISRUPTION: ["BAB_EL_MANDEB", "SUEZ", "HORMUZ"],
    ShockType.SECONDARY_SANCTIONS_RISK: ["RUSSIA", "IRAN"],
    ShockType.BANKING_ISOLATION: ["RUSSIA", "IRAN"],
    ShockType.MILITARY_LOSS_SURGE: ["UKRAINE_CONFLICT"],
    ShockType.SUPPLY_LINE_THREAT: ["UKRAINE_CONFLICT", "TAIWAN_TENSION"],
    ShockType.NUCLEAR_ESCALATION_RISK: ["UKRAINE_CONFLICT", "IRAN"],
    ShockType.DATA_BREACH_SYSTEMIC: ["CYBER"],
    ShockType.LOGISTICS_VISIBILITY_LOSS: ["BAB_EL_MANDEB"],
    ShockType.FINANCIAL_SYSTEM_STRESS: ["GLOBAL_SWIFT", "INTERNET_BACKBONE"],
    ShockType.CLIMATE_DISRUPTION: ["UKRAINE_CONFLICT"],  # No direct node; use proxy
    ShockType.PORT_CLOSURE: ["BAB_EL_MANDEB", "SUEZ"],
    ShockType.CROP_FAILURE: ["WHEAT", "CORN"],
    ShockType.SUPPLY_CHAIN_BREAK: ["MALACCA", "TAIWAN_STRAIT"],
    ShockType.TECH_RESTRICTION_SHOCK: ["CHIPS_ACT", "ENTITY_LIST"],
    ShockType.CHIP_SHORTAGE: ["TSMC_FABRICATION", "SEMICONDUCTORS"],
    ShockType.INNOVATION_GAP: ["CHINA", "RUSSIA"],
    ShockType.ALLIANCE_SHIFT: ["NATO", "BRICS"],
    ShockType.DELISTING_RISK: ["CHINA"],
}

# Shocks that increase asset value for their origin sector (beneficiaries rise).
# NOTE: ENERGY_PRICE_SPIKE is negative for consumers/industry but positive for
# energy producers — handled via per-sector impact direction in the graph edges,
# not here. CYBER_RISK is universally negative.
POSITIVE_SHOCKS: set[ShockType] = {
    ShockType.DEFENSE_DEMAND_SURGE,
}

# Confidence threshold: paths below this are excluded (default; overridden by regime)
MIN_PATH_CONFIDENCE = 0.15

# Dampening factor per hop (each edge traversal decays magnitude by this factor)
DEFAULT_DAMPENING_FACTOR = 0.85

# Lag hours → time horizon mapping
def _lag_to_horizon(lag_hours: float) -> str:
    if lag_hours <= 8:
        return "intraday"
    elif lag_hours <= 72:
        return "short"
    else:
        return "medium"


def _shock_id(origin_trigger_id: str, shock_type: ShockType) -> str:
    raw = f"{origin_trigger_id}:{shock_type.value}"
    return "shock_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_trigger_to_shocks(trigger: GeoTrigger) -> list[ShockType]:
    """Map a geo trigger to its associated shock types."""
    return list(TRIGGER_TO_SHOCKS.get(trigger.trigger_type, []))


def propagate(
    shocks: list[ShockType],
    graph: DependencyGraph,
    trigger_id: str = "unknown",
    max_hops: int = 3,
    magnitude: float = 1.0,
    dampening_factor: float = DEFAULT_DAMPENING_FACTOR,
    regime: str = "sideways",
) -> list[ShockTransmission]:
    """
    Propagate shocks through the dependency graph using BFS.

    Args:
        shocks: Shock types to propagate.
        graph: The geopolitical dependency graph.
        trigger_id: Originating trigger identifier.
        max_hops: Maximum graph hops to follow.
        magnitude: Initial shock magnitude (1.0 = standard, 2.0 = double-strength).
        dampening_factor: Per-hop magnitude decay (default 0.85 = 15% loss per hop).
        regime: Market regime for sensitivity scaling
                ("bull", "bear", "crisis", "reflation", "sideways").

    Returns:
        List of ShockTransmission objects (one per shock that found a path).
    """
    regime_multiplier = REGIME_MAGNITUDE_MULTIPLIER.get(regime, 1.0)
    min_confidence = REGIME_MIN_CONFIDENCE.get(regime, MIN_PATH_CONFIDENCE)
    effective_magnitude = magnitude * regime_multiplier

    transmissions: list[ShockTransmission] = []

    for shock in shocks:
        origin_node_ids = SHOCK_TO_ORIGIN_NODES.get(shock, [])
        is_positive_shock = shock in POSITIVE_SHOCKS

        # BFS: (node_id, hops, path_confidence, visited, total_lag, current_magnitude)
        all_paths: list[tuple[list[TransmissionHop], float, int, float]] = []

        for origin_id in origin_node_ids:
            origin_node = graph.get_node(origin_id)
            if origin_node is None:
                continue

            queue: deque[tuple[str, list[TransmissionHop], float, set[str], int, float]] = deque()
            initial_hop = TransmissionHop(
                node_id=origin_id,
                node_type=origin_node.node_type.value,
                impact_direction="+" if is_positive_shock else "-",
                weight=1.0,
            )
            queue.append((origin_id, [initial_hop], 1.0, {origin_id}, 0, effective_magnitude))

            while queue:
                current_id, hops, path_conf, visited, total_lag, current_mag = queue.popleft()

                current_node = graph.get_node(current_id)
                if current_node and current_node.node_type in {
                    NodeType.SECTOR, NodeType.ASSET, NodeType.MACRO_INDEX
                } and len(hops) > 1:
                    if path_conf >= min_confidence:
                        all_paths.append((list(hops), path_conf, total_lag, current_mag))

                if len(hops) > max_hops:
                    continue

                for edge, neighbor in graph.get_neighbors(current_id):
                    if neighbor.node_id in visited:
                        continue
                    new_conf = path_conf * edge.weight * edge.confidence
                    if new_conf < min_confidence:
                        continue

                    if is_positive_shock:
                        impact_dir = edge.direction
                    else:
                        impact_dir = "-" if edge.direction == "+" else "+"

                    # Apply per-hop dampening to magnitude
                    new_mag = current_mag * dampening_factor

                    new_hop = TransmissionHop(
                        node_id=neighbor.node_id,
                        node_type=neighbor.node_type.value,
                        impact_direction=impact_dir,
                        weight=edge.weight,
                    )
                    queue.append((
                        neighbor.node_id,
                        hops + [new_hop],
                        new_conf,
                        visited | {neighbor.node_id},
                        total_lag + edge.lag_hours,
                        new_mag,
                    ))

        if not all_paths:
            continue

        best_path, best_conf, best_lag, best_mag = max(all_paths, key=lambda x: x[1])

        transmission = ShockTransmission(
            shock_id=_shock_id(trigger_id, shock),
            origin_trigger_id=trigger_id,
            shock_type=shock,
            path=best_path,
            expected_impact_direction="+" if is_positive_shock else "-",
            expected_horizon=_lag_to_horizon(best_lag),
            path_confidence=best_conf,
            magnitude=effective_magnitude,
            dampened_magnitude=best_mag,
            time_to_impact_days=best_lag / 24.0,
        )
        transmissions.append(transmission)

    logger.debug(
        "[ShockPropagation] trigger=%s regime=%s magnitude=%.2f → %d transmissions",
        trigger_id, regime, effective_magnitude, len(transmissions),
    )
    return transmissions


def to_dependency_signal(
    transmissions: list[ShockTransmission],
    trigger_id: str,
    trigger_score: int = 1,
    ttl_hours: int = 6,
    now: datetime | None = None,
) -> DependencySignal:
    """
    Convert shock transmissions into a tradeable DependencySignal.

    Aggregates beneficiaries (assets expected to benefit) and losers
    (assets expected to decline) from all transmissions.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    beneficiaries: dict[str, float] = {}  # asset -> best confidence
    losers: dict[str, float] = {}

    for transmission in transmissions:
        # Walk the path and collect terminal asset nodes
        terminal_hop = transmission.path[-1] if transmission.path else None
        if terminal_hop is None:
            continue

        node_id = terminal_hop.node_id
        impact_dir = terminal_hop.impact_direction
        conf = transmission.path_confidence

        if impact_dir == "+":
            beneficiaries[node_id] = max(beneficiaries.get(node_id, 0.0), conf)
        else:
            losers[node_id] = max(losers.get(node_id, 0.0), conf)

    # Remove overlap: if a node is in both, keep only the higher-confidence bucket
    overlap = set(beneficiaries) & set(losers)
    for node_id in overlap:
        b_conf = beneficiaries[node_id]
        l_conf = losers[node_id]
        if b_conf >= l_conf:
            del losers[node_id]
        else:
            del beneficiaries[node_id]

    # Sort by confidence descending
    sorted_beneficiaries = sorted(beneficiaries, key=lambda n: beneficiaries[n], reverse=True)
    sorted_losers = sorted(losers, key=lambda n: losers[n], reverse=True)

    # Compute overall severity (0-3) and confidence
    all_confs = list(beneficiaries.values()) + list(losers.values())
    overall_conf = max(all_confs) if all_confs else 0.0

    # Scale severity by average magnitude across transmissions
    avg_magnitude = (
        sum(t.magnitude for t in transmissions) / len(transmissions)
        if transmissions else 1.0
    )
    severity = min(3, round(trigger_score * min(avg_magnitude, 2.0)))

    # Compute predominant horizon
    horizons = [t.expected_horizon for t in transmissions]
    if "intraday" in horizons:
        horizon = "intraday"
    elif "short" in horizons:
        horizon = "short"
    else:
        horizon = "medium"

    signal_id = "sig_" + hashlib.sha256(
        f"{trigger_id}:{now.isoformat()}".encode()
    ).hexdigest()[:16]

    return DependencySignal(
        signal_id=signal_id,
        trigger_id=trigger_id,
        beneficiaries=sorted_beneficiaries,
        losers=sorted_losers,
        severity=severity,
        confidence=overall_conf,
        time_horizon=horizon,
        ttl_expires_ts=now + timedelta(hours=ttl_hours),
        basket_overrides={
            "prefer": sorted_beneficiaries[:3],
            "avoid": sorted_losers[:3],
        },
    )
