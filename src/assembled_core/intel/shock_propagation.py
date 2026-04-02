"""Shock propagation through the geopolitical dependency graph."""

from __future__ import annotations

import hashlib
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

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

# ---------------------------------------------------------------------------
# Trigger → Shock mapping
# ---------------------------------------------------------------------------

TRIGGER_TO_SHOCKS: dict[TriggerType, list[ShockType]] = {
    TriggerType.CHOKEPOINT_STRESS: [
        ShockType.OIL_SUPPLY_RISK,
        ShockType.SHIPPING_COST_RISK,
        ShockType.INSURANCE_COST_RISK,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.WAR_ESCALATION: [
        ShockType.DEFENSE_DEMAND_SURGE,
        ShockType.GLOBAL_RISK_OFF,
        ShockType.OIL_SUPPLY_RISK,
    ],
    TriggerType.ENERGY_SUPPLY_RISK: [
        ShockType.ENERGY_PRICE_SPIKE,
        ShockType.OIL_SUPPLY_RISK,
    ],
    TriggerType.SANCTIONS_ESCALATION: [
        ShockType.SANCTIONS_EXPOSURE,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.CYBER_ESCALATION: [
        ShockType.CYBER_RISK,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.SHIPPING_DISRUPTION: [
        ShockType.SHIPPING_COST_RISK,
        ShockType.INSURANCE_COST_RISK,
        ShockType.GLOBAL_RISK_OFF,
    ],
    TriggerType.COUP_RISK: [
        ShockType.GLOBAL_RISK_OFF,
        ShockType.OIL_SUPPLY_RISK,
    ],
    TriggerType.POLICY_SHIFT: [
        ShockType.GLOBAL_RISK_OFF,
        ShockType.SANCTIONS_EXPOSURE,
    ],
}

# ---------------------------------------------------------------------------
# Shock → entry-node mapping (which graph nodes are the "origin" for each shock)
# ---------------------------------------------------------------------------

SHOCK_TO_ORIGIN_NODES: dict[ShockType, list[str]] = {
    ShockType.OIL_SUPPLY_RISK: ["HORMUZ", "GLOBAL_OIL"],
    ShockType.SHIPPING_COST_RISK: ["SUEZ", "HORMUZ"],
    ShockType.INSURANCE_COST_RISK: ["SUEZ", "HORMUZ"],
    ShockType.GLOBAL_RISK_OFF: ["US_EQUITIES", "GOLD"],
    ShockType.ENERGY_PRICE_SPIKE: ["GLOBAL_OIL", "GLOBAL_LNG", "ENERGY_SECTOR"],
    ShockType.DEFENSE_DEMAND_SURGE: ["WAR_ESCALATION_EVENT", "DEFENSE_SECTOR"],
    ShockType.SANCTIONS_EXPOSURE: ["SANCTIONS", "CHINA_MANUFACTURING"],
    ShockType.CYBER_RISK: ["CYBER_ESCALATION", "CYBER_SECTOR"],
}

# Shocks that increase asset value (as opposed to causing risk-off / decrease)
POSITIVE_SHOCKS: set[ShockType] = {
    ShockType.DEFENSE_DEMAND_SURGE,
    ShockType.ENERGY_PRICE_SPIKE,
    ShockType.CYBER_RISK,
}

# Confidence threshold: paths below this are excluded
MIN_PATH_CONFIDENCE = 0.15

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
) -> list[ShockTransmission]:
    """
    Propagate shocks through the dependency graph using BFS.

    Returns a list of ShockTransmission, one per shock type, describing
    the paths from origin nodes to tradeable asset nodes.
    """
    transmissions: list[ShockTransmission] = []

    for shock in shocks:
        origin_node_ids = SHOCK_TO_ORIGIN_NODES.get(shock, [])
        # Determine overall impact direction for this shock
        is_positive_shock = shock in POSITIVE_SHOCKS

        # BFS from each origin node
        # State: (current_node_id, path_of_hops, accumulated_confidence)
        all_paths: list[tuple[list[TransmissionHop], float, int]] = []  # (hops, confidence, max_lag)

        for origin_id in origin_node_ids:
            origin_node = graph.get_node(origin_id)
            if origin_node is None:
                continue

            # BFS queue: (node_id, hops_so_far, path_confidence, visited, total_lag)
            queue: deque[tuple[str, list[TransmissionHop], float, set[str], int]] = deque()
            initial_hop = TransmissionHop(
                node_id=origin_id,
                node_type=origin_node.node_type.value,
                impact_direction="+" if is_positive_shock else "-",
                weight=1.0,
            )
            queue.append((origin_id, [initial_hop], 1.0, {origin_id}, 0))

            while queue:
                current_id, hops, path_conf, visited, total_lag = queue.popleft()

                # Check if current node is a tradeable asset
                current_node = graph.get_node(current_id)
                if current_node and current_node.node_type in {
                    NodeType.SECTOR, NodeType.ASSET, NodeType.MACRO_INDEX
                } and len(hops) > 1:
                    if path_conf >= MIN_PATH_CONFIDENCE:
                        all_paths.append((list(hops), path_conf, total_lag))

                # Don't go deeper than max_hops
                if len(hops) > max_hops:
                    continue

                for edge, neighbor in graph.get_neighbors(current_id):
                    if neighbor.node_id in visited:
                        continue
                    new_conf = path_conf * edge.weight * edge.confidence
                    if new_conf < MIN_PATH_CONFIDENCE:
                        continue

                    # Determine impact direction through this edge
                    # Edge direction "+" means the from_node stress increases the to_node
                    # Edge direction "-" means the from_node stress decreases the to_node
                    if is_positive_shock:
                        impact_dir = edge.direction  # "+" shock propagates as-is
                    else:
                        # Negative shock: invert if edge direction is "+", keep "-" if "-"
                        impact_dir = "-" if edge.direction == "+" else "+"

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
                    ))

        if not all_paths:
            continue

        # Pick the highest-confidence path for this shock
        best_path, best_conf, best_lag = max(all_paths, key=lambda x: x[1])

        transmission = ShockTransmission(
            shock_id=_shock_id(trigger_id, shock),
            origin_trigger_id=trigger_id,
            shock_type=shock,
            path=best_path,
            expected_impact_direction="+" if is_positive_shock else "-",
            expected_horizon=_lag_to_horizon(best_lag),
            path_confidence=best_conf,
        )
        transmissions.append(transmission)

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
    severity = min(3, trigger_score)

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
