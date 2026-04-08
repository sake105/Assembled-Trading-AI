"""Escalation level tracker for active geopolitical conflicts.

Tracks escalation levels over time, detects acceleration, and provides
market impact forecasts based on escalation trajectories.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .escalation_model import ACTIVE_CONFLICTS, compute_market_impact_by_level

logger = logging.getLogger(__name__)


@dataclass
class EscalationEvent:
    """A recorded escalation event."""

    event_id: str
    conflict_id: str
    trigger_type: str
    level_before: int
    level_after: int
    timestamp: datetime
    description: str = ""
    source_tier: str = "T2"


@dataclass
class ConflictTracker:
    """Tracks escalation history for a single conflict."""

    conflict_id: str
    history: deque = field(default_factory=lambda: deque(maxlen=50))
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def record_event(self, event: EscalationEvent) -> None:
        self.history.append(event)
        self.last_updated = event.timestamp

    def get_recent_delta(self, window_days: int = 7) -> float:
        """Return average level change per day over the window."""
        now = datetime.utcnow()
        cutoff = now - timedelta(days=window_days)
        recent = [e for e in self.history if e.timestamp >= cutoff]
        if len(recent) < 2:
            return 0.0
        total_delta = sum(e.level_after - e.level_before for e in recent)
        return total_delta / window_days


# Global registry of conflict trackers
_TRACKERS: dict[str, ConflictTracker] = {
    cid: ConflictTracker(conflict_id=cid)
    for cid in ACTIVE_CONFLICTS
}


def get_tracker(conflict_id: str) -> ConflictTracker | None:
    """Return tracker for a conflict."""
    return _TRACKERS.get(conflict_id)


def update_conflict_level(
    conflict_id: str,
    new_events: list[dict],
) -> int:
    """Update the escalation level of a conflict based on new events.

    Args:
        conflict_id: Conflict identifier
        new_events: List of event dicts with keys: trigger_type, description, source_tier

    Returns:
        New escalation level
    """
    conflict = ACTIVE_CONFLICTS.get(conflict_id)
    tracker = _TRACKERS.get(conflict_id)

    if conflict is None or tracker is None:
        logger.warning("[EscalationTracker] Unknown conflict: %s", conflict_id)
        return 0

    current_level = conflict.current_level

    for raw_event in new_events:
        trigger_type = raw_event.get("trigger_type", "")
        source_tier = raw_event.get("source_tier", "T2")

        # Escalation triggers mapped to level changes
        _escalation_delta = {
            "MILITARY_BUILDUP": +1,
            "TERRITORIAL_ESCALATION": +1,
            "NUCLEAR_THREAT": +2,
            "CAPABILITY_SHIFT": +1,
            "CASUALTY_SPIKE": +1,
            "STRAIT_BLOCKADE": +1,
            "PROXY_WAR_EXPANSION": +1,
            "HEGEMONIC_CHALLENGE": +0,  # No level change but records tension
            "DIPLOMATIC_CRISIS": +0,
            "WAR_ESCALATION": +1,
            # De-escalation triggers
            "PEACE_NEGOTIATIONS": -1,
            "CEASEFIRE": -1,
            "SANCTIONS_RELIEF": -1,
        }
        delta = _escalation_delta.get(trigger_type, 0)

        # T0/T1 sources → full delta, T2 → 50%, T3 → no update
        tier_multiplier = {"T0": 1.0, "T1": 1.0, "T2": 0.5, "T3": 0.0}.get(source_tier, 0.5)
        effective_delta = round(delta * tier_multiplier)

        new_level = max(0, min(10, current_level + effective_delta))

        if new_level != current_level:
            event = EscalationEvent(
                event_id=f"{conflict_id}_{datetime.utcnow().timestamp():.0f}",
                conflict_id=conflict_id,
                trigger_type=trigger_type,
                level_before=current_level,
                level_after=new_level,
                timestamp=datetime.utcnow(),
                description=raw_event.get("description", ""),
                source_tier=source_tier,
            )
            tracker.record_event(event)
            conflict.current_level = new_level
            current_level = new_level

            logger.info(
                "[EscalationTracker] %s: level %d → %d (trigger=%s, tier=%s)",
                conflict_id, event.level_before, new_level, trigger_type, source_tier
            )

    return current_level


def detect_escalation_acceleration(conflict_id: str) -> float:
    """Detect if escalation is accelerating (second derivative positive).

    Returns:
        Positive value = accelerating escalation
        Negative value = de-escalating
        0.0 = stable
    """
    tracker = _TRACKERS.get(conflict_id)
    if tracker is None:
        return 0.0

    # Short-term vs long-term delta
    recent_7d = tracker.get_recent_delta(window_days=7)
    recent_30d = tracker.get_recent_delta(window_days=30)

    # Acceleration = recent rate faster than longer-term rate
    return recent_7d - recent_30d


def compute_deescalation_probability(conflict_id: str) -> float:
    """Estimate probability of de-escalation in next 7 days.

    Based on: current level, recent trend, conflict type.
    """
    conflict = ACTIVE_CONFLICTS.get(conflict_id)
    if conflict is None:
        return 0.0

    level = conflict.current_level
    acceleration = detect_escalation_acceleration(conflict_id)

    # Higher level = harder to de-escalate
    base_prob = max(0, 0.5 - level * 0.04)

    # Accelerating = less likely to de-escalate
    if acceleration > 0:
        base_prob *= 0.6
    elif acceleration < 0:
        base_prob *= 1.5

    return min(max(base_prob, 0.02), 0.60)


def get_market_impact_forecast(
    conflict_id: str,
    horizon_days: int = 30,
) -> dict:
    """Forecast market impact based on escalation trend.

    Returns expected sector impacts weighted by escalation probability.
    """
    conflict = ACTIVE_CONFLICTS.get(conflict_id)
    if conflict is None:
        return {}

    # Current level impact
    current_impact = compute_market_impact_by_level(conflict, conflict.current_level)

    # Probability-weighted escalation impact
    expected_impact: dict[str, float] = dict(current_impact)

    for level, prob in conflict.escalation_probability.items():
        higher_impact = compute_market_impact_by_level(conflict, level)
        for sector, impact in higher_impact.items():
            expected_impact[sector] = (
                expected_impact.get(sector, 0) + prob * impact * 0.3  # Discount future
            )

    # De-escalation probability reduces expected impact
    deesc_prob = compute_deescalation_probability(conflict_id)
    for sector in expected_impact:
        expected_impact[sector] = expected_impact[sector] * (1 - deesc_prob * 0.5)

    return {
        "conflict_id": conflict_id,
        "current_level": conflict.current_level,
        "horizon_days": horizon_days,
        "deescalation_probability": round(deesc_prob, 3),
        "sector_impacts": {
            sector: round(impact, 3)
            for sector, impact in sorted(expected_impact.items(),
                                          key=lambda x: abs(x[1]), reverse=True)
        },
    }
