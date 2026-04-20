"""News-to-position bridge (Point 33).

Translates EvidenceCluster / NewsClassification signals into actionable
position signals (direction, confidence, affected assets, time horizon).

This is a pure signal layer — it does NOT manage positions or send orders.
Downstream consumers (e.g. crisis_alpha_worker, portfolio optimizer) decide
whether to act.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Minimum confidence to emit a non-neutral signal
_MIN_CONFIDENCE = 0.3

# Event types that produce short (bearish) signals by default
_BEARISH_EVENT_TYPES = frozenset({
    "war_escalation",
    "military_strike",
    "sanctions",
    "energy_disruption",
    "political_crisis",
    "market_stress",
    "cyber_attack",
    "natural_disaster",
})

# Event types that produce long (bullish) signals by default
_BULLISH_EVENT_TYPES = frozenset({
    "diplomatic",   # peace talks / ceasefire
    "ma_activity",  # M&A premium
    "earnings",     # positive earnings surprise context
})

# Sectors with known inverse reaction to geo events
_CRISIS_LONG_SECTORS = frozenset({"defense", "materials", "energy"})
_CRISIS_SHORT_SECTORS = frozenset({"consumer", "tech", "industrials"})


@dataclass
class PositionSignal:
    """A single news-derived position signal."""
    signal_id: str
    source_cluster_id: str | None
    direction: str  # "long" / "short" / "flat"
    confidence: float  # 0-1
    affected_assets: list[str] = field(default_factory=list)
    affected_sectors: list[str] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    time_horizon: str = "short"  # "intraday" / "short" / "medium" / "long"
    severity: float = 0.0
    market_direction: str = "neutral"
    generated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    rationale: str = ""

    def is_actionable(self) -> bool:
        return self.direction != "flat" and self.confidence >= _MIN_CONFIDENCE


def cluster_to_signal(cluster: object) -> PositionSignal | None:
    """Convert an EvidenceCluster to a PositionSignal.

    Returns None if the cluster is below minimum confidence or has no
    meaningful direction.

    Args:
        cluster: EvidenceCluster with .cluster_id, .trigger_type,
                 .confidence, .supporting_events attributes.
    """
    cluster_id: str = getattr(cluster, "cluster_id", "unknown")
    trigger_type: str = getattr(cluster, "trigger_type", None)
    if trigger_type is not None:
        trigger_type = getattr(trigger_type, "value", str(trigger_type))
    confidence: float = float(getattr(cluster, "confidence", 0.0))

    if confidence < _MIN_CONFIDENCE:
        return None

    direction = _trigger_to_direction(trigger_type, confidence)

    return PositionSignal(
        signal_id=f"ps_{cluster_id}",
        source_cluster_id=cluster_id,
        direction=direction,
        confidence=confidence,
        event_types=[trigger_type] if trigger_type else [],
        time_horizon="intraday" if trigger_type in ("military_strike", "war_escalation") else "short",
        rationale=f"cluster={cluster_id} trigger={trigger_type} conf={confidence:.2f}",
    )


def classification_to_signal(
    classification: object,
    cluster_id: str | None = None,
) -> PositionSignal | None:
    """Convert a NewsClassification to a PositionSignal.

    Args:
        classification: NewsClassification with .event_types, .severity,
                        .market_direction, .affected_sectors, .affected_assets,
                        .confidence, .time_horizon.
        cluster_id: Optional parent cluster id for tracing.
    """
    confidence: float = float(getattr(classification, "confidence", 0.0))
    if confidence < _MIN_CONFIDENCE:
        return None

    event_types: list[str] = list(getattr(classification, "event_types", []))
    market_direction: str = getattr(classification, "market_direction", "neutral")
    affected_sectors: list[str] = list(getattr(classification, "affected_sectors", []))
    affected_assets: list[str] = list(getattr(classification, "affected_assets", []))
    time_horizon: str = getattr(classification, "time_horizon", "short")
    severity: float = float(getattr(classification, "severity", 0.0))

    # Derive direction from market_direction + event type hints
    direction = _derive_direction(market_direction, event_types, affected_sectors, confidence)

    # Boost assets based on crisis sector logic
    if direction == "short":
        for sector in affected_sectors:
            if sector in _CRISIS_SHORT_SECTORS:
                pass  # already in affected_assets via classifier
    elif direction == "long":
        for sector in affected_sectors:
            if sector in _CRISIS_LONG_SECTORS:
                pass

    signal_id = f"ps_{cluster_id or 'cls'}_{abs(hash(str(event_types)))%10000:04d}"

    return PositionSignal(
        signal_id=signal_id,
        source_cluster_id=cluster_id,
        direction=direction,
        confidence=confidence,
        affected_assets=affected_assets,
        affected_sectors=affected_sectors,
        event_types=event_types,
        time_horizon=time_horizon,
        severity=severity,
        market_direction=market_direction,
        rationale=(
            f"types={event_types[:3]} dir={market_direction} "
            f"sev={severity:.1f} conf={confidence:.2f}"
        ),
    )


def signals_to_basket(
    signals: list[PositionSignal],
    *,
    min_confidence: float = _MIN_CONFIDENCE,
    max_assets: int = 20,
) -> dict[str, float]:
    """Aggregate multiple PositionSignals into a direction-weighted asset basket.

    Returns:
        dict mapping asset ticker → net direction score (-1=full short, +1=full long).
        Score = sum of (conf * direction_sign) across all signals that mention the asset.
    """
    basket: dict[str, float] = {}
    for signal in signals:
        if signal.confidence < min_confidence or signal.direction == "flat":
            continue
        sign = 1.0 if signal.direction == "long" else -1.0
        weight = signal.confidence * sign
        for asset in signal.affected_assets:
            basket[asset] = basket.get(asset, 0.0) + weight

    # Normalise to [-1, +1]
    if basket:
        max_abs = max(abs(v) for v in basket.values())
        if max_abs > 0:
            basket = {k: round(v / max_abs, 4) for k, v in basket.items()}

    # Trim to top N by absolute score
    if len(basket) > max_assets:
        sorted_items = sorted(basket.items(), key=lambda x: abs(x[1]), reverse=True)
        basket = dict(sorted_items[:max_assets])

    return basket


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------


def _trigger_to_direction(trigger_type: str | None, confidence: float) -> str:
    if trigger_type in _BEARISH_EVENT_TYPES:
        return "short"
    if trigger_type in _BULLISH_EVENT_TYPES:
        return "long"
    return "flat"


def _derive_direction(
    market_direction: str,
    event_types: list[str],
    affected_sectors: list[str],
    confidence: float,
) -> str:
    if market_direction == "bearish":
        return "short"
    if market_direction == "bullish":
        return "long"
    if market_direction == "mixed":
        # Mixed: check event type weights
        bearish_count = sum(1 for et in event_types if et in _BEARISH_EVENT_TYPES)
        bullish_count = sum(1 for et in event_types if et in _BULLISH_EVENT_TYPES)
        if bearish_count > bullish_count:
            return "short"
        if bullish_count > bearish_count:
            return "long"
        return "flat"
    # neutral: check event type hints
    for et in event_types:
        if et in _BEARISH_EVENT_TYPES:
            return "short"
        if et in _BULLISH_EVENT_TYPES:
            return "long"
    return "flat"


__all__ = [
    "PositionSignal",
    "cluster_to_signal",
    "classification_to_signal",
    "signals_to_basket",
]
