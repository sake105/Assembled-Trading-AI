"""News-to-position bridge (Point 33).

Translates EvidenceCluster / NewsClassification signals into actionable
position signals (direction, confidence, affected assets, time horizon).

This is a pure signal layer — it does NOT manage positions or send orders.
Downstream consumers (e.g. crisis_alpha_worker, portfolio optimizer) decide
whether to act.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Minimum confidence to emit a non-neutral signal
_MIN_CONFIDENCE = 0.3

# Event types that produce short (bearish) signals by default.
# Must stay in sync with news_classifier._EVENT_KEYWORDS. When the classifier
# gains a new bearish event type, add it here AND update
# tests/test_news_position_bridge.py.
_BEARISH_EVENT_TYPES = frozenset({
    "war_escalation",
    "military_strike",
    "sanctions",
    "energy_disruption",
    "political_crisis",
    "market_stress",
    "cyber_attack",
    "natural_disaster",
    # K7: previously silent (fell through to "flat") — now wired.
    "trade_policy",     # tariffs / trade war typically bearish for risk
    "regulatory",       # investigations / fines are bearish on average
    # Gap fill: capital-structure / labour signals
    "layoffs",          # mass job cuts usually bearish (demand signal)
})

# Event types that produce long (bullish) signals by default
_BULLISH_EVENT_TYPES = frozenset({
    "diplomatic",   # peace talks / ceasefire
    "ma_activity",  # M&A premium
    "earnings",     # positive earnings surprise context
    # Gap fill: capital-return signals
    "buyback",      # share repurchase programs lift per-share value
})

# Event types whose direction depends on market_direction label (classifier
# already decides based on keyword context — central_bank can be either).
# Listed here so reviewers know they were considered but intentionally
# excluded from the static bearish/bullish sets.
_CONTEXT_SENSITIVE_EVENT_TYPES = frozenset({
    "central_bank",      # rate cut = bullish, rate hike = bearish
    "analyst_rating",    # upgrade = bullish, downgrade = bearish
    "ipo",               # IPO can pop or flop; resolved via market_direction
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

    direction = _trigger_to_direction(trigger_type)

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

    # Derive direction from market_direction + event type hints.
    # H12: severity now feeds the "mixed" resolution path.
    direction = _derive_direction(
        market_direction, event_types, affected_sectors, confidence, severity,
    )

    # Deterministic ID: stable across processes (hash() is salted per run).
    # 12 hex chars ≈ 48 bits → ~281T possible values, enough to make
    # collisions negligible across distinct event_type combinations.
    _etype_key = ",".join(sorted(event_types)) or "none"
    _etype_hash = hashlib.sha1(_etype_key.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    _safe_cluster = (cluster_id or "cls").replace("/", "_").replace(" ", "_")
    signal_id = f"ps_{_safe_cluster}_{_etype_hash}"

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


def _trigger_to_direction(trigger_type: str | None) -> str:
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
    severity: float = 0.0,
) -> str:
    if market_direction == "bearish":
        return "short"
    if market_direction == "bullish":
        return "long"
    if market_direction == "mixed":
        # H12: severity-weighted resolution — high-severity bearish event types
        # dominate over low-severity bullish event types (e.g. "military strike"
        # + "ma_activity" should not net to flat).
        bearish_weight = 0.0
        bullish_weight = 0.0
        # Base severity per event type; scaled by headline severity when > 0.
        for et in event_types:
            w = max(0.5, severity / 5.0) if severity > 0 else 1.0
            if et in _BEARISH_EVENT_TYPES:
                bearish_weight += w
            elif et in _BULLISH_EVENT_TYPES:
                bullish_weight += w
        if bearish_weight > bullish_weight * 1.2:
            return "short"
        if bullish_weight > bearish_weight * 1.2:
            return "long"
        return "flat"
    # neutral: check event type hints
    for et in event_types:
        if et in _BEARISH_EVENT_TYPES:
            return "short"
        if et in _BULLISH_EVENT_TYPES:
            return "long"
    return "flat"


def require_corroboration(
    signal: PositionSignal,
    supporting_events: list,
    *,
    min_independent_high_tier: int = 2,
) -> PositionSignal | None:
    """Gate a signal on cross-source confirmation.

    A signal survives the gate only if at least `min_independent_high_tier`
    distinct T0/T1 sources are among `supporting_events`. If not, the signal
    is dropped (returns None) and a debug log line is emitted.

    Tier mapping uses whatever is set on the event's `source_tier`; events
    with no tier are treated as T3 (not counted).
    """
    if signal is None:
        return None
    sources_high_tier: set[str] = set()
    for evt in supporting_events or []:
        tier = getattr(evt, "source_tier", None)
        tier_val = getattr(tier, "value", str(tier) if tier is not None else "")
        if tier_val in ("T0", "T1"):
            src = (getattr(evt, "source_id", "") or "").lower().strip()
            if src:
                sources_high_tier.add(src)
    if len(sources_high_tier) < min_independent_high_tier:
        logger.debug(
            "[SKIP] require_corroboration: signal=%s high_tier_sources=%d < %d",
            signal.signal_id, len(sources_high_tier), min_independent_high_tier,
        )
        return None
    return signal


__all__ = [
    "PositionSignal",
    "cluster_to_signal",
    "classification_to_signal",
    "signals_to_basket",
    "require_corroboration",
]
