"""Intel signal aggregator — combines multiple cluster signals into a unified view.

Takes a list of EvidenceClusters and PositionSignals and produces:
- A single unified intel signal with net direction, confidence, and asset basket
- A ranked sector exposure summary
- A risk level assessment (LOW / MODERATE / HIGH / CRITICAL)

This is the final consolidation step before the portfolio layer consumes
news-derived signals.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_RISK_THRESHOLDS = {
    "CRITICAL": 0.80,
    "HIGH": 0.55,
    "MODERATE": 0.35,
    "LOW": 0.0,
}


@dataclass
class IntelSignal:
    """Unified intel signal derived from aggregated news clusters."""
    signal_id: str
    net_direction: str  # "bearish" / "bullish" / "neutral"
    aggregate_confidence: float  # 0-1, confidence-weighted
    risk_level: str  # "LOW" / "MODERATE" / "HIGH" / "CRITICAL"
    asset_basket: dict[str, float]  # ticker → net direction score [-1, +1]
    sector_exposure: dict[str, float]  # sector → net score [-1, +1]
    top_event_types: list[str]  # most frequent event types
    n_clusters: int
    n_signals: int
    max_severity: float
    generated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def is_actionable(self) -> bool:
        return self.risk_level in ("HIGH", "CRITICAL") and self.net_direction != "neutral"


def aggregate_signals(
    clusters: list,
    position_signals: list | None = None,
    *,
    min_confidence: float = 0.3,
) -> IntelSignal:
    """Aggregate cluster and position signals into a single IntelSignal.

    Args:
        clusters: List of EvidenceCluster objects.
        position_signals: Optional list of PositionSignal objects. If None,
            clusters are converted to direction signals internally.
        min_confidence: Minimum cluster confidence to include.

    Returns:
        IntelSignal with consolidated view.
    """
    from src.assembled_core.intel.news_position_bridge import (
        cluster_to_signal,
        signals_to_basket,
    )

    if position_signals is None:
        position_signals = []
        for cl in clusters:
            sig = cluster_to_signal(cl)
            if sig is not None:
                position_signals.append(sig)

    # Filter by confidence
    active_signals = [s for s in position_signals if s.confidence >= min_confidence]

    if not active_signals:
        return IntelSignal(
            signal_id=_make_signal_id(),
            net_direction="neutral",
            aggregate_confidence=0.0,
            risk_level="LOW",
            asset_basket={},
            sector_exposure={},
            top_event_types=[],
            n_clusters=len(clusters),
            n_signals=0,
            max_severity=0.0,
        )

    # Confidence-weighted direction
    bearish_weight = sum(s.confidence for s in active_signals if s.direction == "short")
    bullish_weight = sum(s.confidence for s in active_signals if s.direction == "long")
    total_weight = bearish_weight + bullish_weight

    if total_weight == 0:
        net_direction = "neutral"
        agg_conf = 0.0
    elif bearish_weight > bullish_weight:
        net_direction = "bearish"
        agg_conf = bearish_weight / total_weight * max(s.confidence for s in active_signals)
    else:
        net_direction = "bullish"
        agg_conf = bullish_weight / total_weight * max(s.confidence for s in active_signals)

    agg_conf = round(min(agg_conf, 1.0), 4)

    # Risk level
    risk_level = "LOW"
    for level, threshold in _RISK_THRESHOLDS.items():
        if agg_conf >= threshold:
            risk_level = level
            break

    # Asset basket
    asset_basket = signals_to_basket(active_signals, min_confidence=min_confidence)

    # Sector exposure
    sector_scores: dict[str, float] = {}
    for sig in active_signals:
        sign = -1.0 if sig.direction == "short" else (1.0 if sig.direction == "long" else 0.0)
        for sector in sig.affected_sectors:
            sector_scores[sector] = sector_scores.get(sector, 0.0) + sig.confidence * sign

    if sector_scores:
        max_abs = max(abs(v) for v in sector_scores.values())
        if max_abs > 0:
            sector_scores = {k: round(v / max_abs, 4) for k, v in sector_scores.items()}

    # Event types frequency
    event_type_counts: dict[str, int] = {}
    for sig in active_signals:
        for et in sig.event_types:
            event_type_counts[et] = event_type_counts.get(et, 0) + 1
    top_event_types = sorted(event_type_counts, key=lambda k: -event_type_counts[k])[:5]

    # Max severity from clusters
    max_severity = max(
        (float(getattr(cl, "confidence", 0.0)) * 10 for cl in clusters),
        default=0.0,
    )

    return IntelSignal(
        signal_id=_make_signal_id(),
        net_direction=net_direction,
        aggregate_confidence=agg_conf,
        risk_level=risk_level,
        asset_basket=asset_basket,
        sector_exposure=sector_scores,
        top_event_types=top_event_types,
        n_clusters=len(clusters),
        n_signals=len(active_signals),
        max_severity=round(max_severity, 2),
    )


def _make_signal_id() -> str:
    now = datetime.now(tz=timezone.utc)
    return f"is_{now.strftime('%Y%m%dT%H%M%S')}"


__all__ = ["IntelSignal", "aggregate_signals"]
