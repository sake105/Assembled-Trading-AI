"""Sector-level news risk overlay for portfolio construction.

Aggregates news signals across all active clusters and the EventStore
to produce a per-sector overlay signal that the portfolio layer can use
to tilt sector weights.

Positive overlay = news-driven tailwind (overweight sector)
Negative overlay = news-driven headwind (underweight sector)
Magnitude [0, 1] where 1.0 = full underweight/overweight constraint.

Usage:
    overlay = SectorNewsOverlay()
    signals = overlay.compute(clusters, event_store=store, now=now)
    # signals: dict[sector, float] e.g. {"energy": -0.3, "defense": +0.4}
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class SectorNewsOverlay:
    """Produces sector-level overlay signals from news.

    Logic:
    1. From EvidenceClusters: cluster confidence * direction → sector weight
    2. From EventStore (if provided): severity-weighted average sentiment per sector
    3. Combine with exponential recency weighting
    4. Normalise to [-1, +1]
    """

    def __init__(
        self,
        decay_hours: float = 12.0,
        max_cluster_contribution: float = 0.5,
        max_store_contribution: float = 0.5,
    ) -> None:
        self._decay_hours = decay_hours
        self._max_cluster = max_cluster_contribution
        self._max_store = max_store_contribution

    def compute(
        self,
        clusters: list | None = None,
        event_store: object | None = None,
        now: datetime | None = None,
        store_lookback_hours: float = 24.0,
    ) -> dict[str, float]:
        """Compute sector overlay signals.

        Args:
            clusters: List of EvidenceCluster objects.
            event_store: Optional NewsEventStore for richer signal.
            now: Reference timestamp (default: utcnow).
            store_lookback_hours: Hours of EventStore history to use.

        Returns:
            dict[sector → overlay_score] where scores are in [-1, +1].
            Negative = headwind (underweight), Positive = tailwind (overweight).
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)

        clusters = clusters or []
        sector_scores: dict[str, float] = {}

        # --- Contribution from EvidenceClusters ---
        cluster_scores = self._from_clusters(clusters, now)
        for sector, score in cluster_scores.items():
            sector_scores[sector] = sector_scores.get(sector, 0.0) + score * self._max_cluster

        # --- Contribution from EventStore ---
        if event_store is not None:
            store_scores = self._from_event_store(event_store, hours=store_lookback_hours, now=now)
            for sector, score in store_scores.items():
                sector_scores[sector] = sector_scores.get(sector, 0.0) + score * self._max_store

        # Normalise to [-1, +1]
        if sector_scores:
            max_abs = max(abs(v) for v in sector_scores.values())
            if max_abs > 0:
                sector_scores = {k: round(v / max_abs, 4) for k, v in sector_scores.items()}

        if sector_scores:
            logger.debug(
                "[OK] SectorNewsOverlay: %d sectors, top=%s",
                len(sector_scores),
                sorted(sector_scores, key=lambda k: -abs(sector_scores[k]))[:3],
            )

        return sector_scores

    def _from_clusters(self, clusters: list, now: datetime) -> dict[str, float]:
        """Extract sector signals from active EvidenceClusters."""
        sector_scores: dict[str, float] = {}

        # Map TriggerType → implicit sector impact
        _TRIGGER_SECTOR_IMPACT: dict[str, dict[str, float]] = {
            "war_escalation": {"defense": +0.8, "energy": -0.4, "consumer": -0.3},
            "military_strike": {"defense": +0.6, "energy": -0.3},
            "sanctions": {"financials": -0.5, "energy": -0.4, "industrials": -0.3},
            "energy_disruption": {"energy": -0.7, "utilities": -0.3, "industrials": -0.2},
            "central_bank": {"financials": -0.3, "tech": -0.2, "utilities": +0.1},
            "trade_policy": {"industrials": -0.4, "consumer": -0.2, "tech": -0.2},
            "political_crisis": {"financials": -0.3},
            "market_stress": {"financials": -0.5, "consumer": -0.3},
            "cyber_attack": {"tech": -0.4},
            "natural_disaster": {"utilities": -0.3, "industrials": -0.2},
            "diplomatic": {"defense": -0.2},  # peace talks reduce defense premium
            "ma_activity": {},
            "regulatory": {},
            "earnings": {},
        }

        for cluster in clusters:
            conf = float(getattr(cluster, "confidence", 0.0))
            if conf < 0.3:
                continue
            trigger_type = getattr(cluster, "trigger_type", None)
            if trigger_type is not None:
                trigger_val = getattr(trigger_type, "value", str(trigger_type))
            else:
                continue

            # Recency decay
            created_at = getattr(cluster, "created_at", None)
            age_hours = 0.0
            if created_at:
                age_hours = (now - created_at).total_seconds() / 3600
            decay = math.exp(-age_hours / max(self._decay_hours, 1.0))

            impacts = _TRIGGER_SECTOR_IMPACT.get(trigger_val, {})
            for sector, base_impact in impacts.items():
                contribution = base_impact * conf * decay
                sector_scores[sector] = sector_scores.get(sector, 0.0) + contribution

        return sector_scores

    def _from_event_store(
        self, event_store: object, hours: float, now: datetime
    ) -> dict[str, float]:
        """Extract sector signals from recent EventStore events."""
        sector_scores: dict[str, float] = {}
        try:
            recent_events = event_store.query_by_time(hours=hours)  # type: ignore[attr-defined]
        except Exception:
            return {}

        for evt in recent_events:
            sectors = getattr(evt, "affected_sectors", []) or []
            market_dir = getattr(evt, "market_direction", "neutral")
            severity = float(getattr(evt, "severity", 0.0) or 0.0)
            confidence = float(getattr(evt, "news_confidence", 0.0) or 0.0)

            direction_sign = -1.0 if market_dir == "bearish" else (1.0 if market_dir == "bullish" else 0.0)
            if direction_sign == 0.0:
                continue

            # Recency decay
            ts = getattr(evt, "published_at", None) or getattr(evt, "ingested_at", None)
            age_hours = 0.0
            if ts:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_hours = (now - ts).total_seconds() / 3600
            decay = math.exp(-age_hours / max(self._decay_hours, 1.0))

            weight = direction_sign * severity / 10.0 * confidence * decay
            for sector in sectors:
                sector_scores[sector] = sector_scores.get(sector, 0.0) + weight

        return sector_scores


__all__ = ["SectorNewsOverlay"]
