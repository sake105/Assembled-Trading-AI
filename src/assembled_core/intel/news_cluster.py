"""News event clustering — groups related events into evidence clusters."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone

from src.assembled_core.intel.geo_trigger import classify_trigger_type
from src.assembled_core.intel.models import (
    EvidenceCluster,
    NewsEvent,
    SourceTier,
    TriggerType,
)

logger = logging.getLogger(__name__)

# Tier boost values used in confidence calculation
_TIER_BOOST: dict[SourceTier, float] = {
    SourceTier.T0: 0.4,
    SourceTier.T1: 0.3,
    SourceTier.T2: 0.15,
    SourceTier.T3: 0.0,
}


# ---------------------------------------------------------------------------
# Cluster ID helpers
# ---------------------------------------------------------------------------


def _make_cluster_id(trigger_type: TriggerType, window_start: datetime) -> str:
    raw = trigger_type.value + window_start.isoformat()
    return "cl_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _floor_to_hour(dt: datetime) -> datetime:
    """Floor a datetime to the start of its hour (UTC)."""
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Cluster manager
# ---------------------------------------------------------------------------


class ClusterManager:
    """
    Groups NewsEvents into EvidenceClusters by TriggerType within 1-hour windows.

    Clusters expire after cluster_ttl_minutes.
    """

    def __init__(self, cluster_ttl_minutes: int = 360) -> None:
        self._ttl_minutes = cluster_ttl_minutes
        # cluster_id → EvidenceCluster
        self.active_clusters: dict[str, EvidenceCluster] = {}

    def update_clusters(
        self,
        events: list[NewsEvent],
        now: datetime | None = None,
    ) -> list[EvidenceCluster]:
        """
        Process new events, group by TriggerType + 1-hour bucket,
        update or create EvidenceClusters, expire old ones.

        Returns list of currently active (non-expired) clusters.
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)

        # --- Group new events by (TriggerType, hour_bucket) ---
        groups: dict[tuple[TriggerType, datetime], list[NewsEvent]] = {}
        for event in events:
            trigger_type = classify_trigger_type(event)
            if trigger_type is None:
                continue
            # Use event.published_at for PIT-safe bucket assignment.
            # In live mode published_at ≈ now; in replay/backtest this prevents
            # all events in a batch from collapsing into a single "now" bucket.
            event_ts = getattr(event, "published_at", None) or now
            bucket = _floor_to_hour(event_ts)
            key = (trigger_type, bucket)
            groups.setdefault(key, []).append(event)

        # --- Update / create clusters ---
        for (trigger_type, bucket), group_events in groups.items():
            cluster_id = _make_cluster_id(trigger_type, bucket)

            if cluster_id in self.active_clusters:
                cluster = self.active_clusters[cluster_id]
                # Add new event IDs (dedupe in-place)
                existing_ids = set(cluster.supporting_events)
                for evt in group_events:
                    if evt.event_id not in existing_ids:
                        cluster.supporting_events.append(evt.event_id)
                        existing_ids.add(evt.event_id)
                # Extend TTL
                cluster.expires_at = now + timedelta(minutes=self._ttl_minutes)
                # Recompute confidence / max_tier
                self._update_confidence(cluster, group_events)
            else:
                # New cluster
                expires_at = now + timedelta(minutes=self._ttl_minutes)
                all_event_ids = [e.event_id for e in group_events]
                cluster = EvidenceCluster(
                    cluster_id=cluster_id,
                    trigger_type=trigger_type,
                    summary=f"{trigger_type.value} cluster at {bucket.isoformat()}",
                    supporting_events=all_event_ids,
                    confidence=0.0,
                    max_tier=SourceTier.T3,
                    created_at=now,
                    expires_at=expires_at,
                )
                self._update_confidence(cluster, group_events)
                self.active_clusters[cluster_id] = cluster
                logger.debug(
                    "[OK] ClusterManager: new cluster %s (%s, %d events)",
                    cluster_id, trigger_type.value, len(group_events),
                )

        # --- Expire old clusters ---
        expired = [cid for cid, cl in self.active_clusters.items() if cl.expires_at < now]
        for cid in expired:
            logger.debug("[SKIP] ClusterManager: expired cluster %s", cid)
            del self.active_clusters[cid]

        return list(self.active_clusters.values())

    def _update_confidence(
        self, cluster: EvidenceCluster, new_events: list[NewsEvent]
    ) -> None:
        """Recompute confidence and max_tier after adding new events."""
        total_events = len(cluster.supporting_events)

        # Determine max_tier across all source_tiers of new events
        # (We only have direct access to new_events here; existing events
        #  contributed their tier when first processed)
        current_max = cluster.max_tier
        for evt in new_events:
            if evt.source_tier.value < current_max.value:
                # Lower string value = higher tier (T0 < T1 < T2 < T3)
                current_max = evt.source_tier
        cluster.max_tier = current_max

        # T2.8: Bayesian confidence (flipped from shadow to production)
        from src.assembled_core.intel.bayesian_confidence import compute_cluster_confidence
        source_tiers = [evt.source_tier.value for evt in new_events]
        n_independent = max(1, len(set(evt.source_id for evt in new_events)))
        cluster.confidence = compute_cluster_confidence(
            trigger_type=cluster.trigger_type.value,
            source_tiers=source_tiers if source_tiers else [current_max.value] * total_events,
            n_independent_sources=n_independent,
            keyword_match_strength=0.7,
        )
        # Step 3: urgency boost — Breaking/Flash events raise confidence up to +0.1
        urgency_boost = max((getattr(evt, "urgency", 0.0) for evt in new_events), default=0.0) * 0.1
        if urgency_boost > 0:
            cluster.confidence = min(cluster.confidence + urgency_boost, 0.99)
        logger.debug(
            "[OK] T2.8 Bayesian confidence: cluster=%s conf=%.3f (urgency_boost=%.3f)",
            cluster.cluster_id, cluster.confidence, urgency_boost,
        )

    def get_active_clusters(self) -> list[EvidenceCluster]:
        """Return all currently active (non-expired) clusters."""
        now = datetime.now(tz=timezone.utc)
        return [cl for cl in self.active_clusters.values() if cl.expires_at >= now]
