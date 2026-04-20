"""NewsEventEnricher — pipeline enrichment step for raw NewsEvent objects.

Takes a batch of raw NewsEvent objects and runs them through:
1. Classification (if not already done)
2. Impact estimation
3. EventStore indexing
4. Velocity tracking
5. Fatigue scoring
6. Source bias discount application

Returns enriched events with all fields populated.

Usage:
    enricher = NewsEventEnricher()
    enriched = enricher.enrich(events)
    for evt in enriched:
        print(evt.news_confidence, evt.severity, evt.event_types)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class NewsEventEnricher:
    """Applies the full enrichment pipeline to a batch of NewsEvents.

    Components (lazy-initialised on first use):
    - NewsClassifier (classification + impact)
    - NewsImpactEstimator (BPS estimates stored as metadata)
    - VelocityTracker (surge detection)
    - NewsDedupeIndex (fatigue scoring)
    """

    def __init__(
        self,
        event_store: object | None = None,
        velocity_tracker: object | None = None,
        dedupe_index: object | None = None,
    ) -> None:
        self._event_store = event_store
        self._velocity_tracker = velocity_tracker
        self._dedupe_index = dedupe_index
        self._impact_estimator = None

    def enrich(
        self,
        events: list,
        now: datetime | None = None,
    ) -> list:
        """Enrich events with classification, impact, and quality signals.

        Args:
            events: List of NewsEvent objects. Modified in-place where possible.
            now: Reference timestamp (default: utcnow).

        Returns:
            Same list with enriched fields. Events that fail enrichment are
            returned as-is (no events are dropped).
        """
        if not events:
            return events

        if now is None:
            now = datetime.now(tz=timezone.utc)

        # Step 1: Classify events that don't have event_types yet
        events = self._run_classification(events)

        # Step 2: Impact estimation (attaches to event as metadata attribute)
        events = self._run_impact_estimation(events)

        # Step 3: Fatigue scoring
        events = self._run_fatigue_scoring(events)

        # Step 4: Velocity update
        if self._velocity_tracker is not None:
            try:
                self._velocity_tracker.update(events, now=now)
            except Exception as exc:
                logger.debug("[SKIP] VelocityTracker: %s", exc)

        # Step 5: EventStore indexing
        if self._event_store is not None:
            try:
                self._event_store.add_many(events)
            except Exception as exc:
                logger.debug("[SKIP] EventStore: %s", exc)

        logger.debug("[OK] NewsEventEnricher: enriched %d events", len(events))
        return events

    def _run_classification(self, events: list) -> list:
        try:
            from src.assembled_core.intel.news_classifier import classify_news_event, apply_source_bias_discount
        except ImportError:
            return events

        for evt in events:
            try:
                if getattr(evt, "event_types", None):
                    continue  # already classified
                geo_tags = list(getattr(evt, "geo_tags", []) or [])
                tickers = list(getattr(evt, "tickers", []) or [])
                source_tier = getattr(evt, "source_tier", None)
                tier_str = getattr(source_tier, "value", str(source_tier)) if source_tier else "T2"
                clf = classify_news_event(
                    evt.title,
                    geo_tags=geo_tags,
                    source_tier=tier_str,
                    tickers=tickers,
                )
                evt.event_types = clf.event_types
                evt.severity = clf.severity
                evt.market_direction = clf.market_direction
                evt.time_horizon = clf.time_horizon
                evt.affected_sectors = clf.affected_sectors
                evt.affected_assets = list({*clf.affected_assets, *tickers})
                source_id = getattr(evt, "source_id", "")
                evt.news_confidence = apply_source_bias_discount(clf.confidence, source_id)
            except Exception as exc:
                logger.debug("[SKIP] Classify %s: %s", getattr(evt, "event_id", "?"), exc)

        return events

    def _run_impact_estimation(self, events: list) -> list:
        if self._impact_estimator is None:
            try:
                from src.assembled_core.intel.news_impact_estimator import NewsImpactEstimator
                self._impact_estimator = NewsImpactEstimator()
            except ImportError:
                return events

        for evt in events:
            try:
                geo_tags = list(getattr(evt, "geo_tags", []) or [])
                source_tier = getattr(evt, "source_tier", None)
                tier_str = getattr(source_tier, "value", str(source_tier)) if source_tier else "T2"
                impact = self._impact_estimator.estimate(evt, geo_tags=geo_tags, source_tier=tier_str)
                # Attach impact as a metadata attribute (not a model field)
                object.__setattr__(evt, "_impact", impact) if hasattr(evt, "__dict__") else None
                try:
                    evt._impact = impact  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception as exc:
                logger.debug("[SKIP] Impact estimation %s: %s", getattr(evt, "event_id", "?"), exc)

        return events

    def _run_fatigue_scoring(self, events: list) -> list:
        if self._dedupe_index is None:
            return events
        for evt in events:
            try:
                score = self._dedupe_index.get_fatigue_score(evt)
                # Discount confidence proportionally to fatigue
                if score > 0 and hasattr(evt, "news_confidence"):
                    evt.news_confidence = round(evt.news_confidence * (1.0 - 0.5 * score), 4)
                    logger.debug(
                        "[OK] Fatigue discount %.2f for %s",
                        score, getattr(evt, "event_id", "?"),
                    )
            except Exception as exc:
                logger.debug("[SKIP] Fatigue score %s: %s", getattr(evt, "event_id", "?"), exc)
        return events


__all__ = ["NewsEventEnricher"]
