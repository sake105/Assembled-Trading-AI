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
        corroboration_tracker: object | None = None,
        decay: object | None = None,
        apply_decay: bool = True,
    ) -> None:
        self._event_store = event_store
        self._velocity_tracker = velocity_tracker
        self._dedupe_index = dedupe_index
        self._corroboration = corroboration_tracker
        self._impact_estimator = None
        self._decay = decay
        self._apply_decay = apply_decay
        self.last_velocity = None  # exposes most recent VelocityResult for callers

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

        # Step 0: Language detection (cheap, always on)
        events = self._run_language_detection(events)

        # Step 1: Classify events that don't have event_types yet
        events = self._run_classification(events)

        # Step 2: Impact estimation (attaches to event as metadata attribute)
        events = self._run_impact_estimation(events)

        # Step 2.1: IC-basierte Severity-Gewichtung (aus ic_loop.json)
        try:
            from src.assembled_core.ml.news_ml_bridge import get_event_type_ic_weights

            _ic_weights = get_event_type_ic_weights()
            if _ic_weights:
                for _evt in events:
                    for _etype in getattr(_evt, "event_types", []):
                        _w = _ic_weights.get(str(_etype), 1.0)
                        if hasattr(_evt, "severity") and _evt.severity is not None:
                            _evt.severity = round(
                                min(10.0, max(0.0, float(_evt.severity) * _w)), 4
                            )
                logger.debug("[OK] IC-Gewichte auf %d Events angewendet", len(events))
        except Exception as _exc:
            logger.debug("[news_enricher] IC-weights step failed: %s", _exc)

        # Step 2.5: Decay discount on impact_bps so stale events don't inflate
        # downstream position sizing.
        if self._apply_decay:
            events = self._run_decay(events, now=now)

        # Step 3: Fatigue scoring
        events = self._run_fatigue_scoring(events)

        # Step 3.5: Cross-source corroboration
        events = self._run_corroboration(events)

        # Step 3.6: Source vote consensus check (warn when sources disagree on direction)
        events = self._run_source_vote(events)

        # Step 4: Velocity update
        if self._velocity_tracker is not None:
            try:
                self.last_velocity = self._velocity_tracker.update(events, now=now)
            except Exception as exc:
                logger.warning("[WARN] VelocityTracker update failed: %s", exc)

        # Step 5: EventStore indexing
        if self._event_store is not None:
            try:
                self._event_store.add_many(events)
            except Exception as exc:
                logger.warning("[WARN] EventStore add_many failed: %s", exc)

        logger.debug("[OK] NewsEventEnricher: enriched %d events", len(events))
        return events

    def _run_language_detection(self, events: list) -> list:
        try:
            from src.assembled_core.intel.news_language import detect_language
        except ImportError:
            return events
        for evt in events:
            try:
                current = getattr(evt, "language", "") or ""
                if current and current != "en":
                    continue  # trust upstream detection
                title = getattr(evt, "title", "") or ""
                evt.language = detect_language(title)
            except Exception as exc:
                logger.debug(
                    "[SKIP] LangDetect %s: %s", getattr(evt, "event_id", "?"), exc
                )
        return events

    def _run_classification(self, events: list) -> list:
        try:
            from src.assembled_core.intel.news_classifier import (
                apply_source_bias_discount,
                classify_news_event,
            )
        except ImportError:
            return events

        for evt in events:
            try:
                if getattr(evt, "event_types", None):
                    # H1: already-classified path still needs downstream-safe
                    # defaults for fields the upstream producer may not have
                    # populated (confidence, direction, horizon).
                    if getattr(evt, "news_confidence", None) in (None, 0.0):
                        evt.news_confidence = 0.0
                    if not getattr(evt, "market_direction", None):
                        evt.market_direction = "neutral"
                    if not getattr(evt, "time_horizon", None):
                        evt.time_horizon = "short"
                    continue  # already classified
                geo_tags = list(getattr(evt, "geo_tags", []) or [])
                tickers = list(getattr(evt, "tickers", []) or [])
                source_tier = getattr(evt, "source_tier", None)
                tier_str = (
                    getattr(source_tier, "value", str(source_tier))
                    if source_tier
                    else "T2"
                )
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
                # Taxonomy category
                try:
                    from src.assembled_core.intel.news_taxonomy import categorize_event

                    evt.category = categorize_event(event_types=clf.event_types)
                except Exception:
                    pass
                source_id = getattr(evt, "source_id", "")
                evt.news_confidence = apply_source_bias_discount(
                    clf.confidence, source_id
                )
            except Exception as exc:
                logger.debug(
                    "[SKIP] Classify %s: %s", getattr(evt, "event_id", "?"), exc
                )

        return events

    def _run_impact_estimation(self, events: list) -> list:
        if self._impact_estimator is None:
            try:
                from src.assembled_core.intel.news_impact_estimator import (
                    NewsImpactEstimator,
                )

                self._impact_estimator = NewsImpactEstimator()
            except ImportError:
                return events

        for evt in events:
            try:
                geo_tags = list(getattr(evt, "geo_tags", []) or [])
                source_tier = getattr(evt, "source_tier", None)
                tier_str = (
                    getattr(source_tier, "value", str(source_tier))
                    if source_tier
                    else "T2"
                )
                impact = self._impact_estimator.estimate(
                    evt, geo_tags=geo_tags, source_tier=tier_str
                )
                # Persist impact on model fields so archive replay retains them
                try:
                    evt.impact_bps = float(impact.bps)
                    evt.impact_horizon_days = int(impact.horizon_days)
                    evt.impact_confidence = float(impact.confidence)
                    evt.impact_geo_premium_bps = float(impact.geo_premium_bps)
                    evt.impact_dominant_event_type = str(impact.dominant_event_type)
                except Exception as exc:
                    logger.debug(
                        "[SKIP] Impact assign %s: %s",
                        getattr(evt, "event_id", "?"),
                        exc,
                    )
            except Exception as exc:
                logger.debug(
                    "[SKIP] Impact estimation %s: %s",
                    getattr(evt, "event_id", "?"),
                    exc,
                )

        return events

    def _run_corroboration(self, events: list) -> list:
        if self._corroboration is None:
            try:
                from src.assembled_core.intel.news_corroboration import (
                    CorroborationTracker,
                )

                self._corroboration = CorroborationTracker()
            except ImportError:
                return events
        try:
            self._corroboration.ingest(events)
        except Exception as exc:
            logger.debug("[SKIP] Corroboration ingest: %s", exc)
            return events
        for evt in events:
            try:
                s = self._corroboration.corroboration_score(evt)
                evt.corroboration_score = float(s.score)
                evt.corroboration_n_sources = int(s.n_sources)
                # Boost confidence for well-corroborated stories (complement to fatigue discount)
                if s.score > 0.5 and hasattr(evt, "news_confidence"):
                    boost = 1.0 + 0.2 * s.score  # up to +20% at score=1.0
                    evt.news_confidence = round(
                        min(1.0, evt.news_confidence * boost), 4
                    )
            except Exception as exc:
                logger.debug(
                    "[SKIP] Corroboration score %s: %s",
                    getattr(evt, "event_id", "?"),
                    exc,
                )
        return events

    def _run_source_vote(self, events: list) -> list:
        """Group events by story fingerprint and check for source direction divergence.

        When the weighted source vote disagrees with an event's market_direction,
        apply a confidence discount proportional to the vote margin difference.
        Low-margin votes (contested stories) get a stronger discount.
        """
        try:
            from src.assembled_core.intel.news_dedupe import content_fingerprint
            from src.assembled_core.intel.news_source_voting import vote_direction
        except ImportError:
            return events

        # Group events by source-agnostic story fingerprint
        groups: dict[str, list] = {}
        for evt in events:
            title = getattr(evt, "title", "") or ""
            fp = content_fingerprint(title, "")
            groups.setdefault(fp, []).append(evt)

        for fp, group in groups.items():
            if len(group) < 2:
                continue
            try:
                vote = vote_direction(group)
                for evt in group:
                    evt_dir = getattr(evt, "market_direction", "neutral") or "neutral"
                    if evt_dir == "neutral" or vote.winner == "neutral":
                        continue
                    if vote.winner != evt_dir and vote.total_weight > 0:
                        # Source vote disagrees — discount confidence by margin
                        discount = max(0.5, 1.0 - vote.margin * 0.3)
                        if hasattr(evt, "news_confidence"):
                            evt.news_confidence = round(
                                evt.news_confidence * discount, 4
                            )
                        logger.debug(
                            "[OK] SourceVote divergence: event=%s dir=%s vote=%s margin=%.2f",
                            getattr(evt, "event_id", "?"),
                            evt_dir,
                            vote.winner,
                            vote.margin,
                        )
            except Exception as exc:
                logger.debug("[SKIP] SourceVote group %s: %s", fp, exc)

        return events

    def _run_decay(self, events: list, now: datetime) -> list:
        if self._decay is None:
            try:
                from src.assembled_core.intel.news_decay import NewsDecay

                self._decay = NewsDecay()
            except ImportError:
                return events
        for evt in events:
            try:
                bps = getattr(evt, "impact_bps", None)
                if bps is None or bps == 0:
                    continue
                dominant = getattr(evt, "impact_dominant_event_type", None)
                if not dominant:
                    etypes = getattr(evt, "event_types", []) or []
                    dominant = etypes[0] if etypes else "default"
                ts = (
                    getattr(evt, "published_at", None)
                    or getattr(evt, "ingested_at", None)
                    or now
                )
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                minutes_since = max(0.0, (now - ts).total_seconds() / 60.0)
                frac = float(self._decay.impact_remaining(str(dominant), minutes_since))
                evt.impact_bps = round(float(bps) * frac, 4)
                evt.impact_decay_fraction = round(frac, 6)
                evt.impact_decay_minutes = round(minutes_since, 2)
            except Exception as exc:
                logger.debug("[SKIP] Decay %s: %s", getattr(evt, "event_id", "?"), exc)
        return events

    def _run_fatigue_scoring(self, events: list) -> list:
        if self._dedupe_index is None:
            return events
        for evt in events:
            try:
                score = self._dedupe_index.get_fatigue_score(evt)
                # Discount confidence proportionally to fatigue
                if score > 0 and hasattr(evt, "news_confidence"):
                    evt.news_confidence = round(
                        evt.news_confidence * (1.0 - 0.5 * score), 4
                    )
                    logger.debug(
                        "[OK] Fatigue discount %.2f for %s",
                        score,
                        getattr(evt, "event_id", "?"),
                    )
            except Exception as exc:
                logger.debug(
                    "[SKIP] Fatigue score %s: %s", getattr(evt, "event_id", "?"), exc
                )
        return events


__all__ = ["NewsEventEnricher"]
