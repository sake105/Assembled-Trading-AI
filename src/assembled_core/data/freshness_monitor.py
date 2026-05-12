"""Data Freshness Monitoring (Plan 10.6).

Per-source last-update tracking with staleness alerts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class SourceFreshness:
    """Freshness status for a data source."""

    source: str
    last_updated: datetime | None = None
    max_age_hours: float = 24.0

    @property
    def age_hours(self) -> float:
        if self.last_updated is None:
            return float("inf")
        now = datetime.now(timezone.utc)
        if self.last_updated.tzinfo is None:
            last = self.last_updated.replace(tzinfo=timezone.utc)
        else:
            last = self.last_updated
        return (now - last).total_seconds() / 3600

    @property
    def is_stale(self) -> bool:
        return self.age_hours > self.max_age_hours


@dataclass
class FreshnessMonitor:
    """Monitor freshness of multiple data sources."""

    sources: dict[str, SourceFreshness] = field(default_factory=dict)

    def register(self, source: str, max_age_hours: float = 24.0) -> None:
        self.sources[source] = SourceFreshness(
            source=source, max_age_hours=max_age_hours
        )

    def update(self, source: str) -> None:
        if source in self.sources:
            self.sources[source].last_updated = datetime.now(timezone.utc)

    def check_all(self) -> list[dict]:
        alerts = []
        for name, sf in self.sources.items():
            if sf.is_stale:
                alerts.append(
                    {
                        "source": name,
                        "age_hours": round(sf.age_hours, 1),
                        "max_age_hours": sf.max_age_hours,
                    }
                )
                logger.warning(
                    "[Freshness] Source '%s' is stale (%.1fh > %.1fh)",
                    name,
                    sf.age_hours,
                    sf.max_age_hours,
                )
        return alerts

    def last_known_good_timestamp(self, source: str) -> datetime | None:
        """Return the last update for a source, or None if never recorded.

        Audit C4-024: callers must distinguish "no data ever seen" from
        "stale but once-fresh data". The freshness alert pipeline always
        considered an unset ``last_updated`` as infinitely-old; this
        helper exposes the underlying value so consumers can react
        differently (e.g. degrade to read-only mode when unknown, alert
        oncall when stale).
        """
        sf = self.sources.get(source)
        if sf is None or sf.last_updated is None:
            return None
        if sf.last_updated.tzinfo is None:
            return sf.last_updated.replace(tzinfo=timezone.utc)
        return sf.last_updated

    def degradation_status(self, source: str) -> str:
        """Return one of ``unknown`` / ``ok`` / ``stale``.

        Operators can wire this into the /ready probe or a dashboard
        without re-implementing the unknown-vs-stale distinction.
        """
        sf = self.sources.get(source)
        if sf is None or sf.last_updated is None:
            return "unknown"
        return "stale" if sf.is_stale else "ok"


__all__ = ["SourceFreshness", "FreshnessMonitor"]
