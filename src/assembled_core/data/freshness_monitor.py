"""Data Freshness Monitoring (Plan 10.6).

Per-source last-update tracking with staleness alerts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

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
        self.sources[source] = SourceFreshness(source=source, max_age_hours=max_age_hours)

    def update(self, source: str) -> None:
        if source in self.sources:
            self.sources[source].last_updated = datetime.now(timezone.utc)

    def check_all(self) -> list[dict]:
        alerts = []
        for name, sf in self.sources.items():
            if sf.is_stale:
                alerts.append({
                    "source": name,
                    "age_hours": round(sf.age_hours, 1),
                    "max_age_hours": sf.max_age_hours,
                })
                logger.warning("[Freshness] Source '%s' is stale (%.1fh > %.1fh)", name, sf.age_hours, sf.max_age_hours)
        return alerts


__all__ = ["SourceFreshness", "FreshnessMonitor"]
