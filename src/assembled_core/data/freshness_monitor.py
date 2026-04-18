"""Data Freshness Monitoring (Plan 10.6).

Per-source last-update tracking with staleness alerts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

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
        else:
            # Silently ignoring an unregistered source name is the classic
            # "we thought we were monitoring it" bug — a typo'd source name
            # leaves the real source stuck at last_updated=None forever.
            logger.warning(
                "[Freshness] update() called for unregistered source %r — "
                "known sources: %s",
                source,
                sorted(self.sources.keys()),
            )

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


def detect_stale_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    stale_days: int = 5,
) -> list[dict]:
    """Detect features that have been constant for too long (possible data feed outage).

    For each (symbol, feature) pair, checks if the feature value has been
    identical for the last ``stale_days`` trading days.  If so, it flags it
    as potentially stale.

    Args:
        df: DataFrame with features, timestamps, and symbols.
        feature_cols: List of feature column names to check.
        timestamp_col: Column with timestamps.
        symbol_col: Column with symbols.
        stale_days: Number of consecutive identical values to trigger alert.

    Returns:
        List of dicts with keys: symbol, feature, last_value, constant_days.
    """
    alerts: list[dict] = []

    if df.empty or timestamp_col not in df.columns:
        return alerts

    for col in feature_cols:
        if col not in df.columns:
            continue

        for symbol, group in df.groupby(symbol_col):
            sorted_group = group.sort_values(timestamp_col).tail(stale_days + 1)
            if len(sorted_group) < stale_days:
                continue

            recent_values = sorted_group[col].dropna()
            if len(recent_values) < stale_days:
                continue

            tail_values = recent_values.tail(stale_days)
            if tail_values.nunique() == 1:
                alerts.append({
                    "symbol": str(symbol),
                    "feature": col,
                    "last_value": float(tail_values.iloc[-1]),
                    "constant_days": stale_days,
                })
                logger.warning(
                    "[Staleness] Feature '%s' for %s has been constant (%.4f) "
                    "for %d consecutive days — possible data feed outage",
                    col, symbol, float(tail_values.iloc[-1]), stale_days,
                )

    return alerts


__all__ = ["SourceFreshness", "FreshnessMonitor", "detect_stale_features"]
