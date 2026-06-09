"""Data Freshness Monitoring (Plan 10.6).

Per-source last-update tracking with staleness alerts.

Two source kinds are supported:

* **in-memory** — registered via :meth:`FreshnessMonitor.register` and stamped
  by an explicit :meth:`FreshnessMonitor.update` call (the original behaviour).
* **file-backed** — registered via :meth:`FreshnessMonitor.register_path`; the
  effective freshness is read live from the backing parquet/cache file's
  modification time (``os.path.getmtime``). No in-process ``update`` is needed,
  so a frozen cache (e.g. ``output/macro.parquet`` when the FRED feed has been
  dead for weeks) or a cache that was never written surfaces as stale/unknown.
  This is the capability audit DAT-003 found missing — the monitor previously
  only ever compared an in-memory ``now()`` stamp and never looked at disk.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# Canonical first-party parquet caches under the project ``output/`` directory,
# mapped to a staleness budget in hours. Budgets are sized to tolerate a 3-day
# holiday weekend (Fri write → Tue read ≈ 84h) without false-alarming, while
# still catching the real failure mode (a feed/job dead for many days). The
# price cache is intentionally absent: its path is a caller parameter
# (``load_eod_prices(price_file=...)``), not a fixed location, so it must be
# registered explicitly via ``register_path``. Override per source by passing
# ``specs`` to :func:`build_cache_freshness_monitor`.
DEFAULT_CACHE_SPECS: dict[str, float] = {
    "macro.parquet": 96.0,
    "macro_gpr.parquet": 96.0,
    "news_sentiment_daily.parquet": 96.0,
    "events_earnings.parquet": 96.0,
    "fundamentals.parquet": 168.0,
    "insider_form4.parquet": 168.0,  # real EDGAR Form 4 feed (legacy insider_trading.parquet retired)
    "dividends.parquet": 168.0,
}


@dataclass
class SourceFreshness:
    """Freshness status for a data source.

    If ``path`` is set the source is *file-backed*: freshness is judged against
    the file's mtime and the in-memory ``last_updated`` stamp is ignored.
    """

    source: str
    last_updated: datetime | None = None
    max_age_hours: float = 24.0
    path: Path | None = None

    def _mtime_utc(self) -> datetime | None:
        """Backing-file mtime as tz-aware UTC, or ``None`` when no path is set
        or the file is absent/unreadable."""
        if self.path is None:
            return None
        try:
            mtime = os.path.getmtime(self.path)
        except OSError as exc:
            # Absent/unreadable file → unknown (visible via degradation_status /
            # check_all), never silently treated as fresh. Logged for forensics.
            logger.debug("[Freshness] mtime read failed for '%s': %s", self.path, exc)
            return None
        return datetime.fromtimestamp(mtime, tz=timezone.utc)

    @property
    def effective_last_updated(self) -> datetime | None:
        """Timestamp freshness is judged against.

        File mtime for file-backed sources (``None`` when the file is missing —
        i.e. "never seen"), otherwise the in-memory ``last_updated`` stamp.
        """
        if self.path is not None:
            return self._mtime_utc()
        return self.last_updated

    @property
    def age_hours(self) -> float:
        last = self.effective_last_updated
        if last is None:
            return float("inf")
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
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

    def register_path(
        self, source: str, path: str | Path, max_age_hours: float = 24.0
    ) -> None:
        """Register a *file-backed* source whose freshness is the file mtime.

        Unlike :meth:`register` + :meth:`update`, no in-process stamp is needed:
        staleness reflects the on-disk cache directly, so a frozen or
        never-written parquet is detectable (audit DAT-003).
        """
        self.sources[source] = SourceFreshness(
            source=source, max_age_hours=max_age_hours, path=Path(path)
        )

    def update(self, source: str) -> None:
        if source in self.sources:
            self.sources[source].last_updated = datetime.now(timezone.utc)

    def check_all(self) -> list[dict[str, object]]:
        """Return one alert dict per source that is not ``ok``.

        Each alert carries an explicit ``status`` (``"stale"`` for an
        existing-but-old source, ``"unknown"`` for one never seen / whose
        backing file is absent) so a consumer can distinguish "frozen feed"
        from "no data ever" — the masking the original monitor could not
        express. ``age_hours`` is ``None`` for ``unknown`` (no finite age).

        The effective timestamp is snapshotted once per source so the reported
        ``status`` and ``age_hours`` are mutually consistent even if a backing
        file is rotated mid-check.
        """
        alerts: list[dict[str, object]] = []
        now = datetime.now(timezone.utc)
        for name, sf in self.sources.items():
            eff = sf.effective_last_updated  # single disk read for file-backed
            if eff is None:
                status = "unknown"
                age: float | None = None
            else:
                last = (
                    eff if eff.tzinfo is not None else eff.replace(tzinfo=timezone.utc)
                )
                age_h = (now - last).total_seconds() / 3600
                if age_h <= sf.max_age_hours:
                    continue  # ok — within budget
                status = "stale"
                age = round(age_h, 1)
            alert: dict[str, object] = {
                "source": name,
                "status": status,
                "age_hours": age,
                "max_age_hours": sf.max_age_hours,
            }
            if sf.path is not None:
                alert["path"] = str(sf.path)
            alerts.append(alert)
            if status == "unknown":
                logger.warning(
                    "[Freshness] Source '%s' has no data yet (unknown)%s",
                    name,
                    f" — expected at {sf.path}" if sf.path is not None else "",
                )
            else:
                logger.warning(
                    "[Freshness] Source '%s' is stale (%.1fh > %.1fh)",
                    name,
                    age,
                    sf.max_age_hours,
                )
        return alerts

    def last_known_good_timestamp(self, source: str) -> datetime | None:
        """Return the effective last-update for a source, or None if never
        recorded (file-backed: file absent).

        Audit C4-024: callers must distinguish "no data ever seen" from
        "stale but once-fresh data". The freshness alert pipeline always
        considered an unset value as infinitely-old; this helper exposes the
        underlying value so consumers can react differently (e.g. degrade to
        read-only mode when unknown, alert oncall when stale).
        """
        sf = self.sources.get(source)
        if sf is None:
            return None
        last = sf.effective_last_updated
        if last is None:
            return None
        if last.tzinfo is None:
            return last.replace(tzinfo=timezone.utc)
        return last

    def degradation_status(self, source: str) -> str:
        """Return one of ``unknown`` / ``ok`` / ``stale``.

        Operators can wire this into the /ready probe or a dashboard
        without re-implementing the unknown-vs-stale distinction.
        """
        sf = self.sources.get(source)
        if sf is None or sf.effective_last_updated is None:
            return "unknown"
        return "stale" if sf.is_stale else "ok"


def build_cache_freshness_monitor(
    output_dir: str | Path = "output",
    *,
    specs: dict[str, float] | None = None,
) -> FreshnessMonitor:
    """Build a monitor pre-wired to the canonical first-party parquet caches.

    ``specs`` (defaults to :data:`DEFAULT_CACHE_SPECS`) maps a file name under
    ``output_dir`` to its staleness budget in hours. The returned monitor reads
    mtimes live, so :meth:`FreshnessMonitor.check_all` surfaces a frozen or
    missing cache with no ``update`` call — closing the DAT-003 gap where cache
    staleness was never detected. Register the price cache separately via
    :meth:`FreshnessMonitor.register_path` since its path is caller-supplied.
    """
    out = Path(output_dir)
    monitor = FreshnessMonitor()
    for fname, max_age in (specs or DEFAULT_CACHE_SPECS).items():
        monitor.register_path(fname, out / fname, max_age_hours=max_age)
    return monitor


__all__ = [
    "SourceFreshness",
    "FreshnessMonitor",
    "build_cache_freshness_monitor",
    "DEFAULT_CACHE_SPECS",
]
