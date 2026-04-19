"""Simple freshness/health tracking for intel pipeline components."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .models import ComponentHealth


class HealthMonitor:
    """
    Tracks health and freshness for multiple pipeline components.

    Implements "stale-on-error" policy: if any tracked component is stale or in
    ERROR state, the overall system health is degraded and crisis_alpha_worker
    cannot go ACTIVE.
    """

    def __init__(self) -> None:
        self._components: dict[str, ComponentHealth] = {}

    def register(
        self,
        component_name: str,
        stale_threshold_minutes: int = 60,
    ) -> None:
        """Register a component with an initial STALE status (not yet updated)."""
        self._components[component_name] = ComponentHealth(
            component_name=component_name,
            last_updated=None,
            status="STALE",
            stale_threshold_minutes=stale_threshold_minutes,
        )

    def update(
        self,
        component: str,
        status: str = "OK",
        now: datetime | None = None,
    ) -> None:
        """Record an update for a component. Status should be 'OK', 'STALE', or 'ERROR'."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        if component not in self._components:
            # Auto-register with default threshold
            self.register(component)
        self._components[component].last_updated = now
        self._components[component].status = status

    def is_stale(self, component: str, now: datetime | None = None) -> bool:
        """Return True if the component is stale or in ERROR state."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        if component not in self._components:
            return True
        return self._components[component].is_stale(now)

    def snapshot(self, now: datetime | None = None) -> dict[str, dict[str, Any]]:
        """Return a snapshot of all component statuses."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        result: dict[str, dict[str, Any]] = {}
        for name, health in self._components.items():
            stale = health.is_stale(now)
            result[name] = {
                "component_name": name,
                "status": health.status,
                "last_updated": health.last_updated.isoformat() if health.last_updated else None,
                "stale_threshold_minutes": health.stale_threshold_minutes,
                "is_stale": stale,
            }
        return result

    def all_ok(self, now: datetime | None = None) -> bool:
        """
        Return True only if all registered components are fresh and OK.
        Implements stale-on-error: any stale or ERROR component returns False.
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)
        if not self._components:
            return False  # No components registered → not safe to proceed
        return all(
            not health.is_stale(now)
            for health in self._components.values()
        )

    def can_go_active(self, now: datetime | None = None) -> bool:
        """
        Explicit check for crisis_alpha_worker: can the system transition to ACTIVE?
        Returns False if any intel component is stale or in error.
        """
        return self.all_ok(now=now)

    def component_names(self) -> list[str]:
        """Return list of registered component names."""
        return list(self._components.keys())

    def mark_degraded(self, component: str) -> None:
        """Mark a component as degraded (WARN-level, not full ERROR)."""
        if component not in self._components:
            self.register(component)
        self._components[component].status = "DEGRADED"


class SourceUptimeTracker:
    """Per-source uptime and fetch-latency tracker (T6.3).

    Tracks consecutive_failures, last_success_ts, and rolling 30-day uptime %.
    """

    def __init__(self, window: int = 30) -> None:
        self._window = window  # rolling observation window
        # {source_id: {"attempts": int, "successes": int, "consecutive_failures": int,
        #               "last_success_ts": str|None, "last_latency_ms": float|None,
        #               "queue_depth": int, "latency_history": list[float]}}
        self._sources: dict[str, dict] = {}

    def record(
        self,
        source_id: str,
        *,
        success: bool,
        latency_ms: float | None = None,
        queue_depth: int | None = None,
    ) -> None:
        """Record a fetch attempt result for a source."""
        now = datetime.now(tz=timezone.utc).isoformat()
        if source_id not in self._sources:
            self._sources[source_id] = {
                "attempts": 0,
                "successes": 0,
                "consecutive_failures": 0,
                "last_success_ts": None,
                "last_latency_ms": None,
                "queue_depth": 0,
                "latency_history": [],
            }
        s = self._sources[source_id]
        s["attempts"] += 1
        s["last_latency_ms"] = latency_ms
        if latency_ms is not None:
            s["latency_history"].append(latency_ms)
            if len(s["latency_history"]) > self._window:
                s["latency_history"] = s["latency_history"][-self._window:]
        if queue_depth is not None:
            s["queue_depth"] = queue_depth
        if success:
            s["successes"] += 1
            s["consecutive_failures"] = 0
            s["last_success_ts"] = now
        else:
            s["consecutive_failures"] += 1

    def uptime_pct(self, source_id: str) -> float | None:
        """Return uptime % (successes/attempts) or None if no data."""
        s = self._sources.get(source_id)
        if not s or s["attempts"] == 0:
            return None
        return round(100.0 * s["successes"] / s["attempts"], 1)

    def is_degraded(self, source_id: str, *, max_consecutive_failures: int = 3) -> bool:
        """Return True if source has too many consecutive failures."""
        s = self._sources.get(source_id)
        if not s:
            return False
        return s["consecutive_failures"] >= max_consecutive_failures

    def p95_latency_ms(self, source_id: str) -> float | None:
        """Return p95 fetch latency in ms over the rolling window, or None if no data."""
        s = self._sources.get(source_id)
        if not s or not s["latency_history"]:
            return None
        hist = sorted(s["latency_history"])
        idx = max(0, int(len(hist) * 0.95) - 1)
        return hist[idx]

    def snapshot(self) -> dict[str, dict]:
        """Return uptime snapshot for all tracked sources."""
        result = {}
        for src, s in self._sources.items():
            result[src] = {
                **s,
                "uptime_pct": self.uptime_pct(src),
                "degraded": self.is_degraded(src),
                "p95_latency_ms": self.p95_latency_ms(src),
            }
        return result
