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
