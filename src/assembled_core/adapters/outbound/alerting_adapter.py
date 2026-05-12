# src/assembled_core/adapters/outbound/alerting_adapter.py
"""AlertChannel adapter — wraps the existing ops/alerting.AlertManager.

This is a *thin* wrapper so the domain can program against the
``AlertChannel`` Protocol while the existing alerting.yaml + AlertManager
machinery keeps running untouched. The audit-mandated migration path:

1. Today  — domain code keeps calling AlertManager().fire() directly.
2. Now    — new domain code receives an ``AlertChannel`` via DI.
3. Later  — existing call-sites migrate one-by-one to the port.

There is no "big bang" — both paths coexist for the entire transition.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.assembled_core.ports.alert_channel import AlertChannel, Severity


class AlertManagerChannel(AlertChannel):
    """AlertChannel implementation backed by ops/alerting.AlertManager."""

    def __init__(self, manager: Any | None = None) -> None:
        # Lazy import — manager creation reads alerting.yaml, which
        # belongs to the adapter, never the domain.
        if manager is None:
            from src.assembled_core.ops.alerting import AlertManager

            manager = AlertManager()
        self._mgr = manager

    def fire(
        self,
        rule_name: str,
        context: Mapping[str, Any] | None = None,
        *,
        severity: Severity | None = None,
    ) -> bool:
        # severity is ignored — the rule's configured severity in
        # alerting.yaml is the source of truth. We accept the kwarg so
        # callers programmed against the port don't bend a knee to the
        # specific adapter implementation.
        del severity
        ctx_dict = dict(context) if context is not None else {}
        return bool(self._mgr.fire(rule_name, ctx_dict))


class NullAlertChannel(AlertChannel):
    """No-op adapter — useful in tests where we just want a port-shaped object."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def fire(
        self,
        rule_name: str,
        context: Mapping[str, Any] | None = None,
        *,
        severity: Severity | None = None,
    ) -> bool:
        del severity
        self.calls.append((rule_name, dict(context) if context else {}))
        return True


__all__ = ["AlertManagerChannel", "NullAlertChannel"]
