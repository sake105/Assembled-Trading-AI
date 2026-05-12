# src/assembled_core/ports/alert_channel.py
"""AlertChannel port — domain-facing API for firing alerts.

The domain says ``alerts.fire("kill_switch_activated", {"reason": ...})``;
the adapter routes that to Slack / Telegram / email / log per the
configured channels (configs/alerting.yaml). The domain does not care
which channel actually received the alert.

Audit C-001 + C-002 + F-007.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class Severity(str, Enum):
    """Severity tiers — match the existing alerting.yaml rule labels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    KILL = "kill"


@runtime_checkable
class AlertChannel(Protocol):
    """Fire a named alert rule with template context."""

    def fire(
        self,
        rule_name: str,
        context: Mapping[str, Any] | None = None,
        *,
        severity: Severity | None = None,
    ) -> bool:
        """Return True if at least one channel dispatched, False on cooldown / no-rule.

        ``severity`` is informational — the adapter may use it to route
        to a different channel group, but the alerting.yaml rule's
        configured severity wins on conflict.
        """
        ...
