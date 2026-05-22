"""Alert Manager (Plan 11.9).

Console/JSON/Webhook alerts with rate limiting.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

AlertLevel = Literal["INFO", "WARNING", "CRITICAL"]


@dataclass
class Alert:
    """A single alert."""

    level: AlertLevel
    source: str
    message: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    details: dict | None = None


class AlertManager:
    """Manages alerts with rate limiting and multiple output channels."""

    def __init__(
        self,
        rate_limit_seconds: float = 60.0,
        output_dir: str = "output/alerts",
    ) -> None:
        self.rate_limit = rate_limit_seconds
        self.output_dir = Path(output_dir)
        self._last_alert_time: dict[str, float] = defaultdict(float)
        self._alerts: list[Alert] = []

    def alert(
        self, level: AlertLevel, source: str, message: str, details: dict | None = None
    ) -> bool:
        """Send an alert if not rate-limited.

        Returns:
            True if alert was sent, False if rate-limited.
        """
        key = f"{source}:{message[:50]}"
        now = time.time()

        if now - self._last_alert_time[key] < self.rate_limit:
            return False

        self._last_alert_time[key] = now
        alert = Alert(level=level, source=source, message=message, details=details)
        self._alerts.append(alert)

        # Console output
        log_fn = (
            logger.info
            if level == "INFO"
            else logger.warning
            if level == "WARNING"
            else logger.critical
        )
        log_fn("[ALERT][%s] %s: %s", level, source, message)

        return True

    def flush_to_json(self) -> str | None:
        """Write accumulated alerts to JSON file."""
        if not self._alerts:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = (
            self.output_dir
            / f"alerts_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )

        data = [
            {
                "level": a.level,
                "source": a.source,
                "message": a.message,
                "timestamp": a.timestamp,
                "details": a.details,
            }
            for a in self._alerts
        ]
        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._alerts.clear()
        return str(filepath)

    @property
    def pending_count(self) -> int:
        return len(self._alerts)


__all__ = ["Alert", "AlertManager", "AlertLevel"]
