"""Alert sinks — Slack webhook + email (Sprint 4 / Plan C14).

Sidecar dispatch helpers that consume :class:`~src.assembled_core.ops.alert_manager.Alert`
objects and push them to external channels. The existing
``AlertManager`` is left untouched — callers opt in by instantiating
one of the sinks below and calling :func:`dispatch_alerts`, or by
calling the sink's ``send`` method directly.

Both sinks degrade gracefully:

* network / import errors are logged, never raised
* missing configuration returns a skip result instead of crashing
* secrets are read from environment variables by name, never hardcoded

The sinks accept either a single ``Alert`` or a list of them. They
return a small result dict so callers can count successes / failures.
"""

from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from typing import Any, Protocol

from .alert_manager import Alert

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
_SEVERITY_COLOR = {
    "INFO": "#2eb886",
    "WARNING": "#daa038",
    "CRITICAL": "#d0342c",
}


class AlertSink(Protocol):
    """Minimal protocol every sink must satisfy."""

    min_severity: str

    def send(self, alert: Alert) -> dict[str, Any]: ...


def _meets_severity(alert_level: str, min_level: str) -> bool:
    return _SEVERITY_ORDER.get(alert_level, 0) >= _SEVERITY_ORDER.get(min_level, 0)


@dataclass
class SlackWebhookSink:
    """Post alerts to a Slack incoming webhook URL.

    The webhook URL is read from an environment variable so the URL
    never lands in config files or commit history.
    """

    webhook_url_env: str = "ASSEMBLED_SLACK_WEBHOOK_URL"
    min_severity: str = "WARNING"
    timeout_seconds: float = 5.0

    def send(self, alert: Alert) -> dict[str, Any]:
        if not _meets_severity(alert.level, self.min_severity):
            return {"status": "skipped", "reason": "below_min_severity"}

        url = os.environ.get(self.webhook_url_env, "").strip()
        if not url:
            return {"status": "skipped", "reason": f"env_var_{self.webhook_url_env}_unset"}

        try:
            import requests  # local import to keep module import-cheap
        except Exception as exc:  # noqa: BLE001
            logger.warning("[alert_sinks] requests not available: %s", exc)
            return {"status": "error", "reason": "requests_missing"}

        payload = {
            "text": f"[{alert.level}] {alert.source}: {alert.message}",
            "attachments": [
                {
                    "color": _SEVERITY_COLOR.get(alert.level, "#888888"),
                    "fields": [
                        {"title": k, "value": str(v), "short": True}
                        for k, v in (alert.details or {}).items()
                    ],
                    "ts": alert.timestamp,
                }
            ],
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[alert_sinks] slack post failed: %s", exc)
            return {"status": "error", "reason": f"post_failed:{exc!r}"}

        if 200 <= resp.status_code < 300:
            return {"status": "sent", "http_status": resp.status_code}
        return {"status": "error", "http_status": resp.status_code}


@dataclass
class EmailSink:
    """Send alerts via SMTP.

    Host / port / credentials are read from environment variables to
    keep them out of config files. The sink is stateless beyond its
    configuration — each ``send`` call opens a fresh SMTP session.
    """

    smtp_host_env: str = "ASSEMBLED_SMTP_HOST"
    smtp_port_env: str = "ASSEMBLED_SMTP_PORT"
    smtp_user_env: str = "ASSEMBLED_SMTP_USER"
    smtp_pass_env: str = "ASSEMBLED_SMTP_PASS"
    from_addr: str = "trading-alerts@example.com"
    to_addrs: list[str] = field(default_factory=list)
    min_severity: str = "CRITICAL"
    use_tls: bool = True
    timeout_seconds: float = 10.0

    def send(self, alert: Alert) -> dict[str, Any]:
        if not _meets_severity(alert.level, self.min_severity):
            return {"status": "skipped", "reason": "below_min_severity"}
        if not self.to_addrs:
            return {"status": "skipped", "reason": "no_recipients"}

        host = os.environ.get(self.smtp_host_env, "").strip()
        if not host:
            return {"status": "skipped", "reason": f"env_var_{self.smtp_host_env}_unset"}

        try:
            port = int(os.environ.get(self.smtp_port_env, "587"))
        except ValueError:
            port = 587

        user = os.environ.get(self.smtp_user_env, "")
        password = os.environ.get(self.smtp_pass_env, "")

        subject = f"[{alert.level}] {alert.source}: {alert.message[:80]}"
        body_lines = [
            f"Source:  {alert.source}",
            f"Level:   {alert.level}",
            f"Time:    {alert.timestamp}",
            f"Message: {alert.message}",
        ]
        if alert.details:
            body_lines.append("")
            body_lines.append("Details:")
            for k, v in alert.details.items():
                body_lines.append(f"  {k}: {v}")
        body = "\n".join(body_lines)

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)

        try:
            with smtplib.SMTP(host, port, timeout=self.timeout_seconds) as smtp:
                if self.use_tls:
                    smtp.starttls()
                if user and password:
                    smtp.login(user, password)
                smtp.sendmail(self.from_addr, self.to_addrs, msg.as_string())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[alert_sinks] smtp send failed: %s", exc)
            return {"status": "error", "reason": f"smtp_failed:{exc!r}"}

        return {"status": "sent", "recipients": len(self.to_addrs)}


def dispatch_alerts(
    alerts: Alert | list[Alert],
    sinks: list[AlertSink],
) -> list[dict[str, Any]]:
    """Send ``alerts`` through every sink. Never raises.

    Returns a flat list of result dicts in ``alert × sink`` order.
    """
    if isinstance(alerts, Alert):
        alerts = [alerts]
    results: list[dict[str, Any]] = []
    for alert in alerts:
        for sink in sinks:
            try:
                res = sink.send(alert)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[alert_sinks] sink %s raised: %s", type(sink).__name__, exc)
                res = {"status": "error", "reason": f"sink_exception:{exc!r}"}
            results.append(
                {
                    "sink": type(sink).__name__,
                    "alert_level": alert.level,
                    "alert_source": alert.source,
                    **res,
                }
            )
    return results


__all__ = [
    "AlertSink",
    "SlackWebhookSink",
    "EmailSink",
    "dispatch_alerts",
]
