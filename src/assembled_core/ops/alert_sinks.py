"""Shared best-effort alert sinks (Discord + SMTP email).

Single source of truth for the human-visible alert path. Both the weekly
synthetic drill (``scripts/run_alert_drill.py``) and the production scheduler
health monitor (``scripts/check_scheduler_health.py``) post through here, so
the alert channel cannot silently diverge between "the drill that proves the
path works" and "the monitor that actually fires in production" — that
divergence is precisely the OPS-03 gap the 2026-04-10 stall postmortem flags.

Both functions are *best-effort*: any transport failure is logged and returns
``False``; they never raise, so an alert-delivery problem can never change a
caller's exit code or crash a monitoring run.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from email.mime.text import MIMEText
from urllib import error, request

logger = logging.getLogger(__name__)

__all__ = ["post_discord", "post_email_fallback"]


def post_discord(webhook: str, content: str) -> bool:
    """POST ``content`` to a Discord webhook. Returns True on 2xx.

    Never raises: a transport error is logged and returns False so the caller
    can fall back to email without special-casing exceptions.
    """
    data = json.dumps({"content": content}).encode("utf-8")
    req = request.Request(
        webhook,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as resp:
            return bool(200 <= resp.status < 300)
    except error.URLError as exc:
        logger.warning("[alert] Discord post failed: %s", exc)
        return False


def post_email_fallback(subject: str, body: str) -> bool:
    """Send an alert via SMTP email — used as Discord failover.

    Reads from ENV: ``SMTP_HOST``, ``SMTP_USER``, ``SMTP_PASS``,
    ``SMTP_PORT`` (default 587), ``ALERT_EMAIL_TO``. Silently skips (returns
    False) if any required credential is missing, so an unconfigured mailer is
    not treated as a delivery failure.
    """
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASS", "").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    to_addr = os.environ.get("ALERT_EMAIL_TO", "").strip()

    if not all([smtp_host, smtp_user, smtp_pass, to_addr]):
        logger.info("[alert] email failover skipped — SMTP credentials not configured")
        return False

    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_addr
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_addr], msg.as_string())
        logger.info("[alert] email fallback delivered to %s", to_addr)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[alert] email fallback failed: %s", exc)
        return False
