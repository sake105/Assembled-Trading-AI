"""Multi-channel alert failover for disaster-recovery drills (Item 163).

Workflow:
1. Try to POST the alert to a Discord webhook (env: DISCORD_WEBHOOK_URL).
2. If Discord is unavailable (timeout, HTTP error, or env-var missing),
   fall back to email via SMTP (env: SMTP_HOST / SMTP_USER / SMTP_PASS / ALERT_EMAIL_TO).
3. Return a structured result dict so callers can log or assert on delivery.

Used by ``fail-drill.yml`` (weekly CI job) and the disaster-recovery runbook.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.text import MIMEText

log = logging.getLogger(__name__)

_DISCORD_TIMEOUT = 5.0  # seconds; Discord should respond fast
_EMAIL_TIMEOUT = 15.0  # seconds; SMTP can be slower


def _send_discord(webhook_url: str, message: str, subject: str) -> bool:
    """POST *message* to a Discord webhook. Returns True on success."""
    try:
        from src.assembled_core.utils.http_client import post

        payload = {"content": f"**[ALERT] {subject}**\n{message}"}
        resp = post(webhook_url, json=payload, timeout=_DISCORD_TIMEOUT)
        if resp.status_code in (200, 204):
            log.info(
                "[alert_failover] Discord delivery OK (status=%d)", resp.status_code
            )
            return True
        log.warning(
            "[alert_failover] Discord returned unexpected status %d", resp.status_code
        )
        return False
    except Exception as exc:
        log.warning("[alert_failover] Discord delivery failed: %s", exc)
        return False


def _send_email(message: str, subject: str) -> bool:
    """Send *message* via SMTP. Returns True on success."""
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    alert_to = os.environ.get("ALERT_EMAIL_TO", "")

    if not (smtp_host and smtp_user and smtp_pass and alert_to):
        log.warning("[alert_failover] Email not configured — missing SMTP env vars")
        return False

    try:
        msg = MIMEText(message)
        msg["Subject"] = f"[ALERT] {subject}"
        msg["From"] = smtp_user
        msg["To"] = alert_to

        with smtplib.SMTP(smtp_host, timeout=_EMAIL_TIMEOUT) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_user, [alert_to], msg.as_string())

        log.info("[alert_failover] Email delivery OK → %s", alert_to)
        return True
    except Exception as exc:
        log.error("[alert_failover] Email delivery failed: %s", exc)
        return False


def send_with_failover(
    message: str,
    subject: str = "Trading System Alert",
) -> dict[str, bool | str]:
    """Send *message* via Discord, falling back to email on failure.

    Args:
        message: Alert body text.
        subject: Short headline used in Discord bold prefix and email Subject.

    Returns:
        Dict with keys:
            ``discord_ok`` (bool), ``email_ok`` (bool), ``channel`` (str).
    """
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    discord_ok = False
    email_ok = False
    channel: str = "none"

    if webhook_url:
        discord_ok = _send_discord(webhook_url, message, subject)
        if discord_ok:
            channel = "discord"

    if not discord_ok:
        log.info("[alert_failover] Discord unavailable — attempting email fallback")
        email_ok = _send_email(message, subject)
        if email_ok:
            channel = "email"

    if not discord_ok and not email_ok:
        log.error(
            "[alert_failover] ALL channels failed — alert NOT delivered: %s", subject
        )

    return {"discord_ok": discord_ok, "email_ok": email_ok, "channel": channel}


def drill_failover_check(simulate_discord_failure: bool = False) -> dict[str, object]:
    """Run a synthetic alert delivery drill (used by fail-drill.yml).

    Args:
        simulate_discord_failure: If True, bypass Discord and go straight to email.

    Returns:
        Result dict from :func:`send_with_failover` plus ``"drill_passed"`` bool.
    """
    import os as _os

    if simulate_discord_failure:
        original = _os.environ.pop("DISCORD_WEBHOOK_URL", None)
        try:
            result = send_with_failover(
                message="Drill: Discord channel forced-failed → testing email fallback.",
                subject="Failover Drill",
            )
        finally:
            if original is not None:
                _os.environ["DISCORD_WEBHOOK_URL"] = original
    else:
        result = send_with_failover(
            message="Drill: synthetic alert to verify delivery path is live.",
            subject="Failover Drill",
        )

    result["drill_passed"] = bool(result.get("discord_ok") or result.get("email_ok"))
    return result


__all__ = ["send_with_failover", "drill_failover_check"]
