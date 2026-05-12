"""Multi-channel alerting system (Plan 11/10 §4.1).

Reads configs/alerting.yaml. Fire events are dispatched to all
matching channels with cooldown logic. Channels: telegram, email, log_only.
Credentials are always read from environment variables — never hardcoded.
"""

from __future__ import annotations

import logging
import os
import smtplib
import time
from collections import defaultdict
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("configs/alerting.yaml")


class AlertManager:
    """Dispatch alerts to configured channels with per-rule cooldowns."""

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG) -> None:
        path = Path(config_path)
        if not path.exists():
            logger.warning("[alerting] config not found at %s — alerts disabled", path)
            self._rules: list[dict] = []
            self._channels: dict[str, list[dict]] = {}
            return

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        alert_cfg = cfg.get("alerts", {})
        self._rules = alert_cfg.get("rules", [])
        self._channels = alert_cfg.get("channels", {})
        self._last_fired: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------
    def fire(self, rule_name: str, context: dict[str, Any] | None = None) -> bool:
        """Fire a named alert rule. Returns True if dispatched, False if skipped.

        Args:
            rule_name: Must match a rule.name in alerting.yaml.
            context: Template variables for the rule's message string.
        """
        context = context or {}
        rule = self._find_rule(rule_name)
        if not rule:
            logger.debug("[alerting] unknown rule '%s'", rule_name)
            return False

        # Cooldown check
        cooldown_s = rule.get("cooldown_minutes", 30) * 60
        since_last = time.time() - self._last_fired[rule_name]
        if cooldown_s > 0 and since_last < cooldown_s:
            logger.debug(
                "[alerting] rule '%s' in cooldown (%.0fs remaining)",
                rule_name,
                cooldown_s - since_last,
            )
            return False

        severity = rule.get("severity", "info")
        try:
            message = rule["message"].format(**context)
        except KeyError as exc:
            message = f"{rule.get('message', rule_name)} [missing key: {exc}]"

        for channel_cfg in self._channels.get(severity, []):
            self._dispatch(channel_cfg, message, rule_name, severity)

        self._last_fired[rule_name] = time.time()
        return True

    # ------------------------------------------------------------------
    def _find_rule(self, name: str) -> dict | None:
        for r in self._rules:
            if r.get("name") == name:
                return r
        return None

    def _dispatch(
        self, channel_cfg: dict, message: str, rule_name: str, severity: str
    ) -> None:
        kind = channel_cfg.get("type", "log_only")

        if kind == "log_only":
            log_fn = (
                logger.critical
                if severity == "critical"
                else (logger.warning if severity == "warning" else logger.info)
            )
            log_fn("[ALERT][%s] %s", rule_name, message)

        elif kind == "telegram":
            try:
                self._send_telegram(channel_cfg, f"[{severity.upper()}] {message}")
            except Exception as exc:
                logger.error("[alerting] telegram dispatch failed: %s", exc)

        elif kind == "email":
            try:
                self._send_email(channel_cfg, rule_name, message)
            except Exception as exc:
                logger.error("[alerting] email dispatch failed: %s", exc)

        elif kind == "slack":
            try:
                self._send_slack(channel_cfg, f"[{severity.upper()}] {message}")
            except Exception as exc:
                logger.error("[alerting] slack dispatch failed: %s", exc)

        else:
            logger.warning("[alerting] unknown channel type '%s'", kind)

    def _send_slack(self, cfg: dict, text: str) -> None:
        """Post to a Slack incoming-webhook URL. Webhook URL read from env."""
        import json as _json
        import urllib.request

        webhook = os.environ.get(cfg.get("webhook_env", "SLACK_WEBHOOK_URL"), "")
        if not webhook:
            logger.warning("[alerting] slack webhook not set in environment")
            return
        payload = _json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            webhook, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 204):
                logger.warning("[alerting] slack returned HTTP %s", resp.status)

    # ------------------------------------------------------------------
    def _send_telegram(self, cfg: dict, text: str) -> None:
        import urllib.request
        import json as _json

        token = os.environ.get(cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN"), "")
        chat_id = os.environ.get(cfg.get("chat_id_env", "TELEGRAM_CHAT_ID"), "")
        if not token or not chat_id:
            logger.warning("[alerting] telegram credentials not set in environment")
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = _json.dumps({"chat_id": chat_id, "text": text}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                logger.warning("[alerting] telegram returned HTTP %s", resp.status)

    def _send_email(self, cfg: dict, subject: str, body: str) -> None:
        to_addr = os.environ.get(cfg.get("to_env", "ALERT_EMAIL_TO"), "")
        smtp_host = os.environ.get(cfg.get("smtp_host_env", "SMTP_HOST"), "")
        smtp_user = os.environ.get(cfg.get("smtp_user_env", "SMTP_USER"), "")
        smtp_pass = os.environ.get(cfg.get("smtp_pass_env", "SMTP_PASS"), "")
        smtp_port = int(cfg.get("smtp_port", 587))

        if not all([to_addr, smtp_host, smtp_user, smtp_pass]):
            logger.warning("[alerting] email credentials incomplete — skipping")
            return

        msg = MIMEText(body)
        msg["Subject"] = f"[Assembled-Trading] {subject}"
        msg["From"] = smtp_user
        msg["To"] = to_addr

        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_addr], msg.as_string())


# Convenience singleton — lazy-initialised on first call
_MANAGER: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = AlertManager()
    return _MANAGER


def fire_alert(rule_name: str, context: dict[str, Any] | None = None) -> bool:
    """Module-level convenience function."""
    return get_alert_manager().fire(rule_name, context)
