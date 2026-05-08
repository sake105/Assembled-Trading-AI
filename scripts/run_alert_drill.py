"""P1 A13 — Weekly synthetic alert-drill (Deep Run v2, 2026-04-18).

Purpose
-------
Verify the alert path by simulating a silent scheduler:

  1. Write a fake ``scheduler_heartbeat.json`` whose timestamp is
     deliberately stale (default: 120 min).
  2. Invoke ``scripts/check_scheduler_health.py`` with
     ``--ignore-market-hours``.
  3. Assert the health-check returned non-zero (the detector works).
  4. Persist a receipt in ``output/ops/alert_drill_<UTC-date>.json``
     so the drill outcome is auditable in CI artefacts.
  5. Optionally post a ``[DRILL]``-tagged notification to
     ``DISCORD_WEBHOOK`` to confirm the human-visible alert path.

Why this exists
---------------
On 2026-04-10 the paper-trading scheduler fell silent for 7 days and
nobody noticed — no heartbeat monitor ran, and even if one had, the
alert-path-to-human had never been exercised. A weekly drill is the
cheapest way to keep that path warm: if Discord silently stops
delivering (token rotated, channel archived, webhook revoked), the
drill is the first place it shows up.

Exit codes
----------
  0 — drill succeeded (stale heartbeat → detector flagged → receipt written)
  1 — drill failed (detector did NOT flag a stale heartbeat; alert path is broken)
  2 — drill could not run (environmental problem)

The workflow treats a **successful drill** as a *pass*, because the
business intent is "the detector fires when it should". A non-zero exit
from the drill means the monitoring pipeline is broken and needs
immediate attention.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import subprocess
import sys
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from urllib import error, request

logger = logging.getLogger("run_alert_drill")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
HEARTBEAT_PATH = REPO_ROOT / "output" / "ops" / "scheduler_heartbeat.json"
BACKUP_PATH = HEARTBEAT_PATH.with_suffix(".drill_backup.json")


def _write_stale_heartbeat(stale_minutes: int) -> dict:
    now = datetime.now(timezone.utc)
    stale_ts = now.timestamp() - stale_minutes * 60.0
    stale_iso = (
        datetime.fromtimestamp(stale_ts, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    payload = {
        "timestamp_utc": stale_iso,
        "status": "DRILL_STALE",
        "source": "run_alert_drill",
        "stale_by_minutes": stale_minutes,
        "note": (
            "This is a synthetic stale heartbeat written by P1 A13 drill. "
            "If you see this in production, the drill forgot to restore "
            "the original heartbeat — check BACKUP_PATH and rerun."
        ),
    }
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_health_check() -> subprocess.CompletedProcess:
    # --ignore-market-hours: the drill must succeed 24/7 regardless of when
    # CI happens to fire. The real detector's market-hours logic is tested
    # elsewhere (check_scheduler_health's own unit tests).
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "check_scheduler_health.py"),
        "--ignore-market-hours",
        "--stale-minutes",
        "10",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _post_discord(webhook: str, content: str) -> bool:
    data = json.dumps({"content": content}).encode("utf-8")
    req = request.Request(
        webhook,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except error.URLError as exc:
        logger.warning("[DRILL] Discord post failed: %s", exc)
        return False


def _post_email_fallback(subject: str, body: str) -> bool:
    """Send alert via SMTP email — used as Discord failover.

    Reads from ENV: SMTP_HOST, SMTP_USER, SMTP_PASS, SMTP_PORT (default 587),
    ALERT_EMAIL_TO.  Silently skips if any credential is missing.
    """
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASS", "").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    to_addr = os.environ.get("ALERT_EMAIL_TO", "").strip()

    if not all([smtp_host, smtp_user, smtp_pass, to_addr]):
        logger.info("[DRILL] email failover skipped — SMTP credentials not configured")
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
        logger.info("[DRILL] email fallback delivered to %s", to_addr)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[DRILL] email fallback failed: %s", exc)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stale-minutes",
        type=int,
        default=120,
        help="How stale the synthetic heartbeat should be (default: 120).",
    )
    parser.add_argument(
        "--no-discord",
        action="store_true",
        help="Skip the Discord notification even if DISCORD_WEBHOOK is set.",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Skip email fallback even if SMTP credentials are configured.",
    )
    parser.add_argument(
        "--receipt-dir",
        default=str(REPO_ROOT / "output" / "ops"),
        help="Where to write the drill receipt JSON.",
    )
    args = parser.parse_args(argv)

    # --- 1. Back up any existing heartbeat so we do not destroy prod state
    original_existed = HEARTBEAT_PATH.exists()
    if original_existed:
        BACKUP_PATH.write_bytes(HEARTBEAT_PATH.read_bytes())
        logger.info("[DRILL] backed up existing heartbeat → %s", BACKUP_PATH)

    try:
        # --- 2. Inject a stale heartbeat
        fake_payload = _write_stale_heartbeat(args.stale_minutes)
        logger.info(
            "[DRILL] wrote synthetic heartbeat stale by %d min", args.stale_minutes
        )

        # --- 3. Run the detector
        result = _run_health_check()
        detector_rc = result.returncode
        detector_stdout = result.stdout.strip()
        detector_stderr = result.stderr.strip()
        logger.info(
            "[DRILL] health-check rc=%d stdout=%r stderr=%r",
            detector_rc,
            detector_stdout[:200],
            detector_stderr[:200],
        )

        drill_ok = detector_rc != 0
        outcome = "pass" if drill_ok else "FAIL"

        # --- 4. Write receipt
        now_utc = datetime.now(timezone.utc)
        receipt = {
            "drill_id": now_utc.strftime("%Y%m%dT%H%M%SZ"),
            "timestamp_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "outcome": outcome,
            "detector_rc": detector_rc,
            "detector_stdout": detector_stdout,
            "detector_stderr": detector_stderr,
            "fake_heartbeat": fake_payload,
            "finding": "P1 A13 (Deep Run v2, 2026-04-18)",
        }
        receipt_dir = Path(args.receipt_dir)
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"alert_drill_{receipt['drill_id']}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        logger.info("[DRILL] receipt → %s", receipt_path)

        # --- 5. Discord (primary channel) with email fallback (item 163)
        webhook = os.environ.get("DISCORD_WEBHOOK", "").strip()
        discord_ok: bool | None = None
        email_ok: bool | None = None
        notification_msg = (
            f"[DRILL] {outcome} — alert-drill receipt "
            f"`{receipt_path.name}`. Detector rc={detector_rc}. "
            f"Scheduled test (P1 A13), no action required on pass. "
            f"On FAIL, the scheduler-health detector is broken."
        )
        if webhook and not args.no_discord:
            discord_ok = _post_discord(webhook, notification_msg)
            logger.info("[DRILL] Discord post delivered=%s", discord_ok)
            if not discord_ok:
                logger.warning("[DRILL] Discord failed — triggering email failover")

        # Email failover: send when Discord is absent, failed, or explicitly disabled
        if not args.no_email:
            should_email = discord_ok is False or (  # Discord present but failed
                webhook == "" and discord_ok is None
            )  # Discord not configured
            if should_email:
                email_ok = _post_email_fallback(
                    subject=f"[Trading Alert Drill] {outcome}",
                    body=notification_msg,
                )
                receipt["email_fallback_delivered"] = email_ok

        receipt["discord_delivered"] = discord_ok
        receipt["email_fallback_delivered"] = email_ok

        # Re-write receipt with channel delivery status
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

        return 0 if drill_ok else 1

    finally:
        # --- 6. Restore the real heartbeat so we never leak drill state
        if original_existed and BACKUP_PATH.exists():
            HEARTBEAT_PATH.write_bytes(BACKUP_PATH.read_bytes())
            BACKUP_PATH.unlink()
            logger.info("[DRILL] restored original heartbeat from backup")
        elif not original_existed and HEARTBEAT_PATH.exists():
            HEARTBEAT_PATH.unlink()
            logger.info("[DRILL] removed synthetic heartbeat (no original)")


if __name__ == "__main__":
    sys.exit(main())
