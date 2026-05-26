"""Dead-Man's Switch: auto-flatten all positions when heartbeat goes stale.

The DMS is a passive, time-based safety mechanism that monitors the heartbeat
file written by the trading cycle. If no heartbeat arrives within
``timeout_seconds``, the DMS triggers an auto-flatten via the kill switch.

The Kill-Switch (P2-1) is operator-initiated; the DMS is fully passive.

Design constraints:
- No circular imports: only imports from ops.heartbeat and execution.kill_switch.
- ``flatten_mode: shadow`` logs the event but does NOT call broker/kill_switch.
- Errors during flatten are logged and retried on the next interval; never
  silently swallowed.
- The monitor loop is interruptible via a threading.Event (stop_event) or
  via KeyboardInterrupt / SIGTERM (handled by the daemon script).

Policy keys read from the ``dead_man_switch`` block in policy.yaml:
    enabled                 (bool)  — false → loop exits immediately
    timeout_seconds         (float) — heartbeat max age before trigger
    check_interval_seconds  (float) — poll cadence
    flatten_mode            (str)   — "market" or "shadow"
    log_path                (str)   — JSONL audit output path
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.assembled_core.ops.heartbeat import check_liveness
from src.assembled_core.execution.kill_switch import activate_kill_switch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe defaults — used when the policy block is absent or incomplete
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "timeout_seconds": 900.0,  # 15 minutes
    "check_interval_seconds": 60.0,
    "flatten_mode": "market",
    "log_path": "output/ops/dms_audit.jsonl",
}


def _cfg(policy: dict[str, Any]) -> dict[str, Any]:
    """Merge policy.dead_man_switch with safe defaults (policy wins)."""
    block: dict[str, Any] = policy.get("dead_man_switch", {}) or {}
    merged: dict[str, Any] = dict(_DEFAULTS)
    merged.update(block)
    return merged


# ---------------------------------------------------------------------------
# Audit record writer
# ---------------------------------------------------------------------------


def record_dms_event(
    reason: str,
    log_path: Path,
    *,
    action_taken: str = "auto_flatten_triggered",
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a JSONL audit record to *log_path*.

    Args:
        reason:       Why the DMS triggered (e.g. "heartbeat_timeout").
        log_path:     Destination JSONL file.  Parent dirs are created if needed.
        action_taken: Short description of the action taken.
        extra:        Optional additional key/value pairs merged into the record.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "DMS_TRIGGER",
        "reason": reason,
        "action_taken": action_taken,
    }
    if extra:
        record.update(extra)

    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception as exc:  # noqa: BLE001
        logger.error("[DMS] Failed to write audit record to %s: %s", log_path, exc)


# ---------------------------------------------------------------------------
# Core flatten action
# ---------------------------------------------------------------------------


def auto_flatten_on_stale(
    policy: dict[str, Any],
    reason: str = "heartbeat_timeout",
) -> None:
    """Trigger auto-flatten when a stale heartbeat is detected.

    In ``shadow`` mode this function logs the event and writes the audit record
    but does NOT call activate_kill_switch — safe for testing and dry-run.

    In ``market`` mode it calls activate_kill_switch(throttle_pct=0.0) which
    blocks all orders system-wide.

    Args:
        policy: Full policy dict (reads ``dead_man_switch`` block).
        reason: Human-readable reason string included in logs and audit record.
    """
    cfg = _cfg(policy)
    flatten_mode: str = str(cfg.get("flatten_mode", "market")).lower()
    log_path = Path(str(cfg.get("log_path", _DEFAULTS["log_path"])))

    logger.critical(
        "[DMS-TRIGGER] Heartbeat stale — initiating auto-flatten. "
        "reason=%s flatten_mode=%s",
        reason,
        flatten_mode,
    )

    if flatten_mode == "shadow":
        logger.warning(
            "[DMS-TRIGGER] Shadow mode active — flatten logged but NOT executed."
        )
        record_dms_event(
            reason=reason,
            log_path=log_path,
            action_taken="shadow_log_only",
        )
        return

    # market mode: activate kill switch (blocks all orders)
    try:
        activate_kill_switch(
            throttle_pct=0.0,
            reason=f"DMS: {reason}",
            actor="dead_man_switch",
        )
        record_dms_event(
            reason=reason,
            log_path=log_path,
            action_taken="kill_switch_activated",
        )
        logger.critical(
            "[DMS-TRIGGER] Kill switch activated via DMS. All orders blocked."
        )
    except Exception as exc:
        # Log the failure clearly but do NOT swallow — caller retries.
        logger.error(
            "[DMS-TRIGGER] FAILED to activate kill switch: %s — will retry next interval.",
            exc,
        )
        record_dms_event(
            reason=reason,
            log_path=log_path,
            action_taken="kill_switch_activation_failed",
            extra={"error": str(exc)},
        )
        raise


# ---------------------------------------------------------------------------
# Monitor loop
# ---------------------------------------------------------------------------


def dms_monitor_loop(
    policy: dict[str, Any],
    *,
    stop_event: threading.Event | None = None,
    heartbeat_path: str | Path | None = None,
) -> None:
    """Infinite loop that monitors heartbeat liveness and triggers auto-flatten.

    The loop polls every ``check_interval_seconds``. It terminates when:
    - ``stop_event`` is set (clean shutdown), or
    - ``policy.dead_man_switch.enabled`` is False (immediate exit).

    If flatten fails, an ERROR is logged and the loop continues retrying on the
    next interval.

    Args:
        policy:         Full policy dict (reads ``dead_man_switch`` block).
        stop_event:     Optional threading.Event to signal clean shutdown.
        heartbeat_path: Override the heartbeat file path (useful for tests).
    """
    cfg = _cfg(policy)

    if not cfg.get("enabled", True):
        logger.info("[DMS] disabled via policy — exiting monitor loop immediately.")
        return

    # Warn when no policy block was provided — using hardcoded safe defaults.
    if "dead_man_switch" not in policy or not policy.get("dead_man_switch"):
        logger.warning(
            "[DMS] No policy block found — using hardcoded safe defaults "
            "(flatten_mode=market, timeout=900s)."
        )

    timeout: float = float(cfg.get("timeout_seconds", _DEFAULTS["timeout_seconds"]))
    interval: float = float(
        cfg.get("check_interval_seconds", _DEFAULTS["check_interval_seconds"])
    )

    logger.info(
        "[DMS] Monitor loop started. timeout=%ss interval=%ss mode=%s",
        timeout,
        interval,
        cfg.get("flatten_mode", "market"),
    )

    _consecutive_outer_failures: int = 0
    _OUTER_FAILURE_ESCALATION_THRESHOLD: int = 3
    _escalation_fired: bool = False

    while True:
        if stop_event is not None and stop_event.is_set():
            logger.info("[DMS] Stop event received — exiting monitor loop.")
            return

        try:
            liveness = check_liveness(
                heartbeat_path,
                max_age_seconds=timeout,
            )
            # Reset outer-failure counter on any successful check_liveness call.
            _consecutive_outer_failures = 0

            if not liveness.get("alive", False):
                reason = liveness.get("reason", "unknown")
                age = liveness.get("age_seconds")
                age_str = f"{age:.0f}s" if age is not None else "N/A"
                logger.critical(
                    "[DMS] Heartbeat stale: reason=%s age=%s — triggering auto-flatten.",
                    reason,
                    age_str,
                )
                try:
                    auto_flatten_on_stale(
                        policy,
                        reason=f"heartbeat_timeout (reason={reason}, age={age_str})",
                    )
                except Exception as flatten_exc:
                    # Already logged inside auto_flatten_on_stale; loop continues.
                    logger.error(
                        "[DMS] auto_flatten_on_stale raised: %s — will retry.",
                        flatten_exc,
                    )
            else:
                age = liveness.get("age_seconds")
                logger.debug(
                    "[DMS] Heartbeat OK (age=%.0fs, status=%s)",
                    age if age is not None else 0,
                    liveness.get("status", "?"),
                )

        except Exception as exc:  # noqa: BLE001
            _consecutive_outer_failures += 1
            logger.error("[DMS] Unexpected error in monitor loop: %s", exc)

            if (
                _consecutive_outer_failures > _OUTER_FAILURE_ESCALATION_THRESHOLD
                and not _escalation_fired
            ):
                logger.critical(
                    "[DMS-CRITICAL] check_liveness has raised %d consecutive times"
                    " — DMS may be inoperative. Triggering conservative auto-flatten.",
                    _consecutive_outer_failures,
                )
                try:
                    auto_flatten_on_stale(
                        policy,
                        reason=f"check_liveness_consecutive_failures_{_consecutive_outer_failures}",
                    )
                except Exception as flatten_exc:
                    logger.error(
                        "[DMS-CRITICAL] auto_flatten_on_stale also raised during escalation: %s",
                        flatten_exc,
                    )
                finally:
                    _escalation_fired = True
                    logger.critical(
                        "[DMS-CRITICAL] DMS is now in degraded mode — escalation already fired. "
                        "Continuing to monitor but kill switch is already active."
                    )

        # Sleep in small increments so stop_event is checked promptly.
        _interruptible_sleep(interval, stop_event)

        if stop_event is not None and stop_event.is_set():
            logger.info("[DMS] Stop event received after sleep — exiting monitor loop.")
            return


def _interruptible_sleep(
    seconds: float,
    stop_event: threading.Event | None,
    granularity: float = 1.0,
) -> None:
    """Sleep for *seconds* but wake early if *stop_event* is set."""
    elapsed = 0.0
    while elapsed < seconds:
        if stop_event is not None and stop_event.is_set():
            return
        chunk = min(granularity, seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk


__all__ = [
    "dms_monitor_loop",
    "auto_flatten_on_stale",
    "record_dms_event",
]
