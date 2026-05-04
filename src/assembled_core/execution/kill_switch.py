"""Kill switch for emergency order blocking.

This module provides a kill switch mechanism to immediately block or throttle
all orders in emergency situations.

Activation sources (checked in order):
1. Environment variable ``ASSEMBLED_KILL_SWITCH`` (truthy value)
2. Sentinel file ``output/ops/.kill_switch_active``
3. Persistent JSON state file ``output/ops/kill_switch_state.json``

The persistent state supports **fractional throttling** (0–100 %) and keeps
an append-only **audit log** of every activation / deactivation event so that
ops can reconstruct what happened.

Usage::

    from src.assembled_core.execution.kill_switch import (
        is_kill_switch_engaged,
        guard_orders_with_kill_switch,
        activate_kill_switch,
        deactivate_kill_switch,
        get_kill_switch_state,
    )
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULT_SENTINEL = Path("output/ops/.kill_switch_active")
_DEFAULT_STATE_FILE = Path("output/ops/kill_switch_state.json")
_DEFAULT_AUDIT_LOG = Path("output/ops/kill_switch_audit.jsonl")


def _sentinel_path() -> Path:
    override = os.environ.get("ASSEMBLED_KILL_SWITCH_SENTINEL", "")
    return Path(override) if override else _DEFAULT_SENTINEL


def _state_path() -> Path:
    override = os.environ.get("ASSEMBLED_KILL_SWITCH_STATE", "")
    return Path(override) if override else _DEFAULT_STATE_FILE


def _audit_path() -> Path:
    override = os.environ.get("ASSEMBLED_KILL_SWITCH_AUDIT", "")
    return Path(override) if override else _DEFAULT_AUDIT_LOG


# ---------------------------------------------------------------------------
# Persistent state  (JSON file with file-level locking on Windows)
# ---------------------------------------------------------------------------

def _read_state() -> dict[str, Any]:
    """Read kill switch state from JSON file. Returns empty dict on error."""
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("[KillSwitch] Failed to read state file %s: %s", p, exc)
        return {}


def _write_state(state: dict[str, Any]) -> bool:
    """Atomically write kill switch state to JSON file.

    Returns:
        True if the write succeeded, False if it failed.
        Callers that perform safety-critical state changes (activate/deactivate)
        MUST check the return value and handle False appropriately.
    """
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(p)
        return True
    except Exception as exc:
        logger.error(
            "[KillSwitch] CRITICAL — failed to persist state file %s: %s. "
            "In-memory activation may not survive restart.",
            p,
            exc,
        )
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass  # cleanup best-effort
        return False


def _append_audit(event: dict[str, Any]) -> None:
    """Append a JSON-lines entry to the audit log."""
    p = _audit_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    event["ts"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception as exc:
        logger.error("[KillSwitch] Failed to write audit log %s: %s", p, exc)


# ---------------------------------------------------------------------------
# Public API: activate / deactivate / query
# ---------------------------------------------------------------------------

def activate_kill_switch(
    *,
    throttle_pct: float = 0.0,
    reason: str = "",
    actor: str = "system",
) -> None:
    """Activate the kill switch with persistent state.

    Args:
        throttle_pct: Fraction of orders to ALLOW (0.0 = block all, 0.25 = allow 25%).
            Must be in [0.0, 1.0].
        reason: Human-readable reason for activation.
        actor: Who/what triggered the activation.
    """
    throttle_pct = max(0.0, min(1.0, throttle_pct))
    state = {
        "engaged": True,
        "throttle_pct": throttle_pct,
        "reason": reason,
        "actor": actor,
        "activated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_ok = _write_state(state)
    _append_audit({"action": "ACTIVATE", "throttle_pct": throttle_pct, "reason": reason, "actor": actor})
    try:
        from src.assembled_core.ops.alerting import AlertManager
        AlertManager().fire("kill_switch_activated", {"reason": reason or "no reason given"})
    except Exception as _ae:
        logger.debug("[KillSwitch] alert dispatch failed: %s", _ae)
    if write_ok:
        logger.warning(
            "[KillSwitch] ACTIVATED — throttle=%.0f%%, reason=%s, actor=%s",
            throttle_pct * 100,
            reason,
            actor,
        )
    else:
        logger.critical(
            "[KillSwitch] ACTIVATION WRITE FAILED — state NOT persisted. "
            "Kill switch will NOT survive process restart. throttle=%.0f%%, reason=%s, actor=%s",
            throttle_pct * 100,
            reason,
            actor,
        )


def deactivate_kill_switch(*, reason: str = "", actor: str = "system") -> None:
    """Deactivate the kill switch and clear persistent state."""
    state = {
        "engaged": False,
        "throttle_pct": 1.0,
        "reason": reason,
        "actor": actor,
        "deactivated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_ok = _write_state(state)
    _append_audit({"action": "DEACTIVATE", "reason": reason, "actor": actor})
    if write_ok:
        logger.info("[KillSwitch] DEACTIVATED — reason=%s, actor=%s", reason, actor)
    else:
        logger.critical(
            "[KillSwitch] DEACTIVATION WRITE FAILED — state NOT persisted. "
            "Kill switch may re-engage on next read from stale state file. reason=%s, actor=%s",
            reason,
            actor,
        )


def get_kill_switch_state() -> dict[str, Any]:
    """Return the current kill switch state (persistent + env + sentinel)."""
    env_engaged = os.environ.get("ASSEMBLED_KILL_SWITCH", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    sentinel_engaged = _sentinel_path().exists()
    persistent = _read_state()
    persistent_engaged = persistent.get("engaged", False)
    throttle_pct = persistent.get("throttle_pct", 1.0) if persistent_engaged else 1.0

    any_engaged = env_engaged or sentinel_engaged or persistent_engaged

    return {
        "engaged": any_engaged,
        "throttle_pct": 0.0 if (env_engaged or sentinel_engaged) else throttle_pct,
        "sources": {
            "env_var": env_engaged,
            "sentinel_file": sentinel_engaged,
            "persistent_state": persistent_engaged,
        },
        "persistent": persistent,
    }


# ---------------------------------------------------------------------------
# Core checks (backward-compatible API)
# ---------------------------------------------------------------------------

def is_kill_switch_engaged() -> bool:
    """Check if kill switch is engaged via env var, sentinel file, or persistent state.

    Returns True if ANY source indicates the kill switch is active.
    """
    # Fast path: env var
    kill_env = os.environ.get("ASSEMBLED_KILL_SWITCH", "").strip().lower()
    if kill_env in {"1", "true", "yes", "on"}:
        return True

    # Sentinel file
    if _sentinel_path().exists():
        logger.warning("[KillSwitch] Sentinel file detected at %s", _sentinel_path())
        return True

    # Persistent state
    state = _read_state()
    if state.get("engaged", False):
        logger.warning(
            "[KillSwitch] Persistent state engaged (throttle=%.0f%%, reason=%s)",
            state.get("throttle_pct", 0.0) * 100,
            state.get("reason", "unknown"),
        )
        return True

    return False


def get_throttle_pct() -> float:
    """Return current throttle percentage (fraction of orders to allow).

    Returns 0.0 if kill switch is fully engaged (env or sentinel),
    the persistent throttle_pct if persistent state is engaged,
    or 1.0 if nothing is engaged.
    """
    state = get_kill_switch_state()
    if not state["engaged"]:
        return 1.0
    return state["throttle_pct"]


def check_drawdown_kill_switch(
    current_equity: float,
    peak_equity: float,
    kill_threshold: float = 0.30,
    auto_activate: bool = True,
) -> bool:
    """Check whether current drawdown breaches the kill-switch threshold.

    If the drawdown exceeds kill_threshold, logs a CRITICAL message
    and (by default) engages the kill switch automatically.

    Args:
        current_equity: Current portfolio equity value.
        peak_equity: Highest equity value observed (high-water mark).
        kill_threshold: Drawdown fraction that triggers the kill flag (default 0.30 = 30%).
        auto_activate: If True (default), automatically engage the kill switch
            when drawdown exceeds threshold.

    Returns:
        True if drawdown >= kill_threshold.
    """
    if peak_equity <= 0 or current_equity <= 0:
        return False
    drawdown = (peak_equity - current_equity) / peak_equity
    if drawdown >= kill_threshold:
        logger.critical(
            "[KillSwitch] Drawdown %.1f%% >= kill threshold %.1f%% "
            "(current=%.2f, peak=%.2f) — orders blocked",
            drawdown * 100,
            kill_threshold * 100,
            current_equity,
            peak_equity,
        )
        if auto_activate:
            activate_kill_switch(
                throttle_pct=0.0,
                reason=f"drawdown {drawdown:.1%} >= threshold {kill_threshold:.1%}",
                actor="drawdown_check",
            )
        return True
    return False


def guard_orders_with_kill_switch(orders: pd.DataFrame) -> pd.DataFrame:
    """Guard orders with kill switch — block or throttle orders.

    Behaviour:
    - Kill switch NOT engaged: return all orders unchanged.
    - Engaged with throttle_pct == 0.0: block ALL orders (empty DataFrame).
    - Engaged with 0 < throttle_pct < 1: scale all order quantities by throttle_pct
      and log which orders were affected.

    Args:
        orders: DataFrame with orders (must have 'qty' column for throttling).

    Returns:
        Filtered/scaled orders DataFrame.
    """
    state = get_kill_switch_state()

    if not state["engaged"]:
        return orders

    throttle = state["throttle_pct"]
    n_orders = len(orders)

    # Audit: record which orders were blocked/throttled
    symbols = list(orders["symbol"].unique()) if "symbol" in orders.columns and not orders.empty else []
    _append_audit({
        "action": "GUARD",
        "orders_count": n_orders,
        "symbols": symbols[:20],  # cap for log size
        "throttle_pct": throttle,
    })

    if throttle <= 0.0 or n_orders == 0:
        logger.warning(
            "[KillSwitch] BLOCKING all %d orders (throttle=0%%). Symbols: %s",
            n_orders,
            ", ".join(symbols[:10]),
        )
        return pd.DataFrame(columns=list(orders.columns))

    # Fractional throttle: scale quantities
    logger.warning(
        "[KillSwitch] THROTTLING %d orders to %.0f%%. Symbols: %s",
        n_orders,
        throttle * 100,
        ", ".join(symbols[:10]),
    )
    result = orders.copy()
    if "qty" in result.columns:
        # Throttle must preserve whole-share semantics: a fractional qty
        # downstream is either silently rounded (bias) or rejected by the
        # broker (order loss). Floor toward zero with sign preserved, then
        # drop orders that floored to zero.
        scaled = result["qty"].astype(float) * throttle
        result["qty"] = np.sign(scaled) * np.floor(np.abs(scaled))
        result = result[result["qty"].abs() >= 1].copy()
    return result
