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

import contextlib
import hashlib
import hmac
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


def _lock_path() -> Path:
    """Resolve the cross-process lock file guarding state + audit writes.

    Co-located with the state file by default so that an env-overridden state
    directory (``ASSEMBLED_KILL_SWITCH_STATE``, e.g. a test ``tmp_path``) gets
    its own lock and never contends with the real ``output/ops`` lock.
    """
    override = os.environ.get("ASSEMBLED_KILL_SWITCH_LOCK", "")
    if override:
        return Path(override)
    return _state_path().parent / ".kill_switch.lock"


@contextlib.contextmanager
def _kill_switch_lock():
    """Serialize state + audit-chain writes across threads AND processes (OPS-04).

    ``_write_state`` and ``_append_audit`` are read-modify-write on shared
    files. Without a lock two concurrent writers — and the DMS daemon and the
    runner drawdown check are *separate processes* (independent Windows Tasks) —
    can (a) clobber each other's shared ``.tmp`` file and (b) read the same
    ``prev_hash`` and fork the hash-chained audit log, which
    ``verify_audit_chain`` then reports as tampered. A cross-process
    ``filelock.FileLock`` (the same primitive ``ops/paper_ledger.py`` uses)
    closes both.

    Degradation contract — a kill switch must stay *live*: if ``filelock`` is
    not installed, or the lock cannot be acquired within the timeout, the
    safety action is NOT blocked. It proceeds unserialized with an explicit
    error/warning log; liveness of the activation outweighs a rare, logged
    race under pathological (>10 s) contention.
    """
    lp = _lock_path()
    try:
        lp.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    try:
        from filelock import FileLock, Timeout
    except ImportError:
        logger.warning(
            "[KillSwitch] filelock not installed — state/audit writes are NOT "
            "serialized across processes (OPS-04 protection inactive)."
        )
        yield
        return

    # Acquire the lock SEPARATELY from the yield so this manager yields exactly
    # once no matter what the wrapped body does. Folding acquisition into a
    # ``with FileLock(): yield`` would let a body-thrown ``Timeout`` re-enter the
    # ``except`` and yield a second time (RuntimeError: generator didn't stop).
    lock = FileLock(str(lp), timeout=10)
    try:
        lock.acquire()
    except Timeout:
        logger.error(
            "[KillSwitch] could not acquire %s within 10s — proceeding WITHOUT "
            "lock (audit-chain fork possible under this contention).",
            lp,
        )
        lock = None
    try:
        yield
    finally:
        if lock is not None:
            lock.release()


# ---------------------------------------------------------------------------
# Persistent state + hash-chained audit (cross-process file lock — OPS-04)
# ---------------------------------------------------------------------------


def _read_state() -> dict[str, Any]:
    """Read kill switch state from JSON file. Returns empty dict on error.

    Back-compat contract: a MISSING file and a CORRUPT/unreadable file both
    return ``{}`` here so existing internal callers (``get_kill_switch_state``)
    keep working. The *safety* distinction between "no file" (legitimately
    disengaged) and "file present but unreadable" (unknown state -> must
    fail-closed) is made separately by ``_persistent_state_corrupt`` and acted
    on in ``is_kill_switch_engaged``. Do NOT collapse a corrupt-present file
    into a silent disengaged decision on the engaged-decision path.
    """
    p = _state_path()
    if not p.exists():
        return {}
    try:
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return data
    except Exception as exc:
        logger.error("[KillSwitch] Failed to read state file %s: %s", p, exc)
        return {}


def _persistent_state_corrupt() -> bool:
    """Return True iff a persistent state file is PRESENT but unreadable/invalid.

    Distinguishes the two cases ``_read_state`` flattens to ``{}``:

    - File MISSING            -> ``False`` (never-engaged default; disengaged).
    - File PRESENT but cannot
      be read / JSON-parsed /
      shaped as a dict        -> ``True`` (unknown switch state -> fail-closed).

    Used only by ``is_kill_switch_engaged`` to fail CLOSED on an unreadable but
    present state file: blocking on unknown state is the safe default for a
    safety kill switch. Uses ``Path.exists()`` to reliably tell "no file" from
    "file present but parse/read failed". A transient ``OSError`` on a present
    path counts as corrupt (block on unknown state), not as missing.
    """
    p = _state_path()
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # Present-but-unparseable (bad JSON) or unreadable (I/O error) on a path
        # that exists() reported as present -> treat as corrupt -> fail-closed.
        return True
    # Parsed but not a JSON object -> not a valid state document -> corrupt.
    return not isinstance(data, dict)


def _write_state(state: dict[str, Any]) -> bool:
    """Atomically write kill switch state to JSON file with fsync durability.

    Survives power-failure / OS-crash mid-write: bytes are flushed to disk,
    the rename is atomic on POSIX (best-effort on Windows), and the parent
    directory entry is fsync'd so the rename is itself durable.

    Returns:
        True if the write succeeded, False if it failed.
        Callers that perform safety-critical state changes (activate/deactivate)
        MUST check the return value and handle False appropriately.
    """
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    # Hold the cross-process lock for the whole tmp+replace: two concurrent
    # writers share the same ``.tmp`` path, so without the lock one would
    # truncate the other's tmp file mid-write (OPS-04).
    with _kill_switch_lock():
        try:
            payload = json.dumps(state, indent=2, sort_keys=True) + "\n"
            # Write + fsync the data file before rename.
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as exc:
                    # fsync can fail on some filesystems (network mounts, /tmp on
                    # certain CI images). The rename is still safer than nothing —
                    # log and continue rather than fail the whole write.
                    logger.warning("[KillSwitch] fsync of tmp file failed: %s", exc)
            os.replace(tmp, p)
            # fsync the directory so the rename is durable across crash.
            try:
                dir_fd = os.open(str(p.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except (OSError, AttributeError):
                # Windows does not support os.fsync on directory handles. Skip
                # silently — os.replace is already best-effort atomic on Win32.
                pass
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


_GENESIS_HASH = "0" * 64  # SHA-256 zero anchor for first audit entry


def _last_audit_hash(p: Path) -> str:
    """Return the SHA-256 ``hash`` of the last JSONL entry, or _GENESIS_HASH."""
    if not p.exists():
        return _GENESIS_HASH
    try:
        # Read the file line-buffered and keep the last non-empty record.
        # Audit logs are append-only, so the simple last-line read is OK;
        # for very large logs callers can rotate to keep this cheap.
        last_obj: dict[str, Any] | None = None
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last_obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
        if last_obj is None:
            return _GENESIS_HASH
        return str(last_obj.get("hash", _GENESIS_HASH))
    except Exception as exc:
        logger.warning("[KillSwitch] could not read prev audit hash: %s", exc)
        return _GENESIS_HASH


def _append_audit(event: dict[str, Any]) -> None:
    """Append a JSON-lines entry to the audit log with hash-chain for tamper detection.

    Each record carries:
        - ``ts``        ISO-8601 UTC timestamp (also used in the digest)
        - ``prev_hash`` SHA-256 of the previous record's full payload
        - ``hash``      SHA-256 of the current record minus the ``hash`` field

    A verifier can recompute the chain by stripping ``hash``, sorting keys, and
    hashing; any tampering between the first record and now breaks the chain.
    File is fsync'd for durability.
    """
    p = _audit_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    event["ts"] = datetime.now(timezone.utc).isoformat()
    # The prev_hash read + append MUST be atomic across writers, else two
    # concurrent appends read the same prev_hash and fork the chain (OPS-04
    # TOCTOU). Hold the cross-process lock around the whole read-modify-write.
    with _kill_switch_lock():
        event["prev_hash"] = _last_audit_hash(p)
        # Compute hash over the record without the "hash" key itself to make the
        # chain self-verifiable.
        digest_payload = json.dumps(event, sort_keys=True).encode("utf-8")
        event["hash"] = hashlib.sha256(digest_payload).hexdigest()
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as exc:
                    logger.warning("[KillSwitch] fsync of audit log failed: %s", exc)
        except Exception as exc:
            logger.error("[KillSwitch] Failed to write audit log %s: %s", p, exc)


def verify_audit_chain(path: Path | str | None = None) -> tuple[bool, int]:
    """Recompute the SHA-256 chain of the audit log and report integrity.

    Args:
        path: optional override of the audit-log path.

    Returns:
        ``(ok, n_records)``. ``ok`` is True iff every record's ``hash`` and
        ``prev_hash`` match the chain; False on any inconsistency. ``n_records``
        is the number of JSONL entries inspected.
    """
    p = Path(path) if path is not None else _audit_path()
    if not p.exists():
        return True, 0
    expected_prev = _GENESIS_HASH
    n = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return False, n
            recorded_hash = rec.pop("hash", None)
            if rec.get("prev_hash") != expected_prev:
                return False, n
            recomputed = hashlib.sha256(
                json.dumps(rec, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if recorded_hash != recomputed:
                return False, n
            expected_prev = recorded_hash
    return True, n


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
    _append_audit(
        {
            "action": "ACTIVATE",
            "throttle_pct": throttle_pct,
            "reason": reason,
            "actor": actor,
        }
    )
    try:
        from src.assembled_core.ops.alerting import AlertManager

        AlertManager().fire(
            "kill_switch_activated", {"reason": reason or "no reason given"}
        )
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


def deactivate_kill_switch(
    *,
    reason: str = "",
    actor: str = "system",
    operator_token: str | None = None,
) -> None:
    """Deactivate the kill switch and clear persistent state.

    Requires a valid OPERATOR_KILL_TOKEN.  If the env var is absent, or the
    supplied token does not match, the call is rejected and a
    REJECT_DEACTIVATE entry is appended to the audit log.

    Raises:
        PermissionError: when the token is missing or incorrect.
    """
    _expected = os.environ.get("OPERATOR_KILL_TOKEN", "")
    if not _expected:
        _append_audit(
            {
                "action": "REJECT_DEACTIVATE",
                "reason": "OPERATOR_KILL_TOKEN env var not set",
                "actor": actor,
            }
        )
        logger.warning(
            "[KillSwitch] REJECT deactivate — OPERATOR_KILL_TOKEN not set. actor=%s",
            actor,
        )
        raise PermissionError(
            "Kill switch deactivation requires OPERATOR_KILL_TOKEN to be set in the environment"
        )
    if not hmac.compare_digest((operator_token or "").encode(), _expected.encode()):
        _append_audit(
            {
                "action": "REJECT_DEACTIVATE",
                "reason": "invalid operator token",
                "actor": actor,
            }
        )
        logger.warning(
            "[KillSwitch] REJECT deactivate — invalid operator token. actor=%s",
            actor,
        )
        raise PermissionError(
            "Kill switch deactivation requires a valid operator token"
        )
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
        "1",
        "true",
        "yes",
        "on",
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

    # Fail-CLOSED on a corrupt/unreadable-but-PRESENT persistent state file.
    # A present state file we cannot read/parse means the switch state is
    # UNKNOWN — a never-engaged switch leaves NO file, so a present-yet-broken
    # file is an anomaly that must conservatively BLOCK trading rather than
    # silently fall through to ``{}.get("engaged") -> False`` (fail-OPEN). A
    # legitimately MISSING file is handled below by ``_read_state() -> {}``
    # and stays disengaged.
    if _persistent_state_corrupt():
        logger.error(
            "[KillSwitch] Persistent state file %s is PRESENT but unreadable/"
            "corrupt — kill-switch state is UNKNOWN, blocking trading "
            "conservatively (fail-closed / treated as ENGAGED).",
            _state_path(),
        )
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
    return float(state["throttle_pct"])


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
    symbols = (
        list(orders["symbol"].unique())
        if "symbol" in orders.columns and not orders.empty
        else []
    )
    _append_audit(
        {
            "action": "GUARD",
            "orders_count": n_orders,
            "symbols": symbols[:20],  # cap for log size
            "throttle_pct": throttle,
        }
    )

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
