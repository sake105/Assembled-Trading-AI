"""Crisis-Alpha state machine — persistent WATCH/ACTIVE/COOLDOWN/PAUSE.

Architecture mirrors the risk state machine (M3) but is scoped entirely
to the crisis-alpha sub-portfolio.  Key design choices:

- Four states: WATCH, ACTIVE, COOLDOWN, PAUSE
- Hysteresis: separate activate/deactivate score thresholds
- Cooldown timer: minimum hours before returning COOLDOWN→WATCH
- Daily loss guard: any loss >= limit forces PAUSE
- Health gate: ERROR health forces WATCH (never ACTIVE)
- Social-only guard: social-only geo signal cannot activate
- Atomic JSON persistence (same pattern as risk/state_machine.py)

State transition rules:
    WATCH → ACTIVE     : geo_score >= activate_threshold
                         AND geo_sources >= min_sources
                         AND NOT social_only
                         AND market_stress_ok
                         AND health_ok
    ACTIVE → COOLDOWN  : geo_score < deactivate_threshold
                         OR NOT health_ok
                         OR daily_loss_breached
    COOLDOWN → WATCH   : cooldown_hours elapsed since last ACTIVE exit
                         AND geo_score < deactivate_threshold
                         AND health_ok
    ANY → PAUSE        : daily_loss_breached AND NOT already PAUSE
    PAUSE → WATCH      : manual reset only (caller must provide reset=True)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext

logger = logging.getLogger(__name__)

CrisisState = Literal["WATCH", "ACTIVE", "COOLDOWN", "PAUSE"]

_DEFAULT_STATE_PATH = Path("output") / "ops" / "crisis_alpha_state.json"


# ---------------------------------------------------------------------------
# Persistent state record
# ---------------------------------------------------------------------------


@dataclass
class CrisisStateRecord:
    """Persistent record of the current crisis-alpha state.

    Attributes:
        state: Current state label.
        entered_at_utc: ISO timestamp when this state was entered.
        last_evaluated_utc: ISO timestamp of the last compute_next_state call.
        reason: Human-readable reason for the most recent transition.
        geo_score_at_entry: geo_score that triggered the last entry (for auditing).
        cooldown_start_utc: ISO timestamp when COOLDOWN was entered (for timer).
    """

    state: CrisisState = "WATCH"
    entered_at_utc: str = ""
    last_evaluated_utc: str = ""
    reason: str = "initial"
    geo_score_at_entry: float = 0.0
    cooldown_start_utc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CrisisStateRecord":
        return cls(
            state=d.get("state", "WATCH"),
            entered_at_utc=d.get("entered_at_utc", ""),
            last_evaluated_utc=d.get("last_evaluated_utc", ""),
            reason=d.get("reason", "loaded"),
            geo_score_at_entry=float(d.get("geo_score_at_entry", 0.0)),
            cooldown_start_utc=d.get("cooldown_start_utc", ""),
        )

    @classmethod
    def default(cls) -> "CrisisStateRecord":
        now = datetime.now(timezone.utc).isoformat()
        return cls(state="WATCH", entered_at_utc=now, last_evaluated_utc=now)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: dict, retries: int = 3) -> None:
    """Write JSON atomically using tempfile + os.replace with retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp", prefix=".crisis_state_"
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt < retries - 1:
                time.sleep(0.05)
            else:
                raise


def load_crisis_state(state_path: Path | str | None = None) -> CrisisStateRecord:
    """Load the persisted crisis-alpha state. Returns WATCH default on missing/corrupt."""
    path = Path(state_path) if state_path else _DEFAULT_STATE_PATH
    if not path.exists():
        return CrisisStateRecord.default()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return CrisisStateRecord.from_dict(data)
    except Exception as exc:
        logger.warning(
            "[WARN] crisis_alpha state file corrupt (%s) — resetting to WATCH: %s",
            path,
            exc,
        )
        return CrisisStateRecord.default()


def save_crisis_state(
    record: CrisisStateRecord, state_path: Path | str | None = None
) -> None:
    """Atomically persist the crisis-alpha state record."""
    path = Path(state_path) if state_path else _DEFAULT_STATE_PATH
    _atomic_write_json(path, record.to_dict())
    logger.info(
        "[CRISIS_STATE] saved state=%s reason=%r path=%s",
        record.state,
        record.reason,
        path,
    )


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def _get(policy: dict, *keys, default=None):
    """Traverse nested dict with dot-path keys, returning default if missing."""
    node = policy
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is None:
            return default
    return node


# ---------------------------------------------------------------------------
# Core transition function
# ---------------------------------------------------------------------------


def compute_next_crisis_state(
    ctx: CrisisAlphaContext,
    policy: dict,
    now_utc: datetime,
    prev: CrisisStateRecord,
    *,
    reset: bool = False,
) -> CrisisStateRecord:
    """Compute the next crisis-alpha state deterministically.

    This is a pure function (no I/O).  Callers must persist the result
    via ``save_crisis_state`` if they want it to survive across runs.

    Args:
        ctx: Current CrisisAlphaContext (geo score, health, daily PnL, …).
        policy: Policy dict (from crisis_alpha.yaml, section ``crisis_alpha``).
        now_utc: Current UTC datetime.
        prev: Previous persisted state record.
        reset: If True and state is PAUSE, force transition back to WATCH.

    Returns:
        New CrisisStateRecord (may be the same state or a new one).
    """
    now_str = now_utc.isoformat()

    # Shorthand for policy values
    cfg = _get(policy, "crisis_alpha", default={})
    activate_threshold: float = _get(
        cfg, "hysteresis", "activate_geo_score", default=2.0
    )
    deactivate_threshold: float = _get(
        cfg, "hysteresis", "deactivate_geo_score", default=1.0
    )
    min_sources: int = _get(cfg, "hysteresis", "min_sources", default=2)
    cooldown_hours: float = _get(cfg, "hysteresis", "cooldown_hours", default=24.0)

    current = prev.state
    reason = "no change"
    next_state = current

    # --- PAUSE reset (manual only) ---
    if current == "PAUSE" and reset:
        next_state = "WATCH"
        reason = "manual PAUSE reset"
        return CrisisStateRecord(
            state=next_state,
            entered_at_utc=now_str,
            last_evaluated_utc=now_str,
            reason=reason,
            geo_score_at_entry=ctx.geo_score,
            cooldown_start_utc="",
        )

    # --- Daily loss guard: any state → PAUSE ---
    if ctx.daily_loss_breached() and current != "PAUSE":
        next_state = "PAUSE"
        reason = f"daily loss breached: pnl={ctx.daily_pnl:.4f} limit={ctx.daily_loss_limit:.4f}"
        logger.warning("[CRISIS_STATE] %s → PAUSE | %s", current, reason)
        return CrisisStateRecord(
            state=next_state,
            entered_at_utc=now_str,
            last_evaluated_utc=now_str,
            reason=reason,
            geo_score_at_entry=ctx.geo_score,
            cooldown_start_utc="",
        )

    # --- ERROR health: force WATCH (never ACTIVE or stay ACTIVE) ---
    if not ctx.health_ok and current == "ACTIVE":
        next_state = "COOLDOWN"
        reason = "health not OK — forced ACTIVE→COOLDOWN"
        logger.warning("[CRISIS_STATE] %s → COOLDOWN | %s", current, reason)
        return CrisisStateRecord(
            state=next_state,
            entered_at_utc=now_str,
            last_evaluated_utc=now_str,
            reason=reason,
            geo_score_at_entry=ctx.geo_score,
            cooldown_start_utc=now_str,
        )

    # --- State-specific transitions ---
    if current == "WATCH":
        can_activate = (
            ctx.geo_score >= activate_threshold
            and ctx.geo_sources >= min_sources
            and not ctx.social_only
            and ctx.market_stress_ok
            and ctx.health_ok
        )
        if can_activate:
            next_state = "ACTIVE"
            reason = (
                f"geo_score={ctx.geo_score:.2f} >= {activate_threshold} | "
                f"sources={ctx.geo_sources} >= {min_sources} | "
                f"market_stress_ok | health_ok"
            )
            logger.info("[CRISIS_STATE] WATCH → ACTIVE | %s", reason)

    elif current == "ACTIVE":
        should_deactivate = ctx.geo_score < deactivate_threshold or not ctx.health_ok
        if should_deactivate:
            next_state = "COOLDOWN"
            reason = (
                f"geo_score={ctx.geo_score:.2f} < {deactivate_threshold} "
                if ctx.geo_score < deactivate_threshold
                else "health not OK"
            )
            logger.info("[CRISIS_STATE] ACTIVE → COOLDOWN | %s", reason)
        else:
            reason = f"still active: geo_score={ctx.geo_score:.2f}"

    elif current == "COOLDOWN":
        # Check cooldown timer
        cooldown_expired = False
        if prev.cooldown_start_utc:
            try:
                cooldown_start = datetime.fromisoformat(prev.cooldown_start_utc)
                elapsed_h = (now_utc - cooldown_start).total_seconds() / 3600
                cooldown_expired = elapsed_h >= cooldown_hours
            except ValueError:
                cooldown_expired = True  # parse failure → treat as expired
        else:
            cooldown_expired = True  # no start recorded → treat as expired

        if cooldown_expired and ctx.geo_score < deactivate_threshold and ctx.health_ok:
            next_state = "WATCH"
            reason = f"cooldown expired | geo_score={ctx.geo_score:.2f} | health_ok"
            logger.info("[CRISIS_STATE] COOLDOWN → WATCH | %s", reason)
        else:
            reason = "cooldown in progress"

    elif current == "PAUSE":
        # PAUSE only exits via manual reset (handled above)
        reason = "PAUSE — waiting for manual reset"

    # --- Build result ---
    state_changed = next_state != current
    cooldown_start = prev.cooldown_start_utc
    if next_state == "COOLDOWN" and current != "COOLDOWN":
        cooldown_start = now_str

    return CrisisStateRecord(
        state=next_state,
        entered_at_utc=now_str if state_changed else prev.entered_at_utc,
        last_evaluated_utc=now_str,
        reason=reason,
        geo_score_at_entry=ctx.geo_score if state_changed else prev.geo_score_at_entry,
        cooldown_start_utc=cooldown_start,
    )
