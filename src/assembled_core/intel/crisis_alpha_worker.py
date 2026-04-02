"""Crisis Alpha Worker — state machine for crisis mode detection and risk posture."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .models import (
    CrisisMode,
    CrisisState,
    DependencySignal,
    GeoTrigger,
)

# ---------------------------------------------------------------------------
# Risk postures per state
# ---------------------------------------------------------------------------

_RISK_POSTURE_NORMAL: dict[str, Any] = {
    "max_daily_loss_pct": 2.0,
    "per_trade_risk_pct": 1.0,
    "max_open_positions": 10,
    "max_trades_day": 20,
    "no_overnight": False,
}

_RISK_POSTURE_WATCH: dict[str, Any] = {
    "max_daily_loss_pct": 1.5,
    "per_trade_risk_pct": 0.5,
    "max_open_positions": 6,
    "max_trades_day": 12,
    "no_overnight": False,
}

_RISK_POSTURE_ACTIVE: dict[str, Any] = {
    "max_daily_loss_pct": 0.8,
    "per_trade_risk_pct": 0.15,
    "max_open_positions": 3,
    "max_trades_day": 6,
    "no_overnight": True,
}

_RISK_POSTURE_COOLDOWN: dict[str, Any] = {
    "max_daily_loss_pct": 1.2,
    "per_trade_risk_pct": 0.3,
    "max_open_positions": 5,
    "max_trades_day": 10,
    "no_overnight": False,
}

_RISK_POSTURES: dict[CrisisMode, dict[str, Any]] = {
    CrisisMode.NORMAL: _RISK_POSTURE_NORMAL,
    CrisisMode.WATCH: _RISK_POSTURE_WATCH,
    CrisisMode.ACTIVE: _RISK_POSTURE_ACTIVE,
    CrisisMode.COOLDOWN: _RISK_POSTURE_COOLDOWN,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CrisisStateConfig:
    geo_score_watch_threshold: int = 2
    geo_score_active_threshold: int = 3
    min_independent_sources_for_active: int = 2
    trigger_ttl_minutes: int = 360
    decay_half_life_minutes: int = 180
    cooldown_min_minutes: int = 720
    # Risk posture per state (can be overridden)
    risk_posture_by_state: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        CrisisMode.NORMAL.value: _RISK_POSTURE_NORMAL,
        CrisisMode.WATCH.value: _RISK_POSTURE_WATCH,
        CrisisMode.ACTIVE.value: _RISK_POSTURE_ACTIVE,
        CrisisMode.COOLDOWN.value: _RISK_POSTURE_COOLDOWN,
    })

    def risk_posture(self, mode: CrisisMode) -> dict[str, Any]:
        return self.risk_posture_by_state.get(mode.value, _RISK_POSTURE_NORMAL)


# ---------------------------------------------------------------------------
# State machine helpers
# ---------------------------------------------------------------------------


def _market_confirm_active(market_confirm: dict[str, Any]) -> bool:
    """
    Returns True if at least one market signal confirms crisis escalation:
    - oil price move > 2%
    - gold price move > 1%
    - VIX spike flag is True
    """
    oil_move = abs(float(market_confirm.get("oil_move", 0.0)))
    gold_move = abs(float(market_confirm.get("gold_move", 0.0)))
    vix_spike = bool(market_confirm.get("vix_spike", False))
    return oil_move > 2.0 or gold_move > 1.0 or vix_spike


def _all_triggers_expired(
    active_triggers: list[GeoTrigger],
    now: datetime,
) -> bool:
    """Return True if all triggers in the list have expired."""
    if not active_triggers:
        return True
    return all(t.is_expired(now) for t in active_triggers)


def _make_audit_entry(
    prev_mode: CrisisMode,
    new_mode: CrisisMode,
    reason: str,
    now: datetime,
    geo_score: int,
) -> dict[str, Any]:
    return {
        "ts": now.isoformat(),
        "prev_mode": prev_mode.value,
        "new_mode": new_mode.value,
        "reason": reason,
        "geo_score": geo_score,
    }


# ---------------------------------------------------------------------------
# Main state update function
# ---------------------------------------------------------------------------


def update_crisis_state(
    prev_state: CrisisState,
    geo_score: int,
    active_triggers: list[GeoTrigger],
    dependency_signal: DependencySignal | None,
    market_confirm: dict[str, Any],
    now: datetime,
    config: CrisisStateConfig | None = None,
) -> CrisisState:
    """
    Compute the next CrisisState given inputs.

    State transitions:
    - NORMAL  → WATCH:    geo_score >= watch_threshold
    - WATCH   → ACTIVE:   geo_score >= active_threshold AND market_confirm shows >=1 signal
    - ACTIVE  → COOLDOWN: geo_score drops below active_threshold OR all triggers expired
    - COOLDOWN→ NORMAL:   cooldown_min_minutes elapsed since entering COOLDOWN
    - Any     → NORMAL:   all triggers expired (hard reset)
    """
    if config is None:
        config = CrisisStateConfig()

    prev_mode = prev_state.mode
    audit_trail = list(prev_state.audit_trail)

    # Determine new mode
    new_mode = prev_mode  # default: no change

    # Check if all triggers are expired → hard reset to NORMAL
    all_expired = _all_triggers_expired(active_triggers, now)

    if all_expired and geo_score == 0:
        new_mode = CrisisMode.NORMAL
        if prev_mode != CrisisMode.NORMAL:
            audit_trail.append(_make_audit_entry(
                prev_mode, new_mode, "all_triggers_expired", now, geo_score
            ))
    elif prev_mode == CrisisMode.NORMAL:
        if geo_score >= config.geo_score_watch_threshold:
            new_mode = CrisisMode.WATCH
            audit_trail.append(_make_audit_entry(
                prev_mode, new_mode, f"geo_score_reached_{geo_score}", now, geo_score
            ))

    elif prev_mode == CrisisMode.WATCH:
        if geo_score >= config.geo_score_active_threshold and _market_confirm_active(market_confirm):
            new_mode = CrisisMode.ACTIVE
            audit_trail.append(_make_audit_entry(
                prev_mode, new_mode, "geo_score_and_market_confirm", now, geo_score
            ))
        elif geo_score < config.geo_score_watch_threshold:
            # Fell back below watch threshold
            new_mode = CrisisMode.NORMAL
            audit_trail.append(_make_audit_entry(
                prev_mode, new_mode, f"geo_score_dropped_{geo_score}", now, geo_score
            ))

    elif prev_mode == CrisisMode.ACTIVE:
        if all_expired or geo_score < config.geo_score_active_threshold:
            new_mode = CrisisMode.COOLDOWN
            audit_trail.append(_make_audit_entry(
                prev_mode, new_mode, "active_threshold_dropped_or_expired", now, geo_score
            ))

    elif prev_mode == CrisisMode.COOLDOWN:
        cooldown_elapsed = (now - prev_state.entered_at).total_seconds() / 60
        if cooldown_elapsed >= config.cooldown_min_minutes:
            new_mode = CrisisMode.NORMAL
            audit_trail.append(_make_audit_entry(
                prev_mode, new_mode, "cooldown_elapsed", now, geo_score
            ))

    # Determine when we entered the new mode
    if new_mode != prev_mode:
        entered_at = now
    else:
        entered_at = prev_state.entered_at

    # Build basket overrides from dependency signal
    basket_overrides: dict[str, list[str]] = {}
    if dependency_signal is not None:
        basket_overrides = dict(dependency_signal.basket_overrides)

    # Risk posture
    risk_posture = config.risk_posture(new_mode)

    return CrisisState(
        mode=new_mode,
        geo_score=geo_score,
        active_triggers=[t.trigger_id for t in active_triggers],
        dependency_signal_id=dependency_signal.signal_id if dependency_signal else None,
        entered_at=entered_at,
        risk_posture=dict(risk_posture),
        basket_overrides=basket_overrides,
        audit_trail=audit_trail,
    )
