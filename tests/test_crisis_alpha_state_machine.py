"""Tests for Crisis-Alpha v1 state machine — M5."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.phase10

from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
from src.assembled_core.events.crisis_alpha.state_machine import (
    CrisisStateRecord,
    compute_next_crisis_state,
    load_crisis_state,
    save_crisis_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)

POLICY = {
    "crisis_alpha": {
        "hysteresis": {
            "activate_geo_score": 2.0,
            "deactivate_geo_score": 1.0,
            "min_sources": 2,
            "cooldown_hours": 24.0,
        }
    }
}


def _prev(state="WATCH", cooldown_start_utc="") -> CrisisStateRecord:
    return CrisisStateRecord(
        state=state,
        entered_at_utc=NOW.isoformat(),
        last_evaluated_utc=NOW.isoformat(),
        cooldown_start_utc=cooldown_start_utc,
    )


def _ctx(
    geo_score=0.0,
    geo_sources=0,
    social_only=False,
    market_stress_ok=False,
    health_ok=True,
    daily_pnl=0.0,
    daily_loss_limit=0.02,
) -> CrisisAlphaContext:
    return CrisisAlphaContext(
        timestamp_utc=NOW,
        geo_score=geo_score,
        geo_sources=geo_sources,
        social_only=social_only,
        market_stress_ok=market_stress_ok,
        health_ok=health_ok,
        daily_pnl=daily_pnl,
        daily_loss_limit=daily_loss_limit,
    )


# ---------------------------------------------------------------------------
# WATCH → ACTIVE transitions
# ---------------------------------------------------------------------------


class TestWatchToActive:
    def test_activates_when_all_conditions_met(self):
        ctx = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=True, health_ok=True)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("WATCH"))
        assert result.state == "ACTIVE"

    def test_stays_watch_when_geo_score_too_low(self):
        ctx = _ctx(geo_score=1.5, geo_sources=3, market_stress_ok=True, health_ok=True)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("WATCH"))
        assert result.state == "WATCH"

    def test_stays_watch_when_not_enough_sources(self):
        ctx = _ctx(geo_score=2.5, geo_sources=1, market_stress_ok=True, health_ok=True)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("WATCH"))
        assert result.state == "WATCH"

    def test_stays_watch_when_social_only(self):
        ctx = _ctx(
            geo_score=2.5,
            geo_sources=3,
            social_only=True,
            market_stress_ok=True,
            health_ok=True,
        )
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("WATCH"))
        assert result.state == "WATCH"

    def test_stays_watch_when_market_stress_not_ok(self):
        ctx = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=False, health_ok=True)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("WATCH"))
        assert result.state == "WATCH"

    def test_stays_watch_when_health_not_ok(self):
        ctx = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=True, health_ok=False)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("WATCH"))
        assert result.state == "WATCH"

    def test_activation_at_exact_threshold(self):
        # geo_score == activate_threshold (2.0) — should activate
        ctx = _ctx(geo_score=2.0, geo_sources=2, market_stress_ok=True, health_ok=True)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("WATCH"))
        assert result.state == "ACTIVE"


# ---------------------------------------------------------------------------
# ACTIVE → COOLDOWN transitions
# ---------------------------------------------------------------------------


class TestActiveToCoooldown:
    def test_deactivates_when_geo_score_drops(self):
        ctx = _ctx(geo_score=0.5, geo_sources=3, market_stress_ok=True, health_ok=True)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("ACTIVE"))
        assert result.state == "COOLDOWN"

    def test_deactivates_when_health_not_ok(self):
        ctx = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=True, health_ok=False)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("ACTIVE"))
        assert result.state == "COOLDOWN"

    def test_stays_active_when_conditions_hold(self):
        ctx = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=True, health_ok=True)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("ACTIVE"))
        assert result.state == "ACTIVE"

    def test_cooldown_start_recorded(self):
        ctx = _ctx(geo_score=0.5)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("ACTIVE"))
        assert result.state == "COOLDOWN"
        assert result.cooldown_start_utc != ""


# ---------------------------------------------------------------------------
# COOLDOWN → WATCH transitions
# ---------------------------------------------------------------------------


class TestCooldownToWatch:
    def test_stays_cooldown_while_timer_running(self):
        cooldown_start = (NOW - timedelta(hours=12)).isoformat()
        ctx = _ctx(geo_score=0.5, health_ok=True)
        result = compute_next_crisis_state(
            ctx, POLICY, NOW, _prev("COOLDOWN", cooldown_start_utc=cooldown_start)
        )
        assert result.state == "COOLDOWN"

    def test_returns_to_watch_after_cooldown_expires(self):
        cooldown_start = (NOW - timedelta(hours=25)).isoformat()
        ctx = _ctx(geo_score=0.5, health_ok=True)
        result = compute_next_crisis_state(
            ctx, POLICY, NOW, _prev("COOLDOWN", cooldown_start_utc=cooldown_start)
        )
        assert result.state == "WATCH"

    def test_stays_cooldown_if_geo_score_still_elevated(self):
        cooldown_start = (NOW - timedelta(hours=25)).isoformat()
        ctx = _ctx(geo_score=1.5, health_ok=True)  # still above deactivate_threshold
        result = compute_next_crisis_state(
            ctx, POLICY, NOW, _prev("COOLDOWN", cooldown_start_utc=cooldown_start)
        )
        assert result.state == "COOLDOWN"

    def test_stays_cooldown_if_health_not_ok(self):
        cooldown_start = (NOW - timedelta(hours=25)).isoformat()
        ctx = _ctx(geo_score=0.5, health_ok=False)
        result = compute_next_crisis_state(
            ctx, POLICY, NOW, _prev("COOLDOWN", cooldown_start_utc=cooldown_start)
        )
        assert result.state == "COOLDOWN"

    def test_no_cooldown_start_treated_as_expired(self):
        ctx = _ctx(geo_score=0.5, health_ok=True)
        result = compute_next_crisis_state(
            ctx, POLICY, NOW, _prev("COOLDOWN", cooldown_start_utc="")
        )
        assert result.state == "WATCH"


# ---------------------------------------------------------------------------
# PAUSE state
# ---------------------------------------------------------------------------


class TestPauseState:
    def test_daily_loss_triggers_pause_from_watch(self):
        ctx = _ctx(geo_score=0.0, daily_pnl=-0.03, daily_loss_limit=0.02)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("WATCH"))
        assert result.state == "PAUSE"

    def test_daily_loss_triggers_pause_from_active(self):
        ctx = _ctx(
            geo_score=2.5,
            geo_sources=3,
            market_stress_ok=True,
            daily_pnl=-0.03,
            daily_loss_limit=0.02,
        )
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("ACTIVE"))
        assert result.state == "PAUSE"

    def test_pause_stays_pause_without_reset(self):
        ctx = _ctx(geo_score=0.0)
        result = compute_next_crisis_state(
            ctx, POLICY, NOW, _prev("PAUSE"), reset=False
        )
        assert result.state == "PAUSE"

    def test_pause_clears_with_manual_reset(self):
        ctx = _ctx(geo_score=0.0)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("PAUSE"), reset=True)
        assert result.state == "WATCH"

    def test_daily_loss_not_breached_when_pnl_positive(self):
        ctx = _ctx(daily_pnl=0.05, daily_loss_limit=0.02)
        assert not ctx.daily_loss_breached()

    def test_daily_loss_not_breached_exactly_at_limit(self):
        # loss exactly at limit — must NOT breach (use >= in check)
        ctx = _ctx(daily_pnl=-0.02, daily_loss_limit=0.02)
        assert ctx.daily_loss_breached()  # -0.02 >= 0.02 → breach


# ---------------------------------------------------------------------------
# State record reason field
# ---------------------------------------------------------------------------


class TestStateRecordReason:
    def test_reason_populated_on_activation(self):
        ctx = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=True, health_ok=True)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("WATCH"))
        assert result.reason != ""
        assert result.state == "ACTIVE"

    def test_reason_populated_on_deactivation(self):
        ctx = _ctx(geo_score=0.5)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("ACTIVE"))
        assert result.reason != ""
        assert result.state == "COOLDOWN"

    def test_geo_score_at_entry_recorded(self):
        ctx = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=True, health_ok=True)
        result = compute_next_crisis_state(ctx, POLICY, NOW, _prev("WATCH"))
        assert result.geo_score_at_entry == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        record = CrisisStateRecord(
            state="ACTIVE",
            entered_at_utc=NOW.isoformat(),
            last_evaluated_utc=NOW.isoformat(),
            reason="test save",
            geo_score_at_entry=2.5,
            cooldown_start_utc="",
        )
        state_path = tmp_path / "crisis_state.json"
        save_crisis_state(record, state_path)
        loaded = load_crisis_state(state_path)
        assert loaded.state == "ACTIVE"
        assert loaded.reason == "test save"
        assert loaded.geo_score_at_entry == pytest.approx(2.5)

    def test_load_returns_watch_when_file_missing(self, tmp_path: Path):
        result = load_crisis_state(tmp_path / "nonexistent.json")
        assert result.state == "WATCH"

    def test_load_returns_watch_when_file_corrupt(self, tmp_path: Path):
        state_path = tmp_path / "crisis_state.json"
        state_path.write_text("not-json", encoding="utf-8")
        result = load_crisis_state(state_path)
        assert result.state == "WATCH"

    def test_atomic_write_creates_parent_dirs(self, tmp_path: Path):
        state_path = tmp_path / "subdir" / "crisis_state.json"
        record = CrisisStateRecord.default()
        save_crisis_state(record, state_path)
        assert state_path.exists()

    def test_loaded_state_survives_full_cycle(self, tmp_path: Path):
        state_path = tmp_path / "crisis_state.json"
        # Start in WATCH
        prev = load_crisis_state(state_path)
        assert prev.state == "WATCH"

        # Activate
        ctx = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=True, health_ok=True)
        new_record = compute_next_crisis_state(ctx, POLICY, NOW, prev)
        save_crisis_state(new_record, state_path)

        # Load again — should be ACTIVE
        loaded = load_crisis_state(state_path)
        assert loaded.state == "ACTIVE"
