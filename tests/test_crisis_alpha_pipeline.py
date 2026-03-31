"""Integration tests for the Crisis-Alpha v1 pipeline — M5.

Tests the full pipeline orchestration including state transitions,
entry generation, exit rules, and deactivation triggers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.phase10

from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
from src.assembled_core.events.crisis_alpha.pipeline import run_crisis_alpha_pipeline
from src.assembled_core.events.crisis_alpha.state_machine import (
    CrisisStateRecord,
    save_crisis_state,
)

NOW = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)

POLICY = {
    "crisis_alpha": {
        "hysteresis": {
            "activate_geo_score": 2.0,
            "deactivate_geo_score": 1.0,
            "min_sources": 2,
            "min_trigger_count": 1,
            "cooldown_hours": 24.0,
        },
        "entry": {"method": "equal_weight"},
        "risk_budget": {"max_gross_exposure": 0.30},
        "exit": {
            "max_hold_hours": 8.0,
            "break_even_pct": 0.005,
            "no_overnight": True,
            "market_close_hour_utc": 21,
        },
        "daily_loss": {"limit": 0.02},
    }
}


def _ctx(**kwargs) -> CrisisAlphaContext:
    defaults = dict(
        timestamp_utc=NOW,
        geo_score=0.0,
        geo_sources=0,
        social_only=False,
        market_stress_ok=False,
        health_ok=True,
        daily_pnl=0.0,
        daily_loss_limit=0.02,
        news_trigger_items=[],
        open_positions=[],
    )
    defaults.update(kwargs)
    return CrisisAlphaContext(**defaults)


# ---------------------------------------------------------------------------
# Basic state transitions via pipeline
# ---------------------------------------------------------------------------

class TestPipelineStateTransitions:
    def test_pipeline_starts_in_watch(self, tmp_path: Path):
        ctx = _ctx()
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=tmp_path / "s.json", dry_run=True)
        assert result["state"] == "WATCH"
        assert result["previous_state"] == "WATCH"

    def test_pipeline_activates_on_sufficient_geo(self, tmp_path: Path):
        ctx = _ctx(
            geo_score=2.5,
            geo_sources=3,
            market_stress_ok=True,
            health_ok=True,
            news_trigger_items=[{"severity": 2, "topic": "geo"}],
        )
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=tmp_path / "s.json")
        assert result["state"] == "ACTIVE"
        assert result["gates_ok"] is True

    def test_pipeline_stays_watch_when_social_only(self, tmp_path: Path):
        ctx = _ctx(
            geo_score=2.5, geo_sources=3, social_only=True,
            market_stress_ok=True, health_ok=True,
            news_trigger_items=[{"severity": 2}],
        )
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=tmp_path / "s.json")
        assert result["state"] == "WATCH"
        assert result["gates_ok"] is False

    def test_pipeline_stays_watch_when_health_error(self, tmp_path: Path):
        ctx = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=True, health_ok=False)
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=tmp_path / "s.json")
        assert result["state"] == "WATCH"
        assert result["gates_ok"] is False


class TestPipelineEntry:
    def test_entry_generated_when_active(self, tmp_path: Path):
        ctx = _ctx(
            geo_score=2.5,
            geo_sources=3,
            market_stress_ok=True,
            health_ok=True,
            news_trigger_items=[{"severity": 2}],
        )
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=tmp_path / "s.json")
        assert result["state"] == "ACTIVE"
        assert len(result["target_weights"]) > 0
        # All weights should be positive (long)
        assert all(w > 0 for w in result["target_weights"].values())

    def test_no_entry_when_watch(self, tmp_path: Path):
        ctx = _ctx()
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=tmp_path / "s.json")
        assert result["state"] == "WATCH"
        assert result["target_weights"] == {}

    def test_gross_exposure_within_cap(self, tmp_path: Path):
        ctx = _ctx(
            geo_score=2.5,
            geo_sources=3,
            market_stress_ok=True,
            health_ok=True,
            news_trigger_items=[{"severity": 2}],
        )
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=tmp_path / "s.json")
        gross = sum(abs(w) for w in result["target_weights"].values())
        assert gross <= 0.30 + 1e-6  # max_gross_exposure from policy


class TestPipelineDeactivation:
    def _make_active_state(self, tmp_path: Path) -> Path:
        """Pre-load an ACTIVE state into the state file."""
        state_path = tmp_path / "s.json"
        active = CrisisStateRecord(
            state="ACTIVE",
            entered_at_utc=NOW.isoformat(),
            last_evaluated_utc=NOW.isoformat(),
            reason="test setup",
            geo_score_at_entry=2.5,
        )
        save_crisis_state(active, state_path)
        return state_path

    def test_deactivates_when_geo_drops(self, tmp_path: Path):
        state_path = self._make_active_state(tmp_path)
        ctx = _ctx(geo_score=0.5, health_ok=True)
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=state_path)
        assert result["state"] == "COOLDOWN"
        assert result["should_flatten_all"] is True

    def test_deactivates_when_health_error(self, tmp_path: Path):
        state_path = self._make_active_state(tmp_path)
        ctx = _ctx(geo_score=2.5, geo_sources=3, health_ok=False)
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=state_path)
        assert result["state"] == "COOLDOWN"
        assert result["should_flatten_all"] is True

    def test_daily_loss_triggers_pause(self, tmp_path: Path):
        state_path = self._make_active_state(tmp_path)
        ctx = _ctx(geo_score=2.5, geo_sources=3, daily_pnl=-0.05, daily_loss_limit=0.02)
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=state_path)
        assert result["state"] == "PAUSE"
        assert result["should_flatten_all"] is True


class TestPipelineExitRules:
    def test_old_positions_flagged_for_time_stop(self, tmp_path: Path):
        state_path = tmp_path / "s.json"
        old_ts = (NOW - timedelta(hours=9)).isoformat()
        open_positions = [
            {"symbol": "GLD", "side": "long", "qty": 100, "entry_price": 200.0, "entry_ts": old_ts}
        ]
        ctx = _ctx(
            geo_score=2.5,
            geo_sources=3,
            market_stress_ok=True,
            health_ok=True,
            news_trigger_items=[{"severity": 2}],
            open_positions=open_positions,
        )
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=state_path)
        # State should still be ACTIVE (position expiry doesn't change state)
        assert result["state"] == "ACTIVE"
        # But the old position should be in positions_to_exit
        assert len(result["positions_to_exit"]) >= 1
        exited_symbols = {p["symbol"] for p, _ in result["positions_to_exit"]}
        assert "GLD" in exited_symbols

    def test_fresh_positions_not_flagged(self, tmp_path: Path):
        state_path = tmp_path / "s.json"
        fresh_ts = (NOW - timedelta(hours=2)).isoformat()
        open_positions = [
            {"symbol": "GLD", "side": "long", "qty": 100, "entry_price": 200.0, "entry_ts": fresh_ts}
        ]
        ctx = _ctx(
            geo_score=2.5,
            geo_sources=3,
            market_stress_ok=True,
            health_ok=True,
            news_trigger_items=[{"severity": 2}],
            open_positions=open_positions,
        )
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=state_path)
        assert result["state"] == "ACTIVE"
        # Fresh position should not be timed out (8h max)
        time_stopped = [r for _, r in result["positions_to_exit"] if "time_stop" in r and "GLD" in r]
        assert len(time_stopped) == 0


class TestPipelineDryRun:
    def test_dry_run_does_not_persist_state(self, tmp_path: Path):
        state_path = tmp_path / "s.json"
        ctx = _ctx(
            geo_score=2.5,
            geo_sources=3,
            market_stress_ok=True,
            health_ok=True,
            news_trigger_items=[{"severity": 2}],
        )
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=state_path, dry_run=True)
        assert result["state"] == "ACTIVE"
        # State file should NOT exist after dry_run
        assert not state_path.exists()

    def test_dry_run_false_does_persist(self, tmp_path: Path):
        state_path = tmp_path / "s.json"
        ctx = _ctx(
            geo_score=2.5,
            geo_sources=3,
            market_stress_ok=True,
            health_ok=True,
            news_trigger_items=[{"severity": 2}],
        )
        run_crisis_alpha_pipeline(ctx, POLICY, state_path=state_path, dry_run=False)
        assert state_path.exists()


class TestPipelineScenarios:
    """Higher-level scenario tests: shock begins, normalizes, health degrades."""

    def test_scenario_geo_shock_activation_and_recovery(self, tmp_path: Path):
        """Scenario: geo shock → ACTIVE → geo normalizes → COOLDOWN → WATCH."""
        state_path = tmp_path / "s.json"

        # T0: geo shock, WATCH → ACTIVE
        ctx0 = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=True, health_ok=True,
                    news_trigger_items=[{"severity": 2}])
        r0 = run_crisis_alpha_pipeline(ctx0, POLICY, state_path=state_path)
        assert r0["state"] == "ACTIVE"

        # T1: geo drops below deactivate threshold → COOLDOWN
        ctx1 = _ctx(geo_score=0.5, health_ok=True)
        r1 = run_crisis_alpha_pipeline(ctx1, POLICY, state_path=state_path)
        assert r1["state"] == "COOLDOWN"

        # T2: 25 hours later, geo still low → WATCH
        later = NOW + timedelta(hours=25)
        ctx2 = CrisisAlphaContext(
            timestamp_utc=later, geo_score=0.3, geo_sources=0,
            social_only=False, market_stress_ok=False, health_ok=True,
        )
        r2 = run_crisis_alpha_pipeline(ctx2, POLICY, state_path=state_path)
        assert r2["state"] == "WATCH"

    def test_scenario_false_activation_blocked(self, tmp_path: Path):
        """Scenario: social-only signal + no market stress → cannot activate."""
        state_path = tmp_path / "s.json"

        # Social-only geo signal, high score but no confirmed news
        ctx = _ctx(geo_score=3.0, geo_sources=1, social_only=True,
                   market_stress_ok=False, health_ok=True)
        result = run_crisis_alpha_pipeline(ctx, POLICY, state_path=state_path)
        # Must stay WATCH regardless of geo_score
        assert result["state"] == "WATCH"
        assert result["gates_ok"] is False

    def test_scenario_health_error_blocks_and_forces_cooldown(self, tmp_path: Path):
        """Scenario: ACTIVE → health degrades to ERROR → forced COOLDOWN."""
        state_path = tmp_path / "s.json"

        # First activate
        ctx0 = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=True, health_ok=True,
                    news_trigger_items=[{"severity": 2}])
        r0 = run_crisis_alpha_pipeline(ctx0, POLICY, state_path=state_path)
        assert r0["state"] == "ACTIVE"

        # Health degrades
        ctx1 = _ctx(geo_score=2.5, geo_sources=3, market_stress_ok=True, health_ok=False)
        r1 = run_crisis_alpha_pipeline(ctx1, POLICY, state_path=state_path)
        assert r1["state"] == "COOLDOWN"
        assert r1["should_flatten_all"] is True

    def test_scenario_daily_loss_pauses_and_manual_reset(self, tmp_path: Path):
        """Scenario: daily loss → PAUSE → manual reset → WATCH."""
        state_path = tmp_path / "s.json"

        # Daily loss
        ctx0 = _ctx(daily_pnl=-0.05, daily_loss_limit=0.02)
        r0 = run_crisis_alpha_pipeline(ctx0, POLICY, state_path=state_path)
        assert r0["state"] == "PAUSE"

        # Manual reset
        ctx1 = _ctx(daily_pnl=0.0)
        r1 = run_crisis_alpha_pipeline(ctx1, POLICY, state_path=state_path, reset_pause=True)
        assert r1["state"] == "WATCH"
