"""Tests for Crisis-Alpha v1 gates — M5."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.phase10

from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
from src.assembled_core.events.crisis_alpha.gates import (
    check_daily_loss_gate,
    check_evidence_gate,
    check_health_gate,
    check_market_stress_gate,
    check_social_only_guard,
    check_source_gate,
    run_all_activation_gates,
)

NOW = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)


def _ctx(**kwargs) -> CrisisAlphaContext:
    defaults = dict(
        timestamp_utc=NOW,
        geo_score=2.5,
        geo_sources=3,
        social_only=False,
        market_stress_ok=True,
        health_ok=True,
        daily_pnl=0.0,
        daily_loss_limit=0.02,
    )
    defaults.update(kwargs)
    return CrisisAlphaContext(**defaults)


class TestHealthGate:
    def test_ok_when_health_ok(self):
        ok, _ = check_health_gate(_ctx(health_ok=True))
        assert ok is True

    def test_fails_when_health_not_ok(self):
        ok, reason = check_health_gate(_ctx(health_ok=False))
        assert ok is False
        assert "ERROR" in reason


class TestSocialOnlyGuard:
    def test_ok_when_not_social_only(self):
        ok, _ = check_social_only_guard(_ctx(social_only=False))
        assert ok is True

    def test_fails_when_social_only(self):
        ok, reason = check_social_only_guard(_ctx(social_only=True))
        assert ok is False
        assert "social" in reason.lower()


class TestEvidenceGate:
    def test_ok_with_qualifying_triggers(self):
        ctx = _ctx(
            news_trigger_items=[
                {"severity": 1, "topic": "geo"},
                {"severity": 2, "topic": "trade"},
            ]
        )
        ok, _ = check_evidence_gate(ctx, min_trigger_count=1)
        assert ok is True

    def test_fails_when_no_qualifying_triggers(self):
        ctx = _ctx(news_trigger_items=[{"severity": 0, "topic": "noise"}])
        ok, reason = check_evidence_gate(ctx, min_trigger_count=1)
        assert ok is False
        assert "evidence gate" in reason

    def test_ok_with_empty_triggers_when_min_is_zero(self):
        ctx = _ctx(news_trigger_items=[])
        ok, _ = check_evidence_gate(ctx, min_trigger_count=0)
        assert ok is True

    def test_counts_only_severity_gte_1(self):
        ctx = _ctx(
            news_trigger_items=[
                {"severity": 0},
                {"severity": 0},
                {"severity": 1},
            ]
        )
        ok, _ = check_evidence_gate(ctx, min_trigger_count=1)
        assert ok is True

        ok2, _ = check_evidence_gate(ctx, min_trigger_count=2)
        assert ok2 is False


class TestSourceGate:
    def test_ok_when_enough_sources(self):
        ok, _ = check_source_gate(_ctx(geo_sources=3), min_sources=2)
        assert ok is True

    def test_fails_when_too_few_sources(self):
        ok, reason = check_source_gate(_ctx(geo_sources=1), min_sources=2)
        assert ok is False
        assert "source gate" in reason

    def test_exactly_at_minimum(self):
        ok, _ = check_source_gate(_ctx(geo_sources=2), min_sources=2)
        assert ok is True


class TestMarketStressGate:
    def test_ok_when_stress_confirmed(self):
        ok, _ = check_market_stress_gate(_ctx(market_stress_ok=True))
        assert ok is True

    def test_fails_when_stress_not_confirmed(self):
        ok, reason = check_market_stress_gate(_ctx(market_stress_ok=False))
        assert ok is False
        assert "market stress" in reason.lower()


class TestDailyLossGate:
    def test_ok_when_no_loss(self):
        ok, _ = check_daily_loss_gate(_ctx(daily_pnl=0.0, daily_loss_limit=0.02))
        assert ok is True

    def test_ok_when_positive_pnl(self):
        ok, _ = check_daily_loss_gate(_ctx(daily_pnl=0.05, daily_loss_limit=0.02))
        assert ok is True

    def test_fails_when_loss_exceeds_limit(self):
        ok, reason = check_daily_loss_gate(_ctx(daily_pnl=-0.03, daily_loss_limit=0.02))
        assert ok is False
        assert "daily loss" in reason.lower()

    def test_small_loss_within_limit(self):
        ok, _ = check_daily_loss_gate(_ctx(daily_pnl=-0.01, daily_loss_limit=0.02))
        assert ok is True


class TestRunAllActivationGates:
    def test_all_pass_when_all_conditions_ok(self):
        ctx = _ctx(
            news_trigger_items=[{"severity": 2, "topic": "geo"}],
            geo_sources=3,
            social_only=False,
            market_stress_ok=True,
            health_ok=True,
            daily_pnl=0.0,
        )
        ok, reasons = run_all_activation_gates(ctx, min_trigger_count=1, min_sources=2)
        assert ok is True
        assert len(reasons) > 0

    def test_fails_fast_on_health_gate(self):
        ctx = _ctx(health_ok=False, market_stress_ok=True)
        ok, reasons = run_all_activation_gates(ctx)
        assert ok is False
        # Should fail on first gate (health)
        assert len(reasons) == 1
        assert "health" in reasons[0].lower()

    def test_fails_on_social_only_after_health_passes(self):
        ctx = _ctx(health_ok=True, social_only=True)
        ok, reasons = run_all_activation_gates(ctx)
        assert ok is False
        assert any("social" in r.lower() for r in reasons)

    def test_fails_on_evidence_gate(self):
        ctx = _ctx(health_ok=True, social_only=False, news_trigger_items=[])
        ok, reasons = run_all_activation_gates(ctx, min_trigger_count=1)
        assert ok is False
        assert any("evidence" in r.lower() for r in reasons)

    def test_fails_on_market_stress_last(self):
        ctx = _ctx(
            health_ok=True,
            social_only=False,
            news_trigger_items=[{"severity": 1}],
            geo_sources=3,
            market_stress_ok=False,
        )
        ok, reasons = run_all_activation_gates(ctx, min_trigger_count=1, min_sources=2)
        assert ok is False
        assert any("market stress" in r.lower() for r in reasons)
