"""Tests for BENCH-1 Intel sim harness (confirm gate blocks vs allows activation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.assembled_core.ops.intel_sim import apply_intel_sim
from src.assembled_core.risk.state_machine import RiskStateRecord, compute_next_state, VERSION


pytestmark = [pytest.mark.unit, pytest.mark.phase6]


@dataclass
class _MockCtx:
    """Minimal ctx for state_machine: news_geo, disclosures_triggers, intel_health_flags, market_stress."""
    news_geo: Any = None
    disclosures_triggers: Any = None
    intel_health_flags: dict[str, str] | None = None
    market_stress: Any = None
    intel_sim_applied: bool = False


def _now_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_confirm_gate_on_blocks_activation_on_non_confirm_days() -> None:
    """With require_disclosures_confirm True and no disclosures (non-confirm day), WATCH stays WATCH."""
    ctx = _MockCtx()
    # day_index=2: stress_ok True (even), geo ACTIVE, but 2 % 5 != 0 -> disclosures_triggers None
    apply_intel_sim(ctx, day_index=2, cfg={"mode": "stress_based", "disclosures_confirm_every_n_days": 5})
    assert ctx.disclosures_triggers is None
    assert ctx.news_geo is not None
    assert ctx.news_geo.get("geo_score") == 2
    assert ctx.market_stress.get("stress_ok") is True

    policy = {
        "risk_state_machine": {
            "enabled": True,
            "hysteresis": {
                "activate_score": 2,
                "require_disclosures_confirm": True,
                "disclosures_min_severity": 1,
            },
        },
    }
    prev = RiskStateRecord(
        state="WATCH",
        since_utc=_now_utc(),
        last_transition_utc=_now_utc(),
        reason="test",
        geo_score=0,
        geo_confidence=0.0,
        version=VERSION,
    )
    now = _now_utc()
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "WATCH"
    assert next_rec.reason == "disclosures_confirm"


def test_confirm_gate_on_allows_activation_on_confirm_days() -> None:
    """With require_disclosures_confirm True and disclosures max_severity=1 (confirm day), can transition to ACTIVE."""
    ctx = _MockCtx()
    apply_intel_sim(ctx, day_index=0, cfg={"mode": "stress_based", "disclosures_confirm_every_n_days": 5})
    assert ctx.disclosures_triggers is not None
    assert getattr(ctx.disclosures_triggers, "summary", {}).get("max_severity") == 1
    assert ctx.news_geo.get("geo_score") == 2
    assert ctx.market_stress.get("stress_ok") is True

    policy = {
        "risk_state_machine": {
            "enabled": True,
            "hysteresis": {
                "activate_score": 2,
                "require_disclosures_confirm": True,
                "disclosures_min_severity": 1,
            },
        },
    }
    prev = RiskStateRecord(
        state="WATCH",
        since_utc=_now_utc(),
        last_transition_utc=_now_utc(),
        reason="test",
        geo_score=0,
        geo_confidence=0.0,
        version=VERSION,
    )
    now = _now_utc()
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "ACTIVE"
    assert next_rec.reason == "activate_score"


def test_intel_sim_sets_news_geo_and_disclosures_deterministically() -> None:
    """apply_intel_sim sets news_geo, disclosures every n days, market_stress, intel_sim_applied."""
    ctx = _MockCtx()
    cfg = {"mode": "stress_based", "disclosures_confirm_every_n_days": 5}
    apply_intel_sim(ctx, 0, cfg)
    assert ctx.news_geo is not None
    assert "geo_score" in ctx.news_geo
    assert "state_hint" in ctx.news_geo
    assert ctx.disclosures_triggers is not None
    assert ctx.market_stress == {"stress_ok": True}
    assert ctx.intel_health_flags == {}
    assert ctx.intel_sim_applied is True

    ctx2 = _MockCtx()
    apply_intel_sim(ctx2, 3, cfg)
    assert ctx2.disclosures_triggers is None
    assert ctx2.intel_sim_applied is True


# --- BENCH-2 geo_spikes ---


def test_geo_spike_emits_score3_on_schedule() -> None:
    """With geo_spikes enabled every_n_days=7, day_index 0 and 7 get geo_score=3; day 1 gets 1 or 2."""
    ctx0 = _MockCtx()
    ctx7 = _MockCtx()
    ctx1 = _MockCtx()
    cfg = {
        "mode": "stress_based",
        "disclosures_confirm_every_n_days": 5,
        "geo_spikes": {"enabled": True, "every_n_days": 7, "geo_score": 3, "geo_confidence": 0.85},
    }
    apply_intel_sim(ctx0, 0, cfg)
    apply_intel_sim(ctx7, 7, cfg)
    apply_intel_sim(ctx1, 1, cfg)
    assert ctx0.news_geo["geo_score"] == 3
    assert ctx0.news_geo["geo_confidence"] == 0.85
    assert ctx7.news_geo["geo_score"] == 3
    assert ctx1.news_geo["geo_score"] in (1, 2)
    assert ctx1.news_geo["geo_score"] != 3


def test_conditional_gate_blocks_only_on_score3_days() -> None:
    """disclosures_confirm_gate min_geo_score=3: day with geo_score=2 and no disclosures => allowed; day with geo_score=3 and no disclosures => blocked."""
    policy: dict[str, Any] = {
        "risk_state_machine": {
            "enabled": True,
            "hysteresis": {
                "activate_score": 2,
                "require_disclosures_confirm": False,
                "disclosures_confirm_gate": {
                    "enabled": True,
                    "min_geo_score": 3,
                    "on_states": ["WATCH", "COOLDOWN"],
                },
            },
        },
    }
    # day_index=2: spike 2%7!=0 -> geo_score=2 (stress_ok), confirm 2%999!=0 -> no disclosures => gate does not apply (2<3) => activation allowed
    cfg_confirm_rare = {
        "mode": "stress_based",
        "disclosures_confirm_every_n_days": 999,
        "geo_spikes": {"enabled": True, "every_n_days": 7, "geo_score": 3, "geo_confidence": 0.85},
    }
    ctx2 = _MockCtx()
    apply_intel_sim(ctx2, 2, cfg_confirm_rare)
    assert ctx2.news_geo["geo_score"] == 2
    assert ctx2.disclosures_triggers is None

    prev = RiskStateRecord(
        state="WATCH",
        since_utc=_now_utc(),
        last_transition_utc=_now_utc(),
        reason="test",
        geo_score=0,
        geo_confidence=0.0,
        version=VERSION,
    )
    next2 = compute_next_state(ctx2, policy, _now_utc(), prev)
    assert next2.state == "ACTIVE"
    assert next2.reason == "activate_score"

    # day_index=7: spike 7%7==0 -> geo_score=3, confirm 7%999!=0 -> no disclosures => gate applies => blocked
    ctx7 = _MockCtx()
    apply_intel_sim(ctx7, 7, cfg_confirm_rare)
    assert ctx7.news_geo["geo_score"] == 3
    assert ctx7.disclosures_triggers is None

    next7 = compute_next_state(ctx7, policy, _now_utc(), prev)
    assert next7.state == "WATCH"
    assert next7.reason == "disclosures_confirm"
