"""Tests for risk state machine (INT-4.1): WATCH / ACTIVE / COOLDOWN / PAUSE."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from src.assembled_core.pipeline.trading_cycle import TradingContext, run_trading_cycle
from src.assembled_core.portfolio.position_sizing import compute_target_positions
from src.assembled_core.risk.state_machine import (
    RiskStateRecord,
    atomic_write_json_with_retry,
    compute_next_state,
    load_risk_state,
    save_risk_state,
)
from src.assembled_core.intel.disclosures_triggers_loader import (
    DisclosuresTriggerSnapshot,
)

import src.assembled_core.pipeline.trading_cycle  # noqa: F401 - ensure submodule is loaded for monkeypatch
import src.assembled_core.risk.state_machine  # noqa: F401 - ensure submodule is loaded for monkeypatch


pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def _policy() -> Dict[str, Any]:
    return {
        "risk_state_machine": {
            "enabled": True,
            "state_path": "output/state/risk_state.json",
            "hysteresis": {
                "activate_score": 2,
                "deactivate_score": 1,
                "pause_score": 3,
                "confidence_floor": 0.60,
                "require_market_stress_confirm": False,
            },
            "cooldown": {"hours": 24},
            "pause": {"enabled": False, "require_confidence": 0.80, "hours": 12},
            "qc": {"if_intel_degraded": "WATCH"},
        }
    }


def _ctx(
    news_geo: Dict[str, Any] | None = None,
    intel_health_flags: Dict[str, str] | None = None,
    market_stress: Dict[str, Any] | None = None,
    disclosures_triggers: Any = None,
) -> TradingContext:
    """Minimal TradingContext with optional news_geo / intel_health_flags / market_stress / disclosures_triggers."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")],
            "symbol": ["X"],
            "close": [100.0],
        }
    )
    ctx = TradingContext(
        prices=df,
        signal_fn=lambda _: pd.DataFrame(
            columns=["timestamp", "symbol", "direction", "score"]
        ),
        position_sizing_fn=lambda s, c: (
            compute_target_positions(s, total_capital=c) if not s.empty else s
        ),
        capital=1.0,
    )
    if news_geo is not None:
        ctx.news_geo = news_geo  # type: ignore[attr-defined]
    if intel_health_flags is not None:
        ctx.intel_health_flags = intel_health_flags  # type: ignore[attr-defined]
    if market_stress is not None:
        ctx.market_stress = market_stress  # type: ignore[attr-defined]
    if disclosures_triggers is not None:
        ctx.disclosures_triggers = disclosures_triggers  # type: ignore[attr-defined]
    return ctx


def test_watch_to_active_when_score_ge_2() -> None:
    """WATCH + geo_score >= activate_score(2) -> ACTIVE (stress not required in this policy)."""
    policy = _policy()
    ctx = _ctx(news_geo={"geo_score": 2, "geo_confidence": 0.8, "state_hint": "ACTIVE"})
    now = "2025-01-15T12:00:00Z"
    prev = RiskStateRecord(
        state="WATCH",
        since_utc="2025-01-01T00:00:00Z",
        last_transition_utc="2025-01-01T00:00:00Z",
        reason="default",
        geo_score=0,
        geo_confidence=0.0,
    )
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "ACTIVE"
    assert next_rec.reason == "activate_score"
    assert next_rec.last_transition_utc == now


def test_watch_to_active_blocked_without_stress() -> None:
    """WATCH + score >= activate_score but require_market_stress_confirm and no stress_ok -> stay WATCH, reason stress_confirm."""
    policy = _policy()
    policy["risk_state_machine"]["hysteresis"]["require_market_stress_confirm"] = True
    ctx = _ctx(
        news_geo={"geo_score": 2, "geo_confidence": 0.8, "state_hint": "ACTIVE"},
        market_stress={"stress_ok": False},
    )
    now = "2025-01-15T12:00:00Z"
    prev = RiskStateRecord(
        state="WATCH",
        since_utc="2025-01-01T00:00:00Z",
        last_transition_utc="2025-01-01T00:00:00Z",
        reason="default",
        geo_score=0,
        geo_confidence=0.0,
    )
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "WATCH"
    assert next_rec.reason == "stress_confirm"


def test_watch_to_active_allowed_with_stress() -> None:
    """WATCH + score >= activate_score and stress_ok True -> ACTIVE."""
    policy = _policy()
    policy["risk_state_machine"]["hysteresis"]["require_market_stress_confirm"] = True
    ctx = _ctx(
        news_geo={"geo_score": 2, "geo_confidence": 0.8, "state_hint": "ACTIVE"},
        market_stress={"stress_ok": True},
    )
    now = "2025-01-15T12:00:00Z"
    prev = RiskStateRecord(
        state="WATCH",
        since_utc="2025-01-01T00:00:00Z",
        last_transition_utc="2025-01-01T00:00:00Z",
        reason="default",
        geo_score=0,
        geo_confidence=0.0,
    )
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "ACTIVE"
    assert next_rec.reason == "activate_score"


def test_active_to_cooldown_when_score_le_1() -> None:
    """ACTIVE + geo_score <= deactivate_score(1) -> COOLDOWN."""
    policy = _policy()
    ctx = _ctx(news_geo={"geo_score": 1, "geo_confidence": 0.7, "state_hint": "WATCH"})
    now = "2025-01-15T12:00:00Z"
    prev = RiskStateRecord(
        state="ACTIVE",
        since_utc="2025-01-10T00:00:00Z",
        last_transition_utc="2025-01-10T00:00:00Z",
        reason="activate_score",
        geo_score=2,
        geo_confidence=0.8,
    )
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "COOLDOWN"
    assert next_rec.reason == "deactivate_score"


def test_cooldown_holds_for_hours() -> None:
    """COOLDOWN stays until cooldown.hours have passed."""
    policy = _policy()
    ctx = _ctx(news_geo={"geo_score": 0, "geo_confidence": 0.0})
    # 12h after last transition -> still COOLDOWN (24h required)
    now = "2025-01-10T12:00:00Z"  # 12h after 2025-01-10T00:00:00Z
    prev = RiskStateRecord(
        state="COOLDOWN",
        since_utc="2025-01-10T00:00:00Z",
        last_transition_utc="2025-01-10T00:00:00Z",
        reason="deactivate_score",
        geo_score=1,
        geo_confidence=0.7,
    )
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "COOLDOWN"
    assert next_rec.reason == "cooldown_timer"


def test_cooldown_to_watch_after_hours() -> None:
    """COOLDOWN -> WATCH after cooldown.hours when score < activate_score."""
    policy = _policy()
    ctx = _ctx(news_geo={"geo_score": 0, "geo_confidence": 0.0})
    now = "2025-01-11T01:00:00Z"  # 25h after 2025-01-10T00:00:00Z
    prev = RiskStateRecord(
        state="COOLDOWN",
        since_utc="2025-01-10T00:00:00Z",
        last_transition_utc="2025-01-10T00:00:00Z",
        reason="deactivate_score",
        geo_score=1,
        geo_confidence=0.7,
    )
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "WATCH"
    assert next_rec.reason == "cooldown_to_watch"


def test_intel_degraded_forces_watch() -> None:
    """Intel degraded -> effective score 0 -> stay or move to WATCH (no activate)."""
    policy = _policy()
    ctx = _ctx(news_geo=None, intel_health_flags={"intel_geo_score": "DEGRADED"})
    now = "2025-01-15T12:00:00Z"
    prev = RiskStateRecord(
        state="WATCH",
        since_utc="2025-01-01T00:00:00Z",
        last_transition_utc="2025-01-01T00:00:00Z",
        reason="default",
        geo_score=0,
        geo_confidence=0.0,
    )
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "WATCH"
    # With score 0 we don't activate
    prev_active = RiskStateRecord(
        state="ACTIVE",
        since_utc="2025-01-10T00:00:00Z",
        last_transition_utc="2025-01-10T00:00:00Z",
        reason="activate_score",
        geo_score=2,
        geo_confidence=0.8,
    )
    next_from_active = compute_next_state(ctx, policy, now, prev_active)
    assert next_from_active.state == "COOLDOWN"  # score 0 <= deactivate_score 1


def test_load_save_risk_state_atomic() -> None:
    """load_risk_state / save_risk_state roundtrip and missing file -> WATCH."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "risk_state.json"
        assert not path.exists()
        rec = load_risk_state(path)
        assert rec.state == "WATCH"
        assert rec.reason == "default"

        rec2 = RiskStateRecord(
            state="ACTIVE",
            since_utc="2025-01-15T12:00:00Z",
            last_transition_utc="2025-01-15T12:00:00Z",
            reason="activate_score",
            geo_score=2,
            geo_confidence=0.8,
        )
        save_risk_state(rec2, path)
        assert path.exists()
        loaded = load_risk_state(path)
        assert loaded.state == "ACTIVE"
        assert loaded.since_utc == rec2.since_utc
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["state"] == "ACTIVE"


def test_run_trading_cycle_fills_risk_state(monkeypatch: Any) -> None:
    """run_trading_cycle fills ctx.risk_state after state machine run."""
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        monkeypatch.setattr(
            "src.assembled_core.pipeline.trading_cycle.get_base_dir", lambda: base
        )
        monkeypatch.setattr(
            "src.assembled_core.pipeline.trading_cycle.load_policy",
            lambda: {
                "risk_state_machine": {
                    "enabled": True,
                    "state_path": "risk_state.json",
                    "persistence": {"mode": "live"},
                }
            },
        )
        prices = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
                "symbol": ["A", "B"],
                "close": [100.0, 200.0],
            }
        )
        ctx = TradingContext(
            prices=prices,
            signal_fn=lambda _: pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
                    "symbol": ["A", "B"],
                    "direction": ["LONG", "LONG"],
                    "score": [1.0, 1.0],
                }
            ),
            position_sizing_fn=lambda s, c: compute_target_positions(
                s, total_capital=c
            ),
            capital=1.0,
        )
        result = run_trading_cycle(ctx)
        assert result.status == "success"
        assert ctx.risk_state is not None
        assert ctx.risk_state.get("state") in ("WATCH", "ACTIVE", "COOLDOWN", "PAUSE")
        assert "since_utc" in ctx.risk_state
        assert "reason" in ctx.risk_state
        assert (base / "risk_state.json").exists()


def test_ephemeral_mode_does_not_write_file(monkeypatch: Any) -> None:
    """When persistence mode is ephemeral, no state file is written under base_dir."""
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        monkeypatch.setattr(
            "src.assembled_core.pipeline.trading_cycle.get_base_dir", lambda: base
        )
        monkeypatch.setattr(
            "src.assembled_core.pipeline.trading_cycle.load_policy",
            lambda: {
                "risk_state_machine": {
                    "enabled": True,
                    "state_path": "output/state/risk_state.json",
                    "persistence": {"mode": "ephemeral"},
                }
            },
        )
        prices = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
                "symbol": ["A", "B"],
                "close": [100.0, 200.0],
            }
        )
        ctx = TradingContext(
            prices=prices,
            signal_fn=lambda _: pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
                    "symbol": ["A", "B"],
                    "direction": ["LONG", "LONG"],
                    "score": [1.0, 1.0],
                }
            ),
            position_sizing_fn=lambda s, c: compute_target_positions(
                s, total_capital=c
            ),
            capital=1.0,
        )
        result = run_trading_cycle(ctx)
        assert result.status == "success"
        assert ctx.risk_state is not None
        assert not (base / "output" / "state" / "risk_state.json").exists()


def test_per_run_mode_writes_to_unique_path(monkeypatch: Any) -> None:
    """When persistence mode is per_run, state is written to per_run_dir/run_id/risk_state.json."""
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        run_id = "test_run_123"
        monkeypatch.setattr(
            "src.assembled_core.pipeline.trading_cycle.get_base_dir", lambda: base
        )
        monkeypatch.setattr(
            "src.assembled_core.pipeline.trading_cycle.load_policy",
            lambda: {
                "risk_state_machine": {
                    "enabled": True,
                    "state_path": "output/state/risk_state.json",
                    "persistence": {
                        "mode": "per_run",
                        "per_run_dir": "output/state/runs",
                    },
                }
            },
        )
        prices = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
                "symbol": ["A", "B"],
                "close": [100.0, 200.0],
            }
        )
        ctx = TradingContext(
            prices=prices,
            signal_fn=lambda _: pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
                    "symbol": ["A", "B"],
                    "direction": ["LONG", "LONG"],
                    "score": [1.0, 1.0],
                }
            ),
            position_sizing_fn=lambda s, c: compute_target_positions(
                s, total_capital=c
            ),
            capital=1.0,
            run_id=run_id,
        )
        result = run_trading_cycle(ctx)
        assert result.status == "success"
        assert ctx.risk_state is not None
        expected_path = base / "output" / "state" / "runs" / run_id / "risk_state.json"
        assert expected_path.exists()


def test_atomic_write_retry_handles_permissionerror(monkeypatch: Any) -> None:
    """atomic_write_json_with_retry retries on PermissionError and then succeeds."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "out.json"
        call_count = 0
        original_replace = os.replace

        def replacing(src: str, dst: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PermissionError(32, "File in use")
            original_replace(src, dst)

        monkeypatch.setattr(
            "src.assembled_core.risk.state_machine.os.replace", replacing
        )
        atomic_write_json_with_retry(path, {"state": "WATCH"}, retries=3, backoff_ms=1)
        assert path.exists()
        assert call_count == 2


# --- Disclosures confirm gate (DISCL-5) ---


def test_activation_blocked_without_disclosures_confirm_when_required() -> None:
    """WATCH -> ACTIVE blocked when require_disclosures_confirm True and disclosures missing/degraded or max_sev < min."""
    policy = _policy()
    policy["risk_state_machine"]["hysteresis"]["require_disclosures_confirm"] = True
    policy["risk_state_machine"]["hysteresis"]["disclosures_min_severity"] = 1
    # No disclosures_triggers -> not confirmed
    ctx = _ctx(
        news_geo={"geo_score": 2, "geo_confidence": 0.8, "state_hint": "ACTIVE"},
        market_stress={"stress_ok": True},
        disclosures_triggers=None,
    )
    now = "2025-01-15T12:00:00Z"
    prev = RiskStateRecord(
        state="WATCH",
        since_utc="2025-01-01T00:00:00Z",
        last_transition_utc="2025-01-01T00:00:00Z",
        reason="default",
        geo_score=0,
        geo_confidence=0.0,
    )
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "WATCH"
    assert next_rec.reason == "disclosures_confirm"

    # DEGRADED flag -> not confirmed
    ctx2 = _ctx(
        news_geo={"geo_score": 2, "geo_confidence": 0.8},
        market_stress={"stress_ok": True},
        intel_health_flags={"intel_disclosures_triggers": "DEGRADED"},
        disclosures_triggers=DisclosuresTriggerSnapshot(
            generated_utc="2025-01-15T12:00:00Z",
            triggers=[{"severity": 2}],
            summary={"max_severity": 2, "count_sev1plus": 1, "count_sev2plus": 1},
        ),
    )
    next_rec2 = compute_next_state(ctx2, policy, now, prev)
    assert next_rec2.state == "WATCH"
    assert next_rec2.reason == "disclosures_confirm"


def test_activation_allowed_with_disclosures_confirm() -> None:
    """WATCH -> ACTIVE when require_disclosures_confirm True and disclosures snapshot has max_severity >= min."""
    policy = _policy()
    policy["risk_state_machine"]["hysteresis"]["require_disclosures_confirm"] = True
    policy["risk_state_machine"]["hysteresis"]["disclosures_min_severity"] = 1
    snap = DisclosuresTriggerSnapshot(
        generated_utc="2025-01-15T12:00:00Z",
        triggers=[{"trigger_id": "dtr_1", "severity": 1}],
        summary={"max_severity": 1, "count_sev1plus": 1, "count_sev2plus": 0},
    )
    ctx = _ctx(
        news_geo={"geo_score": 2, "geo_confidence": 0.8, "state_hint": "ACTIVE"},
        market_stress={"stress_ok": True},
        disclosures_triggers=snap,
    )
    now = "2025-01-15T12:00:00Z"
    prev = RiskStateRecord(
        state="WATCH",
        since_utc="2025-01-01T00:00:00Z",
        last_transition_utc="2025-01-01T00:00:00Z",
        reason="default",
        geo_score=0,
        geo_confidence=0.0,
    )
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "ACTIVE"
    assert next_rec.reason == "activate_score"


def test_activation_not_blocked_when_flag_false() -> None:
    """When require_disclosures_confirm False, activation does not require disclosures (default behavior)."""
    policy = _policy()
    # default: require_disclosures_confirm not set / False
    assert (
        policy["risk_state_machine"]["hysteresis"].get(
            "require_disclosures_confirm", False
        )
        is False
    )
    ctx = _ctx(
        news_geo={"geo_score": 2, "geo_confidence": 0.8, "state_hint": "ACTIVE"},
        market_stress={"stress_ok": True},
        disclosures_triggers=None,  # no disclosures
    )
    now = "2025-01-15T12:00:00Z"
    prev = RiskStateRecord(
        state="WATCH",
        since_utc="2025-01-01T00:00:00Z",
        last_transition_utc="2025-01-01T00:00:00Z",
        reason="default",
        geo_score=0,
        geo_confidence=0.0,
    )
    next_rec = compute_next_state(ctx, policy, now, prev)
    assert next_rec.state == "ACTIVE"
    assert next_rec.reason == "activate_score"


# --- Conditional disclosures confirm gate (DISCL-6) ---


def test_confirm_gate_only_applies_when_geo_score_ge_min() -> None:
    """disclosures_confirm_gate: geo_score < min_geo_score => activation allowed without disclosures; geo_score >= min => blocked without, allowed with."""
    policy = _policy()
    policy["risk_state_machine"]["hysteresis"]["require_disclosures_confirm"] = False
    policy["risk_state_machine"]["hysteresis"]["disclosures_confirm_gate"] = {
        "enabled": True,
        "min_geo_score": 3,
        "on_states": ["WATCH", "COOLDOWN"],
    }
    now = "2025-01-15T12:00:00Z"
    prev = RiskStateRecord(
        state="WATCH",
        since_utc="2025-01-01T00:00:00Z",
        last_transition_utc="2025-01-01T00:00:00Z",
        reason="default",
        geo_score=0,
        geo_confidence=0.0,
    )

    # geo_score=2 < 3 => gate does not apply => activation allowed even without disclosures
    ctx_geo2 = _ctx(
        news_geo={"geo_score": 2, "geo_confidence": 0.8, "state_hint": "ACTIVE"},
        market_stress={"stress_ok": True},
        disclosures_triggers=None,
    )
    next_geo2 = compute_next_state(ctx_geo2, policy, now, prev)
    assert next_geo2.state == "ACTIVE"
    assert next_geo2.reason == "activate_score"

    # geo_score=3 >= 3 => gate applies => blocked without disclosures
    ctx_geo3_no_disc = _ctx(
        news_geo={"geo_score": 3, "geo_confidence": 0.8, "state_hint": "ACTIVE"},
        market_stress={"stress_ok": True},
        disclosures_triggers=None,
    )
    next_geo3_no = compute_next_state(ctx_geo3_no_disc, policy, now, prev)
    assert next_geo3_no.state == "WATCH"
    assert next_geo3_no.reason == "disclosures_confirm"

    # geo_score=3 with disclosures => allowed
    snap = DisclosuresTriggerSnapshot(
        generated_utc="2025-01-15T12:00:00Z",
        triggers=[{"trigger_id": "dtr_1", "severity": 1}],
        summary={"max_severity": 1, "count_sev1plus": 1, "count_sev2plus": 0},
    )
    ctx_geo3_disc = _ctx(
        news_geo={"geo_score": 3, "geo_confidence": 0.8, "state_hint": "ACTIVE"},
        market_stress={"stress_ok": True},
        disclosures_triggers=snap,
    )
    next_geo3_yes = compute_next_state(ctx_geo3_disc, policy, now, prev)
    assert next_geo3_yes.state == "ACTIVE"
    assert next_geo3_yes.reason == "activate_score"


def test_gate_applies_in_cooldown_when_configured() -> None:
    """disclosures_confirm_gate with on_states including COOLDOWN: after cooldown, geo_score >= min and no disclosures => WATCH (disclosures_confirm); with disclosures => ACTIVE."""
    policy = _policy()
    policy["risk_state_machine"]["hysteresis"]["require_disclosures_confirm"] = False
    policy["risk_state_machine"]["hysteresis"]["disclosures_confirm_gate"] = {
        "enabled": True,
        "min_geo_score": 3,
        "on_states": ["WATCH", "COOLDOWN"],
    }
    policy["risk_state_machine"]["cooldown"] = {"hours": 6}
    now = "2025-01-15T12:00:00Z"  # 25h after last_transition -> cooldown expired
    prev = RiskStateRecord(
        state="COOLDOWN",
        since_utc="2025-01-14T00:00:00Z",
        last_transition_utc="2025-01-14T00:00:00Z",
        reason="deactivate_score",
        geo_score=1,
        geo_confidence=0.7,
    )

    # COOLDOWN expired, geo_score=3, no disclosures => gate applies => stay WATCH, reason disclosures_confirm
    ctx_no_disc = _ctx(
        news_geo={"geo_score": 3, "geo_confidence": 0.8, "state_hint": "ACTIVE"},
        market_stress={"stress_ok": True},
        disclosures_triggers=None,
    )
    next_no = compute_next_state(ctx_no_disc, policy, now, prev)
    assert next_no.state == "WATCH"
    assert next_no.reason == "disclosures_confirm"

    # With disclosures => ACTIVE
    snap = DisclosuresTriggerSnapshot(
        generated_utc="2025-01-15T12:00:00Z",
        triggers=[{"severity": 1}],
        summary={"max_severity": 1, "count_sev1plus": 1, "count_sev2plus": 0},
    )
    ctx_disc = _ctx(
        news_geo={"geo_score": 3, "geo_confidence": 0.8, "state_hint": "ACTIVE"},
        market_stress={"stress_ok": True},
        disclosures_triggers=snap,
    )
    next_yes = compute_next_state(ctx_disc, policy, now, prev)
    assert next_yes.state == "ACTIVE"
    assert next_yes.reason == "cooldown_to_active"
