"""Tests for the news_alpha event-driven trading pipeline.

Covers:
- asset_router: topic → correct ETFs
- signal_generator: trigger items → weighted signals
- exit_rules: time/price/stop-loss exits
- pipeline: end-to-end with shadow_only and enabled flags
- Hormuz example: the canonical use case
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Asset router
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_route_shipping_disruption_returns_energy_etfs() -> None:
    from src.assembled_core.events.news_alpha.asset_router import get_route

    route = get_route("shipping_disruption")
    assert route is not None
    assert "XLE" in route["long_etfs"]
    assert (
        "inverse_etfs" in route
    )  # key exists (may be empty — no liquid airline inverse in v1)
    assert route["min_severity"] <= 2


@pytest.mark.fast
def test_route_energy_crisis_returns_oil_etfs() -> None:
    from src.assembled_core.events.news_alpha.asset_router import get_route

    route = get_route("energy_crisis")
    assert route is not None
    assert "XLE" in route["long_etfs"]
    assert "UCO" in route["long_etfs_2x"]


@pytest.mark.fast
def test_route_geopolitical_conflict_includes_defense_and_gold() -> None:
    from src.assembled_core.events.news_alpha.asset_router import get_route

    route = get_route("geopolitical_conflict")
    assert route is not None
    longs = route["long_etfs"]
    assert "GLD" in longs
    assert any(sym in longs for sym in ["LMT", "NOC", "RTX"])


@pytest.mark.fast
def test_route_unknown_topic_returns_none() -> None:
    from src.assembled_core.events.news_alpha.asset_router import get_route

    assert get_route("nonexistent_topic_xyz") is None


@pytest.mark.fast
def test_split_central_bank_hike() -> None:
    from src.assembled_core.events.news_alpha.asset_router import (
        split_central_bank_topic,
    )

    trigger = {"topic": "central_bank", "source": "Fed announces surprise rate hike"}
    assert split_central_bank_topic(trigger) == "central_bank_hike"


@pytest.mark.fast
def test_split_central_bank_cut() -> None:
    from src.assembled_core.events.news_alpha.asset_router import (
        split_central_bank_topic,
    )

    trigger = {"topic": "central_bank", "source": "ECB dovish cut announced"}
    assert split_central_bank_topic(trigger) == "central_bank_cut"


@pytest.mark.fast
def test_split_central_bank_ambiguous_returns_none() -> None:
    """Ambiguous central bank news must not default to hike — skip instead."""
    from src.assembled_core.events.news_alpha.asset_router import (
        split_central_bank_topic,
    )

    trigger = {
        "topic": "central_bank",
        "source": "Fed holds rates steady, signals patience",
    }
    assert split_central_bank_topic(trigger) is None


# ---------------------------------------------------------------------------
# Signal generator — Hormuz use case
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_hormuz_trigger_generates_xle_long() -> None:
    """Hormuz blockade → Long XLE, severity 3."""
    from src.assembled_core.events.news_alpha.signal_generator import generate_signals

    triggers = [
        {
            "severity": 3,
            "topic": "shipping_disruption",
            "source": "reuters",
            "event_id": "t1",
        }
    ]
    signals = generate_signals(triggers, policy=None)
    syms = {s.symbol for s in signals}
    assert "XLE" in syms

    xle = next(s for s in signals if s.symbol == "XLE")
    assert xle.direction == "long"
    assert xle.raw_weight > 0
    assert xle.hold_days > 0


@pytest.mark.fast
def test_nuclear_risk_generates_sh_inverse_long() -> None:
    """Nuclear risk scenario: SH inverse ETF included as direction='long' hedge."""
    from src.assembled_core.events.news_alpha.signal_generator import generate_signals

    triggers = [
        {"severity": 3, "topic": "nuclear_risk", "source": "reuters", "event_id": "t1"}
    ]
    signals = generate_signals(triggers, policy=None)
    syms = {s.symbol for s in signals}
    assert "SH" in syms, "Expected SH inverse ETF for nuclear_risk hedge"
    sh = next(s for s in signals if s.symbol == "SH")
    assert sh.direction == "long", (
        "Inverse ETF must use direction='long' (we BUY the inverse ETF)"
    )
    assert sh.raw_weight > 0


@pytest.mark.fast
def test_leverage_etf_not_included_by_default() -> None:
    from src.assembled_core.events.news_alpha.signal_generator import generate_signals

    triggers = [
        {
            "severity": 3,
            "topic": "shipping_disruption",
            "source": "reuters",
            "event_id": "t1",
        }
    ]
    signals = generate_signals(
        triggers, policy=None
    )  # leverage_etfs_allowed defaults False
    syms = {s.symbol for s in signals}
    assert "UCO" not in syms, (
        "UCO (2x) should not appear when leverage_etfs_allowed=False"
    )


@pytest.mark.fast
def test_leverage_etf_included_when_allowed() -> None:
    from src.assembled_core.events.news_alpha.signal_generator import generate_signals

    policy = {"news_alpha": {"leverage_etfs_allowed": True}}
    triggers = [
        {
            "severity": 3,
            "topic": "shipping_disruption",
            "source": "reuters",
            "event_id": "t1",
        }
    ]
    signals = generate_signals(triggers, policy=policy)
    syms = {s.symbol for s in signals}
    assert "UCO" in syms


@pytest.mark.fast
def test_low_severity_trigger_is_filtered() -> None:
    from src.assembled_core.events.news_alpha.signal_generator import generate_signals

    triggers = [
        {
            "severity": 1,
            "topic": "shipping_disruption",
            "source": "reuters",
            "event_id": "t1",
        }
    ]
    signals = generate_signals(triggers, policy=None)  # min_severity=2
    assert signals == [], "Severity 1 should not generate signals"


@pytest.mark.fast
def test_gross_cap_is_respected() -> None:
    from src.assembled_core.events.news_alpha.signal_generator import (
        generate_signals,
        signals_to_weights,
    )

    # Fire multiple high-severity events to exceed cap
    triggers = [
        {
            "severity": 3,
            "topic": "shipping_disruption",
            "source": "reuters",
            "event_id": "t1",
        },
        {
            "severity": 3,
            "topic": "geopolitical_conflict",
            "source": "reuters",
            "event_id": "t2",
        },
        {"severity": 3, "topic": "market_crash", "source": "reuters", "event_id": "t3"},
    ]
    policy = {"news_alpha": {"max_gross_exposure": 0.40}}
    signals = generate_signals(triggers, policy=policy)
    weights = signals_to_weights(signals, policy=policy)
    gross = sum(abs(w) for w in weights.values())
    assert gross <= 0.41, f"Gross {gross:.3f} exceeds max_gross_exposure 0.40"


# ---------------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_time_exit_fires_after_hold_days() -> None:
    from src.assembled_core.events.news_alpha.exit_rules import check_exits
    from src.assembled_core.events.news_alpha.models import NewsAlphaSignal

    sig = NewsAlphaSignal(
        event_id="e1",
        topic_id="shipping_disruption",
        trigger_type="supply_chain",
        source="reuters",
        symbol="XLE",
        direction="long",
        raw_weight=0.10,
        severity=3,
        hold_days=5,
        entry_day=0,
    )
    # current_day=5, entry_day=0 → held 5 days → should exit
    exits = check_exits([sig], current_day=5)
    assert len(exits) == 1
    assert "time_exit" in exits[0][1]


@pytest.mark.fast
def test_no_exit_before_hold_days() -> None:
    from src.assembled_core.events.news_alpha.exit_rules import check_exits
    from src.assembled_core.events.news_alpha.models import NewsAlphaSignal

    sig = NewsAlphaSignal(
        event_id="e1",
        topic_id="shipping_disruption",
        trigger_type="supply_chain",
        source="reuters",
        symbol="XLE",
        direction="long",
        raw_weight=0.10,
        severity=3,
        hold_days=5,
        entry_day=0,
    )
    exits = check_exits([sig], current_day=3)
    assert exits == []


@pytest.mark.fast
def test_stop_loss_exit() -> None:
    from src.assembled_core.events.news_alpha.exit_rules import check_exits
    from src.assembled_core.events.news_alpha.models import NewsAlphaSignal

    sig = NewsAlphaSignal(
        event_id="e1",
        topic_id="energy_crisis",
        trigger_type="commodity",
        source="reuters",
        symbol="XLE",
        direction="long",
        raw_weight=0.10,
        severity=3,
        hold_days=5,
        entry_day=0,
        entry_price=100.0,
        stop_loss_pct=0.08,
    )
    # Price dropped 10% — below 8% stop
    exits = check_exits([sig], current_day=2, prices={"XLE": 89.0})
    assert len(exits) == 1
    assert "stop_loss" in exits[0][1]


@pytest.mark.fast
def test_take_profit_exit() -> None:
    from src.assembled_core.events.news_alpha.exit_rules import check_exits
    from src.assembled_core.events.news_alpha.models import NewsAlphaSignal

    sig = NewsAlphaSignal(
        event_id="e1",
        topic_id="energy_crisis",
        trigger_type="commodity",
        source="reuters",
        symbol="XLE",
        direction="long",
        raw_weight=0.10,
        severity=3,
        hold_days=5,
        entry_day=0,
        entry_price=100.0,
        take_profit_pct=0.15,
    )
    # Price up 18% — above 15% take profit
    exits = check_exits([sig], current_day=2, prices={"XLE": 118.0})
    assert len(exits) == 1
    assert "take_profit" in exits[0][1]


# ---------------------------------------------------------------------------
# Pipeline end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_pipeline_disabled_returns_empty_result() -> None:
    from src.assembled_core.events.news_alpha.pipeline import run_news_alpha_pipeline

    policy = {"news_alpha": {"enabled": False}}
    triggers = [{"severity": 3, "topic": "shipping_disruption", "source": "reuters"}]
    result = run_news_alpha_pipeline(triggers, policy=policy)
    assert result.signals == []
    assert result.target_weights == {}


@pytest.mark.fast
def test_pipeline_shadow_only_generates_signals_but_flags_them() -> None:
    from src.assembled_core.events.news_alpha.pipeline import run_news_alpha_pipeline

    policy = {"news_alpha": {"enabled": True}}
    triggers = [
        {
            "severity": 3,
            "topic": "shipping_disruption",
            "source": "reuters",
            "event_id": "t1",
        }
    ]
    result = run_news_alpha_pipeline(triggers, policy=policy, shadow_only=True)
    assert result.shadow_only is True
    assert len(result.signals) > 0
    assert "XLE" in result.target_weights


@pytest.mark.fast
def test_pipeline_hormuz_canonical_case() -> None:
    """Canonical use case: Hormuz blockade → XLE long in target_weights."""
    from src.assembled_core.events.news_alpha.pipeline import run_news_alpha_pipeline

    policy = {"news_alpha": {"enabled": True, "base_weight": 0.10}}
    triggers = [
        {
            "severity": 3,
            "topic": "shipping_disruption",
            "source": "reuters: Strait of Hormuz blockade confirmed",
            "event_id": "hormuz-001",
        }
    ]
    result = run_news_alpha_pipeline(triggers, policy=policy, shadow_only=False)
    assert result.errors == []
    assert "XLE" in result.target_weights
    assert result.target_weights["XLE"] > 0, "XLE should be a positive (long) weight"
    # No inverse ETF for shipping_disruption in v1 (no liquid airline inverse)


# ---------------------------------------------------------------------------
# Signal dedup and active=False guard
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_inactive_signal_excluded_from_weights() -> None:
    """signals_to_weights must skip active=False signals."""
    from src.assembled_core.events.news_alpha.models import NewsAlphaSignal
    from src.assembled_core.events.news_alpha.signal_generator import signals_to_weights

    active = NewsAlphaSignal(
        event_id="e1",
        topic_id="shipping_disruption",
        trigger_type="supply_chain",
        source="test",
        symbol="XLE",
        direction="long",
        raw_weight=0.10,
        severity=3,
        hold_days=5,
        entry_day=0,
        active=True,
    )
    inactive = NewsAlphaSignal(
        event_id="e2",
        topic_id="market_crash",
        trigger_type="market_stress",
        source="test",
        symbol="GLD",
        direction="long",
        raw_weight=0.10,
        severity=3,
        hold_days=5,
        entry_day=0,
        active=False,
    )
    weights = signals_to_weights([active, inactive], policy=None)
    assert "XLE" in weights
    assert "GLD" not in weights, "Inactive signal must not appear in weights"


@pytest.mark.fast
def test_signals_to_weights_dedup_takes_higher_weight() -> None:
    """When two signals hit the same symbol, keep the higher absolute weight."""
    from src.assembled_core.events.news_alpha.models import NewsAlphaSignal
    from src.assembled_core.events.news_alpha.signal_generator import signals_to_weights

    low = NewsAlphaSignal(
        event_id="e1",
        topic_id="t1",
        trigger_type="x",
        source="test",
        symbol="XLE",
        direction="long",
        raw_weight=0.05,
        severity=2,
        hold_days=5,
        entry_day=0,
        active=True,
    )
    high = NewsAlphaSignal(
        event_id="e2",
        topic_id="t2",
        trigger_type="x",
        source="test",
        symbol="XLE",
        direction="long",
        raw_weight=0.15,
        severity=3,
        hold_days=5,
        entry_day=0,
        active=True,
    )
    weights = signals_to_weights([low, high], policy=None)
    assert abs(weights["XLE"] - 0.15) < 1e-9, "Higher weight signal should win dedup"


@pytest.mark.fast
def test_central_bank_ambiguous_generates_no_signals() -> None:
    """Ambiguous central bank text must not fire any trade (None route → skip)."""
    from src.assembled_core.events.news_alpha.signal_generator import generate_signals

    triggers = [
        {
            "severity": 3,
            "topic": "central_bank",
            "source": "Fed holds rates steady",
            "event_id": "t1",
        }
    ]
    signals = generate_signals(triggers, policy=None)
    assert signals == [], "Ambiguous central_bank should produce no signals"


@pytest.mark.fast
def test_min_severity_boundary_passes() -> None:
    """Trigger at exact min_severity (2) must generate signals."""
    from src.assembled_core.events.news_alpha.signal_generator import generate_signals

    triggers = [
        {
            "severity": 2,
            "topic": "shipping_disruption",
            "source": "reuters",
            "event_id": "t1",
        }
    ]
    signals = generate_signals(triggers, policy=None)
    assert len(signals) > 0, (
        "Severity at exact min_severity threshold should generate signals"
    )


@pytest.mark.fast
def test_inverse_etf_sized_at_half_vs_directional() -> None:
    """Inverse ETF hedge weight must be half the directional long weight."""
    from src.assembled_core.events.news_alpha.signal_generator import generate_signals

    policy = {"news_alpha": {"leverage_etfs_allowed": False}}
    triggers = [
        {"severity": 3, "topic": "nuclear_risk", "source": "reuters", "event_id": "t1"}
    ]
    signals = generate_signals(triggers, policy=policy)
    gld = next((s for s in signals if s.symbol == "GLD"), None)
    sh = next((s for s in signals if s.symbol == "SH"), None)
    assert gld is not None and sh is not None
    assert abs(sh.raw_weight - gld.raw_weight * 0.5) < 1e-9, (
        "Inverse ETF must be sized at 0.5x directional"
    )


@pytest.mark.fast
def test_tbt_excluded_without_leverage_allowed() -> None:
    """TBT is a 2x leveraged inverse — must be excluded when leverage_etfs_allowed=False."""
    from src.assembled_core.events.news_alpha.signal_generator import generate_signals

    triggers = [
        {
            "severity": 3,
            "topic": "central_bank",
            "source": "Fed surprise hike 50bps",
            "event_id": "t1",
        }
    ]
    signals = generate_signals(
        triggers, policy=None
    )  # leverage_etfs_allowed defaults False
    syms = {s.symbol for s in signals}
    assert "TBT" not in syms, (
        "TBT (2x leveraged inverse) must not appear when leverage_etfs_allowed=False"
    )


@pytest.mark.fast
def test_tbt_included_with_leverage_allowed() -> None:
    """TBT must appear when leverage_etfs_allowed=True."""
    from src.assembled_core.events.news_alpha.signal_generator import generate_signals

    policy = {"news_alpha": {"leverage_etfs_allowed": True}}
    triggers = [
        {
            "severity": 3,
            "topic": "central_bank",
            "source": "Fed surprise hike 50bps",
            "event_id": "t1",
        }
    ]
    signals = generate_signals(triggers, policy=policy)
    syms = {s.symbol for s in signals}
    assert "TBT" in syms, "TBT should appear when leverage_etfs_allowed=True"
    tbt = next(s for s in signals if s.symbol == "TBT")
    assert tbt.is_2x is True
    assert tbt.direction == "long"


@pytest.mark.fast
def test_cross_event_dedup_keeps_higher_severity_weight() -> None:
    """When two events map to the same symbol, the higher-weight signal wins."""
    from src.assembled_core.events.news_alpha.signal_generator import (
        generate_signals,
        signals_to_weights,
    )

    # shipping_disruption (sev=2) and energy_crisis (sev=3) both map to XLE
    triggers = [
        {
            "severity": 2,
            "topic": "shipping_disruption",
            "source": "reuters",
            "event_id": "t1",
        },
        {
            "severity": 3,
            "topic": "energy_crisis",
            "source": "reuters",
            "event_id": "t2",
        },
    ]
    signals = generate_signals(triggers, policy=None)
    # Both events may produce XLE signals — signals_to_weights keeps the higher weight
    weights = signals_to_weights(signals, policy=None)
    xle_w = weights.get("XLE")
    assert xle_w is not None and xle_w > 0
    # severity=3 → higher raw_weight than severity=2; winning weight should be ≥ severity-2 weight
    base = 0.08
    sev2_w = min(base * 1.5 * (2 / 2.0), 0.20)
    assert xle_w >= sev2_w - 1e-9, (
        "Higher-severity event's XLE weight must survive dedup"
    )


@pytest.mark.fast
def test_pipeline_exit_path_fires_and_marks_inactive() -> None:
    """Pipeline must populate positions_to_exit and mark signals inactive."""
    from src.assembled_core.events.news_alpha.models import NewsAlphaSignal
    from src.assembled_core.events.news_alpha.pipeline import run_news_alpha_pipeline

    expired = NewsAlphaSignal(
        event_id="e1",
        topic_id="shipping_disruption",
        trigger_type="supply_chain",
        source="reuters",
        symbol="XLE",
        direction="long",
        raw_weight=0.10,
        severity=3,
        hold_days=5,
        entry_day=0,
        active=True,
    )
    policy = {"news_alpha": {"enabled": True}}
    result = run_news_alpha_pipeline(
        trigger_items=[],
        open_signals=[expired],
        current_day=5,  # exactly at hold_days → time exit fires
        policy=policy,
        shadow_only=False,
    )
    assert len(result.positions_to_exit) == 1
    sig, reason = result.positions_to_exit[0]
    assert "time_exit" in reason
    assert sig.active is False, "Pipeline must mark exited signal as inactive"
