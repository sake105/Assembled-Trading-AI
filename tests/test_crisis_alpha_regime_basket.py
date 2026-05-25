"""Regression tests: regime-aware basket selection in crisis alpha.

Covers:
- detect_rate_regime: neutral when TLT and SHY perform similarly
- detect_rate_regime: rate_hike when TLT underperforms SHY by > threshold
- detect_rate_regime: neutral when data is missing or insufficient
- get_regime_aware_baskets: returns RATE_HIKE_BASKETS when enabled + rate_hike
- get_regime_aware_baskets: returns default baskets when feature disabled
- pipeline: prices_df=None leaves basket unchanged
"""

from __future__ import annotations

import pandas as pd
import pytest


def _make_prices(tlt_ret: float, shy_ret: float, days: int = 20) -> pd.DataFrame:
    """Build synthetic price data for TLT and SHY with given period returns."""
    timestamps = pd.date_range("2022-01-03", periods=days, freq="B", tz="UTC")
    rows = []
    for sym, final_ret in [("TLT", tlt_ret), ("SHY", shy_ret)]:
        start = 100.0
        end = start * (1 + final_ret)
        closes = [start + (end - start) * i / (days - 1) for i in range(days)]
        for ts, c in zip(timestamps, closes):
            rows.append({"timestamp": ts, "symbol": sym, "close": c})
    return pd.DataFrame(rows)


def _regime_policy(enabled: bool = True, threshold: float = -0.05) -> dict:
    return {
        "crisis_alpha": {
            "basket_regime_detection": {
                "enabled": enabled,
                "lookback_days": 30,
                "rate_hike_threshold": threshold,
            }
        }
    }


AS_OF = pd.Timestamp("2022-02-01", tz="UTC")


# ---------------------------------------------------------------------------
# detect_rate_regime
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_detect_rate_regime_neutral_when_tlt_shy_similar() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import detect_rate_regime

    prices = _make_prices(tlt_ret=0.02, shy_ret=0.01)
    regime = detect_rate_regime(prices, AS_OF)
    assert regime == "neutral"


@pytest.mark.fast
def test_detect_rate_regime_rate_hike_when_tlt_lags_shy() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import detect_rate_regime

    # TLT -15%, SHY -1% => gap = -14pp, well below -5% threshold
    prices = _make_prices(tlt_ret=-0.15, shy_ret=-0.01)
    regime = detect_rate_regime(prices, AS_OF)
    assert regime == "rate_hike"


@pytest.mark.fast
def test_detect_rate_regime_neutral_at_boundary() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import detect_rate_regime

    # TLT -4%, SHY 0% => gap = -4pp, above -5% threshold => neutral
    prices = _make_prices(tlt_ret=-0.04, shy_ret=0.00)
    regime = detect_rate_regime(prices, AS_OF)
    assert regime == "neutral"


@pytest.mark.fast
def test_detect_rate_regime_neutral_on_missing_data() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import detect_rate_regime

    # Only SPY rows — no TLT or SHY
    prices = pd.DataFrame(
        {
            "timestamp": [AS_OF],
            "symbol": ["SPY"],
            "close": [400.0],
        }
    )
    regime = detect_rate_regime(prices, AS_OF)
    assert regime == "neutral"


@pytest.mark.fast
def test_detect_rate_regime_neutral_on_empty_df() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import detect_rate_regime

    regime = detect_rate_regime(pd.DataFrame(), AS_OF)
    assert regime == "neutral"


@pytest.mark.fast
def test_detect_rate_regime_neutral_on_none() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import detect_rate_regime

    regime = detect_rate_regime(None, AS_OF)  # type: ignore[arg-type]
    assert regime == "neutral"


# ---------------------------------------------------------------------------
# get_regime_aware_baskets
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_regime_aware_baskets_returns_rate_hike_basket_when_enabled() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import (
        RATE_HIKE_BASKETS,
        get_regime_aware_baskets,
    )

    prices = _make_prices(tlt_ret=-0.16, shy_ret=-0.01)
    baskets = get_regime_aware_baskets(
        prices, AS_OF, policy=_regime_policy(enabled=True)
    )
    symbols = {b["symbol"] for b in baskets}
    expected = {b["symbol"] for b in RATE_HIKE_BASKETS}
    assert symbols == expected, (
        f"Expected RATE_HIKE_BASKETS symbols {expected}, got {symbols}"
    )


@pytest.mark.fast
def test_regime_aware_baskets_returns_default_when_neutral() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import (
        DEFAULT_BASKETS,
        get_regime_aware_baskets,
    )

    prices = _make_prices(tlt_ret=0.01, shy_ret=0.01)
    baskets = get_regime_aware_baskets(
        prices, AS_OF, policy=_regime_policy(enabled=True)
    )
    default_syms = {b["symbol"] for b in DEFAULT_BASKETS}
    result_syms = {b["symbol"] for b in baskets}
    assert result_syms == default_syms


@pytest.mark.fast
def test_regime_aware_baskets_disabled_always_returns_default() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import (
        DEFAULT_BASKETS,
        get_regime_aware_baskets,
    )

    # Even with rate-hike data, feature disabled => default
    prices = _make_prices(tlt_ret=-0.20, shy_ret=-0.01)
    baskets = get_regime_aware_baskets(
        prices, AS_OF, policy=_regime_policy(enabled=False)
    )
    default_syms = {b["symbol"] for b in DEFAULT_BASKETS}
    assert {b["symbol"] for b in baskets} == default_syms


@pytest.mark.fast
def test_regime_aware_baskets_no_prices_returns_default() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import (
        DEFAULT_BASKETS,
        get_regime_aware_baskets,
    )

    baskets = get_regime_aware_baskets(None, AS_OF, policy=_regime_policy(enabled=True))
    default_syms = {b["symbol"] for b in DEFAULT_BASKETS}
    assert {b["symbol"] for b in baskets} == default_syms


# ---------------------------------------------------------------------------
# pipeline: prices_df wiring
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_pipeline_prices_df_none_does_not_change_entry(tmp_path: object) -> None:
    """prices_df=None must not alter entry targets vs. no prices_df arg."""

    from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
    from src.assembled_core.events.crisis_alpha.pipeline import (
        run_crisis_alpha_pipeline,
    )

    ctx = CrisisAlphaContext(
        timestamp_utc=AS_OF.to_pydatetime(),
        geo_score=2.5,
        geo_sources=2,
        social_only=False,
        market_stress_ok=True,
        market_stress_score=2,
        health_ok=True,
        news_trigger_items=[{"severity": 2, "topic": "test", "source": "test"}],
        daily_pnl=0.0,
        daily_loss_limit=0.05,
        open_positions=[],
    )
    policy: dict = {"crisis_alpha": {"enabled": True, "shadow_only": False}}
    import pathlib

    state_path = pathlib.Path(str(tmp_path)) / "state.json"

    r_no_df = run_crisis_alpha_pipeline(
        ctx, policy=policy, state_path=state_path, dry_run=True
    )

    import json

    state_path.write_text(
        json.dumps(
            {
                "state": "WATCH",
                "entered_at_utc": None,
                "last_evaluated_utc": None,
                "reason": "reset",
                "geo_score_at_entry": 0.0,
                "cooldown_start_utc": None,
            }
        ),
        encoding="utf-8",
    )
    r_with_none = run_crisis_alpha_pipeline(
        ctx, policy=policy, state_path=state_path, prices_df=None, dry_run=True
    )

    assert r_no_df["state"] == r_with_none["state"]
    assert r_no_df["target_weights"] == r_with_none["target_weights"]


# ---------------------------------------------------------------------------
# _get_regime_cfg — nested intel format (GAP-4)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_get_regime_cfg_nested_intel_format() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import _get_regime_cfg

    policy = {
        "intel": {
            "crisis_alpha": {
                "basket_regime_detection": {
                    "enabled": True,
                    "lookback_days": 60,
                    "rate_hike_threshold": -0.03,
                }
            }
        }
    }
    cfg = _get_regime_cfg(policy)
    assert cfg == {"enabled": True, "lookback_days": 60, "rate_hike_threshold": -0.03}


@pytest.mark.fast
def test_get_regime_cfg_flat_format() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import _get_regime_cfg

    policy = {"crisis_alpha": {"basket_regime_detection": {"enabled": False}}}
    cfg = _get_regime_cfg(policy)
    assert cfg == {"enabled": False}


# ---------------------------------------------------------------------------
# pipeline: nested intel policy format injection (GAP-2)
# ---------------------------------------------------------------------------


def _make_intel_policy(enabled: bool = True, threshold: float = -0.05) -> dict:
    """Production-style nested policy: policy["intel"]["crisis_alpha"]."""
    return {
        "intel": {
            "crisis_alpha": {
                "enabled": True,
                "shadow_only": False,
                "basket_regime_detection": {
                    "enabled": enabled,
                    "lookback_days": 30,
                    "rate_hike_threshold": threshold,
                },
            }
        }
    }


@pytest.mark.fast
def test_pipeline_injects_rate_hike_basket_via_nested_intel_policy(
    tmp_path: object,
) -> None:
    """Production-format policy (intel.crisis_alpha) — basket injection fires and
    RATE_HIKE_BASKETS symbols reach generate_crisis_entry."""
    import json
    import pathlib
    from unittest.mock import patch

    from src.assembled_core.events.crisis_alpha.baskets import RATE_HIKE_BASKETS
    from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
    from src.assembled_core.events.crisis_alpha.pipeline import (
        run_crisis_alpha_pipeline,
    )

    ctx = CrisisAlphaContext(
        timestamp_utc=AS_OF.to_pydatetime(),
        geo_score=3.5,
        geo_sources=3,
        social_only=False,
        market_stress_ok=True,
        market_stress_score=3,
        health_ok=True,
        news_trigger_items=[{"severity": 3, "topic": "war", "source": "reuters"}],
        daily_pnl=0.0,
        daily_loss_limit=0.05,
        open_positions=[],
    )
    state_path = pathlib.Path(str(tmp_path)) / "state.json"
    # Pre-seed ACTIVE state so generate_crisis_entry is actually called
    state_path.write_text(
        json.dumps(
            {
                "state": "ACTIVE",
                "entered_at_utc": AS_OF.isoformat(),
                "last_evaluated_utc": AS_OF.isoformat(),
                "reason": "test",
                "geo_score_at_entry": 3.5,
                "cooldown_start_utc": None,
            }
        ),
        encoding="utf-8",
    )

    prices = _make_prices(tlt_ret=-0.16, shy_ret=-0.01)  # rate_hike regime
    policy = _make_intel_policy(enabled=True, threshold=-0.05)

    captured: list[dict] = []

    def _capture_entry(ctx_arg, policy_arg):  # noqa: ANN001
        captured.append(policy_arg)
        return {}, ["captured"]

    with patch(
        "src.assembled_core.events.crisis_alpha.pipeline.generate_crisis_entry",
        side_effect=_capture_entry,
    ):
        result = run_crisis_alpha_pipeline(
            ctx, policy=policy, state_path=state_path, prices_df=prices, dry_run=True
        )

    assert result["errors"] == [], f"Unexpected errors: {result['errors']}"
    assert captured, (
        "generate_crisis_entry was never called — state did not reach ACTIVE+gates_ok"
    )
    injected_policy = captured[0]
    injected_baskets = injected_policy["intel"]["crisis_alpha"].get("baskets", [])
    injected_syms = {b["symbol"] for b in injected_baskets}
    expected_syms = {b["symbol"] for b in RATE_HIKE_BASKETS}
    assert injected_syms == expected_syms, (
        f"Nested-intel injection failed: expected {expected_syms}, got {injected_syms}"
    )


@pytest.mark.fast
def test_pipeline_does_not_mutate_caller_policy(tmp_path: object) -> None:
    """Step 3b deepcopy must not mutate the original policy dict."""
    import copy
    import pathlib

    from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
    from src.assembled_core.events.crisis_alpha.pipeline import (
        run_crisis_alpha_pipeline,
    )

    ctx = CrisisAlphaContext(
        timestamp_utc=AS_OF.to_pydatetime(),
        geo_score=2.0,
        geo_sources=2,
        social_only=False,
        market_stress_ok=True,
        market_stress_score=2,
        health_ok=True,
        news_trigger_items=[{"severity": 2, "topic": "test", "source": "test"}],
        daily_pnl=0.0,
        daily_loss_limit=0.05,
        open_positions=[],
    )
    policy: dict = {
        "crisis_alpha": {
            "enabled": True,
            "shadow_only": False,
            "basket_regime_detection": {
                "enabled": True,
                "lookback_days": 30,
                "rate_hike_threshold": -0.05,
            },
        }
    }
    policy_snapshot = copy.deepcopy(policy)
    state_path = pathlib.Path(str(tmp_path)) / "state.json"
    prices = _make_prices(tlt_ret=-0.16, shy_ret=-0.01)  # triggers rate_hike

    run_crisis_alpha_pipeline(
        ctx, policy=policy, state_path=state_path, prices_df=prices, dry_run=True
    )
    assert policy == policy_snapshot, (
        "pipeline must not mutate the caller's policy dict"
    )


# ---------------------------------------------------------------------------
# RATE_HIKE_BASKETS weight invariants (GAP-5)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_rate_hike_baskets_weights_valid() -> None:
    from src.assembled_core.events.crisis_alpha.baskets import RATE_HIKE_BASKETS

    for b in RATE_HIKE_BASKETS:
        w = b["max_weight"]
        assert 0 < w <= 1.0, f"{b['symbol']} max_weight={w} out of range"
    total = sum(b["max_weight"] for b in RATE_HIKE_BASKETS)
    assert total <= 1.0, f"RATE_HIKE_BASKETS total max_weight={total} > 1.0"
