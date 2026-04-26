from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, cast

import pytest

from src.assembled_core.pipeline.trading_cycle_shared import TradingContext
from src.assembled_core.risk.georisk_overlay import compute_exposure_multiplier


pytestmark = [pytest.mark.unit, pytest.mark.phase6]


class DummyCtx(SimpleNamespace):
    """Lightweight stand-in for TradingContext used in unit tests."""

    news_geo: Dict[str, Any] | None
    intel_health_flags: Dict[str, str] | None
    risk_state: Dict[str, Any] | None


def _base_policy() -> Dict[str, Any]:
    return {
        "georisk_overlay": {
            "enabled": True,
            "source": "news_geo",
            "mapping": {
                "WATCH": {
                    "multiplier": 1.00,
                    "hedge": {"enabled": False},
                },
                "ACTIVE": {
                    "multiplier": 0.70,
                    "hedge": {"enabled": False},
                },
                "COOLDOWN": {
                    "multiplier": 0.85,
                    "hedge": {"enabled": False},
                },
                "PAUSE": {
                    "multiplier": 0.00,
                    "hedge": {"enabled": False},
                },
            },
            "by_geo_score": {
                "0": 1.00,
                "1": 0.90,
                "2": 0.70,
                "3": 0.50,
            },
            "confidence_floor": 0.60,
            "qc": {
                "if_intel_degraded": "WATCH",
            },
        }
    }


def test_georisk_multiplier_by_geo_score() -> None:
    """ACTIVE, geo_score=2, conf=0.8 -> by_geo_score multiplier 0.70."""
    ctx = DummyCtx(
        news_geo={
            "geo_score": 2,
            "geo_confidence": 0.8,
            "state_hint": "ACTIVE",
        },
        intel_health_flags={},
    )
    policy = _base_policy()
    m = compute_exposure_multiplier(cast(TradingContext, ctx), policy)
    assert m == pytest.approx(0.70)


def test_georisk_multiplier_below_conf_floor_is_one() -> None:
    """Confidence below floor should yield multiplier 1.0."""
    ctx = DummyCtx(
        news_geo={
            "geo_score": 2,
            "geo_confidence": 0.5,  # below floor 0.60
            "state_hint": "ACTIVE",
        },
        intel_health_flags={},
    )
    policy = _base_policy()
    m = compute_exposure_multiplier(cast(TradingContext, ctx), policy)
    assert m == pytest.approx(1.0)


def test_georisk_degraded_treated_as_watch() -> None:
    """Degraded intel treated as WATCH -> multiplier 1.0 (WATCH mapping)."""
    ctx = DummyCtx(
        news_geo=None,
        intel_health_flags={"intel_geo_score": "DEGRADED"},
    )
    policy = _base_policy()
    m = compute_exposure_multiplier(cast(TradingContext, ctx), policy)
    assert m == pytest.approx(1.0)


def test_overlay_uses_cooldown_multiplier_when_state_machine_sets_cooldown() -> None:
    """ctx.risk_state.state COOLDOWN => multiplier from mapping (0.85) when by_geo_score not overriding."""
    ctx = DummyCtx(
        news_geo={"geo_score": 4, "geo_confidence": 0.8, "state_hint": "ACTIVE"},
        intel_health_flags={},
        risk_state={"state": "COOLDOWN"},
    )
    policy = _base_policy()
    # geo_score 4 not in by_geo_score -> use mapping[COOLDOWN] = 0.85
    m = compute_exposure_multiplier(cast(TradingContext, ctx), policy)
    assert m == pytest.approx(0.85)


def test_pause_multiplier_zero() -> None:
    """ctx.risk_state.state PAUSE => multiplier 0.0."""
    ctx = DummyCtx(
        news_geo={"geo_score": 4, "geo_confidence": 0.9, "state_hint": "ACTIVE"},
        intel_health_flags={},
        risk_state={"state": "PAUSE"},
    )
    policy = _base_policy()
    # geo_score 4 not in by_geo_score -> use mapping[PAUSE] = 0.0
    m = compute_exposure_multiplier(cast(TradingContext, ctx), policy)
    assert m == pytest.approx(0.0)
