"""Regression tests: _load_intel PIT wiring for the disclosures + crisis_state feeds.

Batch 2 of the Item-B preventive PIT fix. _load_intel must NOT inject a
future-dated (non-as_of) intel snapshot into a historical backtest bar:

  * disclosures_triggers: load_disclosures_triggers is now called with
    ``as_of=ctx.as_of``. A snapshot whose ``generated_utc`` is AFTER the bar
    instant must be dropped (ctx.disclosures_triggers stays None); a snapshot at
    or before as_of loads as today.

  * crisis_state.json: the CrisisState artifact carries NO field proving WHEN
    the snapshot became available (its only datetime, ``entered_at``, is the
    state-entry time, not snapshot availability). So in backtest (as_of set) the
    snapshot CANNOT be proven PIT and must NOT be injected; instead the
    ``intel_crisis_alpha`` health flag is set to DEGRADED (observable degrade,
    mirroring the market_stress guard). Live/EOD (as_of None) injects as today.

These tests are discriminating: they FAIL if as_of is not wired into the two
producers (i.e. if the snapshot is injected regardless of the bar instant).
They mirror the style of test_trading_cycle_load_intel_pit.py (MagicMock ctx,
real _load_intel, tmp_path artifacts).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

import src.assembled_core.pipeline.trading_cycle_v2 as tcv2
from src.assembled_core.pipeline.trading_cycle_v2 import _load_intel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(as_of: pd.Timestamp | None) -> MagicMock:
    ctx = MagicMock()
    # market_stress disabled in these tests -> prices content irrelevant, but
    # keep a minimal non-empty frame so nothing trips on an empty slice.
    ctx.prices = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
            "close": [100.0],
            "symbol": ["SPY"],
        }
    )
    ctx.as_of = as_of
    ctx.intel_health_flags = {}
    ctx.intel_sim_applied = False
    ctx.disclosures_triggers = None
    ctx.crisis_state_intel = None
    ctx.news_geo = None
    ctx.market_stress = None
    return ctx


def _policy_disclosures(path: Path) -> dict:
    return {
        "market_stress": {"enabled": False},
        "intel": {
            "disclosures_triggers": {"enabled": True, "path": str(path)},
            "crisis_alpha": {"enabled": False},
        },
        "risk_state_machine": {"enabled": False},
    }


def _policy_crisis(path: Path) -> dict:
    return {
        "market_stress": {"enabled": False},
        "intel": {
            "disclosures_triggers": {"enabled": False},
            "crisis_alpha": {"enabled": True, "crisis_state_path": str(path)},
        },
        "risk_state_machine": {"enabled": False},
    }


def _write_disclosures(path: Path, generated_utc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "disclosures.triggers.v1",
                "generated_utc": generated_utc,
                "items": [{"trigger_id": "d1", "severity": 2}],
            }
        ),
        encoding="utf-8",
    )


def _write_crisis_state(path: Path, entered_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mode": "WATCH",
                "geo_score": 2,
                "active_triggers": ["t1"],
                "entered_at": entered_at,
                "risk_posture": {},
                "basket_overrides": {},
                "audit_trail": [],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _reset_crisis_warn_once():
    """Reset the module-level warn-once flag so each test starts clean."""
    tcv2._CRISIS_STATE_PIT_WARNED = False
    yield
    tcv2._CRISIS_STATE_PIT_WARNED = False


# ---------------------------------------------------------------------------
# disclosures_triggers PIT wiring
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_disclosures_future_snapshot_dropped_in_backtest(tmp_path: Path) -> None:
    """generated_utc > as_of -> snapshot must NOT be injected (PIT drop)."""
    as_of = pd.Timestamp("2024-06-01", tz="UTC")
    snap_path = tmp_path / "triggers_latest.json"
    # Snapshot generated AFTER the bar instant.
    _write_disclosures(snap_path, "2024-09-01T00:00:00Z")

    ctx = _make_ctx(as_of)
    _load_intel(ctx, _policy_disclosures(snap_path), tmp_path, logging.getLogger())

    assert ctx.disclosures_triggers is None, (
        "PIT violation: future-dated disclosures snapshot was injected at "
        f"as_of={as_of.date()}"
    )
    assert ctx.intel_health_flags.get("intel_disclosures_triggers") == "DEGRADED"


@pytest.mark.fast
def test_disclosures_past_snapshot_loaded_in_backtest(tmp_path: Path) -> None:
    """generated_utc <= as_of -> snapshot loads (available at the bar)."""
    as_of = pd.Timestamp("2024-06-01", tz="UTC")
    snap_path = tmp_path / "triggers_latest.json"
    _write_disclosures(snap_path, "2024-03-01T00:00:00Z")

    ctx = _make_ctx(as_of)
    _load_intel(ctx, _policy_disclosures(snap_path), tmp_path, logging.getLogger())

    assert ctx.disclosures_triggers is not None, (
        "past-dated disclosures snapshot should load at as_of"
    )
    assert ctx.disclosures_triggers.generated_utc == "2024-03-01T00:00:00Z"
    assert ctx.intel_health_flags.get("intel_disclosures_triggers") != "DEGRADED"


@pytest.mark.fast
def test_disclosures_live_as_of_none_loads_as_today(tmp_path: Path) -> None:
    """as_of None (live/EOD) -> snapshot loads regardless of date (byte-identical)."""
    snap_path = tmp_path / "triggers_latest.json"
    # Date far in the "future" relative to nothing — with as_of None it loads.
    _write_disclosures(snap_path, "2099-01-01T00:00:00Z")

    ctx = _make_ctx(None)
    _load_intel(ctx, _policy_disclosures(snap_path), tmp_path, logging.getLogger())

    assert ctx.disclosures_triggers is not None, (
        "live path (as_of None) must load the snapshot as today"
    )
    assert ctx.disclosures_triggers.generated_utc == "2099-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# crisis_state PIT wiring (no usable PIT timestamp -> dont-inject + DEGRADED)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_crisis_state_not_injected_in_backtest(tmp_path: Path) -> None:
    """as_of set + no PIT field -> snapshot NOT injected, flag DEGRADED."""
    as_of = pd.Timestamp("2024-06-01", tz="UTC")
    cs_path = tmp_path / "crisis_state.json"
    # entered_at BEFORE as_of would PASS a naive entered_at gate — but entered_at
    # is not snapshot-availability, so it must STILL not be injected.
    _write_crisis_state(cs_path, "2024-01-01T00:00:00Z")

    ctx = _make_ctx(as_of)
    _load_intel(ctx, _policy_crisis(cs_path), tmp_path, logging.getLogger())

    assert ctx.crisis_state_intel is None, (
        "PIT violation: crisis_state snapshot injected in backtest despite no "
        "provable PIT timestamp"
    )
    assert ctx.news_geo is None
    assert ctx.intel_health_flags.get("intel_crisis_alpha") == "DEGRADED"


@pytest.mark.fast
def test_crisis_state_live_as_of_none_injects_as_today(tmp_path: Path) -> None:
    """as_of None (live/EOD) -> crisis_state injects as today (byte-identical)."""
    cs_path = tmp_path / "crisis_state.json"
    _write_crisis_state(cs_path, "2024-01-01T00:00:00Z")

    ctx = _make_ctx(None)
    _load_intel(ctx, _policy_crisis(cs_path), tmp_path, logging.getLogger())

    assert ctx.crisis_state_intel is not None, (
        "live path (as_of None) must inject crisis_state as today"
    )
    assert ctx.crisis_state_intel.get("geo_score") == 2
    assert ctx.news_geo is not None
    assert ctx.news_geo.get("geo_score") == 2
    assert ctx.intel_health_flags.get("intel_crisis_alpha") != "DEGRADED"


@pytest.mark.fast
def test_crisis_state_warn_once(tmp_path: Path, caplog) -> None:
    """The backtest-degrade warning is emitted once, not per-bar."""
    as_of = pd.Timestamp("2024-06-01", tz="UTC")
    cs_path = tmp_path / "crisis_state.json"
    _write_crisis_state(cs_path, "2024-01-01T00:00:00Z")
    pol = _policy_crisis(cs_path)

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            ctx = _make_ctx(as_of)
            _load_intel(ctx, pol, tmp_path, logging.getLogger())

    warn_lines = [
        r
        for r in caplog.records
        if "crisis_state.json has no PIT timestamp" in r.message
    ]
    assert len(warn_lines) == 1, (
        f"expected exactly one warn-once line across 3 bars, got {len(warn_lines)}"
    )
