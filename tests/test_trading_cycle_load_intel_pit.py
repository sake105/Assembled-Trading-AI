"""Regression tests: _load_intel PIT guard for market_stress computation.

Verifies that compute_market_stress receives prices filtered to <= as_of,
not the full historical tail. Without this filter, backtest stress values
are always computed from the end of the dataset rather than the current date.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_ctx(prices: pd.DataFrame, as_of: pd.Timestamp) -> MagicMock:
    ctx = MagicMock()
    ctx.prices = prices
    ctx.as_of = as_of
    ctx.intel_health_flags = {}
    ctx.intel_sim_applied = False
    ctx.disclosures_triggers = None
    ctx.crisis_state_intel = None
    ctx.news_geo = None
    ctx.market_stress = None
    return ctx


def _make_prices(timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [100.0 + i for i in range(len(timestamps))],
            "symbol": ["SPY"] * len(timestamps),
        }
    )


def _policy(enabled: bool = True) -> dict:
    return {
        "market_stress": {
            "enabled": enabled,
            "lookback_days": 5,
            "qc": {"if_data_missing": False},
            "metrics": {
                "vol_z": {"enabled": True, "z_threshold": 1.5},
                "dd_lookback": {"enabled": True, "dd_threshold": -0.05},
            },
            "confirm_rule": {"mode": "any"},
        },
        "intel": {
            "disclosures_triggers": {"enabled": False},
            "crisis_alpha": {"enabled": False},
        },
        "risk_state_machine": {"enabled": False},
    }


@pytest.mark.fast
def test_load_intel_market_stress_pit_filter_excludes_future(tmp_path: Path) -> None:
    """compute_market_stress must NOT receive rows with timestamp > as_of."""
    from src.assembled_core.pipeline.trading_cycle_v2 import _load_intel

    as_of = pd.Timestamp("2024-06-01", tz="UTC")
    past = [as_of - pd.Timedelta(days=d) for d in range(10, 0, -1)]
    future = [as_of + pd.Timedelta(days=d) for d in range(1, 6)]
    prices = _make_prices(past + future)

    ctx = _make_ctx(prices, as_of)

    captured: dict = {}

    from src.assembled_core.risk.market_stress import compute_market_stress as real_cms

    def spy_cms(p: pd.DataFrame, pol: dict) -> dict:
        captured["prices"] = p.copy()
        return real_cms(p, pol)

    with patch(
        "src.assembled_core.pipeline.trading_cycle_v2.compute_market_stress", spy_cms
    ):
        _load_intel(ctx, _policy(), tmp_path, logging.getLogger())

    assert "prices" in captured, "compute_market_stress was not called"
    ts_series = pd.to_datetime(captured["prices"]["timestamp"], utc=True)
    assert ts_series.max() <= as_of, (
        f"PIT violation: compute_market_stress received rows past as_of={as_of.date()}, "
        f"max timestamp={ts_series.max().date()}"
    )
    assert len(captured["prices"]) == 10, "Expected only the 10 past rows, got " + str(
        len(captured["prices"])
    )


@pytest.mark.fast
def test_load_intel_market_stress_pit_filter_includes_as_of(tmp_path: Path) -> None:
    """The as_of day itself must be included in the filtered slice."""
    from src.assembled_core.pipeline.trading_cycle_v2 import _load_intel

    as_of = pd.Timestamp("2024-06-01", tz="UTC")
    prices = _make_prices(
        [as_of - pd.Timedelta(days=2), as_of - pd.Timedelta(days=1), as_of]
    )
    ctx = _make_ctx(prices, as_of)

    captured: dict = {}
    from src.assembled_core.risk.market_stress import compute_market_stress as real_cms

    def spy_cms(p: pd.DataFrame, pol: dict) -> dict:
        captured["prices"] = p.copy()
        return real_cms(p, pol)

    with patch(
        "src.assembled_core.pipeline.trading_cycle_v2.compute_market_stress", spy_cms
    ):
        _load_intel(ctx, _policy(), tmp_path, logging.getLogger())

    ts_series = pd.to_datetime(captured["prices"]["timestamp"], utc=True)
    assert (ts_series == as_of).any(), (
        f"as_of={as_of.date()} not found in filtered slice"
    )


@pytest.mark.fast
def test_load_intel_market_stress_disabled_skips_filter(tmp_path: Path) -> None:
    """When market_stress.enabled=False, compute_market_stress is never called."""
    from src.assembled_core.pipeline.trading_cycle_v2 import _load_intel

    as_of = pd.Timestamp("2024-06-01", tz="UTC")
    prices = _make_prices([as_of - pd.Timedelta(days=i) for i in range(5)])
    ctx = _make_ctx(prices, as_of)

    pol = _policy(enabled=False)
    called = []

    with patch(
        "src.assembled_core.pipeline.trading_cycle_v2.compute_market_stress",
        side_effect=lambda p, pol: called.append(1) or {},
    ):
        _load_intel(ctx, pol, tmp_path, logging.getLogger())

    assert len(called) == 0
    assert ctx.market_stress is None


@pytest.mark.fast
def test_load_intel_market_stress_pit_filter_degraded_on_bad_timestamp(
    tmp_path: Path,
) -> None:
    """When timestamp column is unparseable, health flag must be set to DEGRADED."""
    from src.assembled_core.pipeline.trading_cycle_v2 import _load_intel

    as_of = pd.Timestamp("2024-06-01", tz="UTC")
    prices = pd.DataFrame(
        {
            "timestamp": ["not-a-date", "also-bad"],
            "close": [100.0, 101.0],
            "symbol": ["SPY", "SPY"],
        }
    )
    ctx = _make_ctx(prices, as_of)
    ctx.intel_health_flags = {}

    with patch(
        "src.assembled_core.pipeline.trading_cycle_v2.compute_market_stress",
        return_value={},
    ):
        _load_intel(ctx, _policy(), tmp_path, logging.getLogger())

    assert ctx.intel_health_flags.get("intel_market_stress") == "DEGRADED"
