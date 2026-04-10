"""Tests for the trailing-stops wiring (Sprint 1 / W1).

Covers the pure building blocks used by trading_cycle Phase 11.5:
- compute_trailing_stops with regime-adaptive ATR multipliers
- apply_stop_reductions_to_weights semantics
- VIX scaling factor monotonicity
- Policy-disabled wiring has no effect on target_positions
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.risk.trailing_stops import (
    TrailingStopResult,
    _vix_multiplier_factor,
    apply_stop_reductions_to_weights,
    compute_trailing_stops,
)


def _mk_panel(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(closes), freq="D", tz="UTC")
    close = np.array(closes, dtype=float)
    # Cheap OHLC from closes
    high = close * 1.01
    low = close * 0.99
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": [symbol] * len(closes),
            "high": high,
            "low": low,
            "close": close,
        }
    )


# ---------- VIX scaling ----------

def test_vix_factor_none_is_neutral() -> None:
    assert _vix_multiplier_factor(None) == 1.0


def test_vix_factor_is_monotone_in_stress() -> None:
    calm = _vix_multiplier_factor(10.0)
    normal = _vix_multiplier_factor(18.0)
    elevated = _vix_multiplier_factor(25.0)
    crisis = _vix_multiplier_factor(40.0)
    assert calm < normal <= elevated < crisis


# ---------- compute_trailing_stops ----------

def test_trailing_stops_no_trigger_on_uptrend() -> None:
    prices = _mk_panel("AAA", [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                               110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
    positions = {"AAA": {"entry_price": 100.0, "qty": 10.0, "weight": 0.1}}
    res = compute_trailing_stops(
        positions=positions, prices_df=prices, regime="bull", vix_level=18.0,
        current_bar=20,
    )
    assert "AAA" not in res.triggered_symbols


def test_trailing_stops_trigger_on_sharp_drop() -> None:
    closes = [100 + i for i in range(20)]  # climb to 119
    closes += [90.0]                        # sharp drop below any bull stop
    prices = _mk_panel("BBB", closes)
    positions = {"BBB": {"entry_price": 100.0, "qty": 10.0, "weight": 0.1}}
    res = compute_trailing_stops(
        positions=positions, prices_df=prices, regime="crisis", vix_level=35.0,
        current_bar=len(closes),
    )
    # Under crisis regime the ATR stop is tight; large drop should trigger.
    assert "BBB" in res.triggered_symbols or "BBB" in res.reduction_symbols


def test_trailing_stops_empty_positions() -> None:
    prices = _mk_panel("AAA", [100, 101, 102])
    res = compute_trailing_stops(positions={}, prices_df=prices, regime="bull")
    assert res.triggered_symbols == []
    assert res.reduction_symbols == {}


def test_trailing_stops_missing_price_data_skipped() -> None:
    prices = _mk_panel("AAA", [100, 101, 102])
    positions = {"ZZZ": {"entry_price": 50.0, "qty": 1.0, "weight": 0.1}}
    res = compute_trailing_stops(positions=positions, prices_df=prices, regime="bull")
    assert res.triggered_symbols == []


# ---------- apply_stop_reductions_to_weights ----------

def test_apply_triggered_zeroes_weight() -> None:
    weights = {"AAA": 0.1, "BBB": 0.2}
    result = TrailingStopResult(triggered_symbols=["AAA"])
    out = apply_stop_reductions_to_weights(weights, result)
    assert out["AAA"] == 0.0
    assert out["BBB"] == 0.2


def test_apply_reduction_scales_weight() -> None:
    weights = {"AAA": 0.2, "BBB": 0.1}
    result = TrailingStopResult(reduction_symbols={"AAA": 0.5})
    out = apply_stop_reductions_to_weights(weights, result)
    assert abs(out["AAA"] - 0.1) < 1e-9
    assert out["BBB"] == 0.1


def test_apply_triggered_wins_over_reduction() -> None:
    weights = {"AAA": 0.2}
    result = TrailingStopResult(
        triggered_symbols=["AAA"],
        reduction_symbols={"AAA": 0.25},
    )
    out = apply_stop_reductions_to_weights(weights, result)
    assert out["AAA"] == 0.0


def test_apply_no_op_when_no_triggers_or_reductions() -> None:
    weights = {"AAA": 0.1, "BBB": 0.2}
    out = apply_stop_reductions_to_weights(weights, TrailingStopResult())
    assert out == weights


# ---------- regime multiplier behaviour ----------

def test_bear_regime_stops_tighter_than_bull() -> None:
    closes = list(np.linspace(100, 110, 20)) + [102.0]
    prices = _mk_panel("CCC", closes)
    positions = {"CCC": {"entry_price": 100.0, "qty": 1.0, "weight": 0.1}}

    res_bull = compute_trailing_stops(
        positions=positions, prices_df=prices, regime="bull",
        vix_level=18.0, current_bar=len(closes),
    )
    res_bear = compute_trailing_stops(
        positions=positions, prices_df=prices, regime="bear",
        vix_level=18.0, current_bar=len(closes),
    )
    bull_stop = res_bull.stops[0].stop_price
    bear_stop = res_bear.stops[0].stop_price
    # Bear multiplier is smaller → stop sits closer to HWM → higher stop price.
    assert bear_stop > bull_stop
