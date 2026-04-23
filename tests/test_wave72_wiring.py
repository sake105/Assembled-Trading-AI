"""Tests for wave-72 module wiring into trading_cycle.py.

Covers:
  Step 5.41 — execution.fill_model (PartialFillModel)
  Step 5.42 — execution.intent_store (has_intent / make_daily_key)
  Step 5.43 — execution.pre_open_signals (compute_overnight_gap_signal)
"""

from __future__ import annotations

import pytest

from src.assembled_core.execution.fill_model import (
    PartialFillModel,
    apply_cash_gate,
    ensure_fill_schema,
)
from src.assembled_core.execution.intent_store import (
    has_intent,
    make_daily_key,
    make_run_key,
)
from src.assembled_core.execution.pre_open_signals import (
    PreOpenSignal,
    PreOpenConfig,
    compute_overnight_gap_signal,
    compute_global_market_signal,
)


# ---------------------------------------------------------------------------
# fill_model (Step 5.41)
# ---------------------------------------------------------------------------

def test_partial_fill_model_creates():
    model = PartialFillModel()
    assert isinstance(model, PartialFillModel)


def test_partial_fill_model_defaults():
    model = PartialFillModel()
    assert model.adv_window > 0
    assert 0.0 < model.participation_cap <= 1.0


def test_partial_fill_model_custom():
    model = PartialFillModel(adv_window=10, participation_cap=0.1)
    assert model.adv_window == 10
    assert model.participation_cap == 0.1


def test_apply_cash_gate_importable():
    assert callable(apply_cash_gate)


def test_ensure_fill_schema_importable():
    assert callable(ensure_fill_schema)


# ---------------------------------------------------------------------------
# intent_store (Step 5.42)
# ---------------------------------------------------------------------------

def test_make_daily_key_returns_str():
    key = make_daily_key("cycle_complete")
    assert isinstance(key, str)
    assert len(key) > 0  # key is a hash prefix, not the raw action


def test_make_run_key_returns_str():
    key = make_run_key("backtest_run", "run123")
    assert isinstance(key, str)


def test_has_intent_returns_bool():
    key = make_daily_key("__test_nonexistent_intent__")
    result = has_intent(key)
    assert isinstance(result, bool)


def test_has_intent_nonexistent_is_false():
    key = make_daily_key("__test_nonexistent_intent_xyz__")
    assert has_intent(key) is False


# ---------------------------------------------------------------------------
# pre_open_signals (Step 5.43)
# ---------------------------------------------------------------------------

def test_compute_overnight_gap_signal_returns_tuple():
    signal, confidence = compute_overnight_gap_signal(prev_close=100.0)
    assert isinstance(signal, float)
    assert isinstance(confidence, float)


def test_compute_overnight_gap_signal_zero_futures():
    signal, confidence = compute_overnight_gap_signal(prev_close=100.0, futures_return=0.0)
    assert signal == 0.0
    assert confidence == 0.0


def test_compute_overnight_gap_signal_positive_futures():
    signal, confidence = compute_overnight_gap_signal(prev_close=100.0, futures_return=0.01)
    assert isinstance(signal, float)
    assert confidence >= 0.0


def test_compute_global_market_signal_returns_tuple():
    result = compute_global_market_signal(europe_return=0.005, asia_return=-0.003)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_pre_open_config_creates():
    cfg = PreOpenConfig()
    assert isinstance(cfg, PreOpenConfig)
    assert 0 < cfg.min_strength < 1.0
