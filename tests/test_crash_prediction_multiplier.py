"""Tests for crash-prediction multiplier in multifactor_v1 (Sprint 1 / W6)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.strategies.multifactor_v1 import (
    _crash_prediction_multiplier,
)


def _mk_df(symbol: str = "SPY", n: int = 250) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    closes = 400.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": [symbol] * n,
            "close": closes,
        }
    )


class _FakeSignal:
    def __init__(self, prob: float) -> None:
        self.crash_probability = prob
        self.expected_severity = 0.0
        self.time_horizon_days = 0
        self.confidence = 0.0
        self.contributing_signals: dict[str, float] = {}
        self.recommended_sectors_short: list[str] = []
        self.recommended_instruments: list[str] = []
        self.active = prob >= 0.60


def _patched_engine(prob: float):
    class _FakeEngine:
        def predict(self, *args, **kwargs) -> _FakeSignal:
            return _FakeSignal(prob)

    return _FakeEngine


def test_disabled_returns_1() -> None:
    df = _mk_df()
    assert _crash_prediction_multiplier(df, {}) == 1.0
    assert (
        _crash_prediction_multiplier(df, {"crash_prediction": {"enabled": False}})
        == 1.0
    )


def test_missing_reference_symbol_returns_1() -> None:
    df = _mk_df(symbol="AAPL")
    cfg = {"crash_prediction": {"enabled": True, "reference_symbol": "SPY"}}
    assert _crash_prediction_multiplier(df, cfg) == 1.0


def test_missing_columns_returns_1() -> None:
    df = pd.DataFrame({"foo": [1, 2, 3]})
    cfg = {"crash_prediction": {"enabled": True}}
    assert _crash_prediction_multiplier(df, cfg) == 1.0


@patch("src.assembled_core.signals.crash_prediction.CrashPredictionEngine")
def test_low_prob_returns_1(mock_engine) -> None:
    mock_engine.return_value = _patched_engine(0.10)()
    df = _mk_df()
    cfg = {"crash_prediction": {"enabled": True}}
    assert _crash_prediction_multiplier(df, cfg) == 1.0


@patch("src.assembled_core.signals.crash_prediction.CrashPredictionEngine")
def test_moderate_prob_returns_0_8(mock_engine) -> None:
    mock_engine.return_value = _patched_engine(0.35)()
    df = _mk_df()
    cfg = {"crash_prediction": {"enabled": True}}
    assert _crash_prediction_multiplier(df, cfg) == 0.8


@patch("src.assembled_core.signals.crash_prediction.CrashPredictionEngine")
def test_elevated_prob_returns_0_5(mock_engine) -> None:
    mock_engine.return_value = _patched_engine(0.55)()
    df = _mk_df()
    cfg = {"crash_prediction": {"enabled": True}}
    assert _crash_prediction_multiplier(df, cfg) == 0.5


@patch("src.assembled_core.signals.crash_prediction.CrashPredictionEngine")
def test_high_prob_returns_0_2(mock_engine) -> None:
    mock_engine.return_value = _patched_engine(0.75)()
    df = _mk_df()
    cfg = {"crash_prediction": {"enabled": True}}
    assert _crash_prediction_multiplier(df, cfg) == 0.2


@patch("src.assembled_core.signals.crash_prediction.CrashPredictionEngine")
def test_extreme_prob_returns_0(mock_engine) -> None:
    mock_engine.return_value = _patched_engine(0.90)()
    df = _mk_df()
    cfg = {"crash_prediction": {"enabled": True}}
    assert _crash_prediction_multiplier(df, cfg) == 0.0


@patch("src.assembled_core.signals.crash_prediction.CrashPredictionEngine")
def test_mapping_is_monotone_non_increasing(mock_engine) -> None:
    df = _mk_df()
    cfg = {"crash_prediction": {"enabled": True}}
    last = 1.01
    for p in (0.0, 0.2, 0.35, 0.55, 0.75, 0.9):
        mock_engine.return_value = _patched_engine(p)()
        m = _crash_prediction_multiplier(df, cfg)
        assert m <= last
        last = m


@patch(
    "src.assembled_core.signals.crash_prediction.CrashPredictionEngine",
    side_effect=RuntimeError("boom"),
)
def test_engine_exception_swallowed_returns_1(_mock) -> None:
    df = _mk_df()
    cfg = {"crash_prediction": {"enabled": True}}
    assert _crash_prediction_multiplier(df, cfg) == 1.0


@patch("src.assembled_core.signals.crash_prediction.CrashPredictionEngine")
def test_custom_reference_symbol(mock_engine) -> None:
    mock_engine.return_value = _patched_engine(0.35)()
    df = _mk_df(symbol="QQQ")
    cfg = {"crash_prediction": {"enabled": True, "reference_symbol": "QQQ"}}
    assert _crash_prediction_multiplier(df, cfg) == 0.8
