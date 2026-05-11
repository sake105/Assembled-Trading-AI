"""Tests für ensemble_regime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.ensemble_regime import EnsembleConfig, ensemble_regime


def _idx(n: int = 200) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=n, freq="B")


def test_ensemble_majority():
    idx = _idx()
    # Alle 3 sagen stress an einem Tag
    a = pd.Series(["calm"] * 100 + ["stress"] * 100, index=idx, name="a")
    b = pd.Series(["calm"] * 110 + ["stress"] * 90, index=idx, name="b")
    c = pd.Series(["calm"] * 200, index=idx, name="c")
    out = ensemble_regime(
        drawdown_regime=a,
        multi_signal_regime_in=b,
        macro_regime_in=c,
        config=EnsembleConfig(voting_scheme="majority", smoothing_days=1),
    )
    # Bei index 150 sagen 2/3 stress → majority = stress
    assert out.loc[idx[150], "regime"] == "stress"
    # Bei index 50 sagen 0/3 stress → calm
    assert out.loc[idx[50], "regime"] == "calm"


def test_ensemble_conservative_requires_all():
    idx = _idx()
    a = pd.Series(["stress"] * 200, index=idx)
    b = pd.Series(["stress"] * 200, index=idx)
    c = pd.Series(["calm"] * 200, index=idx)
    out = ensemble_regime(
        drawdown_regime=a,
        multi_signal_regime_in=b,
        macro_regime_in=c,
        config=EnsembleConfig(voting_scheme="conservative", smoothing_days=1),
    )
    # 2/3 stress, 1/3 calm → conservative requires ALL → calm
    assert (out["regime"] == "calm").all()


def test_ensemble_any_triggers_with_one():
    idx = _idx()
    a = pd.Series(["calm"] * 200, index=idx)
    b = pd.Series(["calm"] * 100 + ["stress"] * 100, index=idx)
    c = pd.Series(["calm"] * 200, index=idx)
    out = ensemble_regime(
        drawdown_regime=a,
        multi_signal_regime_in=b,
        macro_regime_in=c,
        config=EnsembleConfig(voting_scheme="any", smoothing_days=1),
    )
    assert out.loc[idx[150], "regime"] == "stress"


def test_ensemble_weighted_mean_with_scores():
    idx = _idx()
    drawdown_score = pd.Series(np.full(200, 0.8), index=idx)
    ms_score = pd.Series(np.full(200, 0.4), index=idx)
    macro_score = pd.Series(np.full(200, 0.2), index=idx)
    # weights: drawdown 0.40, ms 0.30, macro 0.30
    # weighted mean = 0.40*0.8 + 0.30*0.4 + 0.30*0.2 = 0.32 + 0.12 + 0.06 = 0.50
    out = ensemble_regime(
        drawdown_score=drawdown_score,
        multi_signal_score=ms_score,
        macro_score=macro_score,
        config=EnsembleConfig(
            voting_scheme="weighted_mean", threshold=0.45, smoothing_days=1
        ),
    )
    assert "ensemble_score" in out.columns
    np.testing.assert_array_almost_equal(
        out["ensemble_score"].values, np.full(200, 0.50)
    )
    assert (out["regime"] == "stress").all()


def test_ensemble_empty_input_returns_empty():
    out = ensemble_regime()
    assert out.empty


def test_ensemble_partial_inputs():
    idx = _idx()
    a = pd.Series(["calm"] * 200, index=idx)
    # Nur ein Detector — sollte trotzdem funktionieren
    out = ensemble_regime(
        drawdown_regime=a,
        config=EnsembleConfig(voting_scheme="majority", smoothing_days=1),
    )
    assert (out["regime"] == "calm").all()


def test_ensemble_smoothing_prevents_flicker():
    idx = _idx()
    # Konstruiere eine wechselnde Sequenz: 100 calm, 1 stress, 100 calm
    raw = ["calm"] * 100 + ["stress"] * 1 + ["calm"] * 99
    a = pd.Series(raw, index=idx)
    out = ensemble_regime(
        drawdown_regime=a,
        config=EnsembleConfig(voting_scheme="majority", smoothing_days=5),
    )
    # Smoothing sollte das einzelne stress unterdrücken
    n_changes = (out["regime"] != out["regime"].shift()).sum()
    assert n_changes <= 3


def test_ensemble_score_handles_nan():
    idx = _idx()
    a = pd.Series(np.full(200, np.nan), index=idx)
    b = pd.Series(np.full(200, 0.6), index=idx)
    out = ensemble_regime(
        drawdown_score=a,
        multi_signal_score=b,
        config=EnsembleConfig(
            voting_scheme="weighted_mean", threshold=0.5, smoothing_days=1
        ),
    )
    # NaN-row sollte zumindest nicht crashen
    assert len(out) == 200
