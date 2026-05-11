"""Tests für regime_conditional_allocator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.regime_conditional_allocator import (
    RegimeConfig,
    allocate_regime_conditional,
    detect_regime,
    regime_metrics,
)


def _bench_returns(n: int = 500, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    # Drift-positive, niedrige Vol → calm-Regime
    return pd.Series(
        rng.normal(0.0005, 0.005, n), index=pd.date_range("2022-01-01", periods=n)
    )


def test_detect_regime_calm_in_no_stress():
    r = _bench_returns()
    reg = detect_regime(r, RegimeConfig(drawdown_threshold=0.5))  # never triggers
    assert (reg == "calm").all()


def test_detect_regime_stress_kicks_in_on_drawdown():
    # Konstruiere künstlichen Crash
    rng = np.random.default_rng(0)
    base = rng.normal(0.0003, 0.005, 200)
    crash = np.array([-0.04] * 30)  # 30-Tage-Crash je -4 %
    recovery = rng.normal(0.0002, 0.005, 100)
    r = pd.Series(
        np.concatenate([base, crash, recovery]),
        index=pd.date_range("2022-01-01", periods=330),
    )
    reg = detect_regime(r, RegimeConfig(drawdown_threshold=0.10))
    # Mindestens ein paar Tage stress-Regime im Crash-Bereich
    assert (reg.iloc[200:240] == "stress").any()


def test_smoothing_prevents_flicker():
    # Konstruktion: ein einzelner Crash-Tag triggert NICHT bei smoothing_days=10
    # zumindest nicht nachdem die smoothing-Logik greift
    n = 300
    r = pd.Series(np.full(n, 0.0001), index=pd.date_range("2022-01-01", periods=n))
    r.iloc[100] = -0.20
    reg = detect_regime(r, RegimeConfig(drawdown_threshold=0.05, smoothing_days=10))
    # Direkt nach dem Crash sollte stress sein, aber Smoothing erlaubt früh-Flip
    n_changes = (reg != reg.shift()).sum()
    # Smoothing reduziert Tagesflicker
    assert n_changes < 50


def test_allocate_regime_conditional_basic():
    n = 300
    rng = np.random.default_rng(2)
    calm = pd.Series(
        rng.normal(0.0004, 0.005, n), index=pd.date_range("2022-01-01", periods=n)
    )
    stress = pd.Series(rng.normal(0.0001, 0.012, n), index=calm.index)

    # Triggere künstliches Stress-Regime
    crash = np.array([-0.03] * 25)
    calm.iloc[100:125] = crash
    stress.iloc[100:125] = np.full(25, 0.001)

    out = allocate_regime_conditional(
        calm, stress, RegimeConfig(drawdown_threshold=0.08)
    )
    assert "regime" in out.columns
    assert "allocated_return" in out.columns
    # Nach dem Crash sollte mindestens 1 Tag stress-allokiert sein
    assert (out["regime"] == "stress").any()
    # Allokation während stress = stress_return
    stress_rows = out[out["regime"] == "stress"]
    np.testing.assert_array_almost_equal(
        stress_rows["allocated_return"].to_numpy(),
        stress_rows["stress_return"].to_numpy(),
    )


def test_lag_prevents_lookahead():
    rng = np.random.default_rng(3)
    n = 200
    calm = pd.Series(
        rng.normal(0.0, 0.005, n), index=pd.date_range("2022-01-01", periods=n)
    )
    stress = pd.Series(rng.normal(0.0, 0.005, n), index=calm.index)
    out = allocate_regime_conditional(calm, stress, lag_days=1)
    # Erste Zeile muss NaN-Regime sein (kein Lookback)
    assert pd.isna(out["regime"].iloc[0])


def test_regime_metrics_returns_dict_with_keys():
    rng = np.random.default_rng(4)
    n = 300
    calm = pd.Series(
        rng.normal(0.0004, 0.005, n), index=pd.date_range("2022-01-01", periods=n)
    )
    stress = pd.Series(rng.normal(0.0002, 0.01, n), index=calm.index)
    out = allocate_regime_conditional(calm, stress)
    metrics = regime_metrics(out)
    assert "all" in metrics
    assert "calm" in metrics or "stress" in metrics


def test_empty_input_returns_empty_df():
    empty_a = pd.Series(dtype=float)
    empty_b = pd.Series(dtype=float)
    out = allocate_regime_conditional(empty_a, empty_b)
    assert out.empty
