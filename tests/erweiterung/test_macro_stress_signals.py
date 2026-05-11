"""Tests für macro_stress_signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.macro_stress_signals import (
    MacroStressConfig,
    hy_spread_widening_signal,
    macro_stress_composite,
    real_yield_spike_signal,
    vix_spike_signal,
    yield_curve_stress_signal,
)


def _date_idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")


def test_vix_spike_signal_in_unit_range():
    idx = _date_idx(400)
    rng = np.random.default_rng(0)
    base = pd.Series(np.full(400, 15.0) + rng.normal(0, 2, 400), index=idx)
    base.iloc[200:220] = 40.0  # spike
    s = vix_spike_signal(base)
    assert s.min() >= 0
    assert s.max() <= 1.0
    assert s.iloc[210] > 0.5


def test_yield_curve_stress_signal_inverted():
    idx = _date_idx(300)
    yc = pd.Series(np.full(300, 0.5), index=idx)
    yc.iloc[100:150] = -0.3  # inversion
    s = yield_curve_stress_signal(yc)
    assert s.iloc[100:150].mean() == 1.0  # fully inverted block


def test_yield_curve_stress_normal_low():
    idx = _date_idx(300)
    rng = np.random.default_rng(1)
    yc = pd.Series(np.full(300, 1.0) + rng.normal(0, 0.05, 300), index=idx)
    s = yield_curve_stress_signal(yc)
    assert s.mean() < 0.3  # normal regime


def test_hy_spread_widening_signal():
    idx = _date_idx(300)
    hy = pd.Series(np.full(300, 3.0), index=idx)
    hy.iloc[200:220] = 5.5  # 1.8x baseline
    s = hy_spread_widening_signal(hy, baseline_window=60, alarm_ratio=1.3)
    assert (s.iloc[205:215] > 0.5).all()


def test_hy_spread_all_nan_returns_nan_series():
    idx = _date_idx(100)
    hy = pd.Series(np.nan, index=idx)
    s = hy_spread_widening_signal(hy)
    assert s.isna().all()


def test_real_yield_spike_signal_aligned():
    idx = _date_idx(300)
    rng = np.random.default_rng(2)
    nom = pd.Series(np.full(300, 4.0) + rng.normal(0, 0.1, 300), index=idx)
    be = pd.Series(np.full(300, 2.5) + rng.normal(0, 0.1, 300), index=idx)
    # Real yield = ca. 1.5; spike auf 3.5
    nom.iloc[200:220] = 6.0
    s = real_yield_spike_signal(nom, be)
    assert s.max() > 0.5
    assert len(s) == 300


def test_macro_stress_composite_with_full_panel():
    idx = _date_idx(400)
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(
        {
            "vix_close": np.full(400, 16.0) + rng.normal(0, 2, 400),
            "yield_curve_spread": np.full(400, 0.5),
            "hy_spread": np.full(400, 3.0),
            "treasury_10y": np.full(400, 4.0),
            "T10YIE": np.full(400, 2.5),
        },
        index=idx,
    )
    # Konstruiere Stress: VIX spike + YC inversion zwischen 200..240
    panel.iloc[200:240, panel.columns.get_loc("vix_close")] = 40.0
    panel.iloc[200:240, panel.columns.get_loc("yield_curve_spread")] = -0.5

    out = macro_stress_composite(panel)
    assert "composite_score" in out.columns
    assert "regime" in out.columns
    assert (out["regime"].iloc[210:235] == "stress").sum() > 10


def test_macro_stress_composite_partial_data():
    # Nur VIX vorhanden — Composite kommt trotzdem zurück
    idx = _date_idx(300)
    panel = pd.DataFrame({"vix_close": np.full(300, 16.0)}, index=idx)
    out = macro_stress_composite(panel)
    assert out["yield_curve_stress"].isna().all()
    # Composite sollte überwiegend 0 oder calm sein
    assert (out["regime"] == "calm").mean() > 0.8


def test_config_threshold_changes_regime():
    idx = _date_idx(200)
    panel = pd.DataFrame(
        {"vix_close": np.full(200, 16.0), "yield_curve_spread": np.full(200, -0.2)},
        index=idx,
    )
    # Default threshold 0.55; yc_inversion -> signal=1.0 mit gewicht 0.30 + vix=0 mit 0.35
    # = 0.30 / 0.65 = 0.46 (only vix+yc valid). Mit niedrigerem Threshold sollte stress
    out_low = macro_stress_composite(panel, MacroStressConfig(stress_threshold=0.30))
    out_high = macro_stress_composite(panel, MacroStressConfig(stress_threshold=0.80))
    assert (out_low["regime"] == "stress").sum() > (
        out_high["regime"] == "stress"
    ).sum()
