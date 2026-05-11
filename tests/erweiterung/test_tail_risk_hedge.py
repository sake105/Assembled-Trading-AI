"""Tests für tail_risk_hedge."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.tail_risk_hedge import (
    TailHedgeConfig,
    apply_tail_hedge,
    vix_stress_trigger,
)


def _vix_series(
    n: int = 400, spike_idx: int | None = 200, spike_value: float = 60.0
) -> pd.Series:
    rng = np.random.default_rng(0)
    base = np.full(n, 18.0) + rng.normal(0, 2, n)
    if spike_idx is not None:
        base[spike_idx : spike_idx + 30] = spike_value
    return pd.Series(base, index=pd.date_range("2022-01-01", periods=n, freq="B"))


def test_vix_stress_trigger_normal_constant():
    """Bei flacher VIX (keine Variation, Trend, oder Outlier) sollte trigger normal bleiben."""
    vix = pd.Series(
        np.full(400, 18.0),
        index=pd.date_range("2022-01-01", periods=400, freq="B"),
    )
    trig = vix_stress_trigger(vix, TailHedgeConfig(use_zscore=True))
    assert (trig == "normal").all()


def test_vix_stress_trigger_spike_triggers():
    vix = _vix_series(spike_idx=200, spike_value=80)
    trig = vix_stress_trigger(vix, TailHedgeConfig(use_zscore=True))
    # Während spike sollte stress sein
    assert (trig.iloc[210:225] == "stress").any()


def test_vix_absolute_trigger():
    vix = _vix_series(spike_idx=200, spike_value=45)
    trig = vix_stress_trigger(
        vix, TailHedgeConfig(use_zscore=False, vix_absolute_threshold=30)
    )
    assert (trig.iloc[210:225] == "stress").any()


def test_apply_tail_hedge_reduces_exposure_during_stress():
    vix = _vix_series(spike_idx=200, spike_value=70)
    rng = np.random.default_rng(7)
    n = len(vix)
    port = pd.Series(rng.normal(0.0005, 0.012, n), index=vix.index)
    out = apply_tail_hedge(port, vix)
    assert "trigger" in out.columns
    assert "hedged_return" in out.columns
    # In stress: exposure < 1.0
    stress_mask = out["trigger"].shift(1) == "stress"
    if stress_mask.any():
        assert (out.loc[stress_mask, "exposure"] < 1.0).all()


def test_apply_tail_hedge_empty_input():
    out = apply_tail_hedge(pd.Series(dtype=float), pd.Series(dtype=float))
    assert out.empty


def test_tail_hedge_state_persistence():
    """Trigger bleibt 'stress' bis re_engage greift."""
    vix = pd.Series(
        np.concatenate(
            [
                np.full(100, 18.0),
                np.full(50, 50.0),  # spike
                np.full(50, 25.0),  # noch über re_engage (22.0 absolute)
                np.full(100, 18.0),
            ]
        ),  # zurück
        index=pd.date_range("2022-01-01", periods=300, freq="B"),
    )
    trig = vix_stress_trigger(
        vix,
        TailHedgeConfig(
            use_zscore=False,
            vix_absolute_threshold=30.0,
            re_engage_absolute=22.0,
            smoothing_days=1,
        ),
    )
    # Während VIX=50 → stress
    assert (trig.iloc[110:140] == "stress").all()
    # Während VIX=25 → noch stress (über 22.0)
    assert (trig.iloc[160:190] == "stress").all()
    # Zurück bei VIX=18 → normal nach re_engage
    assert (trig.iloc[230:280] == "normal").all()
