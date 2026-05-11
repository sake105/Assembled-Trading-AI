"""Integrationstests für LiveDecisionEngine Geo+News-Overlays."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erweiterung.live.live_decision_engine import (
    LiveDecisionEngine,
    LiveEngineConfig,
)


def _make_returns(n_days: int = 500, n_eq: int = 8, n_xa: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B", tz="UTC")
    eq = pd.DataFrame(
        rng.normal(0.0005, 0.012, (n_days, n_eq)),
        index=idx,
        columns=[f"E{i}" for i in range(n_eq)],
    )
    xa = pd.DataFrame(
        rng.normal(0.0003, 0.008, (n_days, n_xa)),
        index=idx,
        columns=[f"X{i}" for i in range(n_xa)],
    )
    return eq, xa


def _stress_overlay(daily_idx: pd.DatetimeIndex, pause_start: str, pause_end: str):
    """Build daily geo-overlay with PAUSE multiplier on a window."""
    mult = pd.Series(1.0, index=daily_idx)
    s = pd.Timestamp(pause_start, tz="UTC")
    e = pd.Timestamp(pause_end, tz="UTC")
    mask = (mult.index >= s) & (mult.index <= e)
    mult[mask] = 0.50
    return pd.DataFrame({"multiplier": mult, "state": "PAUSE"})


def test_geo_overlay_disabled_by_default():
    eq, xa = _make_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    out = engine.decide_next()
    assert out["geo_multiplier"] == 1.0


def test_attach_geo_overlay_without_enable_is_inert():
    """Attaching overlay but config.enable_geo_overlay=False → no effect."""
    eq, xa = _make_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    overlay = _stress_overlay(eq.index, "2020-06-01", "2021-12-31")
    engine.attach_geo_overlay(overlay)
    next_date = eq.index[-1] + pd.Timedelta(days=1)
    engine.update_with_new_day(
        next_date,
        pd.Series(0.001, index=eq.columns),
        pd.Series(0.0005, index=xa.columns),
    )
    out = engine.decide_next()
    assert out["geo_multiplier"] == 1.0  # disabled → multiplier ignored


def test_geo_overlay_reduces_leverage_during_pause():
    eq, xa = _make_returns()
    cfg = LiveEngineConfig(enable_geo_overlay=True)
    engine = LiveDecisionEngine(cfg)
    engine.bootstrap_from_history(eq, xa)
    # Build overlay covering future date
    future_idx = pd.date_range(
        eq.index[-1] + pd.Timedelta(days=1), periods=10, freq="B", tz="UTC"
    )
    overlay = _stress_overlay(future_idx, str(future_idx[0]), str(future_idx[-1]))
    engine.attach_geo_overlay(overlay)
    engine.update_with_new_day(
        future_idx[0],
        pd.Series(0.001, index=eq.columns),
        pd.Series(0.0005, index=xa.columns),
    )
    out = engine.decide_next()
    assert out["geo_multiplier"] == 0.50

    # Compare to baseline (no overlay)
    engine2 = LiveDecisionEngine(LiveEngineConfig(enable_geo_overlay=False))
    engine2.bootstrap_from_history(eq, xa)
    engine2.update_with_new_day(
        future_idx[0],
        pd.Series(0.001, index=eq.columns),
        pd.Series(0.0005, index=xa.columns),
    )
    out_base = engine2.decide_next()
    # Same input → same vol-target lev. With overlay: half the leverage.
    assert out["sa_leverage"] == pytest.approx(out_base["sa_leverage"] * 0.50, abs=1e-9)
    assert out["xa_ew_leverage"] == pytest.approx(
        out_base["xa_ew_leverage"] * 0.50, abs=1e-9
    )


def test_geo_multiplier_clamped_to_config_range():
    eq, xa = _make_returns()
    cfg = LiveEngineConfig(
        enable_geo_overlay=True, geo_min_multiplier=0.40, geo_max_multiplier=1.05
    )
    engine = LiveDecisionEngine(cfg)
    engine.bootstrap_from_history(eq, xa)
    future_idx = pd.date_range(
        eq.index[-1] + pd.Timedelta(days=1), periods=5, freq="B", tz="UTC"
    )
    overlay = pd.DataFrame(
        {"multiplier": [0.10, 0.50, 1.00, 1.50, 1.05]},
        index=future_idx,
    )
    engine.attach_geo_overlay(overlay)
    multipliers = []
    for d in future_idx:
        engine.update_with_new_day(
            d, pd.Series(0.001, index=eq.columns), pd.Series(0.0005, index=xa.columns)
        )
        multipliers.append(engine.state.current_geo_multiplier)
    # 0.10 → clamped to 0.40; 1.50 → clamped to 1.05
    assert multipliers[0] == pytest.approx(0.40)
    assert multipliers[3] == pytest.approx(1.05)


def test_news_tilt_disabled_by_default():
    eq, xa = _make_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    out = engine.decide_next()
    # Without news tilt, top-N picks come from mom only
    assert isinstance(out["eq_top_weights"], pd.Series)


def test_news_tilt_changes_top_picks():
    """Strong positive news on a low-mom symbol → tilt makes it eligible."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=500, freq="B", tz="UTC")
    # Build returns where E0..E2 have strong mom, E5..E7 weak
    eq = pd.DataFrame(0.0, index=idx, columns=[f"E{i}" for i in range(8)])
    for i in range(8):
        drift = 0.002 if i < 3 else 0.0  # strong drift first 3
        eq[f"E{i}"] = rng.normal(drift, 0.012, len(idx))
    xa = pd.DataFrame(
        rng.normal(0.0003, 0.008, (500, 4)), index=idx, columns=list("ABCD")
    )

    # Baseline: no tilt
    eng_base = LiveDecisionEngine(LiveEngineConfig(eq_quantile_long=0.25))
    eng_base.bootstrap_from_history(eq, xa)
    base_picks = eng_base.decide_next()["eq_top_weights"]
    base_chosen = set(base_picks[base_picks > 0].index)

    # With tilt: massive news boost to E7 (otherwise weakest)
    eng_tilt = LiveDecisionEngine(
        LiveEngineConfig(
            eq_quantile_long=0.25, enable_news_tilt=True, news_tilt_strength=5.0
        )
    )
    eng_tilt.bootstrap_from_history(eq, xa)
    tilt_scores = pd.Series({f"E{i}": -1.0 for i in range(8)} | {"E7": 10.0})
    eng_tilt.attach_news_tilt_scores(tilt_scores)
    tilt_picks = eng_tilt.decide_next()["eq_top_weights"]
    tilt_chosen = set(tilt_picks[tilt_picks > 0].index)

    # E7 not in base picks (low mom), but should be in tilt picks (massive news boost)
    assert "E7" not in base_chosen
    assert "E7" in tilt_chosen


def test_news_tilt_zero_strength_equivalent_to_disabled():
    eq, xa = _make_returns()
    eng_disabled = LiveDecisionEngine(LiveEngineConfig(enable_news_tilt=False))
    eng_disabled.bootstrap_from_history(eq, xa)
    base_picks = eng_disabled.decide_next()["eq_top_weights"]

    eng_zero = LiveDecisionEngine(
        LiveEngineConfig(enable_news_tilt=True, news_tilt_strength=0.0)
    )
    eng_zero.bootstrap_from_history(eq, xa)
    eng_zero.attach_news_tilt_scores(pd.Series({"E0": 5.0, "E1": -3.0}))
    zero_picks = eng_zero.decide_next()["eq_top_weights"]
    pd.testing.assert_series_equal(base_picks, zero_picks)


def test_geo_multiplier_persisted_in_state_summary():
    """state_summary must continue to work after overlay attached."""
    eq, xa = _make_returns()
    cfg = LiveEngineConfig(enable_geo_overlay=True)
    engine = LiveDecisionEngine(cfg)
    engine.bootstrap_from_history(eq, xa)
    summary = engine.state_summary()
    assert "last_date" in summary


def test_overlay_missing_column_raises():
    eq, xa = _make_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    bad = pd.DataFrame({"foo": [1.0]}, index=[pd.Timestamp("2020-01-01", tz="UTC")])
    with pytest.raises(ValueError, match="multiplier"):
        engine.attach_geo_overlay(bad)


def test_geo_overlay_ffill_for_gap_date():
    """Date not in overlay → should ffill from most recent prior entry."""
    eq, xa = _make_returns()
    cfg = LiveEngineConfig(enable_geo_overlay=True)
    engine = LiveDecisionEngine(cfg)
    engine.bootstrap_from_history(eq, xa)

    # Overlay has entries only at month-starts
    monthly_idx = pd.date_range(
        eq.index[-1] - pd.Timedelta(days=90), eq.index[-1], freq="MS", tz="UTC"
    )
    overlay = pd.DataFrame({"multiplier": 0.75}, index=monthly_idx)
    engine.attach_geo_overlay(overlay)

    next_date = eq.index[-1] + pd.Timedelta(days=1)  # not in monthly overlay
    engine.update_with_new_day(
        next_date,
        pd.Series(0.001, index=eq.columns),
        pd.Series(0.0005, index=xa.columns),
    )
    # Should ffill from last monthly entry → 0.75
    assert engine.state.current_geo_multiplier == pytest.approx(0.75)
