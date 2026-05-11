"""Tests für profit_lock_overlay + macro_regime_quadrant."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.macro_regime_quadrant import (
    classify_macro_quadrant,
    regime_returns_summary,
)
from erweiterung.strategies.profit_lock_overlay import (
    ProfitLockConfig,
    apply_profit_lock,
    compute_profit_lock_series,
)


def _ret(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.0005, 0.01, n),
        index=pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC"),
    )


def test_profit_lock_disabled_returns_ones():
    r = _ret()
    m = compute_profit_lock_series(r, ProfitLockConfig(enabled=False))
    assert (m == 1.0).all()


def test_profit_lock_triggers_on_strong_gain():
    # Konstruiere starken Gain: 30 days @ +1% = +35%
    n = 100
    rets = np.concatenate([np.full(30, 0.01), np.zeros(70)])
    r = pd.Series(
        rets, index=pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    )
    m = compute_profit_lock_series(
        r,
        ProfitLockConfig(
            lookback_days=20,
            trigger_return=0.08,
            multiplier_on_trigger=0.7,
            floor=0.5,
        ),
    )
    # Nach den 30 Gewinn-Tagen sollte multiplier < 1.0 sein
    assert (m.iloc[25:35] < 1.0).any()


def test_apply_profit_lock_returns_df():
    r = _ret()
    out = apply_profit_lock(r)
    assert "locked_return" in out.columns
    assert "multiplier" in out.columns


def test_macro_quadrant_classifies():
    n = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    g = pd.Series(rng.normal(0, 1, n), index=idx).rolling(20, min_periods=1).mean()
    i = pd.Series(rng.normal(0, 1, n), index=idx).rolling(20, min_periods=1).mean()
    out = classify_macro_quadrant(g, i)
    if not out.empty:
        assert set(out.unique()).issubset(
            {
                "growth_up_infl_up",
                "growth_up_infl_down",
                "growth_down_infl_up",
                "growth_down_infl_down",
            }
        )


def test_regime_returns_summary():
    n = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    labels = pd.Series(
        np.random.default_rng(2).choice(
            ["growth_up_infl_up", "growth_down_infl_down"], n
        ),
        index=idx,
    )
    summary = regime_returns_summary(rets, labels)
    assert not summary.empty
    assert "ann_return" in summary.columns


def test_macro_quadrant_empty_input():
    out = classify_macro_quadrant(pd.Series(dtype=float), pd.Series(dtype=float))
    assert out.empty
