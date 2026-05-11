"""Tests für intermarket_macro_factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.intermarket_macro_factors import (
    bond_equity_ratio,
    build_intermarket_panel,
    credit_spread_proxy,
    dollar_trend,
    gold_equity_divergence,
    macro_stress_composite_score,
)


def _ts(n: int = 500) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")


def test_bond_equity_ratio_basic():
    idx = _ts()
    tlt = pd.Series(
        100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 500)), index=idx
    )
    spy = pd.Series(
        100 + np.cumsum(np.random.default_rng(1).normal(0.2, 1, 500)), index=idx
    )
    r = bond_equity_ratio(tlt, spy, window=20)
    assert isinstance(r, pd.Series)
    assert r.notna().any()


def test_dollar_trend_returns_series():
    idx = _ts(300)
    s = pd.Series(
        100 + np.cumsum(np.random.default_rng(2).normal(0, 0.5, 300)), index=idx
    )
    out = dollar_trend(s, window=20)
    assert isinstance(out, pd.Series)


def test_credit_spread_proxy():
    idx = _ts(400)
    hyg = pd.Series(
        100 + np.cumsum(np.random.default_rng(3).normal(0, 0.5, 400)), index=idx
    )
    agg = pd.Series(
        100 + np.cumsum(np.random.default_rng(4).normal(0, 0.3, 400)), index=idx
    )
    r = credit_spread_proxy(hyg, agg)
    assert isinstance(r, pd.Series)


def test_gold_equity_divergence():
    idx = _ts(400)
    gld = pd.Series(
        100 + np.cumsum(np.random.default_rng(5).normal(0, 0.5, 400)), index=idx
    )
    spy = pd.Series(
        100 + np.cumsum(np.random.default_rng(6).normal(0.1, 0.8, 400)), index=idx
    )
    r = gold_equity_divergence(gld, spy, window=20)
    assert isinstance(r, pd.Series)


def test_build_intermarket_panel():
    idx = _ts(400)
    rng = np.random.default_rng(7)
    panel = pd.DataFrame(
        {
            "SPY": 100 + np.cumsum(rng.normal(0.1, 0.8, 400)),
            "TLT": 100 + np.cumsum(rng.normal(0, 0.5, 400)),
            "GLD": 100 + np.cumsum(rng.normal(0, 0.5, 400)),
            "HYG": 100 + np.cumsum(rng.normal(0, 0.4, 400)),
            "AGG": 100 + np.cumsum(rng.normal(0, 0.3, 400)),
            "DBC": 100 + np.cumsum(rng.normal(0, 0.6, 400)),
        },
        index=idx,
    )
    out = build_intermarket_panel(panel)
    assert "bond_equity_ratio_20d" in out.columns
    assert "credit_spread_proxy" in out.columns
    assert "gold_equity_divergence_20d" in out.columns
    assert out.notna().any().any()


def test_macro_stress_composite_in_unit_range():
    idx = _ts(500)
    rng = np.random.default_rng(8)
    panel = pd.DataFrame(
        {
            "SPY": 100 + np.cumsum(rng.normal(0.1, 0.8, 500)),
            "TLT": 100 + np.cumsum(rng.normal(0, 0.5, 500)),
            "GLD": 100 + np.cumsum(rng.normal(0, 0.5, 500)),
            "HYG": 100 + np.cumsum(rng.normal(0, 0.4, 500)),
            "AGG": 100 + np.cumsum(rng.normal(0, 0.3, 500)),
        },
        index=idx,
    )
    panel_out = build_intermarket_panel(panel)
    score = macro_stress_composite_score(panel_out)
    valid = score.dropna()
    if not valid.empty:
        assert valid.min() >= 0
        assert valid.max() <= 1.0


def test_empty_panel():
    out = build_intermarket_panel(pd.DataFrame())
    assert out.empty
