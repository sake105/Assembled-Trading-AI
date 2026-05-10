"""Tests for erweiterung.factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.factors import factor_ic, factor_neutralize, fama_french, low_vol


def test_construct_long_short_factor():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=20)
    rows = []
    for d in dates:
        for i in range(20):
            rows.append(
                {
                    "date": d,
                    "symbol": f"S{i}",
                    "size": rng.uniform(1e6, 1e10),
                    "return_t1": rng.normal(0, 0.02),
                }
            )
    panel = pd.DataFrame(rows)
    smb = fama_french.construct_long_short_factor(
        panel, "size", "return_t1", long_high=False
    )
    assert len(smb) > 0


def test_momentum_12_1():
    n = 300
    rng = np.random.default_rng(0)
    rows = []
    for sym in ("A", "B"):
        prices = 100 * np.exp(rng.normal(0.0005, 0.01, n).cumsum())
        for d, p in zip(pd.date_range("2023-01-01", periods=n), prices):
            rows.append({"date": d, "symbol": sym, "close": p})
    panel = pd.DataFrame(rows)
    mom = fama_french.momentum_12_1(panel)
    assert mom.dropna().shape[0] > 0


def test_rolling_beta():
    rng = np.random.default_rng(0)
    n = 500
    market = pd.Series(rng.normal(0, 0.01, n))
    asset = 1.5 * market + pd.Series(rng.normal(0, 0.005, n))
    beta = low_vol.rolling_beta(asset, market, window=100)
    valid = beta.dropna()
    assert abs(valid.mean() - 1.5) < 0.5  # rough


def test_low_vol_signal(synthetic_panel):
    out = low_vol.low_vol_signal(synthetic_panel, window=60)
    assert out.dropna().shape[0] > 0


def test_pearson_ic():
    s = pd.Series(np.linspace(-1, 1, 100))
    r = pd.Series(np.linspace(-0.5, 0.5, 100)) + np.random.default_rng(0).normal(
        0, 0.05, 100
    )
    ic = factor_ic.pearson_ic(s, r)
    assert ic > 0.5


def test_cross_sectional_ic():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2024-01-01", periods=30):
        for i in range(20):
            sig = rng.normal()
            ret = 0.3 * sig + rng.normal(0, 0.5)  # signal predictive
            rows.append({"date": d, "symbol": f"S{i}", "signal": sig, "return_t1": ret})
    panel = pd.DataFrame(rows)
    ic_ts = factor_ic.cross_sectional_ic(panel, "signal", "return_t1")
    assert ic_ts.mean() > 0.1


def test_ic_summary():
    s = pd.Series([0.05, 0.02, -0.01, 0.03, 0.04])
    out = factor_ic.ic_summary(s)
    assert "ic_mean" in out and "ic_ir" in out


def test_sector_demean():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 6,
            "sector": ["A", "A", "A", "B", "B", "B"],
            "value": [1, 2, 3, 10, 12, 14],
        }
    )
    out = factor_neutralize.sector_demean(df, "value", "sector", "date")
    sector_a = out[df["sector"] == "A"]
    assert abs(sector_a.sum()) < 1e-9


def test_industry_rank_normalize():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 8,
            "sector": ["A"] * 4 + ["B"] * 4,
            "value": list(range(8)),
        }
    )
    out = factor_neutralize.industry_rank_normalize(df, "value", "sector", "date")
    assert out.dropna().shape[0] == 8


def test_regress_neutralize():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"] * n,
            "symbol": [f"S{i}" for i in range(n)],
            "y": rng.normal(0, 1, n),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        }
    )
    out = factor_neutralize.regress_neutralize(df, "y", ["x1", "x2"])
    assert out.notna().sum() > 0
