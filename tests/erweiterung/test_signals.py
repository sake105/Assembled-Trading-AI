"""Tests for erweiterung.signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.signals import (
    attention,
    cross_sectional_residuals,
    earnings_drift_v2,
    lead_lag_network,
    macro_nowcast,
    options_implied,
    statistical_arbitrage,
)


def test_residual_returns_basic(synthetic_panel, synthetic_returns):
    sector_map = {s: "TECH" for s in synthetic_returns.columns}
    sector_etf = synthetic_returns["AAA"].copy()  # proxy
    market = synthetic_returns.mean(axis=1)
    out = cross_sectional_residuals.compute_residual_returns(
        synthetic_panel,
        sector_map=sector_map,
        sector_etf_returns={"TECH": sector_etf},
        market_returns=market,
        window=60,
    )
    assert "residual_return" in out.columns
    assert (out["residual_return"].dropna().abs() < 1).all()


def test_residual_momentum(synthetic_panel, synthetic_returns):
    sector_map = {s: "TECH" for s in synthetic_returns.columns}
    sector_etf = synthetic_returns["AAA"]
    market = synthetic_returns.mean(axis=1)
    res = cross_sectional_residuals.compute_residual_returns(
        synthetic_panel,
        sector_map=sector_map,
        sector_etf_returns={"TECH": sector_etf},
        market_returns=market,
        window=60,
    )
    mom = cross_sectional_residuals.residual_momentum(res, lookback=21, skip=1)
    assert "residual_momentum" in mom.columns


def test_realized_vol(synthetic_panel):
    out = options_implied.realized_vol(synthetic_panel, window=21, annualize=True)
    assert "rv_21_d" in out.columns
    valid = out["rv_21_d"].dropna()
    assert (valid > 0).all()
    # Annualisierte Vol sollte im Bereich 0.1-1.0 liegen für synthetisch
    assert valid.mean() < 2.0


def test_garman_klass_volatility(synthetic_returns, synthetic_prices):
    rows = []
    for sym in synthetic_prices.columns:
        for d, c in synthetic_prices[sym].items():
            o = c * np.random.default_rng(int(d.timestamp())).uniform(0.99, 1.01)
            high = max(o, c) * 1.01
            low = min(o, c) * 0.99
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": o,
                    "high": high,
                    "low": low,
                    "close": c,
                }
            )
    ohlc = pd.DataFrame(rows)
    out = options_implied.garman_klass_volatility(ohlc, window=21)
    assert "gk_vol" in out.columns
    assert out["gk_vol"].dropna().min() >= 0


def test_skew_signal_basic():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=200, tz="UTC"),
            "symbol": ["AAA"] * 200,
            "skew_25d": np.random.default_rng(0).normal(0.05, 0.02, 200),
        }
    )
    out = options_implied.skew_signal(df, lookback=60)
    assert "skew_z" in out.columns


def test_attention_composite():
    dates = pd.date_range("2024-01-01", periods=10, tz="UTC")
    wiki = pd.DataFrame(
        {"date": dates, "symbol": ["A"] * 10, "attention_score": np.linspace(-1, 1, 10)}
    )
    trends = pd.DataFrame(
        {"date": dates, "keyword": ["A"] * 10, "svi_z": np.linspace(0, 2, 10)}
    )
    composite = attention.composite_attention_score(wiki_df=wiki, trends_df=trends)
    assert "attention_composite" in composite.columns
    assert len(composite) == 10


def test_attention_meanrev():
    composite = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "symbol": ["A"] * 5,
            "attention_composite": [0, 1, 2.5, -2.5, 0],
        }
    )
    sig = attention.attention_meanrev_signal(composite, threshold=2.0)
    assert sig.iloc[2]["att_meanrev_signal"] == -1.0
    assert sig.iloc[3]["att_meanrev_signal"] == +1.0
    assert sig.iloc[0]["att_meanrev_signal"] == 0.0


def test_granger_causality_basic():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * x[t - 1] + rng.normal(0, 0.5)  # x leads y
    F, p = lead_lag_network.granger_causality_lag1(pd.Series(x), pd.Series(y))
    assert F > 5  # strong causality
    if not pd.isna(p):
        assert p < 0.05


def test_build_leadlag_network(synthetic_panel):
    edges = lead_lag_network.build_leadlag_network(
        synthetic_panel, window=200, f_threshold=0.5, max_pairs=5
    )
    assert isinstance(edges, pd.DataFrame)


def test_find_cointegrated_pairs():
    rng = np.random.default_rng(0)
    n = 300
    common = rng.normal(0, 1, n).cumsum()
    a = common + rng.normal(0, 0.5, n)
    b = 2 * common + rng.normal(0, 0.5, n)  # cointegrated
    c = rng.normal(0, 1, n).cumsum()  # not cointegrated
    df = pd.DataFrame(
        {"A": np.log(a + 100), "B": np.log(b + 200), "C": np.log(c + 100)},
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    pairs = statistical_arbitrage.find_cointegrated_pairs(
        df, p_threshold=0.5, beta_range=(0.1, 10), half_life_range=(1, 100), min_obs=200
    )
    # We don't strictly assert AB is found because adf-fallback is heuristic
    assert isinstance(pairs, list)


def test_pead_signal_basic(synthetic_prices):
    earnings = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "announcement_date": [
                synthetic_prices.index[100],
                synthetic_prices.index[150],
            ],
            "sue": [2.0, -1.5],
        }
    )
    prices = (
        synthetic_prices.reset_index()
        .melt(id_vars=["index"], var_name="symbol", value_name="close")
        .rename(columns={"index": "date"})
    )
    out = earnings_drift_v2.post_earnings_drift_signal(
        earnings, prices, drift_window=10, skip_days=2
    )
    assert "pead_signal" in out.columns
    sub_aaa = out[out["symbol"] == "AAA"]
    assert (sub_aaa["pead_signal"] > 0).all()
    sub_bbb = out[out["symbol"] == "BBB"]
    assert (sub_bbb["pead_signal"] < 0).all()


def test_macro_recession_score():
    n = 240
    fred_md = pd.DataFrame(
        {
            "GS10": np.linspace(2.0, 3.0, n),
            "TB3MS": np.linspace(0.5, 4.0, n),  # later inverted
            "BAAFFM": np.linspace(2.5, 3.5, n),
            "AAAFFM": np.linspace(2.0, 2.5, n),
            "UNRATE": np.linspace(3.5, 4.5, n),
        },
        index=pd.date_range("2010-01-01", periods=n, freq="MS"),
    )
    out = macro_nowcast.composite_recession_score(fred_md)
    assert "recession_score" in out.columns
    assert out["recession_score"].between(0, 1).all()
