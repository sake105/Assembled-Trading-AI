"""Tests for events, nlp (offline parts), bayesian, crossasset, survival."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.bayesian import bayesian_linear
from erweiterung.crossasset import spreads
from erweiterung.events import event_study
from erweiterung.nlp import news_dedup
from erweiterung.survival import hazard_models


def test_market_model_alpha_beta():
    rng = np.random.default_rng(0)
    n = 200
    market = pd.Series(rng.normal(0, 0.01, n))
    asset = 1.2 * market + 0.0002 + pd.Series(rng.normal(0, 0.005, n))
    a, b, sig = event_study.market_model_alpha_beta(asset, market)
    assert abs(b - 1.2) < 0.3
    assert sig > 0


def test_run_event_study():
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.date_range("2023-01-01", periods=n)
    market = pd.Series(rng.normal(0, 0.01, n), index=dates)
    panel_rows = []
    for sym in ("A", "B", "C"):
        for d, mret in zip(dates, market):
            r = 1.0 * mret + rng.normal(0, 0.01)
            panel_rows.append({"date": d, "symbol": sym, "return": r})
    panel = pd.DataFrame(panel_rows)
    events = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "event_date": [dates[300], dates[310], dates[320]],
        }
    )
    res = event_study.run_event_study(events, panel, market, (-50, -5), (-3, 5))
    assert res.n_events >= 1
    assert "aar" in dir(res)


def test_buy_and_hold_abnormal_return():
    a = pd.Series([0.01, 0.02, -0.01])
    m = pd.Series([0.005, 0.005, 0.005])
    bhar = event_study.buy_and_hold_abnormal_return(a, m)
    assert np.isfinite(bhar)


def test_simhash_consistency():
    h1 = news_dedup.simhash("Apple beats Q3 earnings expectations")
    h2 = news_dedup.simhash("Apple beats Q3 earnings expectations")
    assert h1 == h2
    h3 = news_dedup.simhash("Microsoft posts record cloud revenue")
    assert news_dedup.hamming_distance(h1, h3) > 5


def test_jaccard_similarity():
    s = news_dedup.jaccard_similarity("apple beats earnings", "apple beats earnings")
    assert s == 1.0
    s = news_dedup.jaccard_similarity("apple", "google")
    assert s == 0.0


def test_news_dedup():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 4,
            "headline": [
                "Apple beats Q3 earnings expectations strongly",
                "Apple beats Q3 earnings expectations strongly today",
                "Microsoft posts record cloud revenue",
                "Tesla misses delivery target",
            ],
        }
    )
    out = news_dedup.deduplicate(df, jaccard_threshold=0.6)
    assert len(out) <= len(df)
    assert len(out) >= 3


def test_bayesian_linear_basic():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0, 1, (n, 2))
    y = X @ np.array([0.5, -0.3]) + rng.normal(0, 0.1, n)
    fit = bayesian_linear.fit_bayesian_linear(X, y)
    assert abs(fit.mu_n[0] - 0.5) < 0.2
    assert abs(fit.mu_n[1] + 0.3) < 0.2


def test_predictive_distribution():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(0, 1, (n, 3))
    y = X.sum(axis=1) + rng.normal(0, 0.5, n)
    fit = bayesian_linear.fit_bayesian_linear(X, y)
    mean, var = bayesian_linear.predictive_distribution(fit, np.array([1, 0, 0]))
    assert np.isfinite(mean) and var > 0


def test_sharpe_posterior_samples():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    samples = bayesian_linear.sharpe_posterior_samples(r, n_samples=2000)
    assert samples.shape == (2000,)
    lo, hi = bayesian_linear.credible_interval(samples)
    assert lo < hi


def test_cross_asset_spreads():
    n = 100
    gld = pd.Series(np.linspace(170, 190, n))
    slv = pd.Series(np.linspace(20, 25, n))
    gsr = spreads.gold_silver_ratio(gld, slv)
    assert gsr.iloc[0] > 0
    assert gsr.iloc[-1] > 0


def test_kaplan_meier():
    durations = np.array([5, 10, 15, 20, 25, 30])
    events = np.array([1, 0, 1, 1, 0, 1])
    out = hazard_models.kaplan_meier_estimate(durations, events)
    assert "survival" in out.columns
    assert (out["survival"].diff().fillna(0) <= 1e-9).all()  # monotonically decreasing


def test_cox_ph_basic():
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    durations = pd.Series(rng.exponential(10 * np.exp(-0.5 * X["x1"]), n))
    events = pd.Series(rng.binomial(1, 0.7, n))
    beta = hazard_models.fit_cox_ph_simple(X, durations, events, n_iter=20)
    assert "x1" in beta.index
    assert "x2" in beta.index
