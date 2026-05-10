"""Tests for bayesian, crossasset, survival.

events/event_study and nlp/news_dedup wurden in der Cleanup-Phase gelöscht
(siehe DUPLICATE_AUDIT.md). Mainline hat:
- src/assembled_core/qa/event_study.py (446 LoC)
- src/assembled_core/intel/news_dedupe.py (391 LoC)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.bayesian import bayesian_linear
from erweiterung.crossasset import spreads
from erweiterung.survival import hazard_models


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
