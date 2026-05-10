"""Tests for erweiterung.volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.volatility import garch_models, har_rv


def test_garch_fit_and_forecast():
    rng = np.random.default_rng(42)
    n = 1000
    # Synthetic GARCH(1,1) data
    omega, alpha, beta = 0.01, 0.1, 0.85
    var = np.zeros(n)
    var[0] = omega / (1 - alpha - beta)
    eps = np.zeros(n)
    for t in range(1, n):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = rng.normal(0, np.sqrt(var[t]))
    returns = pd.Series(eps / 100)  # decimal

    fit = garch_models.fit_garch(returns, model="GARCH")
    assert fit.persistence > 0.5
    assert len(fit.conditional_vol) == len(returns.dropna())
    fcst = garch_models.garch_forecast(fit, horizon=5)
    assert len(fcst) == 5
    assert (fcst > 0).all()


def test_har_rv_fit():
    rng = np.random.default_rng(0)
    n = 200
    rv = pd.Series(np.exp(rng.normal(-3, 0.4, n)))
    fit = har_rv.fit_har_rv(rv)
    assert -1 <= fit.beta_d <= 2
    assert fit.in_sample_r2 >= 0
    fc = har_rv.har_forecast(fit, rv)
    assert np.isfinite(fc)
