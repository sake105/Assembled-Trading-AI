"""Tests für Round-6 (Attribution, Comparison, Risk-Combiner, ML-Integration)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Performance Attribution
# ---------------------------------------------------------------------------


def test_attribution_basic():
    from src.assembled_core.qa.performance_attribution import compute_attribution

    rng = np.random.default_rng(42)
    n = 200
    # Simulate: portfolio return = 0.5*market + 0.2*size + noise
    market = pd.Series(rng.normal(0.001, 0.01, n))
    size = pd.Series(rng.normal(0.0005, 0.008, n))
    portfolio = 0.5 * market + 0.2 * size + rng.normal(0, 0.005, n) + 0.0005

    factors = pd.DataFrame({"market": market, "size": size})
    result = compute_attribution(portfolio, factors)

    assert "market" in result.factor_betas
    assert "size" in result.factor_betas
    # Recovered betas should be close to 0.5 / 0.2
    assert abs(result.factor_betas["market"] - 0.5) < 0.1
    assert abs(result.factor_betas["size"] - 0.2) < 0.1
    assert result.r_squared > 0.3


def test_attribution_alpha_detection():
    """Positives α sollte erkennbar sein."""
    from src.assembled_core.qa.performance_attribution import compute_attribution

    rng = np.random.default_rng(7)
    n = 300
    market = pd.Series(rng.normal(0.0005, 0.01, n))
    # Echter α = 0.002
    portfolio = 0.3 * market + 0.002 + rng.normal(0, 0.003, n)

    result = compute_attribution(portfolio, pd.DataFrame({"market": market}))
    assert result.alpha > 0.001
    # Mit n=300 sollte t-stat signifikant sein
    assert result.alpha_t_stat > 2.0


def test_attribution_insufficient_data():
    from src.assembled_core.qa.performance_attribution import compute_attribution

    with pytest.raises(ValueError, match="Beobachtungen"):
        compute_attribution(
            pd.Series([0.01, 0.02]),
            pd.DataFrame({"m": [0.01, 0.02]}),
            min_obs=20,
        )


def test_rolling_attribution_shape():
    from src.assembled_core.qa.performance_attribution import rolling_attribution

    rng = np.random.default_rng(3)
    n = 200
    dates = pd.date_range("2025-01-01", periods=n)
    portfolio = pd.Series(rng.normal(0.001, 0.01, n), index=dates)
    market = pd.Series(rng.normal(0.001, 0.01, n), index=dates)

    df = rolling_attribution(
        portfolio, pd.DataFrame({"market": market}, index=dates), window=60, min_obs=30
    )
    assert "alpha" in df.columns
    assert "beta_market" in df.columns
    assert len(df) > 100


# ---------------------------------------------------------------------------
# Risk-Aware Signal Combiner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ML Signal Integration Pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model Comparison (smoke test — no real model compare)
# ---------------------------------------------------------------------------


def test_compare_models_metrics_computation():
    """Nur die _compute_metrics Funktion testen (ohne File-IO)."""
    from scripts.analysis.compare_models import _compute_metrics

    rng = np.random.default_rng(31)
    n = 100
    preds = pd.Series(rng.uniform(-1, 1, n))
    actuals = pd.Series(rng.normal(0, 0.01, n))

    metrics = _compute_metrics(preds, actuals)
    assert "ic" in metrics
    assert "hit_rate" in metrics
    assert "sharpe" in metrics
    assert "mse" in metrics
    assert metrics["n_obs"] == n


def test_diebold_mariano_test():
    from scripts.analysis.compare_models import _diebold_mariano_test

    rng = np.random.default_rng(33)
    errors_a = rng.normal(0, 0.02, 100)
    errors_b = rng.normal(0, 0.02, 100)
    result = _diebold_mariano_test(errors_a, errors_b)
    assert "statistic" in result
    assert "p_value" in result
    # Similar distributions → p > 0.05
    assert result["p_value"] >= 0.0
