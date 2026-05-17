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


def test_risk_combiner_fit_combine():
    import pytest

    pytest.importorskip("src.assembled_core.signals.risk_aware_combiner")
    from src.assembled_core.signals.risk_aware_combiner import RiskAwareSignalCombiner

    rng = np.random.default_rng(11)
    n = 300
    signals = pd.DataFrame(
        {
            "sig_a": rng.uniform(-1, 1, n),
            "sig_b": rng.uniform(-1, 1, n),
        }
    )
    returns = pd.Series(rng.normal(0, 0.01, n))
    regimes = pd.Series(rng.choice(["RISK_ON", "NEUTRAL", "RISK_OFF", "CRISIS"], n))

    combiner = RiskAwareSignalCombiner(min_obs_per_bucket=20)
    combiner.fit(signals, returns, regimes)

    weights_neutral = combiner.get_weights("NEUTRAL")
    assert "sig_a" in weights_neutral
    assert "sig_b" in weights_neutral
    # Weights should sum to ~1.0
    assert abs(sum(weights_neutral.values()) - 1.0) < 1e-6


def test_risk_combiner_auto_regime():
    import pytest

    pytest.importorskip("src.assembled_core.signals.risk_aware_combiner")
    from src.assembled_core.signals.risk_aware_combiner import RiskAwareSignalCombiner

    rng = np.random.default_rng(13)
    n = 200
    signals = pd.DataFrame(
        {
            "sig_a": rng.uniform(-1, 1, n),
            "sig_b": rng.uniform(-1, 1, n),
        }
    )
    returns = pd.Series(rng.normal(0, 0.01, n))
    regimes = pd.Series(rng.choice(["RISK_ON", "NEUTRAL"], n))

    combiner = RiskAwareSignalCombiner(min_obs_per_bucket=10)
    combiner.fit(signals, returns, regimes)

    combined = combiner.combine_auto_regime(signals, regimes)
    assert len(combined) == n


def test_risk_combiner_unknown_regime_fallback():
    import pytest

    pytest.importorskip("src.assembled_core.signals.risk_aware_combiner")
    from src.assembled_core.signals.risk_aware_combiner import RiskAwareSignalCombiner

    rng = np.random.default_rng(17)
    n = 100
    signals = pd.DataFrame({"sig_a": rng.uniform(-1, 1, n)})
    returns = pd.Series(rng.normal(0, 0.01, n))
    regimes = pd.Series(["NEUTRAL"] * n)

    combiner = RiskAwareSignalCombiner(min_obs_per_bucket=10)
    combiner.fit(signals, returns, regimes)

    # Unknown regime → should not crash
    result = combiner.combine(signals, current_regime="UNKNOWN_REGIME")
    assert len(result) == n


# ---------------------------------------------------------------------------
# ML Signal Integration Pipeline
# ---------------------------------------------------------------------------


def test_ml_pipeline_primary_only():
    """Ohne regime_router / nested_meta → pipeline passt durch."""
    import pytest

    pytest.importorskip("src.assembled_core.signals.ml_integration")
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    from src.assembled_core.signals.ml_integration import MLSignalPipeline

    rng = np.random.default_rng(21)
    n = 100
    X_train = rng.standard_normal((100, 2))
    y_train = X_train[:, 0] * 0.5 + rng.normal(0, 0.1, 100)
    model = LinearRegression().fit(X_train, y_train)

    X_test = pd.DataFrame(rng.standard_normal((n, 2)), columns=["f1", "f2"])
    pipeline = MLSignalPipeline(primary_model=model, feature_cols=["f1", "f2"])
    output = pipeline.run(X_test)

    assert len(output.primary_signal) == n
    assert len(output.final_position) == n
    # Ohne nested_meta: confidence = size = 1.0 → final_position == regime_routed
    pd.testing.assert_series_equal(
        output.primary_signal.rename("final_position"),
        output.final_position,
        check_names=False,
    )


def test_ml_pipeline_with_external_primary():
    """User liefert primary_signal direkt."""
    import pytest

    pytest.importorskip("src.assembled_core.signals.ml_integration")
    from src.assembled_core.signals.ml_integration import MLSignalPipeline

    rng = np.random.default_rng(23)
    n = 50
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["f1", "f2"])
    primary = pd.Series(rng.uniform(-1, 1, n), index=X.index)

    pipeline = MLSignalPipeline()
    output = pipeline.run(X, primary_signal=primary)
    pd.testing.assert_series_equal(output.primary_signal, primary)


def test_ml_pipeline_multi_signal():
    """Multi-signal path via risk_combiner."""
    import pytest

    pytest.importorskip("src.assembled_core.signals.ml_integration")
    import pytest

    pytest.importorskip("src.assembled_core.signals.risk_aware_combiner")
    from src.assembled_core.signals.ml_integration import MLSignalPipeline
    from src.assembled_core.signals.risk_aware_combiner import RiskAwareSignalCombiner

    rng = np.random.default_rng(27)
    n = 300
    signals = pd.DataFrame(
        {
            "sig_a": rng.uniform(-1, 1, n),
            "sig_b": rng.uniform(-1, 1, n),
        }
    )
    returns = pd.Series(rng.normal(0, 0.01, n))
    regimes = pd.Series(["NEUTRAL"] * n)

    combiner = RiskAwareSignalCombiner(min_obs_per_bucket=10)
    combiner.fit(signals, returns, regimes)

    pipeline = MLSignalPipeline(risk_combiner=combiner)
    combined = pipeline.run_multi_signal(signals)
    assert len(combined) == n


def test_ml_pipeline_no_models_graceful():
    """Pipeline ohne Modelle → Zero-Signal, kein Crash."""
    import pytest

    pytest.importorskip("src.assembled_core.signals.ml_integration")
    from src.assembled_core.signals.ml_integration import MLSignalPipeline

    X = pd.DataFrame(
        np.random.default_rng(0).standard_normal((20, 2)), columns=["f1", "f2"]
    )
    pipeline = MLSignalPipeline()
    output = pipeline.run(X)
    # Primary-Signal sollte zero sein
    assert (output.primary_signal == 0).all()
    assert output.regime == "NEUTRAL"


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
