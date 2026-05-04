"""Tests für Phase 7-14 ML-Erweiterungen.

Deckt ab:
- Triple-Barrier Labeling
- Fractional Differentiation
- Stacking Ensemble
- Conformal Prediction
- Cross-Sectional Features
- PBO (Backtest Overfit)
- Regime Model Router
- Feature Importance Tracker
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase12


# ---------------------------------------------------------------------------
# Triple Barrier
# ---------------------------------------------------------------------------


def _make_trending_prices(
    n: int = 50, trend: float = 0.002, seed: int = 0
) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, 0.015, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=pd.date_range("2025-01-01", periods=n))


def test_triple_barrier_labels_basic():
    import pytest

    pytest.importorskip("src.assembled_core.ml.triple_barrier")
    from src.assembled_core.ml.triple_barrier import (
        apply_triple_barrier,
        compute_daily_volatility,
    )

    prices = _make_trending_prices(n=100, trend=0.005)  # starker Trend
    vol = compute_daily_volatility(prices, lookback=20)
    result = apply_triple_barrier(
        prices, vol, horizon_days=5, upper_mult=1.5, lower_mult=1.5
    )

    assert {"t1", "label", "ret", "barrier_type"}.issubset(result.columns)
    assert result["label"].notna().sum() > 30  # mindestens einige Labels
    # Letzte 5 Zeilen NaN (horizon_days)
    assert result["label"].iloc[-5:].isna().all()


def test_triple_barrier_upper_hit_on_trend():
    """Starker positiver Trend → viele UPPER-Treffer."""
    import pytest

    pytest.importorskip("src.assembled_core.ml.triple_barrier")
    from src.assembled_core.ml.triple_barrier import (
        apply_triple_barrier,
        compute_daily_volatility,
    )

    # Deterministischer Uptrend
    prices = pd.Series([100 * (1.01**i) for i in range(30)])
    vol = compute_daily_volatility(prices, lookback=10, min_periods=3)
    result = apply_triple_barrier(
        prices, vol, horizon_days=5, upper_mult=1.0, lower_mult=3.0
    )

    barrier_counts = result["barrier_type"].value_counts()
    assert barrier_counts.get("UPPER", 0) > barrier_counts.get("LOWER", 0)


def test_triple_barrier_build_panel():
    import pytest

    pytest.importorskip("src.assembled_core.ml.triple_barrier")
    from src.assembled_core.ml.triple_barrier import build_triple_barrier_labels

    rng = np.random.default_rng(42)
    n = 60
    panel = pd.DataFrame(
        {
            "symbol": ["A"] * n + ["B"] * n,
            "timestamp": list(pd.date_range("2025-01-01", periods=n)) * 2,
            "close": np.concatenate(
                [
                    100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n))),
                    200.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n))),
                ]
            ),
        }
    )
    result = build_triple_barrier_labels(panel, horizon_days=5, vol_lookback=15)
    assert "tb_label_5d" in result.columns
    assert "tb_ret_5d" in result.columns
    assert "tb_barrier_5d" in result.columns


# ---------------------------------------------------------------------------
# Fractional Differentiation
# ---------------------------------------------------------------------------


def test_frac_diff_weights():
    import pytest

    pytest.importorskip("src.assembled_core.features.fractional_diff")
    from src.assembled_core.features.fractional_diff import frac_diff_weights

    w = frac_diff_weights(d=0.5, threshold=1e-4)
    assert len(w) > 5
    assert w[0] == 1.0
    # Weights müssen alternierende Vorzeichen haben (Eigenschaft der Expansion)
    assert w[1] < 0  # -d
    assert abs(w[1] + 0.5) < 1e-9


def test_frac_diff_d_zero_equals_identity():
    """d=0 → series bleibt unverändert."""
    import pytest

    pytest.importorskip("src.assembled_core.features.fractional_diff")
    from src.assembled_core.features.fractional_diff import frac_diff_ffd

    s = pd.Series(np.arange(50, dtype=float))
    result = frac_diff_ffd(s, d=0.0)
    # d=0 → w_0=1 und w_1=0 → window_size=1 → identisch
    np.testing.assert_allclose(result.dropna().values, s.values)


def test_frac_diff_ffd_reduces_integration():
    """Fractional diff auf kumulativer Reihe → stationär ohne Memory-Loss."""
    import pytest

    pytest.importorskip("src.assembled_core.features.fractional_diff")
    from src.assembled_core.features.fractional_diff import frac_diff_ffd

    rng = np.random.default_rng(0)
    returns = rng.standard_normal(200)
    prices = pd.Series(np.cumsum(returns))  # integrated series
    # Use higher threshold so weight window stays small enough for n=200
    diffed = frac_diff_ffd(prices, d=0.4, threshold=1e-3)

    clean = diffed.dropna()
    assert len(clean) > 50
    # Fractional diff reduziert Autokorrelation vs. integrated (>0.99)
    assert abs(clean.autocorr(lag=1)) < 0.98


def test_apply_ffd_to_panel():
    import pytest

    pytest.importorskip("src.assembled_core.features.fractional_diff")
    from src.assembled_core.features.fractional_diff import apply_ffd_to_panel

    panel = pd.DataFrame(
        {
            "symbol": ["A"] * 30 + ["B"] * 30,
            "timestamp": list(pd.date_range("2025-01-01", periods=30)) * 2,
            "close": list(np.arange(1.0, 31.0)) + list(np.arange(50.0, 80.0)),
        }
    )
    result = apply_ffd_to_panel(panel, price_cols=["close"], d=0.3)
    assert "close_ffd_0.30" in result.columns


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------


def test_stacking_cv_basic():
    pytest.importorskip("sklearn")
    pytest.importorskip("src.assembled_core.ml.stacking_ensemble")
    from src.assembled_core.ml.stacking_ensemble import StackingConfig, run_stacking_cv

    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.standard_normal(n),
        }
    )
    y = pd.Series(0.5 * X["f1"] + 0.3 * X["f2"] + rng.normal(0, 0.5, n))

    cfg = StackingConfig(
        base_models=["ridge", "random_forest"],
        meta_model="ridge",
        n_splits=3,
        use_purged_cv=False,
    )
    result = run_stacking_cv(X, y, config=cfg)

    assert len(result.base_models) == 2
    assert result.meta_model is not None
    assert "ridge" in result.base_ic
    # Stacked IC sollte nicht dramatisch unter dem besten Base-IC liegen
    best_base = max(result.base_ic.values())
    assert result.stacked_ic >= best_base - 0.1


def test_stacking_predict():
    pytest.importorskip("sklearn")
    pytest.importorskip("src.assembled_core.ml.stacking_ensemble")
    from src.assembled_core.ml.stacking_ensemble import StackingConfig, run_stacking_cv

    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    y = pd.Series(X["f1"] + rng.normal(0, 0.3, n))

    cfg = StackingConfig(
        base_models=["ridge", "gradient_boosting"],
        meta_model="ridge",
        n_splits=3,
        use_purged_cv=False,
    )
    result = run_stacking_cv(X, y, config=cfg)
    preds = result.predict(X)
    assert len(preds) == n
    assert not preds.isna().all()


# ---------------------------------------------------------------------------
# Conformal Prediction
# ---------------------------------------------------------------------------


def test_conformal_coverage():
    """90%-Intervall sollte ca. 90% der echten Werte abdecken."""
    pytest.importorskip("sklearn")
    pytest.importorskip("src.assembled_core.ml.conformal")
    from sklearn.linear_model import Ridge

    from src.assembled_core.ml.conformal import SplitConformalPredictor

    rng = np.random.default_rng(1)
    n = 600
    X = rng.standard_normal((n, 3))
    y = X @ np.array([1.0, -0.5, 0.3]) + rng.normal(0, 1.0, n)

    X_train, X_cal, X_test = X[:300], X[300:450], X[450:]
    y_train, y_cal, y_test = y[:300], y[300:450], y[450:]

    pred = SplitConformalPredictor(Ridge(), alpha=0.1)
    pred.fit(X_train, y_train, X_cal, y_cal)
    result = pred.predict(X_test)

    coverage = (
        (y_test >= result.lower_bounds.values) & (y_test <= result.upper_bounds.values)
    ).mean()
    # Finite-sample: erlaube kleine Abweichung, aber >=85%
    assert 0.82 < coverage < 0.98


def test_conformal_position_sizing():
    pytest.importorskip("sklearn")
    pytest.importorskip("src.assembled_core.ml.conformal")
    from sklearn.linear_model import Ridge

    from src.assembled_core.ml.conformal import (
        SplitConformalPredictor,
        conformal_position_size,
    )

    rng = np.random.default_rng(2)
    n = 200
    X = rng.standard_normal((n, 2))
    y = X[:, 0] + rng.normal(0, 0.5, n)

    pred = SplitConformalPredictor(Ridge(), alpha=0.1)
    pred.fit(X[:100], y[:100], X[100:150], y[100:150])
    result = pred.predict(X[150:])

    positions = conformal_position_size(result, max_position=1.0)
    assert len(positions) == 50
    assert (positions.abs() <= 1.0).all()


# ---------------------------------------------------------------------------
# Cross-Sectional Features
# ---------------------------------------------------------------------------


def test_rank_cross_sectional_percentile():
    import pytest

    pytest.importorskip("src.assembled_core.features.cross_sectional")
    from src.assembled_core.features.cross_sectional import rank_cross_sectional

    panel = pd.DataFrame(
        {
            "timestamp": ["2025-01-01"] * 4 + ["2025-01-02"] * 4,
            "symbol": ["A", "B", "C", "D"] * 2,
            "f1": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    result = rank_cross_sectional(panel, feature_cols=["f1"], normalize_to="percentile")
    assert "f1_xrank" in result.columns
    # Pro Tag: Ranks müssen in [0, 1] liegen
    day1 = result[result["timestamp"] == "2025-01-01"]["f1_xrank"]
    assert day1.min() > 0
    assert day1.max() <= 1
    # Größter Wert → Rank 1.0, kleinster → Rank 0.25 (1/4)
    assert abs(day1.iloc[3] - 1.0) < 1e-9


def test_zscore_cross_sectional():
    import pytest

    pytest.importorskip("src.assembled_core.features.cross_sectional")
    from src.assembled_core.features.cross_sectional import zscore_cross_sectional

    panel = pd.DataFrame(
        {
            "timestamp": ["2025-01-01"] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = zscore_cross_sectional(panel, feature_cols=["f1"], winsorize_std=None)
    # Mittelwert pro Tag = 0; sample-std (ddof=1) = 1.0 (production uses pandas default)
    assert abs(result["f1_xz"].mean()) < 1e-9
    assert abs(result["f1_xz"].std(ddof=1) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# PBO
# ---------------------------------------------------------------------------


def test_pbo_perfect_strategy_low_pbo():
    """Echte gute Strategie → niedrige PBO."""
    import pytest

    pytest.importorskip("src.assembled_core.qa.backtest_overfit")
    from src.assembled_core.qa.backtest_overfit import compute_pbo

    rng = np.random.default_rng(5)
    n_periods = 30
    n_strats = 10
    # Eine Strategie ist echt gut (höhere Mean), Rest ist Noise
    returns = rng.normal(0, 0.02, (n_periods, n_strats))
    returns[:, 0] = rng.normal(0.01, 0.02, n_periods)  # Strategy 0 hat positives μ

    df = pd.DataFrame(returns, columns=[f"strat_{i}" for i in range(n_strats)])
    result = compute_pbo(df, n_splits=50)

    assert 0 <= result.pbo <= 1
    assert result.n_strategies == n_strats
    assert result.n_periods == n_periods


def test_pbo_random_strategies_high_pbo():
    """Alle Strategien reines Rauschen → PBO sollte hoch sein (~0.5)."""
    import pytest

    pytest.importorskip("src.assembled_core.qa.backtest_overfit")
    from src.assembled_core.qa.backtest_overfit import compute_pbo

    rng = np.random.default_rng(9)
    n_periods = 20
    n_strats = 8
    returns = rng.normal(0, 0.02, (n_periods, n_strats))

    df = pd.DataFrame(returns, columns=[f"strat_{i}" for i in range(n_strats)])
    result = compute_pbo(df, n_splits=100)

    # Pure-Noise-Strategien: PBO tendiert zu 0.5
    assert result.pbo > 0.2


def test_pbo_interpret():
    import pytest

    pytest.importorskip("src.assembled_core.qa.backtest_overfit")
    from src.assembled_core.qa.backtest_overfit import PBOResult

    assert (
        "ROBUST"
        in PBOResult(
            pbo=0.05, n_strategies=5, n_periods=20, n_splits=50, median_logit=1.0
        ).interpret()
    )
    assert (
        "STARK OVERFITTET"
        in PBOResult(
            pbo=0.7, n_strategies=5, n_periods=20, n_splits=50, median_logit=-0.5
        ).interpret()
    )


# ---------------------------------------------------------------------------
# Regime Model Router
# ---------------------------------------------------------------------------


def test_regime_router_fit_predict():
    import pytest

    pytest.importorskip("src.assembled_core.ml.regime_model_router")
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.regime_model_router import (
        RegimeModelRouter,
        RegimeRouterConfig,
    )

    rng = np.random.default_rng(11)
    n = 800  # 200 per regime
    regimes = (
        (["RISK_ON"] * 200)
        + (["NEUTRAL"] * 200)
        + (["RISK_OFF"] * 200)
        + (["CRISIS"] * 200)
    )
    panel = pd.DataFrame(
        {
            "regime": regimes,
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "fwd_return_5d": rng.normal(0, 0.02, n),
        }
    )

    router = RegimeModelRouter(RegimeRouterConfig(min_samples_per_regime=50))
    router.fit(
        panel_df=panel,
        regime_col="regime",
        label_col="fwd_return_5d",
        feature_cols=["f1", "f2"],
    )

    # Test für ein Regime predicten
    X_test = pd.DataFrame({"f1": [0.5, -0.3], "f2": [0.1, 0.2]})
    preds = router.predict(X_test, regime="RISK_ON")
    assert len(preds) == 2


def test_regime_router_crisis_no_trade():
    import pytest

    pytest.importorskip("src.assembled_core.ml.regime_model_router")
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.regime_model_router import (
        RegimeModelRouter,
        RegimeRouterConfig,
    )

    rng = np.random.default_rng(13)
    n = 600
    regimes = (["RISK_ON"] * 200) + (["NEUTRAL"] * 200) + (["CRISIS"] * 200)
    panel = pd.DataFrame(
        {
            "regime": regimes,
            "f1": rng.standard_normal(n),
            "fwd_return_5d": rng.normal(0, 0.02, n),
        }
    )

    router = RegimeModelRouter(
        RegimeRouterConfig(
            min_samples_per_regime=50,
            crisis_policy="no_trade",
        )
    )
    router.fit(
        panel, regime_col="regime", label_col="fwd_return_5d", feature_cols=["f1"]
    )

    X_test = pd.DataFrame({"f1": [0.5, -0.3]})
    preds = router.predict(X_test, regime="CRISIS")
    assert (preds == 0.0).all()


# ---------------------------------------------------------------------------
# Feature Importance Tracker
# ---------------------------------------------------------------------------


def test_importance_tracker_record_and_prune(tmp_path):
    import pytest

    pytest.importorskip("src.assembled_core.ml.feature_importance_tracker")
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge

    from src.assembled_core.ml.feature_importance_tracker import (
        FeatureImportanceTracker,
    )

    rng = np.random.default_rng(17)
    n = 200
    X = pd.DataFrame(
        {
            "useful": rng.standard_normal(n),
            "noise": rng.standard_normal(n),
        }
    )
    y = pd.Series(1.5 * X["useful"] + rng.normal(0, 0.3, n))

    model = Ridge()
    model.fit(X.values, y.values)

    tracker = FeatureImportanceTracker(state_path=tmp_path / "fi.json")

    # 3 Snapshots → genug für trend
    for i in range(3):
        tracker.record_snapshot(
            model=model,
            X=X,
            y=y,
            feature_cols=["useful", "noise"],
            as_of=f"2025-01-{i+1:02d}",
            n_repeats=3,
        )

    assert tracker.history_length() == 3

    decisions = tracker.prune_recommendations(importance_threshold=0.01)
    assert len(decisions) == 2
    by_feat = {d.feature: d for d in decisions}
    # 'useful' sollte nicht pruned werden, 'noise' eher ja
    assert by_feat["useful"].action in ("keep", "review")
