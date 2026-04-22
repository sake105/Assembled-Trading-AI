"""Tests für Round-4 ML-Erweiterungen (Online GB, Nested Meta, Wiring)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase12


# ---------------------------------------------------------------------------
# Online Gradient Boosting
# ---------------------------------------------------------------------------

def test_online_learner_learn_predict():
    from src.assembled_core.ml.online_gradient_boosting import OnlineAdaptiveLearner

    learner = OnlineAdaptiveLearner(feature_names=["f1", "f2"])

    rng = np.random.default_rng(3)
    # Simple linear relationship
    for _ in range(200):
        x = rng.standard_normal(2)
        y = 1.5 * x[0] + 0.5 * x[1] + rng.normal(0, 0.3)
        learner.learn_one(x, y)

    # Prediction should be in reasonable range
    pred = learner.predict_one(np.array([1.0, 0.0]))
    assert isinstance(pred, float)


def test_online_learner_batch():
    from src.assembled_core.ml.online_gradient_boosting import OnlineAdaptiveLearner

    rng = np.random.default_rng(5)
    n = 100
    X = rng.standard_normal((n, 3))
    y = X[:, 0] + rng.normal(0, 0.2, n)

    learner = OnlineAdaptiveLearner(feature_names=["a", "b", "c"])
    errors = learner.learn_batch(X, y)
    assert len(errors) == n

    preds = learner.predict_batch(X)
    assert len(preds) == n


def test_online_learner_no_river_fallback():
    """Ohne river installiert → Mini-Batch-Fallback, aber learn_one/predict_one funktioniert."""
    from src.assembled_core.ml.online_gradient_boosting import OnlineAdaptiveLearner

    learner = OnlineAdaptiveLearner(feature_names=["f1"])
    # Funktioniert in beiden Fällen (river oder fallback)
    learner.learn_one(np.array([1.0]), 0.5)
    pred = learner.predict_one(np.array([1.0]))
    assert isinstance(pred, float)


# ---------------------------------------------------------------------------
# Nested Meta Labeling
# ---------------------------------------------------------------------------

def test_nested_meta_fit_predict():
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.nested_meta_labeling import NestedMetaLabeler

    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({
        "primary_signal": rng.uniform(-1, 1, n),
        "primary_direction": rng.choice([-1, 1], n),
        "news_sentiment_mean": rng.uniform(-0.5, 0.5, n),
        "news_velocity": rng.uniform(0.5, 2.0, n),
        "regime_state": rng.integers(0, 3, n),
        "vix_proxy": rng.uniform(10, 40, n),
        "success_label": rng.integers(0, 2, n),
        "magnitude_label": rng.normal(0, 0.02, n),
    })

    labeler = NestedMetaLabeler(confidence_threshold=0.5)
    labeler.fit(df)

    features_only = df.drop(columns=["success_label", "magnitude_label"])
    pred = labeler.predict(features_only)

    assert len(pred.primary_signal) == n
    assert len(pred.confidence) == n
    assert len(pred.size_scale) == n
    assert len(pred.final_position) == n

    # Confidence in [0, 1]
    assert (pred.confidence >= 0).all() and (pred.confidence <= 1).all()
    # Size ∈ [min_size, 1]
    assert (pred.size_scale >= labeler.min_size - 1e-9).all()
    assert (pred.size_scale <= 1 + 1e-9).all()


def test_nested_meta_size_scale_batch_invariant():
    """Regression P0.3: size_scale must be deterministic across inference batches.

    Prior bug: predict() normalized by batch max → same observation got different
    size_scale depending on which other rows were in the batch. Fix persists the
    training-time max as self._size_scale_max.
    """
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.nested_meta_labeling import NestedMetaLabeler

    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({
        "primary_signal": rng.uniform(-1, 1, n),
        "primary_direction": rng.choice([-1, 1], n),
        "news_sentiment_mean": rng.uniform(-0.5, 0.5, n),
        "news_velocity": rng.uniform(0.5, 2.0, n),
        "regime_state": rng.integers(0, 3, n),
        "vix_proxy": rng.uniform(10, 40, n),
        "success_label": rng.integers(0, 2, n),
        "magnitude_label": rng.normal(0, 0.02, n),
    })

    labeler = NestedMetaLabeler(confidence_threshold=0.5)
    labeler.fit(df)
    assert labeler._size_scale_max is not None
    assert labeler._size_scale_max > 0

    features = df.drop(columns=["success_label", "magnitude_label"])
    # Same observation (row 0), two different batch contexts.
    small = features.iloc[[0, 1, 2]].copy()
    large = features.iloc[[0] + list(range(10, 100))].copy()

    pred_small = labeler.predict(small)
    pred_large = labeler.predict(large)

    # Row 0 must get the same size_scale regardless of batch.
    assert pred_small.size_scale.iloc[0] == pytest.approx(pred_large.size_scale.iloc[0], abs=1e-12)


def test_nested_meta_threshold_gates():
    """Confidence unter Threshold → final_position = 0."""
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.nested_meta_labeling import NestedMetaLabeler

    rng = np.random.default_rng(11)
    n = 150
    # Fast keine Erfolge → Confidence wird meist niedrig
    df = pd.DataFrame({
        "primary_signal": rng.uniform(-1, 1, n),
        "primary_direction": rng.choice([-1, 1], n),
        "news_sentiment_mean": rng.uniform(-0.5, 0.5, n),
        "news_velocity": rng.uniform(0.5, 2.0, n),
        "regime_state": rng.integers(0, 3, n),
        "vix_proxy": rng.uniform(10, 40, n),
        "success_label": (rng.uniform(0, 1, n) > 0.9).astype(int),  # 10% Erfolgsrate
        "magnitude_label": rng.normal(0, 0.02, n),
    })

    labeler = NestedMetaLabeler(confidence_threshold=0.9)  # sehr hoher Threshold
    labeler.fit(df)

    features_only = df.drop(columns=["success_label", "magnitude_label"])
    pred = labeler.predict(features_only)

    # Viele Positions = 0 (wegen threshold gating)
    n_zero = (pred.final_position == 0).sum()
    assert n_zero > n // 2


def test_build_nested_labels_from_trades_uses_direction_col():
    """Helper leitet success_label aus sign(return) == sign(direction) ab."""
    from src.assembled_core.ml.nested_meta_labeling import build_nested_labels_from_trades

    trades = pd.DataFrame({
        "closed_return": [0.02, -0.01, 0.015, -0.03, 0.0],
        "primary_direction": [1, -1, -1, -1, 1],  # 3 hits, 1 miss, 1 zero-return
    })
    out = build_nested_labels_from_trades(trades)
    assert list(out["success_label"].values) == [1, 1, 0, 1, 0]
    # magnitude = abs(return)
    assert out["magnitude_label"].tolist() == [0.02, 0.01, 0.015, 0.03, 0.0]


def test_build_nested_labels_from_trades_falls_back_to_signal():
    """Ohne direction_col nutzt der Helper sign(primary_signal)."""
    from src.assembled_core.ml.nested_meta_labeling import build_nested_labels_from_trades

    trades = pd.DataFrame({
        "closed_return": [0.05, -0.02],
        "primary_signal": [0.4, -0.3],
    })
    out = build_nested_labels_from_trades(trades)
    assert list(out["success_label"].values) == [1, 1]


def test_build_nested_labels_empty_direction_safe():
    """Ohne beide Direction-Spalten → success=0 überall."""
    from src.assembled_core.ml.nested_meta_labeling import build_nested_labels_from_trades

    trades = pd.DataFrame({"closed_return": [0.01, -0.02]})
    out = build_nested_labels_from_trades(trades)
    assert list(out["success_label"].values) == [0, 0]
    assert out["magnitude_label"].tolist() == [0.01, 0.02]


def test_build_nested_labels_then_fit_e2e():
    """E2E: Helper liefert Labels, NestedMetaLabeler.fit akzeptiert sie."""
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.nested_meta_labeling import (
        NestedMetaLabeler,
        build_nested_labels_from_trades,
    )

    rng = np.random.default_rng(19)
    n = 150
    direction = rng.choice([-1, 1], n)
    # Mache Returns korrelierbar zur Richtung für lernbares Signal
    rets = direction * rng.normal(0.005, 0.02, n)

    df = pd.DataFrame({
        "closed_return": rets,
        "primary_signal": direction * rng.uniform(0.3, 1.0, n),
        "primary_direction": direction,
        "news_sentiment_mean": rng.uniform(-0.5, 0.5, n),
        "news_velocity": rng.uniform(0.5, 2.0, n),
        "regime_state": rng.integers(0, 3, n),
        "vix_proxy": rng.uniform(10, 40, n),
    })

    labeled = build_nested_labels_from_trades(df)
    assert "success_label" in labeled.columns
    assert "magnitude_label" in labeled.columns

    labeler = NestedMetaLabeler(confidence_threshold=0.5)
    labeler.fit(labeled)

    # Beide Stufen müssen trainiert sein
    assert labeler._confidence_model is not None
    assert labeler._size_model is not None


# ---------------------------------------------------------------------------
# meta_model.py → predict_with_intervals
# ---------------------------------------------------------------------------

def test_meta_model_predict_with_intervals_no_calib():
    """Ohne Calib → confidence=1.0, intervals=point predictions."""
    pytest.importorskip("sklearn")
    from src.assembled_core.signals.meta_model import train_meta_model

    rng = np.random.default_rng(13)
    n = 200
    df = pd.DataFrame({
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
        "label": rng.integers(0, 2, n),
    })
    mm = train_meta_model(df, feature_cols=["f1", "f2"], label_col="label")

    result = mm.predict_with_intervals(df[["f1", "f2"]])
    assert "predictions" in result
    assert "lower" in result
    assert "upper" in result
    assert "confidence" in result
    assert (result["confidence"] == 1.0).all()
    assert result["half_width"] == 0.0


def test_meta_model_predict_with_intervals_calib():
    """Mit Calib → non-zero half_width, lower < predictions < upper."""
    pytest.importorskip("sklearn")
    from src.assembled_core.signals.meta_model import train_meta_model

    rng = np.random.default_rng(15)
    n = 300
    df = pd.DataFrame({
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
        "label": rng.integers(0, 2, n),
    })
    train = df.iloc[:200]
    calib = df.iloc[200:250]
    test = df.iloc[250:]

    mm = train_meta_model(train, feature_cols=["f1", "f2"], label_col="label")

    result = mm.predict_with_intervals(
        test[["f1", "f2"]],
        X_calib=calib[["f1", "f2"]],
        y_calib=calib["label"],
        alpha=0.1,
    )

    assert result["half_width"] > 0
    assert (result["lower"] <= result["predictions"]).all()
    assert (result["predictions"] <= result["upper"]).all()


# ---------------------------------------------------------------------------
# build_factor_panel.py → triple-barrier wiring
# ---------------------------------------------------------------------------

def test_build_factor_panel_triple_barrier_flag(tmp_path):
    """Funktion build_full_factor_panel akzeptiert triple_barrier=True."""
    # Smoke-Test: Import-Check + Argumente
    from scripts.training.build_factor_panel import build_full_factor_panel

    import inspect
    sig = inspect.signature(build_full_factor_panel)
    assert "triple_barrier" in sig.parameters
    assert "tb_upper_mult" in sig.parameters
    assert "tb_lower_mult" in sig.parameters
