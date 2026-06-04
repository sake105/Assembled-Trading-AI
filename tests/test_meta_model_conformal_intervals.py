"""Regression tests for MetaModel.predict_with_intervals split-conformal path.

Background: ``predict_with_intervals`` previously imported
``ml.conformal.ConformalResult``, which was moved to ``archive/`` — the import
ALWAYS raised ``ModuleNotFoundError`` and the function silently returned
DEGENERATE intervals (lower == upper, half_width == 0.0, confidence == 1.0) for
every call, even when a calibration set was supplied.

The fix removes the archived dependency and computes the split-conformal interval
inline (preds ± q, q = empirical (1-alpha) quantile of |y_calib - pred_calib|).

These tests DISCRIMINATE: they fail if the normal (with-calib) path produces
zero-width / degenerate intervals again, and they assert the no-calib degraded
fallback still behaves gracefully.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_df(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Label correlated with features so the model produces non-trivial,
    # non-constant predict_proba output -> residuals are non-degenerate.
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    logits = 1.3 * f1 - 0.8 * f2 + 0.4 * rng.standard_normal(n)
    label = (logits > 0).astype(int)
    return pd.DataFrame({"f1": f1, "f2": f2, "label": label})


def test_predict_with_intervals_real_intervals_on_normal_path():
    """With a real calib set: half_width > 0, lower < upper element-wise,
    confidence in [0, 1], and point predictions match the model output exactly.
    """
    pytest.importorskip("sklearn")
    from src.assembled_core.signals.meta_model import train_meta_model

    df = _make_df(seed=20260604, n=400)
    train = df.iloc[:250]
    calib = df.iloc[250:330]
    test = df.iloc[330:]

    mm = train_meta_model(train, feature_cols=["f1", "f2"], label_col="label")

    X_test = test[["f1", "f2"]]
    result = mm.predict_with_intervals(
        X_test,
        X_calib=calib[["f1", "f2"]],
        y_calib=calib["label"],
        alpha=0.1,
    )

    # --- DISCRIMINATING assertions: degenerate intervals must FAIL here ---
    assert result["half_width"] > 0.0, "half_width collapsed to 0 -> degenerate"
    assert (result["lower"] < result["upper"]).all(), (
        "lower must be < upper (non-zero width)"
    )

    # Interval is symmetric around the point prediction: lower <= pred <= upper.
    assert (result["lower"] <= result["predictions"]).all()
    assert (result["predictions"] <= result["upper"]).all()

    # Symmetric split-conformal: width == 2 * half_width everywhere.
    width = result["upper"] - result["lower"]
    np.testing.assert_allclose(width.values, 2.0 * result["half_width"])

    # confidence is a per-element Series in [0, 1].
    conf = result["confidence"]
    assert isinstance(conf, pd.Series)
    assert len(conf) == len(X_test)
    assert (conf >= 0.0).all() and (conf <= 1.0).all()

    # point_predictions must equal the underlying model output exactly.
    expected_preds = mm.predict_proba(X_test)
    np.testing.assert_allclose(result["predictions"].values, expected_preds.values)

    # half_width must equal the empirical split-conformal quantile q.
    calib_preds = mm.predict_proba(calib[["f1", "f2"]]).values
    residuals = np.abs(calib["label"].values - calib_preds)
    n = len(residuals)
    q_level = min(1.0, np.ceil((n + 1) * (1 - 0.1)) / n)
    q_expected = float(np.quantile(residuals, q_level))
    assert result["half_width"] == pytest.approx(q_expected)


def test_predict_with_intervals_no_calib_degraded_fallback():
    """Without a calib set: graceful degraded fallback — confidence == 1.0,
    half_width == 0.0, intervals == point predictions.
    """
    pytest.importorskip("sklearn")
    from src.assembled_core.signals.meta_model import train_meta_model

    df = _make_df(seed=99, n=200)
    mm = train_meta_model(df, feature_cols=["f1", "f2"], label_col="label")

    X = df[["f1", "f2"]]
    result = mm.predict_with_intervals(X)

    assert result["half_width"] == 0.0
    assert (result["confidence"] == 1.0).all()
    assert (result["lower"] == result["predictions"]).all()
    assert (result["upper"] == result["predictions"]).all()
    # predictions still match the model output.
    np.testing.assert_allclose(result["predictions"].values, mm.predict_proba(X).values)


def test_predict_with_intervals_insufficient_calib_degrades_gracefully():
    """Empty calibration set must not raise; it degrades to the point-prediction
    fallback (the narrow try/except catches the numeric failure)."""
    pytest.importorskip("sklearn")
    from src.assembled_core.signals.meta_model import train_meta_model

    df = _make_df(seed=7, n=200)
    mm = train_meta_model(df, feature_cols=["f1", "f2"], label_col="label")

    X = df[["f1", "f2"]].iloc[:20]
    empty_calib = df[["f1", "f2"]].iloc[0:0]
    empty_y = df["label"].iloc[0:0]

    result = mm.predict_with_intervals(X, X_calib=empty_calib, y_calib=empty_y)

    # Degraded fallback shape preserved, no exception propagated.
    assert result["half_width"] == 0.0
    assert (result["confidence"] == 1.0).all()
    assert (result["lower"] == result["predictions"]).all()
    assert (result["upper"] == result["predictions"]).all()
