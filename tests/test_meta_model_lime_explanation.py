"""Regression tests for MetaModel.explain_prediction_lime.

These tests pin the corrected contract of ``explain_prediction_lime`` after a
real-bug fix: the previous implementation called ``expl.top_features(...)`` and
read ``expl.predicted_value`` on the ``LimeExplanation`` dataclass, but that
dataclass exposes only ``feature_contributions`` and ``source``. Both wrong
references raised ``AttributeError`` which the surrounding ``try/except`` then
swallowed into ``{"error": ...}`` — i.e. the explainability output was never
produced. The fix maps onto the real wrapper API:

  * ``top_features`` is derived from ``feature_contributions`` (sorted by
    absolute contribution, top-N),
  * the full ``feature_contributions`` map is returned,
  * ``source`` is forwarded,
  * the nonexistent ``predicted_value`` key is dropped.

``lime`` is an optional dependency; when absent the wrapper uses an sklearn
permutation-importance fallback, so the corrected attribute references are
exercised regardless of whether ``lime`` itself is installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.ensemble import GradientBoostingClassifier

from src.assembled_core.signals.meta_model import MetaModel

FEATURES = ["feature_1", "feature_2", "feature_3"]


def _fitted_meta_model() -> MetaModel:
    rng = np.random.default_rng(42)
    n = 120
    X = pd.DataFrame(
        {
            "feature_1": rng.standard_normal(n),
            "feature_2": rng.standard_normal(n),
            "feature_3": rng.standard_normal(n),
        }
    )
    y = ((X["feature_1"] + X["feature_2"]) > 0).astype(int)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X[FEATURES].values, y.values)
    return MetaModel(model=clf, feature_names=list(FEATURES))


@pytest.mark.fast
def test_explain_prediction_lime_returns_corrected_schema():
    """The valid path returns the corrected keys, NOT an error dict."""
    mm = _fitted_meta_model()
    rng = np.random.default_rng(1)
    training = pd.DataFrame(
        rng.standard_normal((60, 3)),
        columns=FEATURES,
    )
    x_row = pd.Series({"feature_1": 0.5, "feature_2": -0.3, "feature_3": 1.1})

    out = mm.explain_prediction_lime(x_row, training_data=training, num_features=3)

    # The old buggy code raised AttributeError -> {"error": ...}. A correct
    # result must NOT carry an error key.
    assert "error" not in out, out
    # Corrected schema.
    assert set(out) == {"top_features", "feature_contributions", "source"}
    assert isinstance(out["feature_contributions"], dict)
    assert out["source"] in {"lime", "permutation_fallback", "zero_fallback"}
    # The removed (nonexistent) attribute must not reappear.
    assert "predicted_value" not in out


@pytest.mark.fast
def test_top_features_sorted_by_absolute_contribution():
    """top_features is a list of (name, contribution) sorted by |contribution|."""
    mm = _fitted_meta_model()
    rng = np.random.default_rng(2)
    training = pd.DataFrame(rng.standard_normal((60, 3)), columns=FEATURES)
    x_row = pd.Series({"feature_1": 1.0, "feature_2": 0.0, "feature_3": -2.0})

    out = mm.explain_prediction_lime(x_row, training_data=training, num_features=2)

    top = out["top_features"]
    assert isinstance(top, list)
    assert len(top) <= 2
    # Each entry is a (name, value) tuple referencing a known feature.
    for name, value in top:
        assert name in FEATURES
        assert isinstance(float(value), float)
    # Sorted descending by absolute contribution.
    abs_vals = [abs(v) for _, v in top]
    assert abs_vals == sorted(abs_vals, reverse=True)


@pytest.mark.fast
def test_explain_prediction_lime_accepts_dict_row():
    """A dict X_row is coerced to an ndarray aligned to feature_names."""
    mm = _fitted_meta_model()
    rng = np.random.default_rng(3)
    training = pd.DataFrame(rng.standard_normal((60, 3)), columns=FEATURES)
    x_row = {"feature_1": 0.2, "feature_2": 0.4, "feature_3": -0.1}

    out = mm.explain_prediction_lime(x_row, training_data=training, num_features=3)

    assert "error" not in out, out
    assert set(out) == {"top_features", "feature_contributions", "source"}
    # feature_contributions covers exactly the model's features.
    assert set(out["feature_contributions"]) == set(FEATURES)
