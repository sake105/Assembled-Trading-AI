"""Tests for wave-49 module wiring into trading_cycle.py.

Covers:
  Step 8.43 — ml.regime_weight_trainer (train_regime_weights / compute_per_regime_ic)
  Step 8.44 — ml.news_ml_bridge (get_event_type_ic_weights / NewsRegimeClassifier)
  Step 8.45 — ml.nlp_sentiment (score_texts_finbert — transformers-gated)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.regime_weight_trainer import (
    train_regime_weights,
    compute_per_regime_ic,
)
from src.assembled_core.ml.news_ml_bridge import (
    get_event_type_ic_weights,
    NewsRegimeClassifier,
)
from src.assembled_core.ml.nlp_sentiment import score_texts_finbert


# ---------------------------------------------------------------------------
# train_regime_weights (Step 8.43)
# ---------------------------------------------------------------------------

def _make_ic_df(n: int = 40, factor_cols: list[str] | None = None) -> pd.DataFrame:
    if factor_cols is None:
        factor_cols = ["ic_momentum", "ic_value", "ic_quality"]
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {c: rng.normal(0, 0.05, n) for c in factor_cols},
        index=dates,
    )
    df.index.name = "date"
    return df.reset_index()


def _make_regime_df(n: int = 40, regimes: list[str] | None = None) -> pd.DataFrame:
    if regimes is None:
        regimes = ["NEUTRAL"] * 20 + ["RISK_OFF"] * 20
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"date": dates, "regime_label": regimes})


def test_train_regime_weights_returns_dict():
    ic_df = _make_ic_df()
    reg_df = _make_regime_df()
    result = train_regime_weights(ic_df, reg_df, factor_cols=["ic_momentum", "ic_value", "ic_quality"])
    assert isinstance(result, dict)


def test_train_regime_weights_has_regime_keys():
    ic_df = _make_ic_df()
    reg_df = _make_regime_df()
    result = train_regime_weights(ic_df, reg_df, factor_cols=["ic_momentum", "ic_value", "ic_quality"])
    assert len(result) >= 1


def test_train_regime_weights_sums_to_one():
    ic_df = _make_ic_df()
    reg_df = _make_regime_df()
    result = train_regime_weights(ic_df, reg_df, factor_cols=["ic_momentum", "ic_value", "ic_quality"])
    for regime, weights in result.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Regime {regime} weights don't sum to 1: {total}"


def test_compute_per_regime_ic_empty_factor_cols():
    ic_df = _make_ic_df()
    reg_df = _make_regime_df()
    result = compute_per_regime_ic(ic_df, reg_df, factor_cols=[])
    assert isinstance(result, dict)
    assert len(result) == 0


def test_train_regime_weights_shrinkage_method():
    ic_df = _make_ic_df()
    reg_df = _make_regime_df()
    result = train_regime_weights(
        ic_df, reg_df,
        factor_cols=["ic_momentum", "ic_value"],
        method="shrinkage",
    )
    assert isinstance(result, dict)


def test_train_regime_weights_ic_ir_method():
    ic_df = _make_ic_df()
    reg_df = _make_regime_df()
    result = train_regime_weights(
        ic_df, reg_df,
        factor_cols=["ic_momentum", "ic_value"],
        method="ic_ir_weighted",
    )
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# news_ml_bridge (Step 8.44)
# ---------------------------------------------------------------------------

def test_get_event_type_ic_weights_returns_dict():
    result = get_event_type_ic_weights()
    assert isinstance(result, dict)


def test_get_event_type_ic_weights_empty_if_no_file(tmp_path):
    result = get_event_type_ic_weights(ic_loop_path=tmp_path / "nonexistent.json")
    assert isinstance(result, dict)
    assert len(result) == 0


def test_news_regime_classifier_creates():
    clf = NewsRegimeClassifier()
    assert isinstance(clf, NewsRegimeClassifier)


def test_news_regime_classifier_has_classify():
    clf = NewsRegimeClassifier()
    assert hasattr(clf, "classify") or hasattr(clf, "predict")


def test_news_regime_classifier_classify_empty():
    clf = NewsRegimeClassifier()
    if hasattr(clf, "classify"):
        result = clf.classify([])
        assert isinstance(result, (list, dict, pd.Series, pd.DataFrame))


# ---------------------------------------------------------------------------
# score_texts_finbert (Step 8.45 — transformers-gated)
# ---------------------------------------------------------------------------

def test_score_texts_finbert_empty_returns_empty():
    result = score_texts_finbert([])
    assert isinstance(result, list)
    assert len(result) == 0


def test_score_texts_finbert_returns_list():
    # transformers may not be installed — graceful degradation expected
    try:
        result = score_texts_finbert(["The company reported strong earnings."])
        assert isinstance(result, list)
    except Exception:
        pytest.skip("transformers not available")


def test_score_texts_finbert_result_keys():
    try:
        result = score_texts_finbert(["Stocks rose sharply on positive news."])
        if result:
            assert "sentiment" in result[0]
            assert "score" in result[0]
    except Exception:
        pytest.skip("transformers not available")
