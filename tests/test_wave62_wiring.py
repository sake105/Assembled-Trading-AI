"""Tests for wave-62 module wiring into trading_cycle.py.

Covers:
  Step 8.63 — ml.lime_explainer (LIMEExplainerWrapper / LIMEExplanation)
  Step 8.64 — ml.online_hpo (OnlineHyperparamAdapter / ArmStats)
  Step 8.65 — ml.nested_meta_labeling (NestedMetaLabeler / NestedPrediction)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.lime_explainer import (
    LIMEExplanation,
    LIMEExplainerWrapper,
)
from src.assembled_core.ml.online_hpo import (
    ArmStats,
    OnlineHyperparamAdapter,
)
from src.assembled_core.ml.nested_meta_labeling import (
    NestedMetaLabeler,
    NestedPrediction,
    build_nested_labels_from_trades,
)


# ---------------------------------------------------------------------------
# lime_explainer (Step 8.63)
# ---------------------------------------------------------------------------

def test_lime_explanation_creates():
    exp = LIMEExplanation(
        feature_contributions=[("momentum", 0.3), ("value", -0.1)],
        predicted_value=0.05,
        source="permutation_fallback",
    )
    assert exp.predicted_value == 0.05


def test_lime_explanation_top_features():
    exp = LIMEExplanation(
        feature_contributions=[("f1", 0.5), ("f2", 0.3), ("f3", 0.1), ("f4", 0.05)],
        predicted_value=0.1,
    )
    top = exp.top_features(k=2)
    assert len(top) == 2


def test_lime_explainer_wrapper_creates():
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X))
    wrapper = LIMEExplainerWrapper(DummyModel(), feature_names=["f1", "f2", "f3"])
    assert wrapper is not None


def test_lime_explainer_explain_returns_explanation():
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X))
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, (30, 3))
    # pass training_data directly to __init__
    wrapper = LIMEExplainerWrapper(DummyModel(), feature_names=["f1", "f2", "f3"], training_data=X_train)
    x = rng.normal(0, 1, 3)
    result = wrapper.explain(x)
    assert isinstance(result, LIMEExplanation)


# ---------------------------------------------------------------------------
# online_hpo (Step 8.64)
# ---------------------------------------------------------------------------

def test_arm_stats_creates():
    arm = ArmStats(arm_id="arm1", params={"alpha": 1.0})
    assert arm.arm_id == "arm1"


def test_online_hpo_creates():
    adapter = OnlineHyperparamAdapter()
    assert isinstance(adapter, OnlineHyperparamAdapter)
    assert len(adapter.arms) > 0


def test_online_hpo_select_arm():
    adapter = OnlineHyperparamAdapter()
    arm = adapter.select_arm()
    assert isinstance(arm, ArmStats)


def test_online_hpo_observe_reward():
    adapter = OnlineHyperparamAdapter()
    arm = adapter.select_arm()
    adapter.observe_reward(arm.arm_id, reward=0.05)
    # no exception


def test_online_hpo_custom_arms():
    arms = [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}]
    adapter = OnlineHyperparamAdapter(arms=arms)
    assert len(adapter.arms) == 3


# ---------------------------------------------------------------------------
# nested_meta_labeling (Step 8.65)
# ---------------------------------------------------------------------------

def test_nested_meta_labeler_creates():
    labeler = NestedMetaLabeler()
    assert isinstance(labeler, NestedMetaLabeler)


def test_nested_meta_labeler_defaults():
    labeler = NestedMetaLabeler()
    assert labeler.confidence_threshold > 0
    assert labeler.min_size > 0


def test_nested_prediction_creates():
    n = 5
    pred = NestedPrediction(
        primary_signal=pd.Series([0.5] * n),
        confidence=pd.Series([0.7] * n),
        size_scale=pd.Series([0.8] * n),
        final_position=pd.Series([0.56] * n),
    )
    assert len(pred.primary_signal) == n


def test_build_nested_labels_empty():
    trades = pd.DataFrame(columns=["symbol", "entry_date", "exit_date", "entry_price", "exit_price"])
    result = build_nested_labels_from_trades(trades)
    assert isinstance(result, pd.DataFrame)
