"""Tests for wave-53 module wiring into trading_cycle.py.

Covers:
  Step 8.52 — ml.bayesian_nn (MCDropoutMLP / BNNPrediction)
  Step 8.53 — ml.hyperopt (OPTUNA_AVAILABLE / tune_model_optuna)
  Step 8.54 — ml.temporal_attention (TemporalAttentionModel / TemporalAttentionConfig)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.bayesian_nn import (
    MCDropoutMLP,
    BNNPrediction,
    TORCH_AVAILABLE,
)
from src.assembled_core.ml.hyperopt import OPTUNA_AVAILABLE
from src.assembled_core.ml.temporal_attention import (
    TemporalAttentionModel,
    TemporalAttentionConfig,
    AttentionResult,
    TORCH_AVAILABLE as TA_TORCH_AVAILABLE,
)


# ---------------------------------------------------------------------------
# MCDropoutMLP (Step 8.52)
# ---------------------------------------------------------------------------

def test_mc_dropout_mlp_creates():
    mlp = MCDropoutMLP()
    assert isinstance(mlp, MCDropoutMLP)


def test_mc_dropout_mlp_defaults():
    mlp = MCDropoutMLP()
    assert mlp.dropout_rate > 0
    assert mlp.n_mc_samples > 0
    assert not mlp._fitted


def test_mc_dropout_mlp_custom_params():
    mlp = MCDropoutMLP(dropout_rate=0.3, n_mc_samples=10)
    assert mlp.dropout_rate == 0.3
    assert mlp.n_mc_samples == 10


def test_torch_available_flag():
    assert isinstance(TORCH_AVAILABLE, bool)


def test_mc_dropout_fit_predict_sklearn_fallback():
    pytest.importorskip("sklearn", reason="scikit-learn required for fallback")
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 4)).astype(np.float32)
    y = rng.normal(0, 1, 40).astype(np.float32)
    mlp = MCDropoutMLP(n_mc_samples=5)
    mlp.fit(X, y)
    assert mlp._fitted
    result = mlp.predict(X[:5])
    assert isinstance(result, BNNPrediction)


def test_bnn_prediction_has_mean_std():
    pytest.importorskip("sklearn", reason="scikit-learn required for fallback")
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 3)).astype(np.float32)
    y = rng.normal(0, 1, 40).astype(np.float32)
    mlp = MCDropoutMLP(n_mc_samples=5)
    mlp.fit(X, y)
    result = mlp.predict(X[:5])
    assert hasattr(result, "mean")
    assert hasattr(result, "std")
    assert len(result.mean) == 5


# ---------------------------------------------------------------------------
# hyperopt (Step 8.53)
# ---------------------------------------------------------------------------

def test_optuna_available_flag():
    assert isinstance(OPTUNA_AVAILABLE, bool)


def test_hyperopt_importable():
    from src.assembled_core.ml.hyperopt import tune_model_optuna, guardrailed_hyperopt
    assert callable(tune_model_optuna)
    assert callable(guardrailed_hyperopt)


# ---------------------------------------------------------------------------
# TemporalAttentionModel (Step 8.54)
# ---------------------------------------------------------------------------

def test_temporal_attention_config_creates():
    cfg = TemporalAttentionConfig()
    assert isinstance(cfg, TemporalAttentionConfig)


def test_temporal_attention_config_defaults():
    cfg = TemporalAttentionConfig()
    assert cfg.seq_len > 0
    assert cfg.d_model > 0
    assert cfg.n_heads > 0


def test_temporal_attention_model_creates():
    model = TemporalAttentionModel()
    assert isinstance(model, TemporalAttentionModel)


def test_temporal_attention_torch_flag():
    assert isinstance(TA_TORCH_AVAILABLE, bool)


def test_temporal_attention_fit_returns_result():
    rng = np.random.default_rng(0)
    n = 60
    features = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=["f1", "f2", "f3", "f4"])
    returns = pd.Series(rng.normal(0, 0.01, n))
    cfg = TemporalAttentionConfig(seq_len=10, epochs=2)
    model = TemporalAttentionModel(config=cfg)
    result = model.fit(features, returns)
    assert isinstance(result, AttentionResult)


def test_temporal_attention_result_has_predictions():
    rng = np.random.default_rng(0)
    n = 60
    features = pd.DataFrame(rng.normal(0, 1, (n, 3)), columns=["f1", "f2", "f3"])
    returns = pd.Series(rng.normal(0, 0.01, n))
    cfg = TemporalAttentionConfig(seq_len=10, epochs=2)
    model = TemporalAttentionModel(config=cfg)
    result = model.fit(features, returns)
    assert hasattr(result, "predictions")
    assert hasattr(result, "attention_weights")


def test_temporal_attention_important_lags():
    rng = np.random.default_rng(0)
    n = 60
    features = pd.DataFrame(rng.normal(0, 1, (n, 3)), columns=["f1", "f2", "f3"])
    returns = pd.Series(rng.normal(0, 0.01, n))
    cfg = TemporalAttentionConfig(seq_len=10, epochs=2)
    model = TemporalAttentionModel(config=cfg)
    result = model.fit(features, returns)
    assert isinstance(result.important_lags, list)
