"""Tests for ML hyperparameter optimization module."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

optuna = pytest.importorskip("optuna")
pytest.importorskip("src.assembled_core.ml.hyperopt")

from src.assembled_core.ml.hyperopt import tune_model_optuna, guardrailed_hyperopt


def _synthetic_panel(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    rows = []
    for sym in ["AAPL", "MSFT"]:
        for i, d in enumerate(dates):
            f0, f1 = rng.normal(0, 1), rng.normal(0, 1)
            target = 0.02 * f0 + 0.01 * f1 + rng.normal(0, 0.05)
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "f0": f0,
                    "f1": f1,
                    "fwd_return_5d": target,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.phase12
class TestTuneModelOptuna:
    def test_basic_ridge(self):
        panel = _synthetic_panel()
        from src.assembled_core.ml.factor_models import MLExperimentConfig

        exp = MLExperimentConfig(
            label_col="fwd_return_5d",
            feature_cols=["f0", "f1"],
            n_splits=3,
            min_train_samples=30,
        )
        best_cfg = tune_model_optuna(
            panel,
            exp,
            model_type="ridge",
            n_trials=5,
        )
        assert best_cfg.model_type == "ridge"

    def test_basic_lasso(self):
        panel = _synthetic_panel()
        from src.assembled_core.ml.factor_models import MLExperimentConfig

        exp = MLExperimentConfig(
            label_col="fwd_return_5d",
            feature_cols=["f0", "f1"],
            n_splits=3,
            min_train_samples=30,
        )
        best_cfg = tune_model_optuna(
            panel,
            exp,
            model_type="lasso",
            n_trials=3,
        )
        assert best_cfg.model_type == "lasso"


@pytest.mark.phase12
class TestGuardrailedHyperopt:
    def test_basic(self):
        panel = _synthetic_panel()
        from src.assembled_core.ml.factor_models import MLExperimentConfig

        exp = MLExperimentConfig(
            label_col="fwd_return_5d",
            feature_cols=["f0", "f1"],
            n_splits=3,
            min_train_samples=30,
        )
        result = guardrailed_hyperopt(panel, exp, model_type="ridge", n_trials=3)
        assert isinstance(result, dict)
        # Should have some result keys
        assert len(result) > 0
