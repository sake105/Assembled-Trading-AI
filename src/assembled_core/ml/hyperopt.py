"""Hyperparameter Optimization for ML Factor Models using Optuna.

Provides Bayesian hyperparameter search over the existing time-series CV
infrastructure in factor_models.py.

Usage:
    from src.assembled_core.ml.hyperopt import tune_model_optuna
    best_cfg = tune_model_optuna(panel_df, experiment_cfg, "xgboost", n_trials=50)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.assembled_core.ml.factor_models import MLExperimentConfig, MLModelConfig

logger = logging.getLogger(__name__)

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

_SEARCH_SPACES: dict[str, dict] = {
    "ridge": {
        "alpha": ("log_float", 1e-3, 1e3),
    },
    "lasso": {
        "alpha": ("log_float", 1e-4, 1e1),
    },
    "random_forest": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int_none", 3, 15),
        "min_samples_leaf": ("int", 1, 20),
        "max_features": ("categorical", ["sqrt", "log2", 0.5, 0.7]),
    },
    "xgboost": {
        "n_estimators": ("int", 100, 1000),
        "learning_rate": ("log_float", 0.01, 0.3),
        "max_depth": ("int", 3, 10),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "reg_alpha": ("log_float", 1e-5, 10.0),
        "reg_lambda": ("log_float", 1e-5, 10.0),
    },
    "lightgbm": {
        "n_estimators": ("int", 100, 1000),
        "learning_rate": ("log_float", 0.01, 0.3),
        "num_leaves": ("int", 16, 256),
        "min_child_samples": ("int", 5, 100),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "reg_alpha": ("log_float", 1e-5, 10.0),
        "reg_lambda": ("log_float", 1e-5, 10.0),
    },
    "catboost": {
        "iterations": ("int", 100, 800),
        "learning_rate": ("log_float", 0.01, 0.3),
        "depth": ("int", 3, 10),
        "l2_leaf_reg": ("log_float", 1e-3, 10.0),
        "subsample": ("float", 0.5, 1.0),
    },
}


def _sample_params(trial: "optuna.Trial", model_type: str) -> dict:
    """Sample hyperparameters for model_type from Optuna trial."""
    space = _SEARCH_SPACES.get(model_type, {})
    params: dict = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "log_float":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "int_none":
            val = trial.suggest_int(name, spec[1], spec[2] + 1)
            params[name] = None if val > spec[2] else val
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[2])
    return params


def tune_model_optuna(
    panel_df: pd.DataFrame,
    experiment: "MLExperimentConfig",
    model_type: str = "xgboost",
    n_trials: int = 50,
    timeout_seconds: int | None = 3600,
    study_name: str | None = None,
    direction: str = "maximize",
    metric: str = "ic",
) -> "MLModelConfig":
    """Run Optuna hyperparameter search for a given model type.

    Uses the existing ``run_time_series_cv`` infrastructure so results
    are always measured via time-series cross-validation — no data leakage.

    Args:
        panel_df: Factor panel DataFrame (same format as factor_models.py)
        experiment: MLExperimentConfig controlling CV splits and label
        model_type: One of the supported model types (default: "xgboost")
        n_trials: Number of Optuna trials (default: 50)
        timeout_seconds: Max wall-clock seconds (None = no limit, default: 3600)
        study_name: Optuna study name (default: auto-generated)
        direction: "maximize" or "minimize" (default: "maximize" for IC)
        metric: Metric to optimise — "ic" (default), "r2", "sharpe"

    Returns:
        MLModelConfig with best hyperparameters found
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "optuna is not installed. Run: pip install 'optuna>=3.0.0'"
        )

    from src.assembled_core.ml.factor_models import MLModelConfig, run_time_series_cv

    if study_name is None:
        study_name = f"tune_{model_type}"

    def objective(trial: "optuna.Trial") -> float:
        params = _sample_params(trial, model_type)
        cfg = MLModelConfig(name=f"trial_{trial.number}", model_type=model_type, params=params)
        try:
            result = run_time_series_cv(panel_df, experiment, cfg)
            gm = result.global_metrics
            if metric == "ic":
                score = float(gm.get("ic_mean", 0.0))
            elif metric == "r2":
                score = float(gm.get("r2_mean", 0.0))
            elif metric == "sharpe":
                score = float(gm.get("long_short_sharpe", 0.0))
            else:
                score = float(gm.get("ic_mean", 0.0))
            if not isinstance(score, float) or score != score:  # nan guard
                return -999.0
            return score
        except Exception as e:
            logger.debug("Trial %d failed: %s", trial.number, e)
            return -999.0

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)

    best_params = study.best_params
    best_value = study.best_value
    logger.info(
        "Optuna tuning complete for %s: best_%s=%.4f, params=%s",
        model_type,
        metric,
        best_value,
        best_params,
    )

    return MLModelConfig(
        name=f"{model_type}_tuned",
        model_type=model_type,  # type: ignore[arg-type]
        params=best_params,
    )


def get_study_summary(study: "optuna.Study") -> dict:
    """Return a compact summary dict of a completed Optuna study."""
    if not OPTUNA_AVAILABLE:
        return {}
    return {
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }
