"""Unified wrappers für XGB / LightGBM / CatBoost mit Time-Series-CV.

Wraps the three industry-standard gradient-boosting libraries behind a single
``fit_predict``-compatible interface, so they slot into our Stacking-Ensemble
or Conformal-Prediction modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BoostConfig:
    objective: str = "regression"
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample: float = 0.8
    early_stopping_rounds: Optional[int] = 20
    random_state: int = 42


def fit_predict_xgb(
    X_train, y_train, X_test, config: Optional[BoostConfig] = None
) -> np.ndarray:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as e:
        raise RuntimeError("pip install xgboost") from e
    cfg = config or BoostConfig()
    params = dict(
        objective=(
            "reg:squarederror" if cfg.objective == "regression" else cfg.objective
        ),
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample,
        random_state=cfg.random_state,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_lgbm(
    X_train, y_train, X_test, config: Optional[BoostConfig] = None
) -> np.ndarray:
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError as e:
        raise RuntimeError("pip install lightgbm") from e
    cfg = config or BoostConfig()
    model = lgb.LGBMRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample,
        random_state=cfg.random_state,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_catboost(
    X_train, y_train, X_test, config: Optional[BoostConfig] = None
) -> np.ndarray:
    try:
        import catboost as cb  # type: ignore
    except ImportError as e:
        raise RuntimeError("pip install catboost") from e
    cfg = config or BoostConfig()
    model = cb.CatBoostRegressor(
        iterations=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        depth=cfg.max_depth,
        random_seed=cfg.random_state,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_random_forest(
    X_train,
    y_train,
    X_test,
    n_estimators: int = 300,
    max_depth: int = 12,
    random_state: int = 42,
) -> np.ndarray:
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
    except ImportError as e:
        raise RuntimeError("scikit-learn required") from e
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def time_series_cv_score(
    fit_predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    metric: str = "rmse",
) -> dict:
    """Time-series CV (rolling) — kein random shuffle."""
    n = len(X)
    fold_size = n // n_splits
    scores = []
    for k in range(1, n_splits):
        end_train = k * fold_size
        end_test = (k + 1) * fold_size if k < n_splits - 1 else n
        X_tr, y_tr = X[:end_train], y[:end_train]
        X_te, y_te = X[end_train:end_test], y[end_train:end_test]
        pred = fit_predict_fn(X_tr, y_tr, X_te)
        if metric == "rmse":
            scores.append(float(np.sqrt(((pred - y_te) ** 2).mean())))
        elif metric == "mae":
            scores.append(float(np.abs(pred - y_te).mean()))
        elif metric == "ic":
            df_pred = (pred - pred.mean()) / (pred.std() + 1e-12)
            df_true = (y_te - y_te.mean()) / (y_te.std() + 1e-12)
            scores.append(float((df_pred * df_true).mean()))
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores,
    }


def optuna_tune_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 4,
    seed: int = 42,
) -> dict:
    """Optuna tuning for LightGBM.

    Returns:
        Dict with best_params, best_value, n_trials.
    """
    try:
        import optuna  # type: ignore
        import lightgbm as lgb  # type: ignore
    except ImportError as e:
        raise RuntimeError("pip install optuna lightgbm") from e

    def _objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "verbose": -1,
        }
        n = len(X)
        fold_size = n // n_splits
        rmses = []
        for k in range(1, n_splits):
            end_train = k * fold_size
            X_tr = X[:end_train]
            y_tr = y[:end_train]
            X_te = X[end_train : (k + 1) * fold_size]
            y_te = y[end_train : (k + 1) * fold_size]
            model = lgb.LGBMRegressor(**params, random_state=seed)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            rmses.append(np.sqrt(((pred - y_te) ** 2).mean()))
        return float(np.mean(rmses))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": n_trials,
    }


__all__ = [
    "BoostConfig",
    "fit_predict_xgb",
    "fit_predict_lgbm",
    "fit_predict_catboost",
    "fit_predict_random_forest",
    "time_series_cv_score",
    "optuna_tune_lgbm",
]
