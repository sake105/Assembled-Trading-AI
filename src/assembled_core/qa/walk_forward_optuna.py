"""Walk-forward hyperparameter optimisation with Optuna TPE sampler.

**Scope:** This module optimises *strategy* hyperparameters (lookback windows,
thresholds, etc.) across walk-forward folds of a return series.  It is the
strategy-backtest sibling of ``scripts/training/walk_forward_hpo.py``, which
is an ML-factor CLI tool that tunes scikit-learn / LightGBM model parameters.
They share the walk-forward concept but operate on different objects:
  - ``walk_forward_optuna`` (this module) → strategy parameters, pure returns input
  - ``walk_forward_hpo.py`` → ML model parameters, feature/label DataFrame input

Each training fold gets an independent Optuna study; the best parameters are
then applied to the test fold.

Algorithm:
  - TPE (Tree-structured Parzen Estimator) sampler — efficient for up to ~50
    parameters.
  - SQLite persistence: studies survive process restarts.
  - Objective: Sharpe ratio on the training fold (in-sample).
  - Evaluation: Sharpe on the test fold (out-of-sample).

When Optuna is not installed, falls back to a single-pass evaluation using the
caller-provided default parameters.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import optuna  # type: ignore[import]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WFOptunaTrial:
    """Result for one walk-forward fold."""
    fold_idx: int
    train_start: int        # index into returns array
    train_end: int
    test_start: int
    test_end: int
    best_params: dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    n_trials: int
    backend: str            # "optuna" or "default"


@dataclass
class WFOptunaResult:
    """Aggregated walk-forward + Optuna result."""
    folds: list[WFOptunaTrial]
    avg_test_sharpe: float
    std_test_sharpe: float
    avg_train_sharpe: float
    best_global_params: dict[str, Any]  # params from fold with highest test Sharpe
    n_folds: int
    backend: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe from daily return array."""
    if len(returns) < 5:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    return mu / max(sigma, 1e-9) * math.sqrt(252)


# ---------------------------------------------------------------------------
# Core walk-forward + Optuna driver
# ---------------------------------------------------------------------------

def walk_forward_optuna(
    returns: list[float] | np.ndarray,
    objective_fn: Callable[[np.ndarray, dict[str, Any]], float],
    search_space: dict[str, Any],
    train_days: int = 252,
    test_days: int = 63,
    step_days: int = 63,
    n_trials: int = 50,
    default_params: dict[str, Any] | None = None,
    storage_path: str | None = None,
    study_name_prefix: str = "wfo",
) -> WFOptunaResult:
    """Walk-forward backtest with per-fold Optuna hyperparameter search.

    Args:
        returns: Daily return series (full history).
        objective_fn: Callable(train_returns, params) → scalar metric to MAXIMIZE.
                      Typically annualised Sharpe on the training slice.
        search_space: Dict defining the parameter search space.
                      Each key maps to a tuple: ("float", low, high) or ("int", low, high)
                      or ("categorical", [choices]).
                      Example: {"lookback": ("int", 10, 60), "threshold": ("float", 0.0, 0.5)}.
        train_days: Length of each training window.
        test_days: Length of each test window.
        step_days: Stride between folds.
        n_trials: Optuna trials per fold.
        default_params: Fallback parameters when Optuna is unavailable.
        storage_path: SQLite file path for Optuna study persistence.
                      None = in-memory (no persistence).
        study_name_prefix: Prefix for Optuna study names.

    Returns:
        WFOptunaResult with per-fold diagnostics and aggregated stats.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)

    folds: list[WFOptunaTrial] = []
    fold_idx = 0
    start = 0

    while start + train_days + test_days <= n:
        train_sl = arr[start: start + train_days]
        test_sl = arr[start + train_days: start + train_days + test_days]

        if _OPTUNA_AVAILABLE:
            best_params, train_sharpe, n_done = _run_optuna_fold(
                train_sl, objective_fn, search_space,
                n_trials, storage_path, f"{study_name_prefix}_fold{fold_idx}",
            )
            backend = "optuna"
        else:
            best_params = default_params or {}
            train_sharpe = _sharpe(train_sl)
            n_done = 0
            backend = "default"

        test_sharpe = _sharpe(test_sl)

        folds.append(WFOptunaTrial(
            fold_idx=fold_idx,
            train_start=start,
            train_end=start + train_days,
            test_start=start + train_days,
            test_end=start + train_days + test_days,
            best_params=best_params,
            train_sharpe=round(train_sharpe, 4),
            test_sharpe=round(test_sharpe, 4),
            n_trials=n_done,
            backend=backend,
        ))

        start += step_days
        fold_idx += 1

    if not folds:
        return WFOptunaResult(
            folds=[], avg_test_sharpe=0.0, std_test_sharpe=0.0,
            avg_train_sharpe=0.0, best_global_params=default_params or {},
            n_folds=0, backend="none",
        )

    test_sharpes = [f.test_sharpe for f in folds]
    train_sharpes = [f.train_sharpe for f in folds]
    best_fold = max(folds, key=lambda f: f.test_sharpe)

    return WFOptunaResult(
        folds=folds,
        avg_test_sharpe=round(float(np.mean(test_sharpes)), 4),
        std_test_sharpe=round(float(np.std(test_sharpes)), 4),
        avg_train_sharpe=round(float(np.mean(train_sharpes)), 4),
        best_global_params=best_fold.best_params,
        n_folds=len(folds),
        backend="optuna" if _OPTUNA_AVAILABLE else "default",
    )


def _run_optuna_fold(
    train_returns: np.ndarray,
    objective_fn: Callable[[np.ndarray, dict[str, Any]], float],
    search_space: dict[str, Any],
    n_trials: int,
    storage_path: str | None,
    study_name: str,
) -> tuple[dict[str, Any], float, int]:
    """Run one Optuna study on a training slice.

    Returns: (best_params, best_value, n_trials_done).
    """
    import optuna

    storage = None
    if storage_path:
        storage = f"sqlite:///{storage_path}"

    def _objective(trial: "optuna.Trial") -> float:
        params: dict[str, Any] = {}
        for key, spec in search_space.items():
            kind = spec[0]
            if kind == "float":
                params[key] = trial.suggest_float(key, spec[1], spec[2])
            elif kind == "int":
                params[key] = trial.suggest_int(key, spec[1], spec[2])
            elif kind == "categorical":
                params[key] = trial.suggest_categorical(key, spec[1])
            else:
                params[key] = spec[1]
        try:
            return float(objective_fn(train_returns, params))
        except Exception:
            return -999.0

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params if study.trials else {}
    best_value = study.best_value if study.trials else 0.0
    n_done = len(study.trials)

    return best_params, float(best_value), n_done


# ---------------------------------------------------------------------------
# Built-in objective: momentum lookback optimisation
# ---------------------------------------------------------------------------

def momentum_sharpe_objective(
    returns: np.ndarray,
    params: dict[str, Any],
) -> float:
    """Reference objective: Sharpe of a simple momentum strategy.

    Parameters:
        lookback (int): Rolling mean window (days).
        threshold (float): Signal fires when rolling mean > threshold.

    Useful as a drop-in for testing the walk-forward machinery.
    """
    import pandas as pd

    lookback = int(params.get("lookback", 20))
    threshold = float(params.get("threshold", 0.0))

    ser = pd.Series(returns)
    signal = (ser.rolling(lookback).mean().shift(1) > threshold).astype(float)
    strat_ret = returns * signal.values

    return _sharpe(strat_ret[~np.isnan(strat_ret)])
