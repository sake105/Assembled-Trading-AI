"""Walk-Forward Hyperparameter Optimization via Optuna.

Optuna führt bayesianische HPO über Sampler TPE. Hier erweitert um
Walk-Forward-Validation:

Für jedes Trial:
1. Teile Daten in rolling windows (Train → Validation)
2. Trainiere Modell mit Trial-Params auf jedem Train-Window
3. Evaluiere auf nächstem Validation-Window
4. Aggregiere IC/Sharpe über alle Folds → Trial-Score
5. Optuna schlägt nächste Params basierend auf TPE-Posterior vor

PIT-Invariante: Validation-Fenster liegt immer ZEITLICH NACH Train.

Verwendung:
    python scripts/training/walk_forward_hpo.py \\
        --panel output/factor_panels/full_panel_7y.parquet \\
        --label fwd_return_5d \\
        --n-trials 50 \\
        --out models/best_params.json

Graceful degradation: Wenn optuna fehlt → einfache Grid-Search über 8 Kombis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _make_walk_forward_splits(
    timestamps: pd.Series,
    n_splits: int = 5,
    val_window: int = 60,
    embargo_days: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generiert expanding-window walk-forward Splits.

    Returns:
        Liste von (train_indices, val_indices) Tupeln.
    """
    n = len(timestamps)
    if n < (n_splits + 1) * val_window:
        raise ValueError(
            f"Zu wenige Samples ({n}) für {n_splits} Splits × {val_window}-Tage-Window"
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    val_size = val_window
    for i in range(n_splits):
        val_end = n - i * val_size
        val_start = val_end - val_size
        train_end = val_start - embargo_days
        if train_end < val_size:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))

    return list(reversed(splits))


def _evaluate_params(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    feature_cols: list[str],
    model_type: str = "lightgbm",
    n_splits: int = 5,
    val_window: int = 60,
    embargo_days: int = 5,
) -> float:
    """Evaluiert Params via walk-forward → mean IC."""
    splits = _make_walk_forward_splits(
        timestamps, n_splits=n_splits, val_window=val_window, embargo_days=embargo_days,
    )
    X_vals = X[feature_cols].fillna(0.0).values
    y_vals = y.values

    fold_ics: list[float] = []
    for train_idx, val_idx in splits:
        try:
            model = _make_model(model_type, params)
            model.fit(X_vals[train_idx], y_vals[train_idx])
            preds = model.predict(X_vals[val_idx])
            if np.std(preds) < 1e-9:
                fold_ics.append(0.0)
                continue
            corr = np.corrcoef(preds, y_vals[val_idx])[0, 1]
            fold_ics.append(float(corr) if not np.isnan(corr) else 0.0)
        except Exception as exc:
            logger.debug("[WF-HPO] Fold failed: %s", exc)
            fold_ics.append(0.0)

    mean_ic = float(np.mean(fold_ics)) if fold_ics else 0.0
    return mean_ic


def _make_model(model_type: str, params: dict) -> object:
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor  # type: ignore
            return LGBMRegressor(
                n_estimators=params.get("n_estimators", 200),
                learning_rate=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 6),
                min_child_samples=params.get("min_child_samples", 20),
                reg_alpha=params.get("reg_alpha", 0.0),
                reg_lambda=params.get("reg_lambda", 0.0),
                random_state=42,
                verbose=-1,
            )
        except ImportError:
            pass
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=params.get("n_estimators", 100),
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=params.get("max_depth", 3),
        random_state=42,
    )


def run_hpo_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    feature_cols: list[str],
    n_trials: int = 50,
    n_splits: int = 5,
    val_window: int = 60,
) -> dict:
    """Optuna TPE-basierte Walk-Forward HPO."""
    try:
        import optuna  # type: ignore
    except ImportError:
        logger.warning("[WF-HPO] optuna fehlt — Fallback auf Grid-Search")
        return run_hpo_grid(X, y, timestamps, feature_cols, n_splits=n_splits, val_window=val_window)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }
        return _evaluate_params(
            params, X, y, timestamps, feature_cols,
            n_splits=n_splits, val_window=val_window,
        )

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
    }


def run_hpo_grid(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    feature_cols: list[str],
    n_splits: int = 5,
    val_window: int = 60,
) -> dict:
    """Fallback-Grid-Search über eine kleine Parameter-Menge."""
    grid = []
    for n_est in [100, 200, 400]:
        for lr in [0.03, 0.05, 0.1]:
            for depth in [4, 6, 8]:
                grid.append({
                    "n_estimators": n_est,
                    "learning_rate": lr,
                    "max_depth": depth,
                })
    best_params = None
    best_score = -np.inf
    for params in grid:
        score = _evaluate_params(
            params, X, y, timestamps, feature_cols,
            n_splits=n_splits, val_window=val_window,
        )
        if score > best_score:
            best_score = score
            best_params = params
    return {
        "best_params": best_params or {},
        "best_value": float(best_score),
        "n_trials": len(grid),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-Forward HPO via Optuna")
    parser.add_argument("--panel", type=Path, required=True, help="Parquet factor_panel")
    parser.add_argument("--label", default="fwd_return_5d", help="Target column")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-window", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("models/best_params.json"))
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--grid-only", action="store_true", help="Skip optuna, use grid search")
    args = parser.parse_args()

    if not args.panel.exists():
        logger.error("Panel nicht gefunden: %s", args.panel)
        return 1

    panel = pd.read_parquet(args.panel)
    panel = panel.dropna(subset=[args.label])

    feature_cols = [
        c for c in panel.select_dtypes(include="number").columns
        if c != args.label and not c.startswith("fwd_return") and c != args.timestamp_col
    ]
    logger.info(
        "[WF-HPO] Panel %d rows, %d features, target=%s",
        len(panel), len(feature_cols), args.label,
    )

    if args.grid_only:
        result = run_hpo_grid(
            panel[feature_cols], panel[args.label], panel[args.timestamp_col],
            feature_cols, n_splits=args.n_splits, val_window=args.val_window,
        )
    else:
        result = run_hpo_optuna(
            panel[feature_cols], panel[args.label], panel[args.timestamp_col],
            feature_cols, n_trials=args.n_trials,
            n_splits=args.n_splits, val_window=args.val_window,
        )

    logger.info("[WF-HPO] Best score: %.4f", result["best_value"])
    logger.info("[WF-HPO] Best params: %s", result["best_params"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("[OK] Saved: %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
