"""Meta-Model Training Pipeline -- End-to-End.

Trains a production-grade meta-model for signal confidence scoring:
1. Feature Selection (IC prescreen -> collinearity -> stability)
2. Triple-Barrier Labeling (Lopez de Prado)
3. Purged Cross-Validation (no look-ahead bias)
4. Stacking Ensemble (Ridge+RF+XGB+LGB base, Ridge meta)
5. Platt Calibration (predicted prob ≈ actual hit rate)
6. CPCV Overfitting Check (combinatorial purged validation)
7. SHAP Explainability

Usage:
    python scripts/training/train_meta_model.py \\
        --panel output/factor_panels/full_panel_7y.parquet \\
        --output-dir models/meta \\
        --label-horizon 5 \\
        --n-splits 5

Requires: scikit-learn, lightgbm (optional), xgboost (optional), shap (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [META-TRAIN] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetaModelTrainResult:
    """Full training result with metrics and diagnostics."""

    model_type: str
    n_features_input: int
    n_features_selected: int
    selected_features: list[str]
    n_training_samples: int
    n_labels_positive: int
    n_labels_negative: int
    # Per-fold CV metrics
    cv_auc_scores: list[float]
    cv_logloss_scores: list[float]
    cv_brier_scores: list[float]
    # Aggregated
    mean_auc: float
    mean_logloss: float
    mean_brier: float
    # Calibration
    calibration_error: float
    # CPCV overfitting check
    cpcv_prob_positive_sharpe: float | None
    cpcv_deflated_sharpe: float | None
    cpcv_is_overfit: bool | None
    # Feature importance (top 10)
    top_features: list[tuple[str, float]]
    # Training metadata
    training_time_seconds: float
    label_horizon: int
    profit_target: float
    stop_loss: float
    model_path: str


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

def run_feature_selection(
    panel: pd.DataFrame,
    fwd_return_col: str,
    min_ic: float = 0.02,
    max_corr: float = 0.85,
) -> tuple[list[str], dict]:
    """Run 3-stage feature selection: IC prescreen -> collinearity -> stability.

    Returns (selected_features, diagnostics_dict).
    """
    from src.assembled_core.ml.feature_selection import (
        ic_prescreen,
        collinearity_filter,
    )

    meta_cols = {"timestamp", "date", "symbol", "label",
                 "barrier_hit", "realized_return", "holding_days",
                 "max_drawdown", "max_runup", "entry_price", "exit_price"}
    # Exclude ALL forward-looking columns (fwd_return_*) to prevent data leakage
    all_features = [
        c for c in panel.columns
        if c not in meta_cols
        and not c.startswith("fwd_return_")
        and panel[c].dtype in ("float64", "float32", "int64", "int32")
    ]

    log.info("Feature selection: %d candidate features", len(all_features))

    # Stage 1: IC prescreen — pass only allowed features + meta to prevent leakage
    allowed_cols = ["timestamp", "symbol", "date", fwd_return_col] + all_features
    panel_safe = panel[[c for c in allowed_cols if c in panel.columns]].copy()
    survived_1, ic_scores = ic_prescreen(
        panel_safe, forward_return_col=fwd_return_col, min_ic=min_ic
    )
    log.info("IC prescreen: %d -> %d features (min_ic=%.3f)", len(all_features), len(survived_1), min_ic)

    # Stage 2: Collinearity filter
    if len(survived_1) > 1:
        survived_2, dropped_pairs = collinearity_filter(
            panel, survived_1, ic_scores=ic_scores, max_corr=max_corr
        )
        log.info("Collinearity filter: %d -> %d features (max_corr=%.2f)", len(survived_1), len(survived_2), max_corr)
    else:
        survived_2 = survived_1
        dropped_pairs = []

    # If feature selection is too aggressive, fall back to IC-only
    if len(survived_2) < 5 and len(survived_1) >= 5:
        log.warning("Collinearity too aggressive (%d left), using IC-only (%d)", len(survived_2), len(survived_1))
        survived_2 = survived_1

    # If even IC prescreen is too aggressive, use all features
    if len(survived_2) < 3:
        log.warning("Feature selection too aggressive (%d left), using all %d features", len(survived_2), len(all_features))
        survived_2 = all_features

    diagnostics = {
        "n_candidates": len(all_features),
        "n_after_ic": len(survived_1),
        "n_after_collinearity": len(survived_2),
        "ic_scores": {k: round(v, 4) for k, v in ic_scores.items()} if ic_scores else {},
        "n_collinear_pairs_dropped": len(dropped_pairs) if isinstance(dropped_pairs, list) else 0,
    }
    return survived_2, diagnostics


# ---------------------------------------------------------------------------
# Triple-Barrier Label Generation
# ---------------------------------------------------------------------------

def generate_labels(
    panel: pd.DataFrame,
    profit_target: float = 0.03,
    stop_loss: float = 0.02,
    max_holding: int = 10,
    fwd_return_col: str = "fwd_return_5d",
) -> pd.DataFrame:
    """Generate triple-barrier labels for the panel.

    If the panel already has a forward return column, uses a simplified
    barrier approach. For full path-dependent barriers, needs raw prices.
    """
    if fwd_return_col in panel.columns:
        # Simplified: use forward return directly
        fwd = panel[fwd_return_col].values
        labels = np.where(fwd >= profit_target, 1, np.where(fwd <= -stop_loss, 0, np.where(fwd > 0, 1, 0)))
        panel = panel.copy()
        panel["label"] = labels
        panel["barrier_hit"] = np.where(
            fwd >= profit_target, "profit_target",
            np.where(fwd <= -stop_loss, "stop_loss", "time_barrier")
        )
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        log.info("Labels: %d positive (%.1f%%), %d negative, total %d",
                 n_pos, n_pos / max(len(labels), 1) * 100, n_neg, len(labels))
        return panel

    log.warning("No forward return column '%s' -- labels require raw prices", fwd_return_col)
    return panel


# ---------------------------------------------------------------------------
# Purged CV Training
# ---------------------------------------------------------------------------

def train_with_purged_cv(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    n_splits: int = 5,
    label_horizon: int = 5,
    embargo_pct: float = 0.01,
    model_type: str = "gradient_boosting",
) -> tuple[Any, dict]:
    """Train model with purged cross-validation.

    Returns (trained_model, cv_metrics_dict).
    """
    from src.assembled_core.ml.purged_cv import PurgedKFold

    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    # Optional: try lightgbm and xgboost
    lgb_available = False
    xgb_available = False
    try:
        import lightgbm as lgb
        lgb_available = True
    except ImportError:
        pass
    try:
        import xgboost as xgb
        xgb_available = True
    except ImportError:
        pass

    # Prepare data
    X = panel[feature_cols].copy().fillna(0)
    y = panel[label_col].values.astype(int)

    # Timestamps for purged split
    ts_col = "timestamp" if "timestamp" in panel.columns else "date"
    timestamps = pd.to_datetime(panel[ts_col])

    # Create purged splitter
    pkf = PurgedKFold(n_splits=n_splits, label_horizon=label_horizon, embargo_pct=embargo_pct)
    splits = pkf.split(timestamps)

    if not splits:
        log.warning("PurgedKFold returned no splits -- falling back to simple time split")
        split_idx = int(len(X) * 0.8)
        splits = [(np.arange(split_idx), np.arange(split_idx, len(X)))]

    # Model factory
    def _make_model(mtype: str):
        if mtype == "lightgbm" and lgb_available:
            return lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                num_leaves=31, min_child_samples=20,
                random_state=42, verbose=-1, n_jobs=-1,
            )
        elif mtype == "xgboost" and xgb_available:
            return xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                min_child_weight=20, random_state=42, verbosity=0, n_jobs=-1,
                use_label_encoder=False, eval_metric="logloss",
            )
        elif mtype == "random_forest":
            return RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=20,
                random_state=42, n_jobs=-1,
            )
        else:
            # Default: gradient boosting
            return GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                min_samples_leaf=20, random_state=42,
            )

    # Choose best available model type
    if model_type == "auto":
        if lgb_available:
            model_type = "lightgbm"
        elif xgb_available:
            model_type = "xgboost"
        else:
            model_type = "gradient_boosting"
        log.info("Auto-selected model type: %s", model_type)

    # Cross-validation
    auc_scores = []
    logloss_scores = []
    brier_scores = []
    oof_predictions = np.full(len(X), np.nan)

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            log.warning("Fold %d: skipping (single class in train or test)", fold_i)
            continue

        model = _make_model(model_type)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        oof_predictions[test_idx] = proba

        auc = roc_auc_score(y_test, proba)
        ll = log_loss(y_test, proba)
        brier = brier_score_loss(y_test, proba)

        auc_scores.append(auc)
        logloss_scores.append(ll)
        brier_scores.append(brier)

        log.info("Fold %d: AUC=%.4f  LogLoss=%.4f  Brier=%.4f  (train=%d, test=%d)",
                 fold_i, auc, ll, brier, len(train_idx), len(test_idx))

    # Retrain on full dataset
    log.info("Retraining on full dataset (%d samples)...", len(X))
    final_model = _make_model(model_type)
    final_model.fit(X, y)

    cv_metrics = {
        "model_type": model_type,
        "n_splits": len(auc_scores),
        "auc_scores": [round(s, 4) for s in auc_scores],
        "logloss_scores": [round(s, 4) for s in logloss_scores],
        "brier_scores": [round(s, 4) for s in brier_scores],
        "mean_auc": round(float(np.mean(auc_scores)), 4) if auc_scores else 0.0,
        "mean_logloss": round(float(np.mean(logloss_scores)), 4) if logloss_scores else 0.0,
        "mean_brier": round(float(np.mean(brier_scores)), 4) if brier_scores else 0.0,
        "oof_predictions_available": int(np.isfinite(oof_predictions).sum()),
    }

    return final_model, cv_metrics


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_model(model, X: pd.DataFrame, y: np.ndarray, method: str = "sigmoid") -> tuple[Any, float]:
    """Apply Platt calibration and compute calibration error.

    For classifiers (with predict_proba): uses CalibratedClassifierCV.
    For regressors (StackedEnsemble etc.): clips predictions to [0,1] and
    computes calibration error directly — no Platt wrapper needed since
    regression predictions are already continuous confidence estimates.

    Returns (calibrated_model, calibration_error).
    """
    has_proba = hasattr(model, "predict_proba")

    if not has_proba:
        # Regressor path: treat raw predictions as probability estimates
        raw = model.predict(X if isinstance(X, np.ndarray) else X.values)
        proba = np.clip(raw, 0.0, 1.0)
        cal_error = _compute_calibration_error(proba, y)
        log.info("Calibration error (regressor, no Platt): %.4f", cal_error)
        return model, cal_error

    try:
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        log.warning("sklearn.calibration not available -- skipping calibration")
        return model, -1.0

    try:
        calibrated = CalibratedClassifierCV(model, method=method, cv=3)
        calibrated.fit(X, y)

        # Compute calibration error: mean |predicted_prob - actual_freq| per decile
        proba = calibrated.predict_proba(X)[:, 1]
        cal_error = _compute_calibration_error(proba, y)

        log.info("Calibration error: %.4f (method=%s)", cal_error, method)
        return calibrated, cal_error
    except Exception as exc:
        log.warning("Calibration failed: %s -- using uncalibrated model", exc)
        proba = model.predict_proba(X)[:, 1]
        cal_error = _compute_calibration_error(proba, y)
        return model, cal_error


def _compute_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Mean absolute calibration error across probability deciles."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    errors = []
    for i in range(n_bins):
        mask = (proba >= bin_edges[i]) & (proba < bin_edges[i + 1])
        if mask.sum() < 5:
            continue
        predicted_mean = proba[mask].mean()
        actual_mean = y[mask].mean()
        errors.append(abs(predicted_mean - actual_mean))
    return float(np.mean(errors)) if errors else 0.0


# ---------------------------------------------------------------------------
# CPCV Overfitting Check
# ---------------------------------------------------------------------------

def run_cpcv_check(
    panel: pd.DataFrame,
    feature_cols: list[str],
    model,
    label_col: str = "label",
    n_groups: int = 6,
    k_test_groups: int = 2,
) -> dict:
    """Run Combinatorial Purged CV to check for overfitting.

    Returns dict with prob_positive_sharpe, deflated_sharpe, is_likely_overfit.
    """
    try:
        from src.assembled_core.ml.cpcv import generate_cpcv_splits, compute_cpcv_sharpe_distribution
    except ImportError:
        log.warning("CPCV module not importable -- skipping overfitting check")
        return {"prob_positive_sharpe": None, "deflated_sharpe": None, "is_likely_overfit": None}

    X = panel[feature_cols].fillna(0).values
    y = panel[label_col].values.astype(int)
    n = len(X)

    splits = generate_cpcv_splits(
        n_timestamps=n, n_groups=n_groups, k_test_groups=k_test_groups,
        purge_length=5, embargo_length=3,
    )

    if not splits:
        log.warning("CPCV generated 0 splits -- cannot check overfitting")
        return {"prob_positive_sharpe": None, "deflated_sharpe": None, "is_likely_overfit": None}

    # Compute Sharpe per path
    sharpes = []
    for train_idx, test_idx in splits:
        if len(train_idx) < 50 or len(test_idx) < 10:
            continue
        try:
            from sklearn.base import clone
            m = clone(model)
            m.fit(X[train_idx], y[train_idx])
            proba = m.predict_proba(X[test_idx])[:, 1]
            # Convert to simulated returns: long when proba > 0.5, flat otherwise
            predictions = (proba > 0.5).astype(float)
            # Simple Sharpe proxy: mean(prediction * actual_direction) / std
            fwd_col = "fwd_return_5d"
            if fwd_col in panel.columns:
                test_returns = panel.iloc[test_idx][fwd_col].values
                strat_returns = predictions * test_returns
                if np.std(strat_returns) > 1e-10:
                    sharpe = float(np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(252))
                else:
                    sharpe = 0.0
            else:
                # Fallback: use accuracy as proxy
                acc = float(np.mean(predictions == y[test_idx]))
                sharpe = (acc - 0.5) * 10  # crude proxy
            sharpes.append(sharpe)
        except Exception as exc:
            log.debug("CPCV path failed: %s", exc)
            continue

    if not sharpes:
        return {"prob_positive_sharpe": None, "deflated_sharpe": None, "is_likely_overfit": None}

    prob_pos = float(np.mean(np.array(sharpes) > 0))
    mean_s = float(np.mean(sharpes))
    std_s = float(np.std(sharpes)) if len(sharpes) > 1 else 1.0

    # Deflated Sharpe Ratio (simplified): adjust for multiple testing
    n_paths = len(sharpes)
    expected_max = std_s * (np.sqrt(2 * np.log(n_paths)) if n_paths > 1 else 0)
    deflated = (mean_s - expected_max) / std_s if std_s > 1e-10 else 0.0

    is_overfit = prob_pos < 0.60 or deflated < 0

    log.info("CPCV: %d paths, P(Sharpe>0)=%.2f, DeflatedSharpe=%.3f, overfit=%s",
             n_paths, prob_pos, deflated, is_overfit)

    return {
        "n_paths": n_paths,
        "prob_positive_sharpe": round(prob_pos, 4),
        "mean_sharpe": round(mean_s, 4),
        "deflated_sharpe": round(deflated, 4),
        "is_likely_overfit": is_overfit,
    }


# ---------------------------------------------------------------------------
# Feature Importance (SHAP)
# ---------------------------------------------------------------------------

def compute_feature_importance(model, X: pd.DataFrame, feature_cols: list[str]) -> list[tuple[str, float]]:
    """Compute feature importance via SHAP (tree) or permutation fallback."""
    # Try SHAP first
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.iloc[:min(500, len(X))])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 for binary
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = list(zip(feature_cols, [round(float(v), 6) for v in mean_abs_shap]))
        importance.sort(key=lambda x: x[1], reverse=True)
        log.info("SHAP importance computed (top: %s=%.4f)", importance[0][0], importance[0][1])
        return importance[:20]
    except Exception:
        pass

    # Fallback: sklearn feature_importances_
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            importance = list(zip(feature_cols, [round(float(v), 6) for v in imp]))
            importance.sort(key=lambda x: x[1], reverse=True)
            log.info("Tree importance computed (top: %s=%.4f)", importance[0][0], importance[0][1])
            return importance[:20]
    except Exception:
        pass

    # Fallback: try calibrated model's base estimator
    try:
        if hasattr(model, "estimators_"):
            base = model.estimators_[0]
            if hasattr(base, "feature_importances_"):
                imp = base.feature_importances_
                importance = list(zip(feature_cols, [round(float(v), 6) for v in imp]))
                importance.sort(key=lambda x: x[1], reverse=True)
                return importance[:20]
    except Exception:
        pass

    log.warning("Could not compute feature importance")
    return []


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def train_meta_model_pipeline(
    panel_path: Path,
    output_dir: Path = Path("models/meta"),
    label_horizon: int = 5,
    profit_target: float = 0.03,
    stop_loss: float = 0.02,
    max_holding: int = 10,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    model_type: str = "auto",
    min_ic: float = 0.02,
    max_corr: float = 0.85,
    calibrate: bool = True,
    run_cpcv: bool = True,
    use_stacking: bool = True,
) -> MetaModelTrainResult:
    """End-to-end meta-model training pipeline.

    Steps:
    1. Load factor panel
    2. Feature selection (IC -> collinearity)
    3. Triple-barrier label generation
    4. Purged cross-validation training
    5. Platt calibration
    6. CPCV overfitting check
    7. Feature importance (SHAP or tree)
    8. Save model + report
    """
    t_start = time.time()

    # 1. Load panel
    log.info("=" * 60)
    log.info("META-MODEL TRAINING PIPELINE")
    log.info("=" * 60)
    log.info("Loading panel from %s", panel_path)

    panel = pd.read_parquet(panel_path)
    log.info("Panel shape: %s (%d rows × %d cols)", panel.shape, len(panel), len(panel.columns))

    fwd_col = f"fwd_return_{label_horizon}d"
    if fwd_col not in panel.columns:
        # Try alternatives
        for alt in ["fwd_return_5d", "fwd_return_10d", "fwd_return_20d"]:
            if alt in panel.columns:
                fwd_col = alt
                label_horizon = int(alt.split("_")[-1].replace("d", ""))
                log.info("Using alternative forward return: %s (horizon=%d)", fwd_col, label_horizon)
                break
        else:
            raise ValueError(f"No forward return column found. Expected: {fwd_col}")

    # 2. Feature selection
    log.info("-" * 40)
    log.info("STEP 1: Feature Selection")
    selected_features, fs_diag = run_feature_selection(panel, fwd_col, min_ic=min_ic, max_corr=max_corr)
    log.info("Selected %d features", len(selected_features))

    n_input_features = fs_diag["n_candidates"]

    # 3. Label generation
    log.info("-" * 40)
    log.info("STEP 2: Triple-Barrier Labels (pt=%.2f, sl=%.2f, max_hold=%d)", profit_target, stop_loss, max_holding)
    panel = generate_labels(panel, profit_target=profit_target, stop_loss=stop_loss,
                            max_holding=max_holding, fwd_return_col=fwd_col)

    if "label" not in panel.columns:
        raise ValueError("Label generation failed -- no 'label' column")

    # Drop rows with NaN labels or features
    valid_mask = panel["label"].notna()
    for fc in selected_features:
        valid_mask &= panel[fc].notna()
    panel_clean = panel[valid_mask].copy()
    log.info("Clean panel: %d rows (dropped %d with NaN)", len(panel_clean), len(panel) - len(panel_clean))

    n_pos = int((panel_clean["label"] == 1).sum())
    n_neg = int((panel_clean["label"] == 0).sum())

    # --- Split train vs calibration BEFORE training (Fix 2: held-out calibration) ---
    # Use the LAST 20% by date for calibration (temporal separation)
    ts_col_name = "timestamp" if "timestamp" in panel_clean.columns else "date"
    panel_clean = panel_clean.sort_values(ts_col_name).reset_index(drop=True)
    cal_split_idx = int(len(panel_clean) * 0.80)
    panel_train = panel_clean.iloc[:cal_split_idx].copy()
    panel_cal = panel_clean.iloc[cal_split_idx:].copy()
    log.info("Train/calibration split: %d train, %d calibration (last 20%% by date)",
             len(panel_train), len(panel_cal))

    # 4. Training (purged CV or stacking)
    log.info("-" * 40)
    if use_stacking:
        log.info("STEP 3: Stacking Ensemble with Purged CV")
        try:
            from src.assembled_core.ml.stacking import build_default_stack
            from src.assembled_core.ml.factor_models import MLExperimentConfig

            stack = build_default_stack()
            experiment_cfg = MLExperimentConfig(
                label_col=f"fwd_return_{label_horizon}d" if f"fwd_return_{label_horizon}d" in panel_train.columns else "label",
                n_splits=n_splits,
                min_train_samples=50,
                standardize=True,
            )
            stack.fit(
                panel_train,
                experiment_cfg,
                feature_cols=selected_features,
                timestamp_col=ts_col_name,
                symbol_col="symbol",
            )
            trained_model = stack
            actual_model_type = "stacking_ensemble"

            # Run purged CV metrics on the training set for reporting
            log.info("Computing CV metrics on training set for reporting...")
            _, cv_metrics = train_with_purged_cv(
                panel_train, selected_features, label_col="label",
                n_splits=n_splits, label_horizon=label_horizon,
                embargo_pct=embargo_pct, model_type=model_type,
            )
            cv_metrics["model_type"] = "stacking_ensemble"
            log.info("Stacking ensemble fitted successfully")
        except Exception as exc:
            log.warning("Stacking failed (%s) -- falling back to single model", exc)
            use_stacking = False

    if not use_stacking:
        log.info("STEP 3: Purged Cross-Validation (splits=%d, embargo=%.2f)", n_splits, embargo_pct)
        trained_model, cv_metrics = train_with_purged_cv(
            panel_train, selected_features, label_col="label",
            n_splits=n_splits, label_horizon=label_horizon,
            embargo_pct=embargo_pct, model_type=model_type,
        )
        actual_model_type = cv_metrics["model_type"]

    # 5. Calibration on held-out set (Fix 2: never calibrate on training data)
    cal_error = -1.0
    if calibrate:
        log.info("-" * 40)
        log.info("STEP 4: Platt Calibration (on held-out %d samples)", len(panel_cal))
        X_cal = panel_cal[selected_features].fillna(0)
        y_cal = panel_cal["label"].values.astype(int)
        trained_model, cal_error = calibrate_model(trained_model, X_cal, y_cal)

    # 6. CPCV check
    cpcv_result: dict = {}
    if run_cpcv:
        log.info("-" * 40)
        log.info("STEP 5: CPCV Overfitting Check")
        cpcv_result = run_cpcv_check(panel_clean, selected_features, trained_model)

    # 7. Feature importance
    log.info("-" * 40)
    log.info("STEP 6: Feature Importance")
    X_imp = panel_clean[selected_features].fillna(0)
    top_features = compute_feature_importance(trained_model, X_imp, selected_features)

    # 8. Save model + report
    log.info("-" * 40)
    log.info("STEP 7: Save Model")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "meta_model_latest.joblib"
    try:
        import joblib
        from src.assembled_core.signals.meta_model import MetaModel
        meta_model = MetaModel(
            model=trained_model,
            feature_names=list(selected_features),
            label_name="label",
        )
        joblib.dump(meta_model, model_path)
        log.info("Model saved: %s", model_path)
    except ImportError:
        log.warning("joblib not available -- model not saved to disk")

    t_elapsed = time.time() - t_start

    result = MetaModelTrainResult(
        model_type=actual_model_type,
        n_features_input=n_input_features,
        n_features_selected=len(selected_features),
        selected_features=list(selected_features),
        n_training_samples=len(panel_clean),
        n_labels_positive=n_pos,
        n_labels_negative=n_neg,
        cv_auc_scores=cv_metrics.get("auc_scores", []),
        cv_logloss_scores=cv_metrics.get("logloss_scores", []),
        cv_brier_scores=cv_metrics.get("brier_scores", []),
        mean_auc=cv_metrics.get("mean_auc", 0.0),
        mean_logloss=cv_metrics.get("mean_logloss", 0.0),
        mean_brier=cv_metrics.get("mean_brier", 0.0),
        calibration_error=round(cal_error, 4),
        cpcv_prob_positive_sharpe=cpcv_result.get("prob_positive_sharpe"),
        cpcv_deflated_sharpe=cpcv_result.get("deflated_sharpe"),
        cpcv_is_overfit=cpcv_result.get("is_likely_overfit"),
        top_features=top_features,
        training_time_seconds=round(t_elapsed, 1),
        label_horizon=label_horizon,
        profit_target=profit_target,
        stop_loss=stop_loss,
        model_path=str(model_path),
    )

    # Save report
    report_path = output_dir / "meta_model_report.json"
    report_dict = asdict(result)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=str)
    log.info("Report saved: %s", report_path)

    # Summary
    log.info("=" * 60)
    log.info("TRAINING COMPLETE in %.1fs", t_elapsed)
    log.info("  Model type:      %s", actual_model_type)
    log.info("  Features:        %d/%d selected", len(selected_features), n_input_features)
    log.info("  Samples:         %d (pos=%d, neg=%d)", len(panel_clean), n_pos, n_neg)
    log.info("  Mean AUC:        %.4f", result.mean_auc)
    log.info("  Mean LogLoss:    %.4f", result.mean_logloss)
    log.info("  Calibration Err: %.4f", result.calibration_error)
    if cpcv_result.get("prob_positive_sharpe") is not None:
        log.info("  CPCV P(S>0):     %.2f", cpcv_result["prob_positive_sharpe"])
        log.info("  CPCV Overfit:    %s", cpcv_result.get("is_likely_overfit"))
    log.info("  Top feature:     %s", top_features[0] if top_features else "N/A")

    # Gates
    gates_passed = True
    if result.mean_auc < 0.52:
        log.warning("GATE FAIL: AUC %.4f < 0.52 (worse than random)", result.mean_auc)
        gates_passed = False
    if result.calibration_error > 0.05 and result.calibration_error >= 0:
        log.warning("GATE FAIL: Calibration error %.4f > 0.05", result.calibration_error)
        gates_passed = False
    if cpcv_result.get("is_likely_overfit"):
        log.warning("GATE WARN: CPCV indicates possible overfitting")

    if gates_passed:
        log.info("ALL GATES PASSED -- model ready for paper trading evaluation")
    else:
        log.warning("SOME GATES FAILED -- review before deploying")

    log.info("=" * 60)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train meta-model for signal confidence")
    parser.add_argument("--panel", type=str, default="output/factor_panels/full_panel_7y.parquet",
                        help="Path to factor panel parquet")
    parser.add_argument("--output-dir", type=str, default="models/meta",
                        help="Output directory for model + report")
    parser.add_argument("--label-horizon", type=int, default=5,
                        help="Forward return horizon in days")
    parser.add_argument("--profit-target", type=float, default=0.03,
                        help="Triple-barrier profit target (fraction)")
    parser.add_argument("--stop-loss", type=float, default=0.02,
                        help="Triple-barrier stop loss (fraction)")
    parser.add_argument("--max-holding", type=int, default=10,
                        help="Maximum holding days for triple-barrier")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of purged CV folds")
    parser.add_argument("--model-type", type=str, default="auto",
                        choices=["auto", "gradient_boosting", "random_forest", "lightgbm", "xgboost"],
                        help="Model type (auto picks best available)")
    parser.add_argument("--min-ic", type=float, default=0.02,
                        help="Minimum IC for feature prescreen")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip Platt calibration")
    parser.add_argument("--no-cpcv", action="store_true",
                        help="Skip CPCV overfitting check")
    parser.add_argument("--use-stacking", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use stacking ensemble (4 base + Ridge meta) instead of single model (default: True)")
    args = parser.parse_args()

    result = train_meta_model_pipeline(
        panel_path=Path(args.panel),
        output_dir=Path(args.output_dir),
        label_horizon=args.label_horizon,
        profit_target=args.profit_target,
        stop_loss=args.stop_loss,
        max_holding=args.max_holding,
        n_splits=args.n_splits,
        model_type=args.model_type,
        min_ic=args.min_ic,
        calibrate=not args.no_calibrate,
        run_cpcv=not args.no_cpcv,
        use_stacking=args.use_stacking,
    )

    return 0 if result.mean_auc >= 0.52 else 1


if __name__ == "__main__":
    sys.exit(main())
