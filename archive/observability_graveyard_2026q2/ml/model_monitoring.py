"""Feature Drift Detection and Model Monitoring (Plan 2.10).

Detects distribution shifts between training and recent data using KS tests.
Triggers retraining when drift exceeds thresholds.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def detect_feature_drift(
    train_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    feature_cols: list[str],
    p_value_threshold: float = 0.01,
) -> dict:
    """Detect feature drift using Kolmogorov-Smirnov test.

    Args:
        train_df: Training data features.
        recent_df: Recent (last 30d) features.
        feature_cols: Columns to test.
        p_value_threshold: KS p-value below this = drift detected.

    Returns:
        Dict with drift_score, drifted_features, n_tested, alert_level.
    """
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return {"drift_score": 0.0, "drifted_features": [], "n_tested": 0, "alert_level": "unknown"}

    drifted = []
    n_tested = 0

    for col in feature_cols:
        if col not in train_df.columns or col not in recent_df.columns:
            continue
        train_vals = train_df[col].dropna().values
        recent_vals = recent_df[col].dropna().values

        if len(train_vals) < 30 or len(recent_vals) < 10:
            continue

        n_tested += 1
        stat, p_val = ks_2samp(train_vals, recent_vals)
        if p_val < p_value_threshold:
            drifted.append({"feature": col, "ks_stat": round(stat, 4), "p_value": round(p_val, 6)})

    drift_score = len(drifted) / max(n_tested, 1)

    if drift_score > 0.50:
        alert_level = "CRITICAL"
    elif drift_score > 0.30:
        alert_level = "WARNING"
    elif drift_score > 0.10:
        alert_level = "INFO"
    else:
        alert_level = "OK"

    if alert_level in ("CRITICAL", "WARNING"):
        logger.warning(
            "[DriftDetection] %s: %.0f%% features drifted (%d/%d)",
            alert_level, drift_score * 100, len(drifted), n_tested,
        )

    return {
        "drift_score": round(drift_score, 4),
        "drifted_features": drifted,
        "n_tested": n_tested,
        "alert_level": alert_level,
        "retrain_recommended": drift_score > 0.30,
    }


def check_shap_drift_monthly(
    model: object,
    panel_df: "pd.DataFrame",
    feature_cols: list,
    timestamp_col: str = "timestamp",
    model_type: str = "auto",
    drift_threshold: float = 0.5,
) -> dict:
    """Monthly SHAP drift check for the autonomous feedback loop.

    Computes SHAP importance per calendar month and detects whether any
    feature has drifted significantly (mean_abs_shap changed > drift_threshold
    relative to the first observed month).

    If SHAP is unavailable, falls back gracefully and returns alert_level="skipped".

    Log prefix: [SHAP-DRIFT]

    Args:
        model: Trained model (sklearn, xgboost, lightgbm, catboost compatible).
        panel_df: Factor panel DataFrame with a timestamp column.
        feature_cols: Feature columns to monitor.
        timestamp_col: Name of the timestamp column (default: "timestamp").
        model_type: Passed to compute_shap_temporal_drift (default: "auto").
        drift_threshold: Relative change in mean_abs_shap to flag as drift (default: 0.5 = 50%).

    Returns:
        Dict with keys:
        - alert_level: "OK" | "WARNING" | "CRITICAL" | "skipped"
        - drifted_features: list of feature names with detected drift
        - n_periods: number of monthly periods evaluated
        - drift_details: list of dicts with per-feature drift info
    """
    _prefix = "[SHAP-DRIFT]"

    try:
        from src.assembled_core.ml.explainability import (  # type: ignore
            compute_shap_temporal_drift,
            SHAP_AVAILABLE,
        )
    except ImportError:
        logger.info("%s explainability module not available — check skipped", _prefix)
        return {
            "alert_level": "skipped",
            "drifted_features": [],
            "n_periods": 0,
            "drift_details": [],
        }

    if not SHAP_AVAILABLE:
        logger.info("%s shap not installed — monthly drift check skipped", _prefix)
        return {
            "alert_level": "skipped",
            "drifted_features": [],
            "n_periods": 0,
            "drift_details": [],
        }

    if panel_df is None or panel_df.empty or not feature_cols:
        logger.warning("%s panel_df empty or no feature_cols — skipped", _prefix)
        return {
            "alert_level": "skipped",
            "drifted_features": [],
            "n_periods": 0,
            "drift_details": [],
        }

    try:
        temporal_df = compute_shap_temporal_drift(
            model=model,
            panel_df=panel_df,
            feature_cols=feature_cols,
            timestamp_col=timestamp_col,
            model_type=model_type,
            freq="ME",
        )
    except Exception as exc:
        logger.warning("%s compute_shap_temporal_drift failed: %s", _prefix, exc)
        return {
            "alert_level": "skipped",
            "drifted_features": [],
            "n_periods": 0,
            "drift_details": [],
        }

    if temporal_df.empty:
        logger.info("%s no monthly periods computed", _prefix)
        return {
            "alert_level": "OK",
            "drifted_features": [],
            "n_periods": 0,
            "drift_details": [],
        }

    periods = temporal_df["period"].unique()
    n_periods = len(periods)

    if n_periods < 2:
        logger.info("%s only %d period(s) — need >= 2 to detect drift", _prefix, n_periods)
        return {
            "alert_level": "OK",
            "drifted_features": [],
            "n_periods": n_periods,
            "drift_details": [],
        }

    # Pivot to (period x feature) matrix
    pivot = temporal_df.pivot_table(
        index="period", columns="feature", values="mean_abs_shap"
    )

    drifted_features: list = []
    drift_details: list = []

    first_period_vals = pivot.iloc[0]
    last_period_vals = pivot.iloc[-1]

    for feat in pivot.columns:
        base = float(first_period_vals.get(feat, 0.0) or 0.0)
        current = float(last_period_vals.get(feat, 0.0) or 0.0)
        if base < 1e-9:
            continue  # skip near-zero base importance
        relative_change = abs(current - base) / base
        drifted = relative_change > drift_threshold
        if drifted:
            drifted_features.append(feat)
        drift_details.append(
            {
                "feature": feat,
                "base_shap": round(base, 6),
                "current_shap": round(current, 6),
                "relative_change": round(relative_change, 4),
                "drifted": drifted,
            }
        )

    n_drifted = len(drifted_features)
    drift_ratio = n_drifted / max(len(pivot.columns), 1)

    if drift_ratio > 0.50:
        alert_level = "CRITICAL"
    elif drift_ratio > 0.25:
        alert_level = "WARNING"
    else:
        alert_level = "OK"

    if alert_level in ("CRITICAL", "WARNING"):
        logger.warning(
            "%s %s: %d/%d features show SHAP drift > %.0f%% over %d months",
            _prefix, alert_level, n_drifted, len(pivot.columns),
            drift_threshold * 100, n_periods,
        )
    else:
        logger.info(
            "%s OK: %d/%d features stable over %d months",
            _prefix, len(pivot.columns) - n_drifted, len(pivot.columns), n_periods,
        )

    return {
        "alert_level": alert_level,
        "drifted_features": drifted_features,
        "n_periods": n_periods,
        "drift_details": drift_details,
    }


__all__ = ["detect_feature_drift", "check_shap_drift_monthly"]
