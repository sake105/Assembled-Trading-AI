"""Feature Drift Detection and Model Monitoring (Plan 2.10).

Detects distribution shifts between training and recent data using KS tests.
Triggers retraining when drift exceeds thresholds.
"""

from __future__ import annotations

import logging

import numpy as np
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


__all__ = ["detect_feature_drift"]
