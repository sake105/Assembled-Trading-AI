"""SHAP explainability for ML models in the trading pipeline.

From 11_FREE_MODELLE.md §11.18.
Pattern: TreeExplainer on LightGBM/XGBoost (fast, exact).
Per-trade Shapley values stored (~KB/trade) → P&L attribution by feature.

Install: pip install shap==0.48
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_shap():
    try:
        import shap
        return shap
    except ImportError:
        logger.warning("shap not installed — pip install shap==0.48")
        return None


def compute_shap_values(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    model_type: str = "tree",
) -> np.ndarray | None:
    """Compute SHAP values for feature importance attribution.

    Args:
        model: Fitted model (LightGBM, XGBoost, RandomForest, etc.)
        X: Feature matrix for which to compute SHAP values
        model_type: 'tree' (fast, exact) | 'kernel' (slow, model-agnostic)

    Returns:
        SHAP values array (n_samples × n_features). None if shap unavailable.
    """
    shap = _try_shap()
    if shap is None:
        return None

    X_arr = X.values if isinstance(X, pd.DataFrame) else X

    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X_arr[:50])

        shap_values = explainer.shap_values(X_arr)
        # For binary classification, shap_values may be a list [neg_class, pos_class]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        return np.asarray(shap_values)
    except Exception as exc:
        logger.debug("SHAP computation failed: %s", exc)
        return None


def feature_attribution_per_trade(
    model: Any,
    X_trade: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame | None:
    """Compute per-trade SHAP values as a DataFrame.

    Args:
        model: Fitted LightGBM/XGBoost model
        X_trade: Feature row(s) for the trade(s) — shape (n_trades, n_features)
        feature_names: Column names (from X_trade if DataFrame)

    Returns:
        DataFrame with SHAP values: index=trade_idx, columns=feature_names.
        Returns None if computation fails.
    """
    names = feature_names or (list(X_trade.columns) if isinstance(X_trade, pd.DataFrame) else None)
    shap_vals = compute_shap_values(model, X_trade)
    if shap_vals is None:
        return None

    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)

    df = pd.DataFrame(shap_vals, columns=names)
    return df


def top_features_by_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    n: int = 10,
    aggregate: str = "mean_abs",
) -> pd.Series:
    """Return top-N features by aggregate SHAP importance.

    Args:
        shap_values: 2D SHAP array (n_samples × n_features)
        feature_names: List of feature names
        n: Number of top features to return
        aggregate: 'mean_abs' (default) | 'sum_abs'

    Returns:
        Series: feature_name → importance, sorted descending.
    """
    if aggregate == "sum_abs":
        importances = np.sum(np.abs(shap_values), axis=0)
    else:
        importances = np.mean(np.abs(shap_values), axis=0)

    s = pd.Series(importances, index=feature_names)
    return s.nlargest(n)


def pnl_attribution_waterfall(
    trade_shap: pd.DataFrame,
    pnl_column: str | None = None,
) -> pd.DataFrame:
    """Compute P&L attribution by feature group using SHAP values.

    Groups features by prefix (e.g. 'news_', 'ta_', 'insider_') and
    computes each group's contribution to the SHAP sum.

    Returns DataFrame: feature_group → shap_contribution, pct_of_total.
    """
    if trade_shap.empty:
        return pd.DataFrame()

    # Mean SHAP per feature across all trades
    mean_shap = trade_shap.mean()

    # Group by prefix
    groups: dict[str, float] = {}
    for feat, val in mean_shap.items():
        prefix = str(feat).split("_")[0] if "_" in str(feat) else "other"
        groups[prefix] = groups.get(prefix, 0.0) + float(val)

    total = sum(abs(v) for v in groups.values()) + 1e-9
    rows = [
        {"feature_group": k, "shap_contribution": v, "pct_of_total": abs(v) / total * 100}
        for k, v in sorted(groups.items(), key=lambda x: abs(x[1]), reverse=True)
    ]
    return pd.DataFrame(rows)


__all__ = [
    "compute_shap_values",
    "feature_attribution_per_trade",
    "top_features_by_shap",
    "pnl_attribution_waterfall",
]
