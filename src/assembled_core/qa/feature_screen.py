# src/assembled_core/qa/feature_screen.py
"""Feature-screen utilities (audit C2-052).

Single function ``mutual_info_screen`` that ranks features by their mutual
information against a target. Wraps ``sklearn.feature_selection.mutual_info_regression``
(Kraskov-Stogbauer-Grassberger kNN estimator) when available, falls back
to a stdlib binned-histogram estimator when sklearn is absent.

The MI estimate is **PIT-blind** by construction (a function of two
columns, no time alignment). Callers are responsible for passing the
already-aligned X and y at the same as_of.

Returns a DataFrame ranked descending by MI so callers can pick the
top-N for further consideration.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ksg_mi_via_sklearn(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Try sklearn KSG-based MI; return None if sklearn is not importable."""
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        return None
    return np.asarray(
        mutual_info_regression(X, y, discrete_features=False, random_state=42)
    )


def _binned_mi_fallback(X: np.ndarray, y: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Histogram MI estimator (Kraskov-Stogbauer-Grassberger is preferred but
    requires sklearn). For each column j compute the 2D histogram of
    (X[:, j], y), turn into joint/marginal probabilities, sum p·log(p/(p_x*p_y)).

    Less precise than KSG-kNN but dependency-free; good enough for
    feature *ranking* (which is what callers actually need).
    """
    n, p = X.shape
    out = np.zeros(p)
    for j in range(p):
        xj = X[:, j]
        # Drop pairs with NaN to keep histogram2d well-defined.
        mask = ~np.isnan(xj) & ~np.isnan(y)
        if mask.sum() < 30:
            out[j] = np.nan
            continue
        xj_m, y_m = xj[mask], y[mask]
        joint, x_edges, y_edges = np.histogram2d(xj_m, y_m, bins=n_bins)
        joint = joint + 1e-12  # avoid log(0)
        joint = joint / joint.sum()
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        out[j] = float(np.sum(joint * np.log(joint / (px * py + 1e-12))))
    return out


def mutual_info_screen(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    top_n: int | None = None,
    prefer_sklearn: bool = True,
) -> pd.DataFrame:
    """Rank columns of X by mutual information against y.

    Args:
        X: feature DataFrame (rows = samples, cols = features). NaNs are
            dropped pair-wise per feature.
        y: target Series, aligned with X by index.
        top_n: optional — return only the top-N features. Default: all.
        prefer_sklearn: if True, use sklearn KSG (more accurate);
            otherwise use the histogram fallback. Auto-falls-back to
            histogram if sklearn is missing.

    Returns:
        DataFrame with columns ``feature`` + ``mi`` + ``rank`` sorted
        descending by ``mi``. Features with NaN MI are listed last with
        rank = -1.
    """
    if y.name is None:
        y = y.rename("target")
    aligned = X.join(y.rename("__y__"), how="inner").dropna(subset=["__y__"])
    if len(aligned) < 30:
        logger.warning(
            "[feature_screen] too few aligned rows (%d) for stable MI estimate",
            len(aligned),
        )
        return pd.DataFrame(columns=["feature", "mi", "rank"])

    feat_cols = [c for c in X.columns if c in aligned.columns]
    X_arr = aligned[feat_cols].to_numpy(dtype=float, na_value=np.nan)
    y_arr = aligned["__y__"].to_numpy(dtype=float)
    # Drop columns that are all-NaN before passing to KSG (sklearn refuses
    # all-NaN inputs). Histograms tolerate per-column NaNs already.
    valid_cols = [j for j in range(X_arr.shape[1]) if not np.isnan(X_arr[:, j]).all()]
    if not valid_cols:
        return pd.DataFrame(columns=["feature", "mi", "rank"])
    X_arr_valid = X_arr[:, valid_cols]
    feat_cols_valid = [feat_cols[j] for j in valid_cols]
    # sklearn KSG cannot handle NaNs at all — clean them.
    X_arr_clean = np.where(np.isnan(X_arr_valid), 0.0, X_arr_valid)

    mi: np.ndarray | None = None
    if prefer_sklearn:
        mi = _ksg_mi_via_sklearn(X_arr_clean, y_arr)
    if mi is None:
        mi = _binned_mi_fallback(X_arr_clean, y_arr)

    df: dict[str, Any] = {
        "feature": feat_cols_valid,
        "mi": list(mi),
    }
    out = pd.DataFrame(df).sort_values("mi", ascending=False, na_position="last")
    out["rank"] = np.where(out["mi"].isna(), -1, np.arange(1, len(out) + 1))
    out = out.reset_index(drop=True)
    if top_n is not None:
        out = out.head(top_n).reset_index(drop=True)
    return out


__all__ = ["mutual_info_screen"]
