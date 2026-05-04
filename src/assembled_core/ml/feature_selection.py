"""Feature selection pipeline with stability filtering.

Implements three complementary selection methods (V5):
1. IC-based pre-screening: Drop features with low predictive power.
2. Collinearity filter: Remove redundant highly-correlated features.
3. Cross-validated stability: Only keep features consistently important across CV folds.

Reference: Lopez de Prado — *Advances in Financial Machine Learning* (Ch. 8, MDA/MDI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Output of the feature selection pipeline."""

    selected_features: list[str]
    dropped_features: dict[str, str]  # feature -> reason
    stability_scores: dict[str, float]  # feature -> stability score (0-1)
    ic_scores: dict[str, float]  # feature -> mean IC
    collinear_pairs: list[tuple[str, str, float]]  # (kept, dropped, corr)


# ---------------------------------------------------------------------------
# 1. IC-based pre-screening
# ---------------------------------------------------------------------------


def ic_prescreen(
    factor_panel: pd.DataFrame,
    forward_return_col: str = "fwd_return_1m",
    min_ic: float = 0.02,
    ic_window: int | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Drop features whose mean absolute cross-sectional IC is below threshold.

    Args:
        factor_panel: Panel with timestamp, symbol, feature columns, and forward return.
        forward_return_col: Name of the forward return column.
        min_ic: Minimum mean |IC| to keep a feature.
        ic_window: If provided, use only the last ic_window timestamps.

    Returns:
        Tuple of (kept_features, ic_scores_dict).
    """
    _SKIP_COLS = {"timestamp", "symbol", "date"}
    _NUMERIC_DTYPES = (np.float64, np.float32, np.int64, np.int32, float, int)

    if forward_return_col not in factor_panel.columns:
        _log.warning(
            "Forward return column '%s' not found — skipping IC prescreen",
            forward_return_col,
        )
        feature_cols = [c for c in factor_panel.columns if c not in _SKIP_COLS]
        return feature_cols, {}

    skip = _SKIP_COLS | {forward_return_col}
    feature_cols = [
        c
        for c in factor_panel.columns
        if c not in skip and factor_panel[c].dtype in _NUMERIC_DTYPES
    ]

    panel = factor_panel
    if ic_window and "timestamp" in panel.columns:
        ts_unique = panel["timestamp"].sort_values().unique()
        if len(ts_unique) > ic_window:
            cutoff = ts_unique[-ic_window]
            panel = panel[panel["timestamp"] >= cutoff]

    ic_scores: dict[str, float] = {}
    for feat in feature_cols:
        if "timestamp" in panel.columns:
            ic_per_ts = (
                panel.groupby("timestamp")
                .apply(
                    lambda g: g[feat].corr(g[forward_return_col]),
                    include_groups=False,
                )
                .dropna()
            )
        else:
            ic_per_ts = pd.Series([panel[feat].corr(panel[forward_return_col])])
        ic_scores[feat] = float(ic_per_ts.abs().mean()) if len(ic_per_ts) > 0 else 0.0

    kept = [f for f in feature_cols if ic_scores.get(f, 0.0) >= min_ic]
    _log.info(
        "IC prescreen: %d/%d features pass (min_ic=%.3f)",
        len(kept),
        len(feature_cols),
        min_ic,
    )
    return kept, ic_scores


# ---------------------------------------------------------------------------
# 2. Collinearity filter
# ---------------------------------------------------------------------------


def collinearity_filter(
    factor_panel: pd.DataFrame,
    features: list[str],
    ic_scores: dict[str, float],
    max_corr: float = 0.85,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Remove the weaker member of highly correlated feature pairs.

    Args:
        factor_panel: Panel containing feature columns.
        features: Features to check.
        ic_scores: Feature IC scores (higher = better).
        max_corr: Maximum pairwise correlation before dropping.

    Returns:
        Tuple of (kept_features, dropped_pairs).
    """
    if len(features) < 2:
        return features, []

    corr_matrix = factor_panel[features].corr().abs()
    to_drop: set[str] = set()
    dropped_pairs: list[tuple[str, str, float]] = []

    for i, f1 in enumerate(features):
        if f1 in to_drop:
            continue
        for j, f2 in enumerate(features):
            if j <= i or f2 in to_drop:
                continue
            corr_val = corr_matrix.loc[f1, f2]
            if corr_val > max_corr:
                ic1 = ic_scores.get(f1, 0.0)
                ic2 = ic_scores.get(f2, 0.0)
                victim = f2 if ic1 >= ic2 else f1
                to_drop.add(victim)
                winner = f1 if victim == f2 else f2
                dropped_pairs.append((winner, victim, float(corr_val)))

    kept = [f for f in features if f not in to_drop]
    _log.info(
        "Collinearity filter: dropped %d features (max_corr=%.2f)",
        len(to_drop),
        max_corr,
    )
    return kept, dropped_pairs


# ---------------------------------------------------------------------------
# 3. Cross-validated stability filter
# ---------------------------------------------------------------------------


def stability_filter(
    factor_panel: pd.DataFrame,
    features: list[str],
    forward_return_col: str = "fwd_return_1m",
    n_splits: int = 5,
    top_k: int = 30,
    min_stability: float = 0.5,
) -> tuple[list[str], dict[str, float]]:
    """Keep only features that appear in top-K across >min_stability of CV folds.

    Uses IC-based importance (permutation-IC proxy) as the ranking metric.

    Args:
        factor_panel: Panel with features and forward returns.
        features: Candidate features.
        forward_return_col: Target column.
        n_splits: Number of time-series CV splits.
        top_k: Top-K features per fold.
        min_stability: Minimum fraction of folds a feature must appear in top-K.

    Returns:
        Tuple of (stable_features, stability_scores).
    """
    if forward_return_col not in factor_panel.columns or len(features) == 0:
        return features, {f: 1.0 for f in features}

    if "timestamp" not in factor_panel.columns:
        return features, {f: 1.0 for f in features}

    timestamps = sorted(factor_panel["timestamp"].unique())
    n_ts = len(timestamps)
    if n_ts < n_splits * 2:
        _log.warning(
            "Not enough timestamps (%d) for %d-fold stability filter",
            n_ts,
            n_splits,
        )
        return features, {f: 1.0 for f in features}

    fold_size = n_ts // n_splits
    appearance_count: dict[str, int] = {f: 0 for f in features}

    for fold_idx in range(n_splits):
        start_idx = fold_idx * fold_size
        end_idx = min((fold_idx + 1) * fold_size, n_ts)
        fold_ts = timestamps[start_idx:end_idx]
        fold_data = factor_panel[factor_panel["timestamp"].isin(fold_ts)]

        if fold_data.empty:
            continue

        fold_ics: dict[str, float] = {}
        for feat in features:
            if feat in fold_data.columns:
                ic_vals = (
                    fold_data.groupby("timestamp")
                    .apply(
                        lambda g: g[feat].corr(g[forward_return_col]),
                        include_groups=False,
                    )
                    .dropna()
                )
                fold_ics[feat] = (
                    float(ic_vals.abs().mean()) if len(ic_vals) > 0 else 0.0
                )
            else:
                fold_ics[feat] = 0.0

        sorted_feats = sorted(fold_ics, key=fold_ics.get, reverse=True)[:top_k]
        for f in sorted_feats:
            appearance_count[f] += 1

    stability_scores = {f: count / n_splits for f, count in appearance_count.items()}
    stable = [f for f in features if stability_scores.get(f, 0.0) >= min_stability]

    _log.info(
        "Stability filter: %d/%d features stable (min_stability=%.1f, top_k=%d, splits=%d)",
        len(stable),
        len(features),
        min_stability,
        top_k,
        n_splits,
    )
    return stable, stability_scores


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


def run_feature_selection(
    factor_panel: pd.DataFrame,
    forward_return_col: str = "fwd_return_1m",
    min_ic: float = 0.02,
    max_corr: float = 0.85,
    n_splits: int = 5,
    top_k: int = 30,
    min_stability: float = 0.5,
    ic_window: int | None = None,
) -> FeatureSelectionResult:
    """Run the full feature selection pipeline: IC -> collinearity -> stability.

    Args:
        factor_panel: Panel with timestamp, symbol, feature columns, forward_return_col.
        forward_return_col: Name of the forward return column.
        min_ic: Minimum mean |IC| for IC prescreen.
        max_corr: Maximum pairwise correlation for collinearity filter.
        n_splits: Number of CV folds for stability.
        top_k: Top-K features per fold for stability.
        min_stability: Minimum fraction of folds for stability.
        ic_window: Optional IC lookback window.

    Returns:
        FeatureSelectionResult with selected features, dropped features, and scores.
    """
    _SKIP_COLS = {"timestamp", "symbol", "date"}
    _NUMERIC_DTYPES = (np.float64, np.float32, np.int64, np.int32, float, int)

    dropped: dict[str, str] = {}

    skip = _SKIP_COLS | {forward_return_col}
    all_numeric = [
        c
        for c in factor_panel.columns
        if c not in skip and factor_panel[c].dtype in _NUMERIC_DTYPES
    ]

    ic_kept, ic_scores = ic_prescreen(
        factor_panel, forward_return_col, min_ic, ic_window
    )
    for f in all_numeric:
        if f not in ic_kept:
            dropped[f] = f"IC too low ({ic_scores.get(f, 0.0):.4f} < {min_ic})"

    collinear_kept, collinear_pairs = collinearity_filter(
        factor_panel, ic_kept, ic_scores, max_corr
    )
    for _, victim, corr_val in collinear_pairs:
        if victim not in dropped:
            dropped[victim] = f"Collinear (rho={corr_val:.3f} > {max_corr})"

    stable_kept, stability_scores = stability_filter(
        factor_panel, collinear_kept, forward_return_col, n_splits, top_k, min_stability
    )
    for f in collinear_kept:
        if f not in stable_kept and f not in dropped:
            dropped[f] = (
                f"Unstable ({stability_scores.get(f, 0.0):.2f} < {min_stability})"
            )

    result = FeatureSelectionResult(
        selected_features=stable_kept,
        dropped_features=dropped,
        stability_scores=stability_scores,
        ic_scores=ic_scores,
        collinear_pairs=collinear_pairs,
    )

    _log.info(
        "Feature selection complete: %d selected, %d dropped "
        "(IC: %d->%d, collinear: %d->%d, stability: %d->%d)",
        len(stable_kept),
        len(dropped),
        len(all_numeric),
        len(ic_kept),
        len(ic_kept),
        len(collinear_kept),
        len(collinear_kept),
        len(stable_kept),
    )

    return result


def check_factor_diversification(
    weights: dict[str, float],
    *,
    min_effective_n: float = 4.0,
) -> tuple[bool, float]:
    """Factor diversification gate: Herfindahl-based effective N.

    Checks that portfolio factor weights are not over-concentrated.
    effective_N = (sum |w_i|)^2 / sum(w_i^2)

    Args:
        weights: Factor name -> weight mapping.
        min_effective_n: Minimum effective number of factors (default: 4).

    Returns:
        Tuple of (passed: bool, effective_n: float).
    """
    w = np.array([abs(v) for v in weights.values() if v != 0], dtype=float)
    if len(w) == 0:
        _log.warning("[Diversification] No active factors — gate fails")
        return False, 0.0

    sum_abs = w.sum()
    sum_sq = (w**2).sum()
    effective_n = (sum_abs**2) / sum_sq if sum_sq > 1e-15 else 0.0

    passed = effective_n >= min_effective_n
    _log.info(
        "[Diversification] effective_N=%.1f (min=%.1f) -> %s",
        effective_n,
        min_effective_n,
        "PASS" if passed else "FAIL",
    )
    return passed, round(float(effective_n), 2)


def mutual_information_ranking(
    features: pd.DataFrame,
    target: pd.Series,
    top_k: int | None = None,
    discrete_features: bool = False,
) -> list[tuple[str, float]]:
    """Rank features by Mutual Information with target.

    Non-linear alternative to Pearson IC.

    Args:
        features: (T, N) feature DataFrame.
        target: (T,) target returns.
        top_k: Return top-K features. None = all.
        discrete_features: Whether features are discrete.

    Returns:
        List of (feature_name, MI_score) sorted descending.
    """
    cols = list(features.columns)
    X = features.values.astype(float)
    y = np.asarray(target, dtype=float)

    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[valid]
    y_clean = y[valid]

    if len(y_clean) < 30:
        _log.warning("[MI] Insufficient valid data (%d rows)", len(y_clean))
        return [(c, 0.0) for c in cols]

    try:
        from sklearn.feature_selection import mutual_info_regression

        mi_scores = mutual_info_regression(
            X_clean,
            y_clean,
            discrete_features=discrete_features,
            random_state=42,
            n_neighbors=5,
        )
    except ImportError:
        _log.info("[MI] sklearn not available, using Spearman proxy")
        mi_scores = np.array(
            [
                abs(np.corrcoef(X_clean[:, i], y_clean)[0, 1])
                for i in range(X_clean.shape[1])
            ]
        )
        mi_scores = np.nan_to_num(mi_scores, 0.0)

    ranked = sorted(zip(cols, mi_scores), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]

    _log.info("[MI] Top features: %s", [(n, round(s, 4)) for n, s in ranked[:5]])
    return ranked


def conditional_mutual_information(
    features: pd.DataFrame,
    target: pd.Series,
    already_selected: list[str],
    candidate: str,
) -> float:
    """Conditional MI: I(candidate; target | already_selected).

    Fast heuristic: residualize candidate on selected features, then compute
    absolute correlation of residual with target.

    Args:
        features: Feature DataFrame.
        target: Target returns.
        already_selected: Already chosen features.
        candidate: Candidate feature to evaluate.

    Returns:
        Approximate conditional MI score.
    """
    y = np.asarray(target, dtype=float)

    if not already_selected:
        ranked = mutual_information_ranking(features[[candidate]], target)
        return ranked[0][1] if ranked else 0.0

    cand_vals = features[candidate].values.astype(float)
    sel_mean = features[already_selected].mean(axis=1).values.astype(float)

    valid = ~(np.isnan(cand_vals) | np.isnan(y) | np.isnan(sel_mean))
    if valid.sum() < 30:
        return 0.0

    from numpy.linalg import lstsq

    X_sel = features[already_selected].values[valid].astype(float)
    cand_clean = cand_vals[valid]
    y_clean = y[valid]

    coef = lstsq(X_sel, cand_clean, rcond=None)[0]
    residual = cand_clean - X_sel @ coef

    if np.std(residual) < 1e-10:
        return 0.0

    corr = abs(np.corrcoef(residual, y_clean)[0, 1])
    return float(np.nan_to_num(corr, 0.0))


__all__ = [
    "FeatureSelectionResult",
    "ic_prescreen",
    "collinearity_filter",
    "stability_filter",
    "run_feature_selection",
    "check_factor_diversification",
    "mutual_information_ranking",
    "conditional_mutual_information",
]
