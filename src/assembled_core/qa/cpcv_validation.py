"""CombinatorialPurgedCV (CPCV) validation via skfolio.

From 11_FREE_MODELLE.md §11.5.
Gold standard for financial ML cross-validation — purges and embargoes to prevent
information leakage between training and test sets.

Install: pip install skfolio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CPCVResult:
    """Results from CPCV cross-validation."""

    n_splits: int
    scores: np.ndarray
    mean_score: float
    std_score: float
    oos_predictions: np.ndarray | None


def _try_skfolio():
    try:
        from skfolio.model_selection import CombinatorialPurgedCV

        return CombinatorialPurgedCV
    except ImportError:
        try:
            # Try alternative import path
            from skfolio.model_selection._combinatorial import CombinatorialPurgedCV

            return CombinatorialPurgedCV
        except ImportError:
            logger.warning("skfolio not installed — pip install skfolio")
            return None


def _try_sklearn_walk_forward():
    """Walk-forward CV as fallback if skfolio unavailable."""
    try:
        from sklearn.model_selection import TimeSeriesSplit

        return TimeSeriesSplit
    except ImportError:
        return None


def combinatorial_purged_cv(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int = 6,
    n_test_splits: int = 2,
    purged_window: int = 5,
    embargo: int = 1,
    scoring: str = "accuracy",
) -> CPCVResult:
    """Combinatorial Purged Cross-Validation for financial ML.

    Args:
        estimator: Sklearn-compatible estimator.
        X: Feature matrix.
        y: Labels.
        n_splits: Total number of folds (k in CPCV(k, p)).
        n_test_splits: Number of test splits per combination (p).
        purged_window: Bars to purge from training near test boundary.
        embargo: Bars to embargo after test split.
        scoring: Scoring metric.

    Returns:
        CPCVResult with OOS scores and predictions.
    """
    CPCV = _try_skfolio()

    if CPCV is not None:
        try:
            from sklearn.model_selection import cross_val_score

            cv = CPCV(
                n_splits=n_splits,
                n_test_splits=n_test_splits,
                purged_window=purged_window,
                embargo=embargo,
            )
            scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
            return CPCVResult(
                n_splits=len(scores),
                scores=scores,
                mean_score=float(scores.mean()),
                std_score=float(scores.std()),
                oos_predictions=None,
            )
        except Exception as exc:
            logger.debug("skfolio CPCV failed, using TimeSeriesSplit fallback: %s", exc)

    # Fallback: TimeSeriesSplit
    return _walk_forward_cv(estimator, X, y, n_splits, scoring)


def _walk_forward_cv(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int,
    scoring: str,
) -> CPCVResult:
    """Walk-forward cross-validation fallback."""
    TSCV = _try_sklearn_walk_forward()
    if TSCV is None:
        logger.warning("sklearn not installed — returning dummy CV result")
        return CPCVResult(0, np.array([]), 0.0, 0.0, None)

    from sklearn.model_selection import cross_val_score

    cv = TSCV(n_splits=n_splits)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    return CPCVResult(
        n_splits=len(scores),
        scores=scores,
        mean_score=float(scores.mean()),
        std_score=float(scores.std()),
        oos_predictions=None,
    )


def walk_forward_oos_score(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "accuracy",
) -> CPCVResult:
    """Simple walk-forward OOS evaluation.

    Args:
        estimator: Sklearn estimator.
        X: Feature matrix (time-sorted).
        y: Labels (time-sorted).
        n_splits: Number of walk-forward splits.
        scoring: Scoring metric.

    Returns:
        CPCVResult with walk-forward scores.
    """
    return _walk_forward_cv(estimator, X, y, n_splits, scoring)


def purged_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    embargo_bars: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Simple purged train/test split with embargo.

    Args:
        X: Feature matrix (time-indexed).
        y: Labels (time-indexed).
        test_size: Fraction of data for test set.
        embargo_bars: Bars to drop between train end and test start.

    Returns:
        X_train, X_test, y_train, y_test
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))
    embargo_end = min(split_idx + embargo_bars, n)

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[embargo_end:]
    y_test = y.iloc[embargo_end:]

    return X_train, X_test, y_train, y_test


def meta_labeling_pipeline(
    X_primary: pd.DataFrame,
    y_primary: pd.Series,
    primary_predictions: pd.Series,
    meta_labels: pd.Series,
    meta_estimator: Any,
    n_splits: int = 5,
) -> CPCVResult:
    """Full meta-labeling validation pipeline.

    Step 1: Primary model already trained, predictions provided.
    Step 2: Train meta-model on primary predictions + features → 0/1 bet-or-skip.
    Step 3: Validate meta-model with CPCV.

    Args:
        X_primary: Feature matrix for the primary model.
        y_primary: Primary labels (not used directly — meta_labels are).
        primary_predictions: Primary model predictions as a feature for meta-model.
        meta_labels: Binary labels (1 = primary was right, 0 = skip).
        meta_estimator: Meta-model estimator.
        n_splits: Number of CV splits.

    Returns:
        CPCVResult for the meta-model.
    """
    # Augment features with primary predictions
    X_meta = X_primary.copy()
    X_meta["primary_pred"] = primary_predictions.reindex(X_meta.index, fill_value=0)

    common_idx = X_meta.index.intersection(meta_labels.index)
    X_meta = X_meta.loc[common_idx]
    y_meta = meta_labels.loc[common_idx]

    return combinatorial_purged_cv(
        meta_estimator,
        X_meta,
        y_meta,
        n_splits=n_splits,
        scoring="roc_auc",
    )


__all__ = [
    "CPCVResult",
    "combinatorial_purged_cv",
    "walk_forward_oos_score",
    "purged_train_test_split",
    "meta_labeling_pipeline",
]
