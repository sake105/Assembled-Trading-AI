"""Stacking-Ensemble — Meta-Learner über Base-Predictions.

Theorie
-------
Wolpert (1992): Statt einzelnen Modellen, kombiniere sie via Meta-Learner.
Wenn Base-Modelle unkorrelierte Fehler haben, reduziert Stacking
Vorhersagevarianz substantiell.

Implementation
--------------
1. K-Fold-CV-Predictions für jedes Base-Model auf Training (out-of-fold).
2. Meta-Learner trainiert auf OOF-Predictions als Features.
3. Bei Inference: Base-Models full-fit auf Train -> Meta nimmt deren Predictions.

Wichtig in Time-Series: Purged-K-Fold (siehe ``backtest.cpcv``) statt random
KFold, sonst Look-ahead-Bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BaseModel:
    name: str
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


class StackingRegressor:
    """Ensemble mit OOF-Predictions als Meta-Features."""

    def __init__(
        self,
        base_models: Sequence[BaseModel],
        meta_fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        n_splits: int = 5,
        embargo: int = 0,
        min_train_size: int = 2,
    ) -> None:
        """Stacking regressor with strict-pre-validation time-series folds.

        Args:
            base_models: list of base models with ``fit_predict``.
            meta_fit_predict: meta-learner callable.
            n_splits: number of expanding-window folds.
            embargo: bars to drop immediately before each validation fold
                (audit C4-002 hardening — protects against feature-side
                leakage when features use a rolling lookback).
            min_train_size: skip folds with fewer training rows than this.
        """
        self.base_models = list(base_models)
        self.meta_fit_predict = meta_fit_predict
        self.n_splits = n_splits
        self.embargo = int(embargo)
        self.min_train_size = int(min_train_size)
        self._fitted = False
        self._meta_train_features: np.ndarray | None = None
        self._meta_train_y: np.ndarray | None = None

    def _kfold_indices(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Expanding-window Time-Series K-Fold (strict pre-validation).

        Fix (audit C4-002 / C3-027):
            The previous implementation concatenated ``[0, val_start)`` and
            ``[val_end, n)`` into the training set — i.e. post-validation
            rows leaked into training. Time-series CV forbids that: training
            MUST consist only of bars strictly earlier than the validation
            fold. We additionally honour an optional embargo immediately
            before each fold to protect against rolling-feature leakage.
        """
        fold_size = n // self.n_splits
        out = []
        for k in range(self.n_splits):
            val_start = k * fold_size
            val_end = val_start + fold_size if k < self.n_splits - 1 else n
            val_idx = np.arange(val_start, val_end)
            train_end = max(0, val_start - self.embargo)
            train_idx = np.arange(0, train_end)
            out.append((train_idx, val_idx))
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Erzeuge OOF-Predictions und trainiere Meta-Learner.

        Folds whose training set is shorter than ``min_train_size`` (e.g.
        the very first expanding fold) are skipped; their OOF predictions
        remain at the zero-init value. The meta-learner sees those rows
        but can be masked downstream.
        """
        n = len(X)
        oof_preds = np.zeros((n, len(self.base_models)))
        oof_mask = np.zeros(n, dtype=bool)
        for tr_idx, va_idx in self._kfold_indices(n):
            if len(tr_idx) < self.min_train_size:
                logger.debug(
                    "[stacking] skipping fold: train size %d < %d",
                    len(tr_idx),
                    self.min_train_size,
                )
                continue
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va = X[va_idx]
            for j, bm in enumerate(self.base_models):
                preds = bm.fit_predict(X_tr, y_tr, X_va)
                oof_preds[va_idx, j] = preds
            oof_mask[va_idx] = True
        self._meta_train_features = oof_preds
        self._meta_train_y = y
        self._oof_mask = oof_mask
        self._fitted = True

    @property
    def oof_mask(self) -> np.ndarray:
        """Boolean mask: True where OOF predictions are valid (fold ran)."""
        if not self._fitted or not hasattr(self, "_oof_mask"):
            raise RuntimeError("call fit() first")
        return self._oof_mask

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        # full-fit each base model on entire training set, then predict X_test
        # we need access to original X_train and y_train; we keep meta_train_features
        # as a reasonable proxy (since we're a stacker, we re-call fit_predict with
        # the original train data).  For simplicity here, we accept X_train and y_train
        # need to be passed via predict_full.
        raise NotImplementedError("use predict_full(X_train, y_train, X_test)")

    def predict_full(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        base_preds_test = np.zeros((len(X_test), len(self.base_models)))
        for j, bm in enumerate(self.base_models):
            base_preds_test[:, j] = bm.fit_predict(X_train, y_train, X_test)
        return self.meta_fit_predict(
            self._meta_train_features, self._meta_train_y, base_preds_test
        )


__all__ = ["BaseModel", "StackingRegressor"]
