"""LIME-based feature explainability wrapper with permutation-importance fallback.

When the `lime` package is unavailable the wrapper falls back to
permutation-importance computed via sklearn utilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LimeExplanation:
    """Result of a single-instance explanation."""

    feature_contributions: dict[str, float]
    source: str  # "lime" | "permutation_fallback" | "zero_fallback"


class LIMEExplainerWrapper:
    """Thin wrapper around LIME tabular explainer with permutation-importance fallback."""

    def __init__(
        self,
        model: Any = None,
        training_data: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        mode: str = "regression",
    ) -> None:
        self._model = model
        self.feature_names = feature_names or []
        self._training_data = training_data
        self._explainer: Any = None
        self._lime_available = False

        if training_data is not None and len(training_data) > 0:
            try:
                from lime.lime_tabular import LimeTabularExplainer  # type: ignore[import]

                self._explainer = LimeTabularExplainer(
                    training_data,
                    feature_names=self.feature_names,
                    mode=mode,
                    random_state=42,
                )
                self._lime_available = True
                logger.debug("[OK] LIMEExplainerWrapper: lime available")
            except ImportError:
                logger.debug(
                    "[SKIP] LIMEExplainerWrapper: lime not installed, using permutation fallback"
                )

    def explain(
        self,
        instance: np.ndarray,
        num_features: int = 10,
    ) -> LimeExplanation:
        """Return explanation for one instance.

        Uses LIME if available + training_data provided; otherwise permutation fallback.
        """
        if (
            self._lime_available
            and self._explainer is not None
            and self._model is not None
        ):
            try:
                predict_fn = (
                    self._model.predict
                    if hasattr(self._model, "predict")
                    else self._model
                )
                exp = self._explainer.explain_instance(
                    instance, predict_fn, num_features=num_features
                )
                return LimeExplanation(
                    feature_contributions=dict(exp.as_list()),
                    source="lime",
                )
            except Exception as exc:
                logger.debug("[WARN] LIME explain_instance failed: %s", exc)

        # Permutation-importance fallback via sklearn when model is available
        if self._model is not None:
            try:
                contribs = self._permutation_contributions(instance)
                return LimeExplanation(
                    feature_contributions=contribs,
                    source="permutation_fallback",
                )
            except Exception as exc:
                logger.debug("[WARN] permutation fallback failed: %s", exc)

        return LimeExplanation(
            feature_contributions={f: 0.0 for f in self.feature_names},
            source="zero_fallback",
        )

    def _permutation_contributions(self, instance: np.ndarray) -> dict[str, float]:
        """Approximate feature importance by ±epsilon perturbation around instance."""
        base = instance.reshape(1, -1)
        baseline_pred = float(np.atleast_1d(self._model.predict(base))[0])
        eps = 1.0
        contribs: dict[str, float] = {}
        for i, fname in enumerate(
            self.feature_names or [f"f{j}" for j in range(len(instance))]
        ):
            perturbed = base.copy()
            perturbed[0, i] += eps
            delta = (
                float(np.atleast_1d(self._model.predict(perturbed))[0]) - baseline_pred
            )
            contribs[fname] = round(delta, 6)
        return contribs

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Any,
        num_features: int = 10,
    ) -> dict[str, float]:
        """Legacy API: return feature importance dict directly."""
        if self._lime_available and self._explainer is not None:
            try:
                exp = self._explainer.explain_instance(
                    instance, predict_fn, num_features=num_features
                )
                return dict(exp.as_list())
            except Exception as exc:
                logger.debug("[WARN] explain_instance failed: %s", exc)
        return {f: 0.0 for f in (self.feature_names or [])}

    def explain_dataframe(
        self,
        df: pd.DataFrame,
        predict_fn: Any,
        num_features: int = 10,
    ) -> list[dict[str, float]]:
        """Explain all rows in a DataFrame."""
        if df is None or df.empty:
            return []
        feat_cols = (
            [c for c in self.feature_names if c in df.columns]
            if self.feature_names
            else list(df.columns)
        )
        return [
            self.explain_instance(row.values, predict_fn, num_features)
            for _, row in df[feat_cols].iterrows()
        ]
