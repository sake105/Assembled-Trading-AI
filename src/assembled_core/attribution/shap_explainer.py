"""SHAP-based feature attribution explainer.

From 38_FEATURE_ATTRIBUTION_DASHBOARD.md §3.3–§3.4.

DimensionExplainer wraps a trained tree model (sklearn/xgboost/lightgbm)
with SHAP TreeExplainer. Falls back gracefully if shap is not installed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

NEWS_SUB_FEATURES: list[str] = [
    "sentiment_score",
    "news_volume_spike",
    "source_quality_weight",
    "headline_uncertainty",
    "topic_cluster_signal",
    "cross_source_corroboration",
]


class DimensionExplainer:
    """SHAP TreeExplainer wrapper for a single composite dimension model.

    Latency: ~1–10 ms per call (TreeExplainer, interventional perturbation).
    Falls back to uniform zero SHAP values if shap is not installed.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_data: "pd.DataFrame",
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self._explainer: Any = None

        try:
            import shap  # type: ignore[import]
            self._explainer = shap.TreeExplainer(
                model,
                data=background_data,
                feature_perturbation="interventional",
            )
        except ImportError:
            logger.warning("shap not installed — DimensionExplainer returns zero SHAP values")

    def explain_single(self, feature_values: dict[str, float]) -> dict[str, float]:
        """Compute SHAP values for a single input.

        Returns:
            {feature_name: shap_value} dict.
            All zeros if shap is not available.
        """
        if self._explainer is None:
            return {name: 0.0 for name in self.feature_names}

        x = np.array([[feature_values.get(name, 0.0) for name in self.feature_names]])
        shap_values = self._explainer.shap_values(x)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return dict(zip(self.feature_names, shap_values[0].tolist()))

    def explain_batch(self, X: "pd.DataFrame") -> "pd.DataFrame":
        """Compute SHAP values for a batch of inputs.

        Returns:
            DataFrame with same index as X, columns = feature_names.
            All zeros if shap is not available.
        """
        import pandas as pd

        if self._explainer is None:
            return pd.DataFrame(
                np.zeros((len(X), len(self.feature_names))),
                columns=self.feature_names,
                index=X.index,
            )

        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return pd.DataFrame(shap_values, columns=self.feature_names, index=X.index)


class NewsDimensionExplainer:
    """SHAP-based explanation for the news composite dimension.

    Wraps DimensionExplainer with the news-specific sub-feature set.
    Requires a trained model and background sample parquet at model_dir.
    """

    from pathlib import Path as _Path

    def __init__(self, model: Any, background_data: "pd.DataFrame") -> None:
        self.explainer = DimensionExplainer(
            model=model,
            feature_names=NEWS_SUB_FEATURES,
            background_data=background_data,
        )

    def explain(self, feature_values: dict[str, float]) -> dict[str, float]:
        """Return SHAP values explaining the news model output."""
        return self.explainer.explain_single(feature_values)

    def top_drivers(self, feature_values: dict[str, float], n: int = 3) -> dict[str, float]:
        """Return the n strongest SHAP drivers (by absolute value)."""
        shap_values = self.explain(feature_values)
        sorted_vals = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return dict(sorted_vals[:n])


__all__ = ["DimensionExplainer", "NewsDimensionExplainer", "NEWS_SUB_FEATURES"]
