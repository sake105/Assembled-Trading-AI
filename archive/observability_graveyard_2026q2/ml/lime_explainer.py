"""LIME (Local Interpretable Model-Agnostic Explanations) Wrapper.

SHAP (bereits in model_monitoring.py) erklärt GLOBALE Feature-Wichtigkeit.
LIME erklärt LOKAL — warum hat das Modell für DIESE Prediction X entschieden?

Anwendung:
- Single-Prediction-Erklärung für Trader/Analyst
- Debugging von Edge-Cases (ausreißer-Predictions)
- Komplementär zu SHAP

Graceful degradation: Wenn `lime` nicht installiert → Rückfall auf
Permutation-basierte lokale Importance (näherungsweise).

PIT-Invariante: LIME operiert auf bereits trainierten Modellen, nur
Inference-Zeit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LIMEExplanation:
    """Lokale Erklärung einer einzelnen Prediction."""

    feature_contributions: list[tuple[str, float]] = field(default_factory=list)
    """Sortiert nach |contribution| desc."""

    predicted_value: float = 0.0
    source: str = "lime"
    """'lime' oder 'permutation_fallback'"""

    def top_features(self, k: int = 5) -> list[tuple[str, float]]:
        return self.feature_contributions[:k]


class LIMEExplainerWrapper:
    """Wrapper um lime.lime_tabular mit graceful fallback."""

    def __init__(
        self,
        model: object,
        feature_names: list[str],
        training_data: pd.DataFrame | np.ndarray | None = None,
        mode: str = "regression",
    ) -> None:
        """Args:
            model: sklearn-kompatibles Modell (muss predict_proba / predict haben)
            feature_names: Featurename-Liste
            training_data: Baseline-Distribution (samples × features). Benötigt für LIME.
            mode: 'regression' oder 'classification'
        """
        self.model = model
        self.feature_names = feature_names
        self.mode = mode
        self._explainer: object | None = None
        self._available = False
        self._training_data = training_data
        self._try_init()

    def _try_init(self) -> None:
        if self._training_data is None:
            logger.debug("[LIME] Kein training_data — fallback mode")
            return
        try:
            from lime.lime_tabular import LimeTabularExplainer  # type: ignore

            X = (
                self._training_data.values
                if isinstance(self._training_data, pd.DataFrame)
                else np.asarray(self._training_data)
            )
            self._explainer = LimeTabularExplainer(
                training_data=X,
                feature_names=self.feature_names,
                mode=self.mode,
                random_state=42,
                discretize_continuous=True,
            )
            self._available = True
            logger.info("[LIME] Initialisiert (mode=%s)", self.mode)
        except ImportError:
            logger.info("[LIME] lime nicht installiert — Permutation-Fallback")
            self._available = False

    def explain(
        self,
        instance: pd.Series | np.ndarray | dict,
        num_features: int = 10,
    ) -> LIMEExplanation:
        """Erklärt eine einzelne Prediction."""
        if isinstance(instance, dict):
            x = np.array([instance.get(f, 0.0) for f in self.feature_names])
        elif isinstance(instance, pd.Series):
            x = np.array([instance.get(f, 0.0) for f in self.feature_names])
        else:
            x = np.asarray(instance, dtype=float).flatten()

        if self._available and self._explainer is not None:
            return self._explain_lime(x, num_features)
        return self._explain_permutation(x, num_features)

    def _explain_lime(self, x: np.ndarray, num_features: int) -> LIMEExplanation:
        try:
            if self.mode == "regression":
                explanation = self._explainer.explain_instance(  # type: ignore[attr-defined]
                    x, self.model.predict, num_features=num_features,  # type: ignore[attr-defined]
                )
            else:
                predict_fn = getattr(self.model, "predict_proba", self.model.predict)  # type: ignore[attr-defined]
                explanation = self._explainer.explain_instance(  # type: ignore[attr-defined]
                    x, predict_fn, num_features=num_features,
                )

            contribs = explanation.as_list()
            pred_val = 0.0
            try:
                pred_val = float(self.model.predict(x.reshape(1, -1))[0])  # type: ignore[attr-defined]
            except Exception:
                pass

            # Sort by |contribution| desc
            contribs_sorted = sorted(contribs, key=lambda t: abs(t[1]), reverse=True)
            return LIMEExplanation(
                feature_contributions=[(str(k), float(v)) for k, v in contribs_sorted],
                predicted_value=pred_val,
                source="lime",
            )
        except Exception as exc:
            logger.warning("[LIME] explain failed (%s) — fallback", exc)
            return self._explain_permutation(x, num_features)

    def _explain_permutation(self, x: np.ndarray, num_features: int) -> LIMEExplanation:
        """Fallback: 1-feature-at-a-time Permutation um baseline."""
        try:
            base_pred = float(self.model.predict(x.reshape(1, -1))[0])  # type: ignore[attr-defined]
        except Exception:
            return LIMEExplanation(source="permutation_fallback")

        contribs: list[tuple[str, float]] = []
        rng = np.random.default_rng(42)

        for i, fname in enumerate(self.feature_names):
            x_perm = x.copy()
            # Perturbe via noise
            noise_scale = abs(x[i]) * 0.5 if abs(x[i]) > 1e-6 else 0.1
            perturbed_preds = []
            for _ in range(5):
                x_perm[i] = x[i] + rng.normal(0, noise_scale)
                try:
                    perturbed_preds.append(float(self.model.predict(x_perm.reshape(1, -1))[0]))  # type: ignore[attr-defined]
                except Exception:
                    perturbed_preds.append(base_pred)
            avg_delta = float(np.mean(perturbed_preds) - base_pred)
            contribs.append((fname, avg_delta))

        contribs_sorted = sorted(contribs, key=lambda t: abs(t[1]), reverse=True)[:num_features]
        return LIMEExplanation(
            feature_contributions=contribs_sorted,
            predicted_value=base_pred,
            source="permutation_fallback",
        )


__all__ = [
    "LIMEExplanation",
    "LIMEExplainerWrapper",
]
