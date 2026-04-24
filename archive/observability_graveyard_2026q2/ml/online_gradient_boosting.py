"""Online Gradient Boosting / Adaptive Tree für nichtlineare Online-Learning.

Komplementär zu EWRLS (linear) in online_learning.py:
- EWRLS lernt lineare Koeffizienten incrementell
- Hier: Hoeffding-Adaptive-Tree / Adaptive-Random-Forest für nichtlineare Beziehungen

Nutzt river (wenn installiert) für echte streaming-Algorithmen mit:
- Concept-Drift-Detection (ADWIN built-in)
- Incremental Split-Decisions
- Bounded memory (kein unendliches Wachstum)

Graceful degradation: Wenn river fehlt → Mini-Batch-GradientBoosting als Fallback
(nicht true online, aber verwendbar).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OnlineAdaptiveLearner:
    """Online-nichtlineares Lernen via river.Hoeffding-Tree.

    State wird intern in river gehalten (nicht direkt persistierbar ohne pickle).
    Für Persistenz empfiehlt sich externes joblib.dump() auf das ._model Attribut.
    """

    def __init__(
        self,
        model_type: str = "adaptive_tree",
        feature_names: list[str] | None = None,
    ) -> None:
        """Args:
            model_type: 'adaptive_tree' (HoeffdingAdaptiveTreeRegressor) oder
                        'adaptive_forest' (AdaptiveRandomForestRegressor)
            feature_names: Feature-Spaltennamen für dict-basierte river-Interface
        """
        self.model_type = model_type
        self.feature_names = feature_names or []
        self._model: object | None = None
        self._available = False
        self._buffer: deque = deque(maxlen=1000)  # Fallback-Mini-Batch-Buffer
        self._build_model()

    def _build_model(self) -> None:
        try:
            if self.model_type == "adaptive_tree":
                from river.tree import HoeffdingAdaptiveTreeRegressor  # type: ignore
                self._model = HoeffdingAdaptiveTreeRegressor(seed=42)
            elif self.model_type == "adaptive_forest":
                from river.forest import AdaptiveRandomForestRegressor  # type: ignore
                self._model = AdaptiveRandomForestRegressor(
                    n_models=10, seed=42,
                )
            else:
                raise ValueError(f"Unbekannter model_type: {self.model_type}")
            self._available = True
            logger.info("[OnlineTree] %s initialisiert (river)", self.model_type)
        except ImportError:
            logger.info("[OnlineTree] river fehlt — Mini-Batch-GradientBoosting-Fallback")
            self._available = False

    def _to_river_dict(self, x: np.ndarray | dict) -> dict:
        if isinstance(x, dict):
            return x
        if self.feature_names and len(x) == len(self.feature_names):
            return dict(zip(self.feature_names, [float(v) for v in x]))
        return {f"f{i}": float(v) for i, v in enumerate(x)}

    def learn_one(self, x: np.ndarray | dict, y: float) -> float:
        """Lernt einen Einzel-Sample. Gibt Prediction-Error zurück."""
        y_hat = self.predict_one(x)
        error = y - y_hat

        if self._available and self._model is not None:
            x_dict = self._to_river_dict(x)
            try:
                self._model.learn_one(x_dict, y)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.debug("[OnlineTree] learn_one failed: %s", exc)
        else:
            # Fallback: in Buffer sammeln
            x_arr = x if isinstance(x, np.ndarray) else np.array(list(self._to_river_dict(x).values()))
            self._buffer.append((x_arr, float(y)))

        return float(error)

    def predict_one(self, x: np.ndarray | dict) -> float:
        """Prediction für Einzel-Sample."""
        if self._available and self._model is not None:
            x_dict = self._to_river_dict(x)
            try:
                pred = self._model.predict_one(x_dict)  # type: ignore[attr-defined]
                return float(pred) if pred is not None else 0.0
            except Exception:
                return 0.0

        # Fallback: Mini-Batch-Modell aus Buffer
        if len(self._buffer) < 10:
            return 0.0
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            X_buf = np.array([rec[0] for rec in self._buffer])
            y_buf = np.array([rec[1] for rec in self._buffer])
            model = GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=42)
            model.fit(X_buf, y_buf)
            x_arr = x if isinstance(x, np.ndarray) else np.array(list(self._to_river_dict(x).values()))
            return float(model.predict(x_arr.reshape(1, -1))[0])
        except Exception:
            return 0.0

    def learn_batch(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Sequential-Update über eine Batch. Gibt Error-Array zurück."""
        errors = np.empty(len(y))
        for i in range(len(y)):
            errors[i] = self.learn_one(X[i], float(y[i]))
        return errors

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Prediction für Batch."""
        preds = np.empty(len(X))
        for i in range(len(X)):
            preds[i] = self.predict_one(X[i])
        return preds

    @property
    def available(self) -> bool:
        return self._available


__all__ = [
    "OnlineAdaptiveLearner",
]
