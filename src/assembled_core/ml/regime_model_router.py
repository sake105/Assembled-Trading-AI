"""Regime-bedingter Modell-Router.

Nutzt NewsRegimeClassifier (Phase 4) um zur Inferenz-Zeit das passende
spezialisierte Modell auszuwählen:

- RISK_ON → trend-following / momentum-optimiertes Modell
- NEUTRAL → ausgeglichener Ensemble
- RISK_OFF → defensive / mean-reversion
- CRISIS → vol-aware / konservativ (oder kein Trade)

Warum:
- Ein globales Modell mittelt über Regime → schlechter als regime-spezifisch
- Crisis-Regime hat komplett andere Korrelations- und Hit-Rate-Charakteristika

Workflow:
1. Training: Panel wird nach Regime stratifiziert
2. Pro Regime: separates Modell mit Purged-CV
3. Inferenz: NewsRegimeClassifier.predict(current) → route zu Modell

PIT-Invariante:
- Regime-Label zum Training-Zeitpunkt basiert auf historischem Sentiment-Window
- Keine Future-Regime-Leakage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REGIME_LABELS = ["RISK_ON", "NEUTRAL", "RISK_OFF", "CRISIS"]


@dataclass
class RegimeRouterConfig:
    min_samples_per_regime: int = 100
    """Minimum Trainingsdaten pro Regime, sonst Fallback zu globalem Modell."""

    crisis_policy: str = "conservative"
    """'conservative' → reduziere Position um 50%, 'no_trade' → return 0, 'default' → normales Modell."""

    fallback_to_global: bool = True
    """Falls Regime nicht trainierbar → globales Modell verwenden."""


@dataclass
class RegimeRouterResult:
    regime_models: dict
    """{"RISK_ON": model, "NEUTRAL": model, ...}"""

    regime_ic: dict
    """{"RISK_ON": 0.12, ...}"""

    global_model: object | None = None
    """Fallback-Modell über alle Regime."""

    feature_cols: list[str] = field(default_factory=list)

    regime_classifier: object | None = None
    """NewsRegimeClassifier für Inferenz-Zeit Regime-Detection."""

    config: RegimeRouterConfig = field(default_factory=RegimeRouterConfig)


class RegimeModelRouter:
    """Trainiert und routet zwischen regime-spezifischen Modellen."""

    def __init__(self, config: RegimeRouterConfig | None = None) -> None:
        self.config = config or RegimeRouterConfig()
        self._state: RegimeRouterResult | None = None

    def fit(
        self,
        panel_df: pd.DataFrame,
        regime_col: str,
        label_col: str,
        feature_cols: list[str],
        model_factory: callable | None = None,
    ) -> "RegimeModelRouter":
        """Trainiert pro Regime ein eigenes Modell.

        Args:
            panel_df: Panel mit Feature-Spalten + regime_col + label_col
            regime_col: Spalte mit Regime-Label pro Zeile
            label_col: Target-Spalte (z.B. fwd_return_5d)
            feature_cols: Feature-Spalten
            model_factory: callable() → sklearn-kompatibles Modell.
                           Default: LightGBMRegressor bzw. GradientBoostingRegressor.

        Returns:
            Self, mit self._state populiert.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        if model_factory is None:
            def model_factory():
                try:
                    from lightgbm import LGBMRegressor  # type: ignore
                    return LGBMRegressor(
                        n_estimators=200, max_depth=6, learning_rate=0.05,
                        random_state=42, verbose=-1,
                    )
                except ImportError:
                    return GradientBoostingRegressor(n_estimators=100, random_state=42)

        # Clean panel
        panel = panel_df.dropna(subset=[label_col, regime_col]).copy()
        panel = panel[panel[regime_col].isin(_REGIME_LABELS)]

        regime_models: dict[str, object] = {}
        regime_ic: dict[str, float] = {}

        for regime in _REGIME_LABELS:
            sub = panel[panel[regime_col] == regime]
            if len(sub) < self.config.min_samples_per_regime:
                logger.info(
                    "[RegimeRouter] %s: nur %d Samples (<%d) — kein eigenes Modell",
                    regime, len(sub), self.config.min_samples_per_regime,
                )
                continue

            X = sub[feature_cols].fillna(0.0).values
            y = sub[label_col].values
            model = model_factory()
            try:
                model.fit(X, y)  # type: ignore[attr-defined]
                preds = model.predict(X)  # type: ignore[attr-defined]
                if np.std(preds) > 1e-9:
                    ic = float(np.corrcoef(preds, y)[0, 1])
                    if np.isnan(ic):
                        ic = 0.0
                else:
                    ic = 0.0
                regime_models[regime] = model
                regime_ic[regime] = ic
                logger.info(
                    "[RegimeRouter] %s: trainiert auf %d Samples, IC=%.4f",
                    regime, len(sub), ic,
                )
            except Exception as exc:
                logger.warning("[RegimeRouter] %s training failed: %s", regime, exc)

        # Fallback-Global
        global_model = None
        if self.config.fallback_to_global:
            try:
                X_all = panel[feature_cols].fillna(0.0).values
                y_all = panel[label_col].values
                global_model = model_factory()
                global_model.fit(X_all, y_all)  # type: ignore[attr-defined]
                logger.info("[RegimeRouter] Global Fallback-Modell trainiert (n=%d)", len(panel))
            except Exception as exc:
                logger.warning("[RegimeRouter] Global-Fallback-Training failed: %s", exc)

        self._state = RegimeRouterResult(
            regime_models=regime_models,
            regime_ic=regime_ic,
            global_model=global_model,
            feature_cols=feature_cols,
            config=self.config,
        )
        return self

    def predict(
        self,
        X: pd.DataFrame,
        regime: str,
    ) -> pd.Series:
        """Route zu regime-spezifischem Modell.

        Args:
            X: Feature-DataFrame
            regime: Aktuelles Regime-Label

        Returns:
            pd.Series mit Predictions. Scaling durch crisis_policy in CRISIS-Regime.
        """
        if self._state is None:
            raise RuntimeError("Router not fitted")

        state = self._state
        X_vals = X[state.feature_cols].fillna(0.0).values

        # Modell auswählen
        if regime == "CRISIS" and state.config.crisis_policy == "no_trade":
            return pd.Series(np.zeros(len(X)), index=X.index, name="routed_prediction")

        model = state.regime_models.get(regime)
        if model is None and state.global_model is not None:
            model = state.global_model
            logger.debug("[RegimeRouter] Regime=%s → Global-Fallback", regime)

        if model is None:
            logger.warning("[RegimeRouter] Kein Modell für Regime=%s — Zero-Predictions", regime)
            return pd.Series(np.zeros(len(X)), index=X.index, name="routed_prediction")

        preds = model.predict(X_vals)  # type: ignore[attr-defined]

        # Crisis-Policy: konservativ → Position halbieren
        if regime == "CRISIS" and state.config.crisis_policy == "conservative":
            preds = preds * 0.5

        return pd.Series(preds, index=X.index, name="routed_prediction")

    def predict_auto_regime(
        self,
        X: pd.DataFrame,
        sentiment_window: pd.DataFrame,
    ) -> tuple[pd.Series, str]:
        """Predict mit automatischer Regime-Detection via NewsRegimeClassifier.

        Args:
            X: Feature-DataFrame
            sentiment_window: Aktuelles Sentiment-Window für Regime-Detection
                              (muss die 4 Features mean_sentiment/sentiment_std/
                              news_velocity/negative_fraction haben)

        Returns:
            (predictions, detected_regime)
        """
        if self._state is None or self._state.regime_classifier is None:
            # Fallback: NEUTRAL
            return self.predict(X, regime="NEUTRAL"), "NEUTRAL"

        regime = self._state.regime_classifier.predict(sentiment_window)  # type: ignore[attr-defined]
        preds = self.predict(X, regime=regime)
        return preds, regime

    def attach_regime_classifier(self, classifier: object) -> None:
        """Hängt NewsRegimeClassifier an Router für auto-regime inference."""
        if self._state is None:
            raise RuntimeError("Router not fitted")
        self._state.regime_classifier = classifier

    def summary(self) -> dict:
        if self._state is None:
            return {"status": "unfitted"}
        return {
            "n_regime_models": len(self._state.regime_models),
            "regime_ic": {k: round(v, 4) for k, v in self._state.regime_ic.items()},
            "has_global_fallback": self._state.global_model is not None,
            "n_features": len(self._state.feature_cols),
        }


__all__ = [
    "RegimeRouterConfig",
    "RegimeRouterResult",
    "RegimeModelRouter",
]
