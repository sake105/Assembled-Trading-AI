"""Integration der neuen ML-Stacks (Regime-Router, Nested-Meta, BMA) in den Signal-Layer.

Zentrale Einstiegspunkte für Pipeline-Nutzer:
- MLSignalPipeline: Orchestriert Primary → Regime-Route → Nested-Meta → Risk-Aware-Combining

Workflow:
1. Primary Model erzeugt rohe Richtungs-Signale
2. RegimeRouter wählt regime-spezifisches Modell
3. NestedMetaLabeler berechnet Confidence + Size-Scale
4. RiskAwareSignalCombiner (optional) aggregiert mehrere Primary-Signale

Alle Komponenten sind OPTIONAL — pipeline fällt auf einfacheres Verhalten zurück,
wenn einzelne Bausteine fehlen.

PIT-Invariante: Inference-Zeit, keine Retrain-Logik hier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MLPipelineOutput:
    """Ausgabe einer vollständigen ML-Signal-Pipeline."""

    primary_signal: pd.Series
    """Roh-Prediction aus Primary-Modell."""

    regime_routed_signal: pd.Series
    """Signal nach Regime-Router (oder primary wenn kein Router)."""

    meta_confidence: pd.Series
    """P(success | primary_signal) ∈ [0, 1]."""

    size_scale: pd.Series
    """Position-Size-Scale ∈ [0, 1]."""

    final_position: pd.Series
    """Finale Position: sign × confidence × size × router-adjustment."""

    regime: str
    """Aktuelles Regime-Label (wenn regime_classifier gesetzt)."""


class MLSignalPipeline:
    """End-to-End ML-Signal-Pipeline für Inference.

    Kombiniert Regime-Router (regime-spezifische Modelle) + Nested-Meta
    (Confidence + Size) + Risk-Aware-Combiner (Multi-Signal-Aggregation).

    Alles optional — jedes Modul gracefully fallback wenn nicht vorhanden.
    """

    def __init__(
        self,
        primary_model: object | None = None,
        regime_router: object | None = None,
        nested_meta: object | None = None,
        risk_combiner: object | None = None,
        regime_classifier: object | None = None,
        combined_regime_classifier: object | None = None,
        feature_cols: list[str] | None = None,
    ) -> None:
        """Args:
            primary_model: sklearn-kompatibles Modell ODER None
            regime_router: RegimeModelRouter-Instanz ODER None
            nested_meta: NestedMetaLabeler-Instanz ODER None
            risk_combiner: RiskAwareSignalCombiner-Instanz ODER None
            regime_classifier: NewsRegimeClassifier-Instanz ODER None (legacy)
            combined_regime_classifier: CombinedRegimeClassifier (Round 7J). Falls
                gesetzt, überschreibt regime_classifier für Regime-Detection.
            feature_cols: Feature-Spalten für primary_model
        """
        self.primary_model = primary_model
        self.regime_router = regime_router
        self.nested_meta = nested_meta
        self.risk_combiner = risk_combiner
        self.regime_classifier = regime_classifier
        self.combined_regime_classifier = combined_regime_classifier
        self.feature_cols = feature_cols or []

    def run(
        self,
        features: pd.DataFrame,
        sentiment_window: pd.DataFrame | None = None,
        primary_signal: pd.Series | None = None,
        context_features: pd.DataFrame | None = None,
        market_returns: pd.Series | None = None,
    ) -> MLPipelineOutput:
        """Führt vollständige ML-Signal-Pipeline aus.

        Args:
            features: Haupt-Feature-DataFrame (für Primary + Regime-Router)
            sentiment_window: News-Sentiment-Window für Regime-Detection (optional)
            primary_signal: Externes Primary-Signal (überschreibt primary_model)
            context_features: Kontext für Nested-Meta (Regime, VIX-Proxy, etc.)

        Returns:
            MLPipelineOutput mit allen Zwischenresultaten.
        """
        # ---------- 1. Primary ----------
        if primary_signal is not None:
            primary = primary_signal.reindex(features.index).fillna(0.0)
        elif self.primary_model is not None:
            try:
                feat_cols = self.feature_cols or [
                    c for c in features.select_dtypes(include="number").columns
                ]
                X_vals = features[feat_cols].fillna(0.0).values
                if hasattr(self.primary_model, "predict_proba"):
                    try:
                        proba = self.primary_model.predict_proba(X_vals)
                        if proba.ndim == 2 and proba.shape[1] == 2:
                            primary = pd.Series(proba[:, 1] * 2 - 1.0, index=features.index)
                        else:
                            primary = pd.Series(self.primary_model.predict(X_vals), index=features.index)
                    except Exception:
                        primary = pd.Series(self.primary_model.predict(X_vals), index=features.index)
                else:
                    primary = pd.Series(self.primary_model.predict(X_vals), index=features.index)
            except Exception as exc:
                logger.warning("[MLPipeline] primary_model failed: %s — Zero-Signal", exc)
                primary = pd.Series(np.zeros(len(features)), index=features.index)
        else:
            logger.debug("[MLPipeline] Kein Primary-Model/Signal — Zero-Signal")
            primary = pd.Series(np.zeros(len(features)), index=features.index)

        # ---------- 2. Regime Detection ----------
        regime = "NEUTRAL"
        if self.combined_regime_classifier is not None:
            try:
                combined_out = self.combined_regime_classifier.predict(
                    sentiment_window=sentiment_window,
                    returns=market_returns,
                )
                regime = combined_out.combined_regime
                logger.debug(
                    "[MLPipeline] Combined regime: %s (news=%s, hmm=%s, agreement=%s)",
                    regime, combined_out.news_regime, combined_out.hmm_regime, combined_out.agreement,
                )
            except Exception as exc:
                logger.debug("[MLPipeline] combined_regime failed: %s", exc)
                regime = "NEUTRAL"
        elif self.regime_classifier is not None and sentiment_window is not None and not sentiment_window.empty:
            try:
                regime = self.regime_classifier.predict(sentiment_window)
            except Exception as exc:
                logger.debug("[MLPipeline] regime_classifier failed: %s", exc)
                regime = "NEUTRAL"

        # ---------- 3. Regime Router ----------
        regime_routed = primary.copy()
        if self.regime_router is not None:
            try:
                regime_routed = self.regime_router.predict(features, regime=regime)
                regime_routed.index = features.index
            except Exception as exc:
                logger.warning("[MLPipeline] regime_router failed: %s — primary-passthrough", exc)

        # ---------- 4. Nested Meta ----------
        confidence = pd.Series(np.ones(len(features)), index=features.index, name="confidence")
        size_scale = pd.Series(np.ones(len(features)), index=features.index, name="size_scale")
        final_position = regime_routed.copy()

        if self.nested_meta is not None:
            try:
                # Merge primary_signal + context into features
                meta_features = features.copy()
                meta_features["primary_signal"] = regime_routed
                meta_features["primary_direction"] = np.sign(regime_routed).astype(int)

                if context_features is not None:
                    for col in context_features.columns:
                        meta_features[col] = context_features[col].reindex(features.index).fillna(0.0)

                pred = self.nested_meta.predict(meta_features, primary_signal=regime_routed)
                confidence = pred.confidence
                size_scale = pred.size_scale
                final_position = pred.final_position
            except Exception as exc:
                logger.warning("[MLPipeline] nested_meta failed: %s — kein Meta-Scaling", exc)

        return MLPipelineOutput(
            primary_signal=primary,
            regime_routed_signal=regime_routed,
            meta_confidence=confidence,
            size_scale=size_scale,
            final_position=final_position,
            regime=regime,
        )

    def run_multi_signal(
        self,
        signal_df: pd.DataFrame,
        sentiment_window: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Multi-Signal-Variante: nutzt risk_combiner für Aggregation.

        Args:
            signal_df: DataFrame mit mehreren Signalen (Spalten = Signal-Namen)
            sentiment_window: für Regime-Detection

        Returns:
            Kombiniertes Signal.
        """
        regime = "NEUTRAL"
        if self.regime_classifier is not None and sentiment_window is not None and not sentiment_window.empty:
            try:
                regime = self.regime_classifier.predict(sentiment_window)
            except Exception:
                regime = "NEUTRAL"

        if self.risk_combiner is not None:
            try:
                return self.risk_combiner.combine(signal_df, current_regime=regime)
            except Exception as exc:
                logger.warning("[MLPipeline] risk_combiner failed: %s — equal-weight fallback", exc)

        # Equal-weight fallback
        return signal_df.mean(axis=1).rename("combined_signal")


__all__ = [
    "MLPipelineOutput",
    "MLSignalPipeline",
]
