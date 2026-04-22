"""Combined Regime Classifier — Ensemble aus News-Sentiment und HMM-Returns.

News-basiertes Regime (Round 4: NewsRegimeClassifier) und Returns-basiertes
HMM-Regime (Round 7J: OnlineHMMRegimeDetector) liefern jeweils eigene Sichten.

Kombi-Ansatz:
- Beide stimmen überein → hohe Konfidenz
- Sie divergieren → Mischmodus, vorsichtigere Position-Scaling
- Bei CRISIS in beiden → maximale Defensive

Labels werden auf kompatible 4-Klassen-Taxonomie gemapped (RISK_ON/NEUTRAL/
RISK_OFF/CRISIS), damit existierende RegimeModelRouter-Wiring reuse.

PIT-Invariante: Beide Einzel-Classifier haben eigene PIT-Schutzmechanismen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# Map HMM labels to NewsRegimeClassifier labels
_HMM_TO_NEWS = {
    "LOW_VOL": "RISK_ON",
    "NORMAL": "NEUTRAL",
    "HIGH_VOL": "CRISIS",
}


@dataclass
class CombinedRegimeOutput:
    news_regime: str
    hmm_regime: str
    combined_regime: str
    agreement: bool
    """True wenn news und hmm identischen regime mappen."""
    confidence: float
    """0.0 (divergent) .. 1.0 (identical)."""


class CombinedRegimeClassifier:
    """Ensemble von News- und HMM-Regime.

    Falls einer fehlt: Single-Source-Fallback.
    """

    def __init__(
        self,
        news_classifier: object | None = None,
        hmm_detector: object | None = None,
    ) -> None:
        self.news_classifier = news_classifier
        self.hmm_detector = hmm_detector

    def predict(
        self,
        sentiment_window: pd.DataFrame | None = None,
        returns: pd.Series | None = None,
    ) -> CombinedRegimeOutput:
        """Inference mit beiden Signalen.

        Args:
            sentiment_window: Input für NewsRegimeClassifier (optional)
            returns: Zeitreihe für HMM (optional)

        Returns:
            CombinedRegimeOutput
        """
        news_regime = "NEUTRAL"
        hmm_regime_mapped = "NEUTRAL"
        hmm_raw = "NORMAL"

        if self.news_classifier is not None and sentiment_window is not None:
            try:
                news_regime = self.news_classifier.predict(sentiment_window)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.debug("[CombinedRegime] news_classifier failed: %s", exc)

        if self.hmm_detector is not None and returns is not None:
            try:
                state = self.hmm_detector.predict_current_regime(returns)  # type: ignore[attr-defined]
                hmm_raw = state.regime_label
                hmm_regime_mapped = _HMM_TO_NEWS.get(hmm_raw, "NEUTRAL")
            except Exception as exc:
                logger.debug("[CombinedRegime] hmm_detector failed: %s", exc)

        agreement = news_regime == hmm_regime_mapped
        combined = news_regime if agreement else self._resolve_disagreement(news_regime, hmm_regime_mapped)
        confidence = 1.0 if agreement else 0.5

        return CombinedRegimeOutput(
            news_regime=news_regime,
            hmm_regime=hmm_raw,
            combined_regime=combined,
            agreement=agreement,
            confidence=confidence,
        )

    def _resolve_disagreement(self, news: str, hmm: str) -> str:
        """Conservative Resolution bei Divergenz."""
        # Priorisiere defensive Signale
        if "CRISIS" in (news, hmm):
            return "CRISIS"
        if "RISK_OFF" in (news, hmm):
            return "RISK_OFF"
        # Beide sind bullish aber unterschiedlich → NEUTRAL
        if "RISK_ON" in (news, hmm) and "NEUTRAL" in (news, hmm):
            return "NEUTRAL"
        return "NEUTRAL"


__all__ = [
    "CombinedRegimeOutput",
    "CombinedRegimeClassifier",
]
