"""Nested Meta-Labeling: Zwei-Stufen-Klassifikation.

Kern-Idee: Trennt "Should we trade?" und "How big should the trade be?" in
zwei spezialisierte Modelle statt einem gemischten Signal.

Stufe 1 — Primary Model: Richtung (long/short/neutral)
    Input: Technische + Fundamental + News Features
    Output: primary_signal ∈ [-1, +1]

Stufe 2 — Meta Model 1: "Wird das Primary-Signal erfolgreich sein?"
    Input: Primary-Signal + Kontext-Features (Regime, Vol, Sentiment)
    Output: meta_confidence_1 ∈ [0, 1]

Stufe 3 — Meta Model 2: "Wenn erfolgreich, wie groß sollte die Position sein?"
    Input: Kontext + Erwartung (aus Stufe 1+2)
    Output: position_scale ∈ [0, 1]

Finale Position: sign(primary) × confidence_1 × position_scale × max_pos

Vorteile:
- Bessere Entkopplung von Richtung vs. Size
- Unterschiedliche Kostenfunktionen: Direction = classification, Size = regression
- Robuster bei kleinem Datensatz — weniger Parameter pro Modell

PIT-Invariante: Stufe 2 + 3 werden auf Records mit `closed_at` trainiert
(realisierter Return bekannt), wie bei MetaLabeler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PREFIX = "[NestedMeta]"


@dataclass
class NestedPrediction:
    """Output einer nested-meta-Prediction."""

    primary_signal: pd.Series
    """Richtungs-Score (aus Stufe 1)."""

    confidence: pd.Series
    """P(success | primary_signal) aus Stufe 2."""

    size_scale: pd.Series
    """Position-Size-Scale in [0, 1] aus Stufe 3."""

    final_position: pd.Series
    """sign(primary) × confidence × size_scale."""


class NestedMetaLabeler:
    """Drei gestapelte Modelle für Direction + Confidence + Size.

    Die Primary-Stufe ist EXTERN (meta_model.py oder factor_models.py);
    diese Klasse kümmert sich nur um Stufe 2 + 3.
    """

    DIRECTION_FEATURES = [
        "primary_signal",
        "primary_direction",
    ]
    CONTEXT_FEATURES = [
        "news_sentiment_mean",
        "news_velocity",
        "regime_state",
        "vix_proxy",
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        min_size: float = 0.1,
    ) -> None:
        """Args:
            confidence_threshold: Stufe 2 muss mindestens diesen Wert liefern für Trade.
            min_size: Minimum Position-Size bei aktivem Trade (vermeidet 0-Trades).
        """
        self.confidence_threshold = confidence_threshold
        self.min_size = min_size
        self._confidence_model: object | None = None
        self._size_model: object | None = None

    def fit(
        self,
        training_df: pd.DataFrame,
        label_success_col: str = "success_label",
        label_magnitude_col: str = "magnitude_label",
    ) -> "NestedMetaLabeler":
        """Trainiert beide Stufen gemeinsam.

        Args:
            training_df: DataFrame mit direction_features + context_features +
                         label_success_col (0/1) + label_magnitude_col (magnitude)
            label_success_col: Binäre Spalte "Trade war profitabel?"
            label_magnitude_col: Kontinuierliche Spalte "wie groß war Return?"

        Returns:
            Self
        """
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        feat_cols = self.DIRECTION_FEATURES + self.CONTEXT_FEATURES
        missing = [c for c in feat_cols if c not in training_df.columns]
        if missing:
            logger.warning("%s Fehlende Features: %s — mit 0 gefüllt", _PREFIX, missing)

        X = pd.DataFrame({c: training_df.get(c, pd.Series(np.zeros(len(training_df)))) for c in feat_cols}).fillna(0.0).values

        # Stufe 2: Confidence
        if label_success_col in training_df.columns:
            y_conf = training_df[label_success_col].fillna(0).astype(int).values
            self._confidence_model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42,
            )
            self._confidence_model.fit(X, y_conf)
            logger.info("%s Stufe 2 (Confidence) trainiert: n=%d", _PREFIX, len(y_conf))
        else:
            logger.warning("%s %s fehlt — Stufe 2 nicht trainierbar", _PREFIX, label_success_col)

        # Stufe 3: Size (nur auf erfolgreichen Trades)
        if label_magnitude_col in training_df.columns and label_success_col in training_df.columns:
            success_mask = training_df[label_success_col].fillna(0).astype(int) == 1
            if success_mask.sum() >= 30:
                X_size = X[success_mask.values]
                y_size = np.abs(training_df.loc[success_mask, label_magnitude_col].fillna(0).values)
                self._size_model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42,
                )
                self._size_model.fit(X_size, y_size)
                logger.info("%s Stufe 3 (Size) trainiert: n=%d", _PREFIX, len(y_size))
            else:
                logger.warning(
                    "%s Nur %d erfolgreiche Trades — Stufe 3 nicht trainierbar",
                    _PREFIX, int(success_mask.sum()),
                )

        return self

    def predict(
        self,
        features: pd.DataFrame,
        primary_signal: pd.Series | None = None,
    ) -> NestedPrediction:
        """Vorhersage über alle 3 Stufen.

        Args:
            features: DataFrame mit direction_features + context_features
            primary_signal: Wenn None, wird 'primary_signal' aus features genutzt.

        Returns:
            NestedPrediction
        """
        feat_cols = self.DIRECTION_FEATURES + self.CONTEXT_FEATURES
        X = pd.DataFrame({c: features.get(c, pd.Series(np.zeros(len(features)))) for c in feat_cols}).fillna(0.0).values
        idx = features.index

        primary = primary_signal if primary_signal is not None else features.get(
            "primary_signal",
            pd.Series(np.zeros(len(features)), index=idx),
        )

        # Stufe 2
        if self._confidence_model is not None:
            try:
                conf = self._confidence_model.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("%s Stufe 2 predict failed: %s — conf=0.5", _PREFIX, exc)
                conf = np.full(len(X), 0.5)
        else:
            conf = np.full(len(X), 0.5)
        confidence = pd.Series(conf, index=idx, name="confidence")

        # Stufe 3
        if self._size_model is not None:
            try:
                size_raw = self._size_model.predict(X)  # type: ignore[attr-defined]
                # Normalize: map to [0, 1]
                max_abs = max(np.abs(size_raw).max(), 1e-9)
                size_norm = np.clip(np.abs(size_raw) / max_abs, self.min_size, 1.0)
            except Exception as exc:
                logger.warning("%s Stufe 3 predict failed: %s — size=0.5", _PREFIX, exc)
                size_norm = np.full(len(X), 0.5)
        else:
            size_norm = np.full(len(X), 0.5)
        size = pd.Series(size_norm, index=idx, name="size_scale")

        # Final Position
        gated_conf = confidence.where(confidence >= self.confidence_threshold, 0.0)
        final = np.sign(primary.values) * gated_conf.values * size.values
        final_pos = pd.Series(final, index=idx, name="final_position")

        return NestedPrediction(
            primary_signal=primary,
            confidence=confidence,
            size_scale=size,
            final_position=final_pos,
        )

    def save(self, path) -> None:
        import joblib
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path) -> "NestedMetaLabeler":
        import joblib
        return joblib.load(path)


def build_nested_labels_from_trades(
    trades_df: pd.DataFrame,
    return_col: str = "closed_return",
    direction_col: str = "primary_direction",
    signal_col: str = "primary_signal",
    success_col: str = "success_label",
    magnitude_col: str = "magnitude_label",
) -> pd.DataFrame:
    """Leitet ``success_label`` + ``magnitude_label`` aus geschlossenen Trades ab.

    Ohne diesen Helper erwartet ``NestedMetaLabeler.fit`` zwei Spalten, die der
    learning_store normalerweise nicht schreibt. Hier wird aus realisiertem
    Return + Richtung:

    - ``success_label``: 1 wenn ``sign(return) == sign(direction)``, sonst 0.
      Return = 0 → 0 (kein Erfolg, kein Fehlschlag zählt als miss).
    - ``magnitude_label``: ``abs(return)``.

    Richtung wird aus ``direction_col`` bevorzugt; fehlt sie, wird
    ``sign(signal_col)`` genutzt. Existiert keines von beiden, wird eine
    leere Spalte zurückgegeben (success_label=0 überall).

    Args:
        trades_df: DataFrame mit geschlossenen Trades (closed_at vorhanden).
        return_col: Spalte mit realisiertem Return (z.B. closed_return oder pnl).
        direction_col: Spalte mit Richtung (−1/0/+1), optional.
        signal_col: Fallback-Spalte, wenn direction_col fehlt.
        success_col: Name der zu erzeugenden Success-Spalte.
        magnitude_col: Name der zu erzeugenden Magnitude-Spalte.

    Returns:
        Kopie von ``trades_df`` mit zusätzlichen Spalten ``success_col`` und
        ``magnitude_col``. Bereits vorhandene Spalten werden überschrieben.
    """
    df = trades_df.copy()
    if return_col not in df.columns:
        df[success_col] = 0
        df[magnitude_col] = 0.0
        return df

    rets = pd.to_numeric(df[return_col], errors="coerce").fillna(0.0)
    if direction_col in df.columns:
        direction = pd.to_numeric(df[direction_col], errors="coerce").fillna(0.0)
    elif signal_col in df.columns:
        direction = np.sign(pd.to_numeric(df[signal_col], errors="coerce").fillna(0.0))
    else:
        direction = pd.Series(np.zeros(len(df)), index=df.index)

    ret_sign = np.sign(rets)
    # success: direction and return both non-zero and same sign
    same_sign = (ret_sign == np.sign(direction)) & (ret_sign != 0) & (np.sign(direction) != 0)
    df[success_col] = same_sign.astype(int).values
    df[magnitude_col] = np.abs(rets.values)
    return df


__all__ = [
    "NestedPrediction",
    "NestedMetaLabeler",
    "build_nested_labels_from_trades",
]
