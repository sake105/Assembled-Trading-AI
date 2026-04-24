"""Meta-Labeling für Signal-Filterung (Lopez de Prado, AIFML Chapter 3).

Sekundärklassifikator: Gegeben Primary-Signal + Kontext-Features,
wie hoch ist die Wahrscheinlichkeit, dass das Signal korrekt ist?

Position-Sizing: position = primary_signal × meta_confidence
(wenn meta_confidence < threshold → 0.0, kein Trade)

PIT-Invariante:
- Training ausschließlich auf `closed_at`-Records (realisierter Return bekannt)
- as_of-Parameter filtert alle Records nach diesem Datum
- Minimum 100 Records, sonst ValueError
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PREFIX = "[MetaLabel]"
_DEFAULT_THRESHOLD = 0.55
_MIN_SAMPLES = 100


@dataclass
class MetaLabelRecord:
    primary_signal: float
    primary_direction: int
    news_sentiment_mean: float
    news_velocity: float
    regime_state: int
    vix_proxy: float
    meta_label: int


def _load_records_from_store(
    path: Path,
    as_of: pd.Timestamp | None = None,
) -> list[MetaLabelRecord]:
    """Lädt Trade-Records aus JSONL learning_store.

    PIT-Guard: `closed_at`-Timestamp muss <= as_of sein.
    Nur Records mit `closed_at` UND `closed_return` werden genutzt.
    """
    records: list[MetaLabelRecord] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                closed_at = rec.get("closed_at")
                if not closed_at:
                    continue
                if as_of is not None:
                    closed_ts = pd.Timestamp(closed_at)
                    if closed_ts.tz is None:
                        closed_ts = closed_ts.tz_localize("UTC")
                    as_of_tz = as_of.tz_localize("UTC") if as_of.tz is None else as_of
                    if closed_ts > as_of_tz:
                        continue
                closed_return = rec.get("closed_return") or rec.get("pnl")
                if closed_return is None:
                    continue
                records.append(MetaLabelRecord(
                    primary_signal=float(rec.get("score", 0.0)),
                    primary_direction=int(rec.get("direction", 0) or rec.get("label", 0)),
                    news_sentiment_mean=float(rec.get("news_sentiment_mean", 0.0)),
                    news_velocity=float(rec.get("news_velocity", 0.0)),
                    regime_state=int(rec.get("regime_state", -1)),
                    vix_proxy=float(rec.get("vix_proxy", 0.0)),
                    meta_label=1 if float(closed_return) > 0 else 0,
                ))
            except Exception:
                continue
    return records


class MetaLabeler:
    """Sekundärklassifikator für Signal-Filterung.

    Trainiert auf historischen Trade-Outcomes.
    Verwendet GradientBoostingClassifier (aus sklearn, immer verfügbar).
    """

    FEATURE_NAMES = [
        "primary_signal",
        "primary_direction",
        "news_sentiment_mean",
        "news_velocity",
        "regime_state",
        "vix_proxy",
    ]

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        confidence_threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self._model: object | None = None

    def fit(self, dataset: pd.DataFrame) -> "MetaLabeler":
        """Auf DataFrame mit MetaLabelRecord-Spalten trainieren."""
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore

        X = dataset[self.FEATURE_NAMES].fillna(0.0).values
        y = dataset["meta_label"].values

        self._model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )
        self._model.fit(X, y)  # type: ignore[union-attr]

        train_score = float(self._model.score(X, y))  # type: ignore[union-attr]
        logger.info(
            "%s Trainiert auf %d Records, train_score=%.3f, threshold=%.2f",
            _PREFIX, len(y), train_score, self.confidence_threshold,
        )
        return self

    def predict_confidence(self, features: pd.DataFrame) -> pd.Series:
        """Meta-Konfidenz-Scores vorhersagen.

        Returns:
            Series mit Werten in [0, 1] — Wahrscheinlichkeit dass Primary Signal korrekt.
        """
        if self._model is None:
            return pd.Series(np.full(len(features), 0.5), index=features.index)

        X = features[self.FEATURE_NAMES].fillna(0.0).values
        proba = self._model.predict_proba(X)[:, 1]  # type: ignore[union-attr]
        return pd.Series(proba, index=features.index, name="meta_confidence")

    def scale_position(
        self,
        primary_signal: float,
        meta_confidence: float,
    ) -> float:
        """Position-Scaling durch Meta-Konfidenz.

        Returns:
            primary_signal × meta_confidence wenn confidence >= threshold, sonst 0.0.

        Examples:
            primary=0.8, confidence=0.40 → 0.0 (unter threshold 0.55)
            primary=0.8, confidence=0.70 → 0.56
            primary=-0.5, confidence=0.65 → -0.325 (Short skaliert ebenfalls)
        """
        if meta_confidence < self.confidence_threshold:
            return 0.0
        return primary_signal * meta_confidence

    def save(self, path: Path) -> None:
        import joblib  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("%s Gespeichert: %s", _PREFIX, path)

    @classmethod
    def load(cls, path: Path) -> "MetaLabeler":
        import joblib  # type: ignore
        return joblib.load(path)

    @classmethod
    def from_learning_store(
        cls,
        learning_store_path: Path,
        as_of: pd.Timestamp | None = None,
        confidence_threshold: float = _DEFAULT_THRESHOLD,
    ) -> tuple["MetaLabeler", dict]:
        """Lade und trainiere aus learning_store.jsonl.

        Args:
            learning_store_path: Pfad zur JSONL-Datei
            as_of: PIT-Cutoff (Default: gestern)
            confidence_threshold: Threshold für scale_position

        Returns:
            (labeler, report_dict)

        Raises:
            ValueError: Wenn weniger als _MIN_SAMPLES Records verfügbar
        """
        if as_of is None:
            as_of = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1)

        raw_records = _load_records_from_store(learning_store_path, as_of=as_of)

        if len(raw_records) < _MIN_SAMPLES:
            raise ValueError(
                f"{_PREFIX} Nur {len(raw_records)} Records in learning_store "
                f"(Minimum: {_MIN_SAMPLES}). Mehr Trades sammeln bevor "
                "Meta-Labeler trainiert werden kann."
            )

        dataset = pd.DataFrame([vars(r) for r in raw_records])

        labeler = cls(confidence_threshold=confidence_threshold)
        labeler.fit(dataset)

        hit_rate = float(dataset["meta_label"].mean())
        report = {
            "n_records": len(raw_records),
            "as_of": str(as_of),
            "hit_rate": round(hit_rate, 4),
            "confidence_threshold": confidence_threshold,
            "class_balance": {
                "positive_fraction": round(hit_rate, 4),
                "negative_fraction": round(1.0 - hit_rate, 4),
            },
        }
        return labeler, report


__all__ = [
    "MetaLabelRecord",
    "MetaLabeler",
]
