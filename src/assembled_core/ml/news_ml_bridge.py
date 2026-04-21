"""News → ML Feature Bridge.

Verbindet die News Engine mit dem ML-Feature-Layer. Drei Komponenten:

1. FinBERT Embedding Extraction + PCA Kompression (32 Komponenten)
2. Event-Type IC-Weight Reader (aus ic_loop.json, für Severity-Gewichtung)
3. News Regime Classifier (KMeans 4 Zustände: RISK_ON / NEUTRAL / RISK_OFF / CRISIS)

PIT-Invariante:
- PCA wird NUR auf historischen Daten gefittet (as_of-Parameter).
- Zur Inferenz ausschließlich transform_embeddings_pca() verwenden.
- get_event_type_ic_weights() liest realisierte historische Returns → PIT-safe.
- NewsRegimeClassifier.fit() wird auf Trainingsdaten aufgerufen, predict() auf aktuellem Window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

_FINBERT_MODEL = "ProsusAI/finbert"
_MODEL_CACHE: dict = {}   # Separater Cache für AutoModel (nicht _PIPELINE_CACHE in nlp_sentiment)
_DEFAULT_PCA_COMPONENTS = 32
_DEFAULT_IC_LOOP_PATH = Path("output/intel/ic_loop.json")

# ---------------------------------------------------------------------------
# Komponente A: FinBERT Embedding Extraktion + PCA
# ---------------------------------------------------------------------------


def _get_embedding_model(model_name: str = _FINBERT_MODEL, device: str | None = None) -> dict:
    """Lädt AutoModel mit output_hidden_states=True. Gecacht per Prozess.

    Unterscheidet sich von _PIPELINE_CACHE in nlp_sentiment.py:
    - Dort: pipeline("text-classification") für Scalar-Sentiment-Scores
    - Hier: AutoModel für Hidden-State-Embeddings (768-dim CLS-Token)
    """
    if model_name not in _MODEL_CACHE:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers und torch erforderlich für Embedding-Extraktion. "
                "pip install 'transformers>=4.35.0' torch"
            ) from exc
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[EmbBridge] Lade %s für Hidden States auf %s", model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(device).eval()
        _MODEL_CACHE[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
        }
    return _MODEL_CACHE[model_name]


def extract_finbert_embeddings(
    texts: list[str],
    model_name: str = _FINBERT_MODEL,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Extrahiert [CLS] Hidden States aus FinBERT letzter Schicht.

    Args:
        texts: Liste von Texten (Headlines, Abstracts, etc.)
        batch_size: Batch-Größe für Inferenz (Default: 32)
        device: "cuda" oder "cpu" (auto-detect wenn None)

    Returns:
        np.ndarray der Form (n_texts, 768).
        Zero-Array wenn transformers nicht installiert (graceful degradation).

    PIT-Hinweis: Nur Inferenz, kein State, PIT-safe per Design.
    """
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    try:
        import torch  # type: ignore
        cache = _get_embedding_model(model_name, device)
        model = cache["model"]
        tokenizer = cache["tokenizer"]
        dev = cache["device"]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            ).to(dev)
            with torch.no_grad():
                outputs = model(**inputs)
            # Letzter Hidden State, CLS-Token (Index 0): (batch, 768)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

        return np.vstack(all_embeddings).astype(np.float32)

    except ImportError:
        logger.debug("[EmbBridge] transformers nicht installiert — Zero-Embeddings zurück")
        return np.zeros((len(texts), 768), dtype=np.float32)
    except Exception as exc:
        logger.warning("[EmbBridge] Embedding-Fehler: %s — Zero-Embeddings zurück", exc)
        return np.zeros((len(texts), 768), dtype=np.float32)


def fit_embedding_pca(
    embeddings: np.ndarray,
    n_components: int = _DEFAULT_PCA_COMPONENTS,
    save_path: Path | None = None,
    as_of: pd.Timestamp | None = None,
) -> object:
    """PCA auf Embedding-Matrix fitten.

    WICHTIG: Nur auf Trainingsdaten fitten (as_of = Train-Cutoff).
    Der as_of-Parameter ist für Logging; der Aufrufer muss embeddings bereits gefiltert haben.
    Für Inferenz ausschließlich transform_embeddings_pca() verwenden.

    Args:
        embeddings: (n_samples, 768) Matrix
        n_components: Anzahl PCA-Komponenten (Default: 32)
        save_path: Wenn gesetzt, PCA-Model nach Fit speichern
        as_of: Nur zur Dokumentation/Logging

    Returns:
        Gefittetes sklearn.decomposition.PCA Objekt
    """
    from sklearn.decomposition import PCA  # type: ignore
    import joblib  # type: ignore

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings)

    var_explained = float(pca.explained_variance_ratio_.sum())
    logger.info(
        "[EmbBridge] PCA(%d) gefittet: %.1f%% Varianz erklärt (as_of=%s)",
        n_components, var_explained * 100, as_of,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, save_path)
        logger.info("[EmbBridge] PCA gespeichert: %s", save_path)

    return pca


def transform_embeddings_pca(
    embeddings: np.ndarray,
    pca: object,
) -> np.ndarray:
    """Embeddings in PCA-Raum projizieren.

    Args:
        embeddings: (n_samples, 768) Matrix
        pca: Gefittetes PCA-Objekt (aus fit_embedding_pca)

    Returns:
        (n_samples, n_components) Array
    """
    return pca.transform(embeddings).astype(np.float32)  # type: ignore[attr-defined]


def load_pca(path: Path) -> object:
    """Gespeicherte PCA aus Disk laden."""
    import joblib  # type: ignore
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Komponente B: Event-Type IC-Gewichte aus ic_loop.json
# ---------------------------------------------------------------------------


def get_event_type_ic_weights(
    ic_loop_path: Path | None = None,
    min_obs: int = 10,
    fallback_weight: float = 1.0,
    weight_min: float = 0.5,
    weight_max: float = 1.5,
) -> dict[str, float]:
    """Liest per-Trigger-Type IC aus ic_loop.json und normalisiert zu Gewichten.

    Normalisierung auf [weight_min, weight_max]:
        weight = weight_min + (IC - IC_min) / (IC_max - IC_min) * (weight_max - weight_min)

    Kein Trigger-Type wird vollständig ausgeschaltet (Minimum weight_min=0.5).
    Starke Prediktoren erhalten max weight_max=1.5.

    Args:
        ic_loop_path: Pfad zu ic_loop.json (Default: output/intel/ic_loop.json)
        min_obs: Minimum Beobachtungen für gültigen IC-Wert
        fallback_weight: Nicht direkt genutzt — Aufrufer nutzt 1.0 bei leerem Dict
        weight_min: Untergrenze (Default: 0.5)
        weight_max: Obergrenze (Default: 1.5)

    Returns:
        Dict {trigger_type: weight_multiplier}
        Leeres Dict wenn ic_loop.json nicht existiert.

    PIT-Hinweis: ic_loop.json enthält nur realisierte historische Returns → PIT-safe.
    """
    path = ic_loop_path or _DEFAULT_IC_LOOP_PATH
    if not path.exists():
        logger.debug("[EmbBridge] ic_loop.json nicht gefunden → leere Gewichte")
        return {}

    try:
        from src.assembled_core.intel.ic_loop import ICTracker  # type: ignore

        tracker = ICTracker(state_path=path)
        report = tracker.compute_report()
        results = report.get("results", {})

        ic_values: dict[str, float] = {}
        for ttype, info in results.items():
            ic = info.get("ic")
            n = info.get("n_obs", 0)
            if ic is not None and n >= min_obs:
                ic_values[ttype] = float(ic)

        if not ic_values:
            return {}

        ic_min = min(ic_values.values())
        ic_max = max(ic_values.values())
        ic_range = ic_max - ic_min

        weights: dict[str, float] = {}
        for ttype, ic in ic_values.items():
            if ic_range < 1e-9:
                weights[ttype] = 1.0
            else:
                normalized = (ic - ic_min) / ic_range
                weights[ttype] = weight_min + normalized * (weight_max - weight_min)

        logger.debug(
            "[EmbBridge] IC-Gewichte geladen: %d Trigger-Types, min=%.2f max=%.2f",
            len(weights),
            min(weights.values()),
            max(weights.values()),
        )
        return weights

    except Exception as exc:
        logger.warning("[EmbBridge] IC-Gewichte Ladefehler: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Komponente C: News Regime Klassifikator
# ---------------------------------------------------------------------------


@dataclass
class NewsRegimeClassifier:
    """Klassifiziert die tägliche Sentiment-Verteilung in 4 Markt-Regime.

    Features:
        mean_sentiment: Mittelwert über Intraday-Sentiment (FinBERT-Score)
        sentiment_std: Sentiment-Volatilität
        news_velocity: Artikelanzahl relativ zum 30-Tage-Durchschnitt
        negative_fraction: Anteil negativer Artikel (finbert_score < -0.3)

    Cluster → Label-Zuweisung nach Fit:
        Höchster mean_sentiment → RISK_ON
        Niedrigster mean_sentiment → CRISIS
        Höchste news_velocity unter den mittleren → RISK_OFF
        Rest → NEUTRAL

    random_state=42 für Reproduzierbarkeit.
    """

    _kmeans: object = field(default=None, repr=False)
    _scaler: object = field(default=None, repr=False)
    _label_map: dict = field(default_factory=dict, repr=False)
    _feature_cols: list = field(default_factory=list, repr=False)

    FEATURE_COLS = [
        "mean_sentiment",
        "sentiment_std",
        "news_velocity",
        "negative_fraction",
    ]

    def fit(self, sentiment_history: pd.DataFrame) -> "NewsRegimeClassifier":
        """Auf historischer Sentiment-History fitten.

        Args:
            sentiment_history: DataFrame mit Spalten:
                mean_sentiment, sentiment_std, news_velocity, negative_fraction
                (täglich, empfohlen >= 60 Tage)

        PIT-Hinweis: Nur auf Trainingsdaten fitten, nie auf Live-Daten.
        """
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        self._feature_cols = self.FEATURE_COLS
        X = sentiment_history[self._feature_cols].dropna().values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scaler = scaler

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        self._kmeans = kmeans

        # Cluster-zu-Label-Zuweisung via mean_sentiment der Cluster-Zentren
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        sentiment_idx = self._feature_cols.index("mean_sentiment")
        velocity_idx = self._feature_cols.index("news_velocity")

        cluster_sentiments = [
            (i, float(centers[i, sentiment_idx])) for i in range(4)
        ]
        cluster_sentiments.sort(key=lambda x: x[1])

        # Niedrigster → CRISIS, Höchster → RISK_ON
        label_map: dict[int, str] = {}
        label_map[cluster_sentiments[0][0]] = "CRISIS"
        label_map[cluster_sentiments[3][0]] = "RISK_ON"

        # Mittlere zwei: höhere Velocity → RISK_OFF
        mid_clusters = [cluster_sentiments[1][0], cluster_sentiments[2][0]]
        velocities = [
            (ci, float(centers[ci, velocity_idx])) for ci in mid_clusters
        ]
        velocities.sort(key=lambda x: x[1], reverse=True)
        label_map[velocities[0][0]] = "RISK_OFF"
        label_map[velocities[1][0]] = "NEUTRAL"

        self._label_map = label_map
        logger.info(
            "[RegimeClassifier] Gefittet: cluster→label=%s",
            {v: k for k, v in label_map.items()},
        )
        return self

    def predict(self, current_window: pd.DataFrame) -> str:
        """Aktuelles Regime vorhersagen.

        Args:
            current_window: Einzelne Zeile oder kleines Window,
                            gleiche Spalten wie bei fit()

        Returns:
            "RISK_ON" | "NEUTRAL" | "RISK_OFF" | "CRISIS"
        """
        if self._kmeans is None or self._scaler is None:
            return "NEUTRAL"

        try:
            row = current_window[self._feature_cols].mean().values.reshape(1, -1)
            row_scaled = self._scaler.transform(row)
            cluster = int(self._kmeans.predict(row_scaled)[0])
            return self._label_map.get(cluster, "NEUTRAL")
        except Exception as exc:
            logger.debug("[RegimeClassifier] predict Fehler: %s → NEUTRAL", exc)
            return "NEUTRAL"

    def save(self, path: Path) -> None:
        import joblib  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("[RegimeClassifier] Gespeichert: %s", path)

    @classmethod
    def load(cls, path: Path) -> "NewsRegimeClassifier":
        import joblib  # type: ignore
        return joblib.load(path)


__all__ = [
    "extract_finbert_embeddings",
    "fit_embedding_pca",
    "transform_embeddings_pca",
    "load_pca",
    "get_event_type_ic_weights",
    "NewsRegimeClassifier",
]
