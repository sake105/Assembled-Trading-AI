"""NLP Sentiment Analysis using FinBERT for financial news texts.

Provides financial domain sentiment scoring via the ProsusAI/finbert
pretrained model. Designed as an optional module — gracefully degrades
when transformers/torch are not installed.

Usage:
    from src.assembled_core.ml.nlp_sentiment import score_news_store, build_finbert_sentiment_factors
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline  # type: ignore

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None  # type: ignore
    pipeline = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore

_FINBERT_MODEL = "ProsusAI/finbert"
_LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# Module-level cache so the model is loaded at most once per process
_PIPELINE_CACHE: dict[str, object] = {}


def _get_pipeline(model_name: str = _FINBERT_MODEL, device: str | None = None):
    """Load or retrieve cached FinBERT sentiment pipeline."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers and torch are required. "
            "Install with: pip install 'transformers>=4.35.0' torch"
        )
    if model_name not in _PIPELINE_CACHE:
        if device is None:
            device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        logger.info("[NLP] Loading %s on %s", model_name, device)
        pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=0 if device == "cuda" else -1,
            top_k=None,
        )
        _PIPELINE_CACHE[model_name] = pipe
    return _PIPELINE_CACHE[model_name]


def score_texts_finbert(
    texts: list[str],
    batch_size: int = 32,
    model_name: str = _FINBERT_MODEL,
    max_length: int = 128,
    device: str | None = None,
) -> list[dict]:
    """Score a list of text strings with FinBERT.

    Args:
        texts: List of text strings (e.g., news headlines)
        batch_size: Inference batch size (default: 32)
        model_name: HuggingFace model identifier (default: ProsusAI/finbert)
        max_length: Max token length per text (default: 128)
        device: "cuda" or "cpu" (default: auto-detect)

    Returns:
        List of dicts with keys:
          - text: original text
          - sentiment: "positive" | "neutral" | "negative"
          - score: float in [-1, 1] (weighted sum of class probabilities)
          - confidence: float in [0, 1] (max class probability)
    """
    if not texts:
        return []

    pipe = _get_pipeline(model_name, device)
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # truncate texts to avoid OOM
        batch_trunc = [t[:512] if isinstance(t, str) else "" for t in batch]
        try:
            raw = pipe(batch_trunc, truncation=True, max_length=max_length)
        except Exception as e:
            logger.warning("[NLP] FinBERT batch %d failed: %s", i // batch_size, e)
            for t in batch:
                results.append({"text": t, "sentiment": "neutral", "score": 0.0, "confidence": 0.0})
            continue

        for text, label_list in zip(batch, raw):
            # label_list is a list of {"label": str, "score": float}
            proba = {item["label"].lower(): item["score"] for item in label_list}
            weighted = (
                proba.get("positive", 0.0) * 1.0
                + proba.get("neutral", 0.0) * 0.0
                + proba.get("negative", 0.0) * -1.0
            )
            dominant = max(proba, key=lambda k: proba[k])
            results.append(
                {
                    "text": text,
                    "sentiment": dominant,
                    "score": float(weighted),
                    "confidence": float(proba.get(dominant, 0.0)),
                }
            )

    return results


def score_news_store(
    news_df: pd.DataFrame,
    text_col: str = "headline",
    symbol_col: str | None = "symbol",
    timestamp_col: str = "published_at",
    batch_size: int = 32,
    model_name: str = _FINBERT_MODEL,
) -> pd.DataFrame:
    """Score a news DataFrame with FinBERT and return enriched DataFrame.

    Args:
        news_df: News DataFrame. Expected columns depend on your news pipeline.
        text_col: Name of the text column to score (default: "headline")
        symbol_col: Optional symbol column for attribution (default: "symbol")
        timestamp_col: Timestamp column name (default: "published_at")
        batch_size: Inference batch size (default: 32)
        model_name: HuggingFace model (default: ProsusAI/finbert)

    Returns:
        Original DataFrame with added columns: finbert_sentiment, finbert_score, finbert_confidence
    """
    if news_df.empty:
        result = news_df.copy()
        result["finbert_sentiment"] = pd.Series(dtype="str")
        result["finbert_score"] = pd.Series(dtype="float64")
        result["finbert_confidence"] = pd.Series(dtype="float64")
        return result

    if text_col not in news_df.columns:
        logger.warning("[NLP] Column '%s' not found in news_df", text_col)
        result = news_df.copy()
        result["finbert_sentiment"] = "neutral"
        result["finbert_score"] = 0.0
        result["finbert_confidence"] = 0.0
        return result

    texts = news_df[text_col].fillna("").tolist()
    scored = score_texts_finbert(texts, batch_size=batch_size, model_name=model_name)

    result = news_df.copy()
    result["finbert_sentiment"] = [s["sentiment"] for s in scored]
    result["finbert_score"] = [s["score"] for s in scored]
    result["finbert_confidence"] = [s["confidence"] for s in scored]
    return result


def build_finbert_sentiment_factors(
    news_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    text_col: str = "headline",
    symbol_col: str = "symbol",
    timestamp_col: str = "published_at",
    price_timestamp_col: str = "timestamp",
    lookback_days: list[int] | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Build FinBERT-based sentiment factors aligned to price panel dates.

    Scores news with FinBERT, then computes rolling-window aggregations
    aligned to the price panel's (symbol, timestamp) grid.

    Returns:
        DataFrame with columns: symbol, timestamp,
          finbert_sentiment_mean_{d}d, finbert_shock_flag_{d}d (for each d in lookback_days)

    Args:
        news_df: News DataFrame with text, symbol, timestamp
        prices_df: Price panel (symbol, timestamp) used for alignment
        text_col: News text column (default: "headline")
        symbol_col: Symbol column in news_df (default: "symbol")
        timestamp_col: Timestamp column in news_df (default: "published_at")
        price_timestamp_col: Timestamp column in prices_df (default: "timestamp")
        lookback_days: Rolling windows in calendar days (default: [5, 20])
        batch_size: FinBERT inference batch size (default: 32)
    """
    if lookback_days is None:
        lookback_days = [5, 20]

    if not TRANSFORMERS_AVAILABLE:
        logger.warning("[NLP] transformers not available — returning empty factors")
        return pd.DataFrame(columns=[symbol_col, price_timestamp_col])

    # Score news
    scored_df = score_news_store(
        news_df,
        text_col=text_col,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        batch_size=batch_size,
    )
    scored_df[timestamp_col] = pd.to_datetime(scored_df[timestamp_col])
    scored_df["_date"] = scored_df[timestamp_col].dt.normalize()

    # Build daily mean score per symbol
    daily = (
        scored_df.groupby([symbol_col, "_date"])["finbert_score"]
        .mean()
        .reset_index()
        .rename(columns={"_date": price_timestamp_col})
    )

    # Align to price panel
    price_dates = prices_df[[symbol_col, price_timestamp_col]].drop_duplicates().copy()
    price_dates[price_timestamp_col] = pd.to_datetime(price_dates[price_timestamp_col])

    merged = price_dates.merge(daily, on=[symbol_col, price_timestamp_col], how="left")
    merged = merged.sort_values([symbol_col, price_timestamp_col])
    merged["finbert_score"] = merged["finbert_score"].fillna(0.0)

    for d in lookback_days:
        col_mean = f"finbert_sentiment_mean_{d}d"
        col_shock = f"finbert_shock_flag_{d}d"
        merged[col_mean] = (
            merged.groupby(symbol_col)["finbert_score"]
            .transform(lambda x: x.rolling(d, min_periods=1).mean())
        )
        # Shock flag: sentiment mean in bottom 10th percentile (cross-sectional) at this date
        def _shock_flag(df: pd.DataFrame) -> pd.Series:
            threshold = df[col_mean].quantile(0.10)
            return (df[col_mean] < threshold).astype(float)

        merged[col_shock] = merged.groupby(price_timestamp_col, group_keys=False).apply(
            lambda g: pd.Series(
                (g[col_mean] < g[col_mean].quantile(0.10)).astype(float).values,
                index=g.index,
            )
        )

    drop_cols = ["finbert_score"] if "finbert_score" in merged.columns else []
    return merged.drop(columns=drop_cols)


def score_news_store_with_embeddings(
    news_df: pd.DataFrame,
    text_col: str = "headline",
    pca_path: Path | str | None = None,
    n_pca_components: int = 32,
) -> pd.DataFrame:
    """score_news_store() + PCA-komprimierte FinBERT-Embedding-Features.

    Calls score_news_store() unverändert, then appends finbert_emb_pc_0 …
    finbert_emb_pc_{n-1}.  Falls transformers nicht installiert oder PCA
    fehlt → graceful degradation (Spalten fehlen einfach).
    """
    result = score_news_store(news_df, text_col=text_col)

    if not TRANSFORMERS_AVAILABLE or text_col not in news_df.columns:
        return result

    try:
        from src.assembled_core.ml.news_ml_bridge import (
            extract_finbert_embeddings,
            load_pca,
            transform_embeddings_pca,
        )

        texts = news_df[text_col].fillna("").tolist()
        embeddings = extract_finbert_embeddings(texts)

        if pca_path and Path(pca_path).exists():
            pca = load_pca(Path(pca_path))
            compressed = transform_embeddings_pca(embeddings, pca)
        else:
            compressed = embeddings[:, :n_pca_components]

        n_comp = compressed.shape[1]
        for i in range(n_comp):
            result[f"finbert_emb_pc_{i}"] = compressed[:, i]

        logger.info("[NLP] %d finbert_emb_pc_* Spalten hinzugefügt", n_comp)
    except Exception as exc:
        logger.debug("[NLP] Embedding-Features übersprungen: %s", exc)

    return result
