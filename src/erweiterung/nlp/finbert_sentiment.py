"""FinBERT — Finanz-domänen-spezifisches Sentiment.

Modell
------
ProsusAI/finbert — BERT fine-tuned auf Reuters-/Bloomberg-Finanznachrichten.
3 Klassen: positive, negative, neutral.

Repository: https://huggingface.co/ProsusAI/finbert (frei, Apache 2.0).

Performance vs. VADER
---------------------
- VADER: lexikalisch, schnell, auf Tweets trainiert.
- FinBERT: kontextuell, auf Finanztexte spezialisiert, besser für formale News.
- Akademisch: FinBERT outperformt VADER um ~10-15 % F1 auf Finanz-Sentiment.

Performance-Notiz
-----------------
FinBERT-Inference ist ~10-50 ms/Sample auf CPU, ~1-3 ms/Sample auf GPU.
Für 100k+ News-Items also CPU = 1000+ Sekunden.  Für High-Throughput-Pipelines
sollte Batching + ONNX-Export verwendet werden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    label: str  # 'positive', 'negative', 'neutral'
    score: float  # confidence
    polarity: float  # +1 / -1 / 0 mapped numeric


_pipeline = None


def _load_pipeline():
    """Lazy-load FinBERT pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "transformers + torch required: pip install transformers torch"
        ) from e
    _pipeline = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        device=-1,  # CPU; set 0 for GPU
        truncation=True,
        max_length=512,
    )
    return _pipeline


def score_finbert(text: str) -> SentimentResult:
    """Score one text snippet."""
    if not text or not isinstance(text, str):
        return SentimentResult(label="neutral", score=0.0, polarity=0.0)
    pipe = _load_pipeline()
    out = pipe(text[:1024])[0]
    label = out["label"].lower()
    score = float(out["score"])
    polarity = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}.get(label, 0.0)
    return SentimentResult(label=label, score=score, polarity=polarity)


def batch_score_finbert(texts: list[str], batch_size: int = 16) -> pd.DataFrame:
    """Batch FinBERT-Scoring."""
    pipe = _load_pipeline()
    out_rows = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:1024] for t in texts[i : i + batch_size] if isinstance(t, str)]
        if not batch:
            continue
        results = pipe(batch)
        for j, r in enumerate(results):
            label = r["label"].lower()
            score = float(r["score"])
            polarity = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}.get(
                label, 0.0
            )
            out_rows.append(
                {
                    "idx": i + j,
                    "label": label,
                    "score": score,
                    "polarity": polarity,
                    "weighted_polarity": polarity * score,
                }
            )
    return pd.DataFrame(out_rows)


def aggregate_daily_sentiment(
    news_df: pd.DataFrame,
    text_col: str = "headline",
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Aggregiere FinBERT-Sentiment auf Tagesbasis je Symbol.

    Args:
        news_df: DataFrame [date, symbol, headline].
        text_col, date_col, symbol_col: column names.

    Returns:
        DataFrame [date, symbol, mean_polarity, weighted_polarity, n_articles].
    """
    if news_df.empty:
        return pd.DataFrame()
    texts = news_df[text_col].fillna("").astype(str).tolist()
    sent = batch_score_finbert(texts)
    if sent.empty:
        return pd.DataFrame()
    sent = sent.set_index("idx")
    df = news_df.copy()
    df["polarity"] = sent["polarity"].values
    df["weighted_polarity"] = sent["weighted_polarity"].values
    return (
        df.groupby([date_col, symbol_col])
        .agg(
            mean_polarity=("polarity", "mean"),
            weighted_polarity=("weighted_polarity", "mean"),
            n_articles=("polarity", "size"),
        )
        .reset_index()
    )


__all__ = [
    "SentimentResult",
    "score_finbert",
    "batch_score_finbert",
    "aggregate_daily_sentiment",
]
