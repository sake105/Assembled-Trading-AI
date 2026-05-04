"""FinBERT-based sentiment scoring with automatic fallbacks.

Priority order:
1. HuggingFace transformers + ProsusAI/finbert (best — finance-domain BERT)
2. VADER sentiment (good general-purpose rule-based)
3. Keyword-based fallback (always available, no deps)

The public API is intentionally the same regardless of which backend runs.
Results include a ``backend`` field so callers can log which path was used.

Usage:
    scorer = get_sentiment_scorer()
    result = scorer.score("Earnings beat expectations by a wide margin.")
    # {'score': 0.82, 'label': 'positive', 'backend': 'finbert'}

    batch = scorer.score_batch(["text1", "text2", ...])
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

_log = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "negative", "neutral"]

# ---------------------------------------------------------------------------
# Keyword lists for fallback scorer
# ---------------------------------------------------------------------------
_POSITIVE_KEYWORDS = frozenset(
    [
        "beat",
        "beats",
        "exceeded",
        "exceeds",
        "surpassed",
        "record",
        "growth",
        "profit",
        "revenue",
        "gain",
        "gains",
        "rally",
        "upgraded",
        "upgrade",
        "outperform",
        "strong",
        "robust",
        "optimistic",
        "upside",
        "bullish",
        "expansion",
        "recovery",
        "raised",
        "raise",
        "dividend",
        "buyback",
        "acquisition",
        "partnership",
        "approved",
        "approval",
        "positive",
        "higher",
        "increased",
        "increases",
        "breakthrough",
    ]
)

_NEGATIVE_KEYWORDS = frozenset(
    [
        "miss",
        "missed",
        "misses",
        "fell",
        "fell short",
        "below",
        "declined",
        "decline",
        "loss",
        "losses",
        "cut",
        "cuts",
        "downgraded",
        "downgrade",
        "underperform",
        "weak",
        "warning",
        "risk",
        "lawsuit",
        "investigation",
        "recall",
        "layoff",
        "layoffs",
        "bankruptcy",
        "default",
        "delay",
        "delays",
        "writeoff",
        "impairment",
        "bearish",
        "slowdown",
        "contraction",
        "concern",
        "concerns",
        "negative",
        "lower",
        "decreased",
        "decreases",
        "penalty",
        "fine",
        "fraud",
    ]
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SentimentResult:
    """Single-text sentiment result."""

    text: str
    score: float  # -1.0 (most negative) to +1.0 (most positive)
    label: SentimentLabel
    confidence: float  # 0.0 to 1.0
    backend: str


@dataclass
class SentimentScorer:
    """Sentiment scorer with automatic backend selection."""

    backend: str
    _pipeline: object = field(default=None, repr=False)

    def score(self, text: str) -> SentimentResult:
        """Score a single text string."""
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Score a batch of text strings."""
        if self.backend == "finbert":
            return _score_finbert(texts, self._pipeline)
        elif self.backend == "vader":
            return _score_vader(texts, self._pipeline)
        else:
            return _score_keywords(texts)


# ---------------------------------------------------------------------------
# Backend: FinBERT
# ---------------------------------------------------------------------------


def _score_finbert(texts: list[str], pipeline) -> list[SentimentResult]:
    results = []
    try:
        raw = pipeline(texts, truncation=True, max_length=512)
        if isinstance(raw, dict):
            raw = [raw]
        for text, r in zip(texts, raw):
            label_raw = r["label"].lower()
            conf = float(r["score"])
            if label_raw == "positive":
                label: SentimentLabel = "positive"
                score = conf
            elif label_raw == "negative":
                label = "negative"
                score = -conf
            else:
                label = "neutral"
                score = 0.0
            results.append(
                SentimentResult(
                    text=text,
                    score=round(score, 4),
                    label=label,
                    confidence=round(conf, 4),
                    backend="finbert",
                )
            )
    except Exception as exc:
        _log.warning("[FinBERT] batch scoring failed, falling back: %s", exc)
        return _score_keywords(texts)
    return results


# ---------------------------------------------------------------------------
# Backend: VADER
# ---------------------------------------------------------------------------


def _score_vader(texts: list[str], analyzer) -> list[SentimentResult]:
    results = []
    for text in texts:
        try:
            scores = analyzer.polarity_scores(text)
            compound = float(scores["compound"])
            if compound >= 0.05:
                label: SentimentLabel = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            conf = abs(compound)
            results.append(
                SentimentResult(
                    text=text,
                    score=round(compound, 4),
                    label=label,
                    confidence=round(conf, 4),
                    backend="vader",
                )
            )
        except Exception as exc:
            _log.warning("[VADER] failed on text, using keyword fallback: %s", exc)
            results.extend(_score_keywords([text]))
    return results


# ---------------------------------------------------------------------------
# Backend: Keyword fallback (no external deps)
# ---------------------------------------------------------------------------


def _score_keywords(texts: list[str]) -> list[SentimentResult]:
    results = []
    for text in texts:
        words = set(re.findall(r"\b\w+\b", text.lower()))
        pos_hits = len(words & _POSITIVE_KEYWORDS)
        neg_hits = len(words & _NEGATIVE_KEYWORDS)
        total = pos_hits + neg_hits
        if total == 0:
            score = 0.0
            label: SentimentLabel = "neutral"
            conf = 0.0
        else:
            score = round((pos_hits - neg_hits) / total, 4)
            conf = round(min(total / 5.0, 1.0), 4)
            if score > 0.1:
                label = "positive"
            elif score < -0.1:
                label = "negative"
            else:
                label = "neutral"
        results.append(
            SentimentResult(
                text=text,
                score=score,
                label=label,
                confidence=conf,
                backend="keyword",
            )
        )
    return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_sentiment_scorer(
    prefer_backend: Literal["finbert", "vader", "keyword"] | None = None,
    finbert_model: str = "ProsusAI/finbert",
    device: int = -1,
) -> SentimentScorer:
    """Build a SentimentScorer using the best available backend.

    Args:
        prefer_backend: Force a specific backend. None = auto-detect.
        finbert_model: HuggingFace model ID for FinBERT (default: ProsusAI/finbert).
        device: Device index for transformers (-1 = CPU, 0 = first GPU).

    Returns:
        SentimentScorer ready for use.
    """
    if prefer_backend == "keyword":
        _log.info("[Sentiment] Using keyword backend (forced)")
        return SentimentScorer(backend="keyword")

    if prefer_backend != "vader":
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import]

            _log.info("[Sentiment] Loading FinBERT model: %s", finbert_model)
            pipe = hf_pipeline(
                "text-classification",
                model=finbert_model,
                device=device,
                top_k=None,
            )
            # Warm-up to confirm model loaded
            pipe(["test"], truncation=True, max_length=16)
            _log.info("[Sentiment] FinBERT loaded OK (model=%s)", finbert_model)
            return SentimentScorer(backend="finbert", _pipeline=pipe)
        except Exception as exc:
            _log.info("[Sentiment] FinBERT unavailable (%s), trying VADER", exc)

    if prefer_backend != "finbert":
        try:
            from vaderSentiment.vaderSentiment import (  # type: ignore[import]
                SentimentIntensityAnalyzer,
            )

            analyzer = SentimentIntensityAnalyzer()
            _log.info("[Sentiment] VADER loaded OK")
            return SentimentScorer(backend="vader", _pipeline=analyzer)
        except Exception as exc:
            _log.info("[Sentiment] VADER unavailable (%s), using keyword fallback", exc)

    _log.info("[Sentiment] Using keyword fallback backend")
    return SentimentScorer(backend="keyword")


def score_news_items(
    items: list[dict],
    *,
    text_key: str = "headline",
    prefer_backend: Literal["finbert", "vader", "keyword"] | None = None,
) -> list[dict]:
    """Score a list of news dicts in place, adding sentiment fields.

    Args:
        items: List of dicts with at least a text field (e.g. "headline").
        text_key: Key in each dict that contains the text to score.
        prefer_backend: Force a specific backend (default: auto).

    Returns:
        Same list with added keys: sentiment_score, sentiment_label,
        sentiment_confidence, sentiment_backend.
    """
    if not items:
        return items

    scorer = get_sentiment_scorer(prefer_backend=prefer_backend)
    texts = [str(item.get(text_key, "")) for item in items]
    results = scorer.score_batch(texts)

    for item, result in zip(items, results):
        item["sentiment_score"] = result.score
        item["sentiment_label"] = result.label
        item["sentiment_confidence"] = result.confidence
        item["sentiment_backend"] = result.backend

    _log.info(
        "[Sentiment] Scored %d items via %s backend",
        len(items),
        results[0].backend if results else "n/a",
    )
    return items


__all__ = [
    "SentimentResult",
    "SentimentScorer",
    "get_sentiment_scorer",
    "score_news_items",
]
