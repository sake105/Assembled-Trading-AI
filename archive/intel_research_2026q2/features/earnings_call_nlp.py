"""Earnings Call NLP Analysis (M26 Task 26.2).

Extracts alpha signals from earnings call transcripts and press releases:
1. Sentiment scoring (positive/negative word ratios)
2. Uncertainty language detection
3. Forward guidance tone extraction
4. Management confidence scoring
5. Topic shift detection (quarter-over-quarter)

Reference:
    Loughran & McDonald (2011) financial sentiment lexicon
    Price et al. (2012) conference call tone and stock returns
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EarningsNLPResult:
    """NLP analysis result for one earnings call."""
    sentiment_score: float         # Net sentiment [-1, 1]
    uncertainty_score: float       # Uncertainty language ratio [0, 1]
    forward_guidance_tone: float   # Forward-looking sentiment [-1, 1]
    management_confidence: float   # Confidence indicator [0, 1]
    word_count: int
    positive_ratio: float
    negative_ratio: float


# Loughran-McDonald financial sentiment words (curated subset)
LM_POSITIVE = {
    "achieve", "achieved", "achievement", "achievements", "attain", "attained",
    "benefit", "beneficial", "best", "better", "boost", "breakthrough",
    "confident", "constructive", "creative", "deliver", "delivered",
    "efficient", "enhance", "enhanced", "excellent", "exceed", "exceeded",
    "exceptional", "favorable", "gain", "gained", "gains", "great", "greater",
    "grew", "grow", "growing", "growth", "highest", "improve", "improved",
    "improvement", "improvements", "increase", "increased", "innovation",
    "leadership", "momentum", "opportunity", "opportunities", "optimal",
    "optimistic", "outpace", "outperform", "outperformed", "outstanding",
    "positive", "profitable", "profitability", "progress", "record",
    "recovery", "robust", "solid", "strength", "strong", "stronger",
    "strongest", "succeed", "success", "successful", "superior", "surpass",
    "surpassed", "upbeat", "upgrade", "upturn",
}

LM_NEGATIVE = {
    "adverse", "adversely", "against", "challenge", "challenged", "challenges",
    "closure", "closures", "concern", "concerned", "concerns", "decline",
    "declined", "declining", "decrease", "decreased", "default", "deficit",
    "delay", "delayed", "deteriorate", "deteriorated", "deterioration",
    "difficult", "difficulties", "difficulty", "disappoint", "disappointed",
    "disappointing", "disappointment", "downturn", "drop", "dropped",
    "failure", "fall", "fallen", "falling", "fell", "headwind", "headwinds",
    "impair", "impaired", "impairment", "inability", "incur", "incurred",
    "lawsuit", "layoff", "layoffs", "litigation", "lose", "loss", "losses",
    "lost", "lower", "lowest", "negative", "negatively", "penalty",
    "restructure", "restructuring", "risk", "risks", "severe", "shortage",
    "shortfall", "shrink", "slowed", "slowdown", "struggle", "struggled",
    "suffer", "suffered", "terminate", "terminated", "threat", "threats",
    "trouble", "uncertain", "uncertainty", "underperform", "unfavorable",
    "unfortunately", "volatile", "volatility", "vulnerability", "weak",
    "weaken", "weakened", "weakness", "worse", "worsen", "worsened", "worst",
}

LM_UNCERTAINTY = {
    "approximately", "assume", "assumption", "assumptions", "believe",
    "conceivable", "conditional", "contingent", "could", "depend", "depends",
    "dependent", "doubt", "doubtful", "estimate", "estimated", "estimates",
    "expect", "expected", "fluctuate", "fluctuations", "forecast",
    "foreseeable", "indefinite", "indefinitely", "indicate", "likelihood",
    "may", "maybe", "might", "nearly", "occasionally", "perhaps",
    "possibility", "possible", "possibly", "predict", "predicted",
    "prediction", "preliminary", "presumably", "probabilistic", "probable",
    "probably", "project", "projected", "projection", "prospect",
    "roughly", "seem", "seems", "somewhat", "suggest", "tentative",
    "uncertain", "uncertainty", "unclear", "undetermined", "unknown",
    "unlikely", "unpredictable", "unsure", "variability", "variable",
}

FORWARD_LOOKING = {
    "ahead", "anticipate", "anticipated", "anticipation", "believe",
    "continue", "estimate", "expect", "expected", "forecast", "forward",
    "future", "goal", "goals", "guidance", "intend", "intention",
    "long-term", "next", "objective", "objectives", "outlook", "pipeline",
    "plan", "planned", "planning", "plans", "project", "projected",
    "projection", "strategy", "target", "targets", "upcoming", "will",
}


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer for financial text."""
    text = text.lower()
    text = re.sub(r"[^a-z\s\-]", " ", text)
    return [w for w in text.split() if len(w) > 2]


def analyze_earnings_text(text: str) -> EarningsNLPResult:
    """Analyze a single earnings call transcript or press release.

    Args:
        text: Full text of earnings call or press release.

    Returns:
        EarningsNLPResult with sentiment and uncertainty metrics.
    """
    words = _tokenize(text)
    total = len(words)
    if total < 10:
        return EarningsNLPResult(
            sentiment_score=0.0, uncertainty_score=0.0,
            forward_guidance_tone=0.0, management_confidence=0.5,
            word_count=total, positive_ratio=0.0, negative_ratio=0.0,
        )

    word_counts = Counter(words)

    # Sentiment
    pos_count = sum(word_counts.get(w, 0) for w in LM_POSITIVE)
    neg_count = sum(word_counts.get(w, 0) for w in LM_NEGATIVE)
    pos_ratio = pos_count / total
    neg_ratio = neg_count / total
    sentiment = (pos_count - neg_count) / max(pos_count + neg_count, 1)

    # Uncertainty
    unc_count = sum(word_counts.get(w, 0) for w in LM_UNCERTAINTY)
    uncertainty = unc_count / total

    # Forward guidance tone
    fwd_words = [w for w in words if w in FORWARD_LOOKING]  # noqa: F841
    fwd_context_pos = 0
    fwd_context_neg = 0
    for i, w in enumerate(words):
        if w in FORWARD_LOOKING:
            # Check surrounding 5 words for sentiment
            context = words[max(0, i - 5):i + 6]
            fwd_context_pos += sum(1 for c in context if c in LM_POSITIVE)
            fwd_context_neg += sum(1 for c in context if c in LM_NEGATIVE)

    fwd_total = fwd_context_pos + fwd_context_neg
    fwd_tone = (fwd_context_pos - fwd_context_neg) / max(fwd_total, 1)

    # Management confidence: high positive, low uncertainty, low hedging
    confidence = max(0, min(1, 0.5 + 0.3 * sentiment - 0.5 * uncertainty))

    return EarningsNLPResult(
        sentiment_score=round(float(np.clip(sentiment, -1, 1)), 4),
        uncertainty_score=round(float(np.clip(uncertainty, 0, 1)), 4),
        forward_guidance_tone=round(float(np.clip(fwd_tone, -1, 1)), 4),
        management_confidence=round(confidence, 4),
        word_count=total,
        positive_ratio=round(pos_ratio, 6),
        negative_ratio=round(neg_ratio, 6),
    )


def compute_earnings_nlp_features(
    transcripts: dict[str, list[tuple[str, str]]],
    price_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute NLP features for multiple stocks across time.

    Args:
        transcripts: {ticker: [(date_str, text), ...]} earnings transcripts.
        price_dates: Trading dates for output alignment.

    Returns:
        MultiIndex DataFrame (date, ticker) with NLP features.
    """
    records = []

    for ticker, calls in transcripts.items():
        prev_result = None
        for date_str, text in sorted(calls, key=lambda x: x[0]):
            result = analyze_earnings_text(text)

            # Compute quarter-over-quarter change
            sentiment_change = 0.0
            if prev_result is not None:
                sentiment_change = result.sentiment_score - prev_result.sentiment_score

            records.append({
                "date": pd.Timestamp(date_str),
                "ticker": ticker,
                "earnings_sentiment": result.sentiment_score,
                "earnings_uncertainty": result.uncertainty_score,
                "earnings_fwd_tone": result.forward_guidance_tone,
                "earnings_confidence": result.management_confidence,
                "earnings_sentiment_change": round(sentiment_change, 4),
            })
            prev_result = result

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Forward-fill features to daily frequency
    output = []
    for ticker in df["ticker"].unique():
        t_data = df[df["ticker"] == ticker].set_index("date").sort_index()
        t_daily = t_data.reindex(price_dates, method="ffill")
        t_daily["ticker"] = ticker
        output.append(t_daily)

    if not output:
        return pd.DataFrame()

    return pd.concat(output).reset_index().rename(columns={"index": "date"})


def compute_earnings_surprise_drift(
    actual_eps: pd.Series,
    expected_eps: pd.Series,
    post_days: int = 60,
) -> pd.Series:
    """Post-earnings announcement drift signal.

    Positive surprises tend to drift higher for 60 trading days.

    Args:
        actual_eps: Actual EPS values with date index.
        expected_eps: Consensus expected EPS.
        post_days: Number of days for drift signal.

    Returns:
        Standardized surprise signal.
    """
    surprise = (actual_eps - expected_eps) / (expected_eps.abs() + 0.01)
    # Forward-fill the surprise for post_days
    signal = surprise.reindex(
        pd.date_range(surprise.index.min(), surprise.index.max() + pd.Timedelta(days=post_days * 2), freq="B"),
    ).ffill(limit=post_days).fillna(0.0)

    return signal.clip(-3, 3)


__all__ = [
    "EarningsNLPResult",
    "analyze_earnings_text",
    "compute_earnings_nlp_features",
    "compute_earnings_surprise_drift",
]
