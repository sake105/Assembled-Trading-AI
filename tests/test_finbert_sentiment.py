"""Tests for intel/finbert_sentiment.py — KNOWN_ISSUES §6.5.4 closure.

Covers the 3-tier fallback (FinBERT → VADER → keyword):
- Keyword backend always available (no external deps)
- VADER backend optional (vaderSentiment installed)
- FinBERT backend optional (transformers + ProsusAI/finbert)

Tests use `prefer_backend="keyword"` for deterministic behavior in CI.
The optional-dep paths use `pytest.importorskip` to skip cleanly when the
backend is not installed.
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.finbert_sentiment import (
    SentimentResult,
    SentimentScorer,
    get_sentiment_scorer,
    score_news_items,
)


# ---------------------------------------------------------------------------
# Keyword backend (always available)
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestKeywordBackend:
    def test_force_keyword_backend(self):
        scorer = get_sentiment_scorer(prefer_backend="keyword")
        assert isinstance(scorer, SentimentScorer)
        assert scorer.backend == "keyword"

    def test_positive_text_scores_positive(self):
        scorer = get_sentiment_scorer(prefer_backend="keyword")
        result = scorer.score("Earnings beat expectations with strong growth.")
        assert isinstance(result, SentimentResult)
        assert result.backend == "keyword"
        assert result.label == "positive"
        assert result.score > 0.0

    def test_negative_text_scores_negative(self):
        scorer = get_sentiment_scorer(prefer_backend="keyword")
        # Use words exclusively in the negative list (avoid "revenue"/"profit"
        # which are POSITIVE keywords in the module's lists).
        result = scorer.score("Bankruptcy and fraud investigation; layoffs announced.")
        assert result.label == "negative"
        assert result.score < 0.0

    def test_neutral_text_scores_neutral(self):
        scorer = get_sentiment_scorer(prefer_backend="keyword")
        result = scorer.score("The company published its quarterly report on Tuesday.")
        assert result.label == "neutral"
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_score_batch_returns_list(self):
        scorer = get_sentiment_scorer(prefer_backend="keyword")
        results = scorer.score_batch(
            [
                "Strong revenue growth and dividend raise.",
                "Bankruptcy filing announced today.",
                "Conference scheduled for next week.",
            ]
        )
        assert len(results) == 3
        assert results[0].label == "positive"
        assert results[1].label == "negative"
        assert results[2].label == "neutral"

    def test_keyword_confidence_capped_at_one(self):
        """Confidence should never exceed 1.0 even with many keyword hits."""
        scorer = get_sentiment_scorer(prefer_backend="keyword")
        # Many positive keywords in one text
        text = "beat exceeded surpassed record growth profit gain upgraded outperform"
        result = scorer.score(text)
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# VADER backend (optional)
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestVaderBackend:
    def test_vader_backend_when_available(self):
        pytest.importorskip("vaderSentiment")
        scorer = get_sentiment_scorer(prefer_backend="vader")
        assert scorer.backend == "vader"

    def test_vader_polarity_for_positive(self):
        pytest.importorskip("vaderSentiment")
        scorer = get_sentiment_scorer(prefer_backend="vader")
        result = scorer.score("This is fantastic, excellent, brilliant news!")
        assert result.backend == "vader"
        # VADER's compound score should be clearly positive
        assert result.score > 0.0
        assert result.label == "positive"


# ---------------------------------------------------------------------------
# FinBERT backend (optional, requires transformers + network/cache)
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestFinbertBackend:
    def test_finbert_backend_when_available(self):
        """If transformers is installed, finbert backend should be selectable.

        Skipped if transformers is not present or if loading the model fails
        (network unavailable, weights not cached). The auto-fallback in
        get_sentiment_scorer handles model-load failures by falling back to
        VADER/keyword, so we cannot assert backend == "finbert" without
        explicit prefer_backend="finbert" AND a successful pipeline init.
        """
        pytest.importorskip("transformers")
        # Use auto-detection to avoid forcing finbert when weights are missing
        scorer = get_sentiment_scorer()
        # Auto-detect may have fallen back; just confirm we got a usable scorer
        assert scorer.backend in {"finbert", "vader", "keyword"}
        result = scorer.score("test")
        assert isinstance(result, SentimentResult)


# ---------------------------------------------------------------------------
# score_news_items wrapper (signal-layer integration helper)
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestScoreNewsItems:
    def test_adds_sentiment_fields_in_place(self):
        items = [
            {"headline": "Earnings beat expectations", "ticker": "AAPL"},
            {"headline": "Profit warning issued", "ticker": "MSFT"},
        ]
        result = score_news_items(items, prefer_backend="keyword")
        assert result is items  # In-place mutation
        for item in items:
            assert "sentiment_score" in item
            assert "sentiment_label" in item
            assert "sentiment_confidence" in item
            assert "sentiment_backend" in item
            assert item["sentiment_backend"] == "keyword"

    def test_empty_list_returns_empty(self):
        result = score_news_items([], prefer_backend="keyword")
        assert result == []

    def test_custom_text_key(self):
        items = [{"title": "Strong revenue growth and dividend raise.", "id": 1}]
        score_news_items(items, text_key="title", prefer_backend="keyword")
        assert items[0]["sentiment_label"] == "positive"

    def test_missing_text_key_is_treated_as_empty_string(self):
        """Missing 'headline' should not crash — items get neutral score."""
        items = [{"other_field": "value"}]
        score_news_items(items, prefer_backend="keyword")
        # Empty string scores as neutral with the keyword backend
        assert items[0]["sentiment_label"] == "neutral"
        assert items[0]["sentiment_score"] == 0.0


# ---------------------------------------------------------------------------
# Auto-detection contract
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestAutoDetection:
    def test_default_returns_a_valid_scorer(self):
        """Without prefer_backend, the factory must return SOME working scorer.

        The exact backend depends on what's installed; we only assert that
        the returned scorer can produce a valid result.
        """
        scorer = get_sentiment_scorer()
        assert scorer.backend in {"finbert", "vader", "keyword"}
        result = scorer.score("This is good news.")
        assert isinstance(result, SentimentResult)
        assert result.label in {"positive", "negative", "neutral"}
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
