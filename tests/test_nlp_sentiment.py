"""Tests for NLP sentiment module (FinBERT)."""

from __future__ import annotations

import pytest; pytest.importorskip("src.assembled_core.ml.nlp_sentiment")
import pandas as pd

from src.assembled_core.ml.nlp_sentiment import (
    score_texts_finbert,
    score_news_store,
)


@pytest.mark.phase12
class TestScoreTextsFinbert:
    def test_without_transformers(self):
        """Without transformers installed, should raise ImportError."""
        try:
            import transformers  # noqa: F401
            pytest.skip("transformers is installed — this test is for no-transformers env")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="transformers"):
            score_texts_finbert(["Stock market rises"])

    def test_empty_input(self):
        result = score_texts_finbert([])
        assert result == []

    def test_single_text(self):
        try:
            import transformers  # noqa: F401
            pytest.skip("transformers required for actual scoring test")
        except ImportError:
            with pytest.raises(ImportError):
                score_texts_finbert(["Market is doing well"])


@pytest.mark.phase12
class TestScoreNewsStore:
    def test_basic_requires_transformers(self):
        """score_news_store requires transformers — should raise without it."""
        try:
            import transformers  # noqa: F401
            pytest.skip("transformers installed — skip no-transformers test")
        except ImportError:
            pass
        news_df = pd.DataFrame({
            "headline": ["Stock rises sharply", "Market crashes"],
            "symbol": ["AAPL", "MSFT"],
            "timestamp": pd.date_range("2024-01-01", periods=2),
        })
        with pytest.raises(ImportError):
            score_news_store(news_df)

    def test_empty_news(self):
        result = score_news_store(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
