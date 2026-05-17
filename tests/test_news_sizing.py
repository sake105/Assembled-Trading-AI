"""Tests for T4.4 apply_news_sentiment_weight_adjustment."""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.portfolio.position_sizing import (
    apply_news_sentiment_weight_adjustment,
)


class _MockLinker:
    """Minimal EntityLinker stub that maps entity name directly to uppercase."""

    def link(self, entity: str) -> str | None:
        if not entity:
            return None
        return entity.upper()


def _make_positions(symbols: list[str], weight: float = 0.25) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": sym, "target_weight": weight, "target_qty": weight}
            for sym in symbols
        ]
    )


def _make_news(entities: list[str], scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"entity": e, "sentiment_score": s} for e, s in zip(entities, scores)]
    )


@pytest.mark.fast
class TestNewsWeightAdjustment:
    def test_shadow_only_returns_unchanged(self):
        pos = _make_positions(["AAPL", "MSFT"])
        news = _make_news(["AAPL"], [0.8])
        result = apply_news_sentiment_weight_adjustment(
            pos, news, entity_linker=_MockLinker(), shadow_only=True
        )
        # Shadow: positions unchanged
        assert list(result["target_weight"]) == list(pos["target_weight"])

    def test_no_news_returns_unchanged(self):
        pos = _make_positions(["AAPL"])
        result = apply_news_sentiment_weight_adjustment(
            pos, None, entity_linker=_MockLinker()
        )
        assert list(result["target_weight"]) == list(pos["target_weight"])

    def test_no_linker_returns_unchanged(self):
        pos = _make_positions(["AAPL"])
        news = _make_news(["AAPL"], [0.5])
        result = apply_news_sentiment_weight_adjustment(pos, news, entity_linker=None)
        assert list(result["target_weight"]) == list(pos["target_weight"])

    def test_live_mode_adjusts_up(self):
        pos = _make_positions(["AAPL"], weight=0.20)
        news = _make_news(["AAPL"], [1.0])  # max positive
        result = apply_news_sentiment_weight_adjustment(
            pos,
            news,
            entity_linker=_MockLinker(),
            max_adjustment=0.10,
            shadow_only=False,
        )
        # Weight increased by up to 0.10
        assert result.iloc[0]["target_weight"] > 0.20

    def test_live_mode_adjusts_down(self):
        # Use two symbols so renormalization doesn't trivially collapse to 1.0
        pos = _make_positions(["AAPL", "MSFT"], weight=0.30)
        news = _make_news(["AAPL"], [-1.0])  # max negative on AAPL only
        result = apply_news_sentiment_weight_adjustment(
            pos,
            news,
            entity_linker=_MockLinker(),
            max_adjustment=0.10,
            shadow_only=False,
        )
        aapl_w = result[result["symbol"] == "AAPL"].iloc[0]["target_weight"]
        msft_w = result[result["symbol"] == "MSFT"].iloc[0]["target_weight"]
        # AAPL reduced, MSFT unchanged → after renorm AAPL < MSFT
        assert aapl_w < msft_w

    def test_weight_never_below_zero(self):
        pos = _make_positions(["AAPL"], weight=0.01)
        news = _make_news(["AAPL"], [-1.0])
        result = apply_news_sentiment_weight_adjustment(
            pos,
            news,
            entity_linker=_MockLinker(),
            max_adjustment=0.10,
            shadow_only=False,
        )
        assert result.iloc[0]["target_weight"] >= 0.0

    def test_unknown_symbol_not_adjusted(self):
        pos = _make_positions(["AAPL", "MSFT"])
        news = _make_news(["GOOG"], [1.0])  # GOOG not in positions
        result = apply_news_sentiment_weight_adjustment(
            pos, news, entity_linker=_MockLinker(), shadow_only=False
        )
        # AAPL and MSFT weights unchanged (GOOG not matched)
        assert (
            pytest.approx(result.iloc[0]["target_weight"], abs=1e-6)
            == result.iloc[1]["target_weight"]
        )
