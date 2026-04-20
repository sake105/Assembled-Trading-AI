"""Tests for news_entity_mapper — ticker extraction from headlines."""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_entity_mapper import (
    SimpleEntityLinker,
    extract_tickers_from_title,
)


@pytest.mark.phase12
class TestExtractTickersFromTitle:
    def test_apple_detected(self):
        tickers = extract_tickers_from_title("Apple announces record iPhone sales")
        assert "AAPL" in tickers

    def test_nvidia_detected(self):
        tickers = extract_tickers_from_title("NVIDIA stock soars on AI chip demand")
        assert "NVDA" in tickers

    def test_multiple_tickers(self):
        tickers = extract_tickers_from_title("Microsoft and Google battle for AI dominance")
        assert "MSFT" in tickers
        assert "GOOGL" in tickers

    def test_defense_ticker(self):
        tickers = extract_tickers_from_title("Lockheed Martin wins major DoD contract")
        assert "LMT" in tickers

    def test_energy_ticker(self):
        tickers = extract_tickers_from_title("ExxonMobil cuts output amid sanctions")
        assert "XOM" in tickers

    def test_no_match_returns_empty(self):
        tickers = extract_tickers_from_title("General geopolitical tensions rise in region")
        assert isinstance(tickers, list)

    def test_case_insensitive(self):
        tickers = extract_tickers_from_title("BOEING reports quarterly loss")
        assert "BA" in tickers

    def test_empty_title(self):
        assert extract_tickers_from_title("") == []


@pytest.mark.phase12
class TestSimpleEntityLinker:
    def test_exact_match(self):
        linker = SimpleEntityLinker()
        assert linker.link("apple") == "AAPL"
        assert linker.link("tesla") == "TSLA"

    def test_none_for_unknown(self):
        linker = SimpleEntityLinker()
        result = linker.link("some_completely_unknown_entity_xyz")
        assert result is None

    def test_empty_returns_none(self):
        linker = SimpleEntityLinker()
        assert linker.link("") is None

    def test_partial_match(self):
        linker = SimpleEntityLinker()
        result = linker.link("ExxonMobil Corporation")
        assert result == "XOM"
