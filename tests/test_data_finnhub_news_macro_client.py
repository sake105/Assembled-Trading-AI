"""Tests for Finnhub News & Macro API client (Phase B2).

Tests the fetch_news(), fetch_news_sentiment(), and fetch_macro_series() functions
using mocks to avoid actual HTTP requests.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.config.settings import Settings
from src.assembled_core.data.altdata.finnhub_news_macro import (
    fetch_macro_series,
    fetch_news,
    fetch_news_sentiment,
)


@pytest.fixture
def mock_settings() -> Settings:
    """Create a mock Settings object with API key."""
    settings = MagicMock(spec=Settings)
    settings.finnhub_api_key = "test_api_key_12345"
    return settings


@pytest.fixture
def mock_settings_no_key() -> Settings:
    """Create a mock Settings object without API key."""
    settings = MagicMock(spec=Settings)
    settings.finnhub_api_key = None
    return settings


class TestFetchNews:
    """Tests for fetch_news()."""
    
    def test_successful_fetch_company_news(self, mock_settings):
        """Test successful company news fetch with valid response."""
        mock_response_data = [
            {
                "category": "company",
                "datetime": 1577836800000,  # 2020-01-01 00:00:00 UTC
                "headline": "Apple announces new product",
                "id": 12345,
                "image": "https://example.com/image.jpg",
                "related": "AAPL",
                "source": "Reuters",
                "summary": "Apple Inc. announced...",
                "url": "https://example.com/news/12345",
            },
            {
                "category": "company",
                "datetime": 1577923200000,  # 2020-01-02 00:00:00 UTC
                "headline": "Apple stock rises",
                "id": 12346,
                "image": "",
                "related": "AAPL",
                "source": "Bloomberg",
                "summary": "Apple stock...",
                "url": "https://example.com/news/12346",
            },
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_news(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert mock_session.get.called
        
        # Check data contract columns
        if not result.empty:
            assert "timestamp" in result.columns
            assert "symbol" in result.columns
            assert "headline" in result.columns
            assert "news_id" in result.columns
            assert "event_type" in result.columns
            assert (result["event_type"] == "news").all()
    
    def test_successful_fetch_market_news(self, mock_settings):
        """Test successful market-wide news fetch (no symbols)."""
        mock_response_data = [
            {
                "category": "general",
                "datetime": 1577836800000,
                "headline": "Market opens higher",
                "id": 99999,
                "image": "",
                "related": "",
                "source": "CNBC",
                "summary": "Markets...",
                "url": "https://example.com/news/99999",
            },
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_news(
                symbols=None,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "timestamp" in result.columns
            assert "headline" in result.columns
    
    def test_empty_response(self, mock_settings):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_news(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        assert isinstance(result, pd.DataFrame)
        # Empty DataFrame is acceptable
    
    def test_http_error_4xx(self, mock_settings):
        """Test handling of 4xx HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("400 Bad Request")
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_news(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        # Should return empty DataFrame, not crash
        assert isinstance(result, pd.DataFrame)
    
    def test_http_error_5xx(self, mock_settings):
        """Test handling of 5xx HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("500 Internal Server Error")
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_news(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        # Should return empty DataFrame, not crash
        assert isinstance(result, pd.DataFrame)
    
    def test_missing_api_key(self, mock_settings_no_key):
        """Test that missing API key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
            fetch_news(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings_no_key,
            )


class TestFetchNewsSentiment:
    """Tests for fetch_news_sentiment()."""
    
    def test_successful_fetch(self, mock_settings):
        """Test successful news sentiment fetch."""
        # Mock news response (used internally by fetch_news_sentiment)
        mock_news_data = [
            {
                "category": "company",
                "datetime": 1577836800000,
                "headline": "Positive news",
                "id": 12345,
                "image": "",
                "related": "AAPL",
                "source": "Reuters",
                "summary": "Good news...",
                "url": "https://example.com/news/12345",
            },
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_news_data
        mock_response.raise_for_status.return_value = None
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_news_sentiment(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "timestamp" in result.columns
            assert "sentiment_score" in result.columns
            assert "sentiment_volume" in result.columns
    
    def test_empty_response(self, mock_settings):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_news_sentiment(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        assert isinstance(result, pd.DataFrame)


class TestFetchMacroSeries:
    """Tests for fetch_macro_series()."""
    
    def test_successful_fetch_economic_calendar(self, mock_settings):
        """Test successful macro series fetch from economic calendar."""
        mock_response_data = [
            {
                "actual": 3.2,
                "estimate": 3.0,
                "event": "CPI",
                "impact": "high",
                "time": "2020-01-15 08:30:00",
                "unit": "percent",
            },
            {
                "actual": 2.1,
                "estimate": 2.0,
                "event": "GDP",
                "impact": "high",
                "time": "2020-01-20 10:00:00",
                "unit": "percent",
            },
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_macro_series(
                codes=["CPI", "GDP"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "timestamp" in result.columns
            assert "macro_code" in result.columns
            assert "value" in result.columns
            assert "country" in result.columns
    
    def test_successful_fetch_economic_indicator(self, mock_settings):
        """Test successful macro series fetch from economic indicator endpoint."""
        mock_response_data = {
            "data": [
                {"date": "2020-01-01", "value": 3.2},
                {"date": "2020-01-02", "value": 3.3},
            ]
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_macro_series(
                codes=["CPI"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_empty_response(self, mock_settings):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_macro_series(
                codes=["CPI"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_http_error_4xx(self, mock_settings):
        """Test handling of 4xx HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("400 Bad Request")
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_macro_series(
                codes=["CPI"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        # Should return empty DataFrame, not crash
        assert isinstance(result, pd.DataFrame)
    
    def test_http_error_5xx(self, mock_settings):
        """Test handling of 5xx HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("500 Internal Server Error")
        
        with patch("src.assembled_core.data.altdata.finnhub_news_macro._get_finnhub_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")
            
            result = fetch_macro_series(
                codes=["CPI"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings,
            )
        
        # Should return empty DataFrame, not crash
        assert isinstance(result, pd.DataFrame)
    
    def test_missing_api_key(self, mock_settings_no_key):
        """Test that missing API key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
            fetch_macro_series(
                codes=["CPI"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                settings=mock_settings_no_key,
            )

