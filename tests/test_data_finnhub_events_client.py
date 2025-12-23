"""Tests for Finnhub Events API client (Phase B1).

Tests the fetch_earnings_events() and fetch_insider_events() functions
using mocks to avoid actual HTTP requests.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.config.settings import Settings
from src.assembled_core.data.altdata.finnhub_events import (
    fetch_earnings_events,
    fetch_insider_events,
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


class TestFetchEarningsEvents:
    """Tests for fetch_earnings_events()."""

    def test_successful_fetch(self, mock_settings):
        """Test successful earnings events fetch with valid response."""
        # Mock response data
        mock_response_data = {
            "earningsCalendar": [
                {
                    "date": "2020-01-15",
                    "symbol": "AAPL",  # Add symbol to match filter
                    "epsActual": 2.1,
                    "epsEstimate": 2.0,
                    "epsSurprise": 0.1,
                    "epsSurprisePercent": 5.0,
                    "revenueActual": 100.0,
                    "revenueEstimate": 95.0,
                    "revenueSurprise": 5.0,
                    "revenueSurprisePercent": 5.26,
                    "fiscalPeriod": "2020 Q1",
                },
                {
                    "date": "2020-03-15",
                    "symbol": "AAPL",  # Add symbol to match filter
                    "epsActual": 1.8,
                    "epsEstimate": 2.0,
                    "epsSurprise": -0.2,
                    "epsSurprisePercent": -10.0,
                    "revenueActual": 90.0,
                    "revenueEstimate": 95.0,
                    "revenueSurprise": -5.0,
                    "revenueSurprisePercent": -5.26,
                    "fiscalPeriod": "2020 Q2",
                },
            ]
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        with patch(
            "src.assembled_core.data.altdata.finnhub_events._get_finnhub_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")

            result = fetch_earnings_events(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                settings=mock_settings,
            )

        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        # Check that mock was called
        assert mock_session.get.called

        # The result should contain events if the mock data includes matching symbols
        # Since we're mocking with "AAPL" in the earnings calendar, it should be found
        if not result.empty:
            # Check required columns
            assert "timestamp" in result.columns
            assert "symbol" in result.columns
            assert "event_type" in result.columns
            assert "event_id" in result.columns

            # Check that event_type is "earnings"
            assert (result["event_type"] == "earnings").all()

            # Check that symbols are correct
            assert (result["symbol"] == "AAPL").all()

        # Check required columns
        assert "timestamp" in result.columns
        assert "symbol" in result.columns
        assert "event_type" in result.columns
        assert "event_id" in result.columns

        # Check that event_type is "earnings"
        assert (result["event_type"] == "earnings").all()

        # Check that symbols are correct
        assert (result["symbol"] == "AAPL").all()

        # Check that timestamps are UTC-aware
        assert result["timestamp"].dtype.tz == pd.Timestamp.now(tz="UTC").tz

        # Check that EPS data is present
        if "eps_actual" in result.columns:
            assert result["eps_actual"].notna().any()

    def test_empty_response(self, mock_settings):
        """Test handling of empty API response."""
        mock_response_data = {"earningsCalendar": []}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        with patch(
            "src.assembled_core.data.altdata.finnhub_events._get_finnhub_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")

            result = fetch_earnings_events(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                settings=mock_settings,
            )

        # Should return empty DataFrame, not crash
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_http_error_4xx(self, mock_settings):
        """Test handling of 4xx HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        with patch(
            "src.assembled_core.data.altdata.finnhub_events._get_finnhub_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")

            # Should not crash, but return empty DataFrame
            result = fetch_earnings_events(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                settings=mock_settings,
            )

        # Should return empty DataFrame on error
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_http_error_5xx(self, mock_settings):
        """Test handling of 5xx HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception(
            "500 Internal Server Error"
        )

        with patch(
            "src.assembled_core.data.altdata.finnhub_events._get_finnhub_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")

            # Should not crash, but return empty DataFrame
            result = fetch_earnings_events(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                settings=mock_settings,
            )

        # Should return empty DataFrame on error
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_api_key(self, mock_settings_no_key):
        """Test that missing API key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="FINNHUB_API_KEY not set"):
            fetch_earnings_events(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                settings=mock_settings_no_key,
            )

    def test_multiple_symbols(self, mock_settings):
        """Test fetching events for multiple symbols."""
        mock_response_data = {
            "earningsCalendar": [
                {
                    "date": "2020-01-15",
                    "epsActual": 2.1,
                    "epsEstimate": 2.0,
                },
            ]
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        with patch(
            "src.assembled_core.data.altdata.finnhub_events._get_finnhub_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")

            fetch_earnings_events(
                symbols=["AAPL", "MSFT", "GOOGL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                settings=mock_settings,
            )

        # Earnings calendar endpoint is called once (not per symbol)
        # The function filters results by symbol after fetching
        assert mock_session.get.call_count >= 1


class TestFetchInsiderEvents:
    """Tests for fetch_insider_events()."""

    def test_successful_fetch(self, mock_settings):
        """Test successful insider events fetch with valid response."""
        # Mock response data
        mock_response_data = {
            "data": [
                {
                    "name": "John Doe",
                    "share": 10000,
                    "change": 10000,
                    "filingDate": "2020-01-20",
                    "transactionDate": "2020-01-20",
                    "transactionCode": "P",  # Purchase
                    "transactionPrice": 100.0,
                },
                {
                    "name": "Jane Smith",
                    "share": -5000,
                    "change": -5000,
                    "filingDate": "2020-01-25",
                    "transactionDate": "2020-01-25",
                    "transactionCode": "S",  # Sale
                    "transactionPrice": 100.0,
                },
            ]
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        with patch(
            "src.assembled_core.data.altdata.finnhub_events._get_finnhub_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")

            result = fetch_insider_events(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                settings=mock_settings,
            )

        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        # Check required columns
        assert "timestamp" in result.columns
        assert "symbol" in result.columns
        assert "event_type" in result.columns
        assert "event_id" in result.columns

        # Check that event_type starts with "insider_"
        assert result["event_type"].str.startswith("insider_").all()

        # Check that symbols are correct
        assert (result["symbol"] == "AAPL").all()

        # Check that timestamps are UTC-aware
        assert result["timestamp"].dtype.tz == pd.Timestamp.now(tz="UTC").tz

    def test_empty_response(self, mock_settings):
        """Test handling of empty API response."""
        mock_response_data = {"data": []}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        with patch(
            "src.assembled_core.data.altdata.finnhub_events._get_finnhub_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")

            result = fetch_insider_events(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                settings=mock_settings,
            )

        # Should return empty DataFrame, not crash
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_http_error_4xx(self, mock_settings):
        """Test handling of 4xx HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

        with patch(
            "src.assembled_core.data.altdata.finnhub_events._get_finnhub_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")

            # Should not crash, but return empty DataFrame
            result = fetch_insider_events(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                settings=mock_settings,
            )

        # Should return empty DataFrame on error
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_http_error_5xx(self, mock_settings):
        """Test handling of 5xx HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = Exception(
            "503 Service Unavailable"
        )

        with patch(
            "src.assembled_core.data.altdata.finnhub_events._get_finnhub_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_get_session.return_value = (mock_session, "test_api_key_12345")

            # Should not crash, but return empty DataFrame
            result = fetch_insider_events(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                settings=mock_settings,
            )

        # Should return empty DataFrame on error
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_api_key(self, mock_settings_no_key):
        """Test that missing API key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="FINNHUB_API_KEY not set"):
            fetch_insider_events(
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                settings=mock_settings_no_key,
            )
