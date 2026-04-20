"""Tests for NewsAPIFetcher (gated; network is mocked via monkeypatch)."""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_newsapi_fetcher import NewsAPIFetcher


class _FakeResp:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


@pytest.mark.phase12
class TestNewsAPIFetcher:
    def test_disabled_without_key(self, monkeypatch):
        monkeypatch.delenv("NEWSAPI_KEY", raising=False)
        f = NewsAPIFetcher(api_key="")
        assert f.enabled is False
        assert f.fetch(query="anything") == []

    def test_enabled_with_key(self):
        f = NewsAPIFetcher(api_key="KEY123")
        assert f.enabled is True

    def test_fetch_returns_events(self, monkeypatch):
        f = NewsAPIFetcher(api_key="KEY123")
        payload = {
            "status": "ok",
            "articles": [
                {
                    "source": {"id": "reuters", "name": "Reuters"},
                    "title": "Russia-Ukraine peace talks break down",
                    "url": "https://example.com/a1",
                    "publishedAt": "2026-04-20T12:00:00Z",
                },
                {
                    "source": {"id": None, "name": "AP"},
                    "title": "Fed signals rate cut",
                    "url": "https://example.com/a2",
                    "publishedAt": "2026-04-20T12:05:00Z",
                },
            ],
        }
        import src.assembled_core.intel.news_newsapi_fetcher as mod

        class _FakeRequests:
            @staticmethod
            def get(url, params=None, timeout=None):
                return _FakeResp(payload)

        monkeypatch.setattr(mod, "requests", _FakeRequests, raising=False)
        # Also inject into the local `import requests` path used in fetch()
        monkeypatch.setitem(__import__("sys").modules, "requests", _FakeRequests)

        events = f.fetch(query="russia")
        assert len(events) == 2
        assert events[0].source_id == "reuters"
        assert events[0].url == "https://example.com/a1"

    def test_bad_http_returns_empty(self, monkeypatch):
        f = NewsAPIFetcher(api_key="KEY123")

        class _FakeRequests:
            @staticmethod
            def get(url, params=None, timeout=None):
                return _FakeResp({}, status_code=429, text="rate limited")

        monkeypatch.setitem(__import__("sys").modules, "requests", _FakeRequests)
        assert f.fetch() == []

    def test_malformed_article_skipped(self, monkeypatch):
        f = NewsAPIFetcher(api_key="KEY123")
        payload = {
            "articles": [
                {"title": "", "url": ""},                       # empty
                {"title": "Good title", "url": "https://x"},    # keeper
            ],
        }

        class _FakeRequests:
            @staticmethod
            def get(url, params=None, timeout=None):
                return _FakeResp(payload)

        monkeypatch.setitem(__import__("sys").modules, "requests", _FakeRequests)
        events = f.fetch()
        assert len(events) == 1
        assert events[0].title == "Good title"
