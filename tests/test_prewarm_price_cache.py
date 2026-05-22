"""Tests for scripts/ops/prewarm_price_cache.py.

Covers the new stale-row refresh path (F-RX-6 §9.12 (d)): prewarm previously
only refreshed missing symbols; now it can also refresh symbols PRESENT in
cache but with per-symbol stale rows (e.g. KO/PEP/BRK-B/PG that aren't in
the master_universe_panel and therefore can't be refreshed offline).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "ops" / "prewarm_price_cache.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("prewarm_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_cache(tmp_path: Path, per_sym_latest: dict[str, str]) -> Path:
    """Write a tiny cache where each symbol's latest bar = per_sym_latest[sym]."""
    rows = []
    for sym, latest in per_sym_latest.items():
        dates = pd.date_range(end=pd.Timestamp(latest, tz="UTC"), periods=3, freq="D")
        for d in dates:
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "adj_close": 100.5,
                    "volume": 1_000_000,
                }
            )
    cache_path = tmp_path / "daily.parquet"
    pd.DataFrame(rows).to_parquet(cache_path, index=False)
    return cache_path


def test_stale_cache_symbols_identifies_per_symbol_stale(tmp_path):
    mod = _load_module()
    today = pd.Timestamp.now("UTC").normalize()
    fresh = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    medium = (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    old = (today - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    cache_path = _write_cache(
        tmp_path,
        {"AAPL": fresh, "MSFT": fresh, "KO": medium, "PEP": old},
    )

    stale = mod.stale_cache_symbols(
        ["AAPL", "MSFT", "KO", "PEP"], max_age_days=5, path=cache_path
    )
    # PEP older than KO → PEP first
    assert stale == ["PEP", "KO"]


def test_stale_cache_symbols_respects_max_age_days(tmp_path):
    mod = _load_module()
    today = pd.Timestamp.now("UTC").normalize()
    fresh = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    medium = (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    cache_path = _write_cache(tmp_path, {"AAPL": fresh, "KO": medium})

    # With max_age_days=15, the 10d-old sym is still fresh enough
    assert mod.stale_cache_symbols(["AAPL", "KO"], 15, cache_path) == []
    # With max_age_days=5, the 10d-old sym IS stale
    assert mod.stale_cache_symbols(["AAPL", "KO"], 5, cache_path) == ["KO"]


def test_stale_cache_symbols_filters_to_watchlist(tmp_path):
    """Symbols not in the watchlist must be ignored even if stale."""
    mod = _load_module()
    today = pd.Timestamp.now("UTC").normalize()
    old = (today - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    cache_path = _write_cache(tmp_path, {"AAPL": old, "OUTDATED_SYM": old})

    # Watchlist only contains AAPL; OUTDATED_SYM is not in our trading universe
    stale = mod.stale_cache_symbols(["AAPL"], 5, cache_path)
    assert stale == ["AAPL"]
    assert "OUTDATED_SYM" not in stale


def test_stale_cache_symbols_missing_cache_returns_empty(tmp_path):
    mod = _load_module()
    cache_path = tmp_path / "does_not_exist.parquet"
    assert mod.stale_cache_symbols(["AAPL"], 5, cache_path) == []


# ---------------------------------------------------------------------------
# YFinanceRateLimitError — 429 fast-abort (no retry waste)
# ---------------------------------------------------------------------------


def test_yfinance_rate_limit_error_raised_on_429():
    """_fetch_single_symbol must raise YFinanceRateLimitError on 429, not retry."""
    import sys
    from unittest.mock import MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.assembled_core.data.sources.yfinance_source import (
        YFinanceRateLimitError,
        _fetch_single_symbol,
    )

    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = Exception("HTTP Error 429: Too Many Requests")

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(YFinanceRateLimitError):
            _fetch_single_symbol("AAPL", "2024-01-01", "2024-12-31", "1d")

    # Must NOT have called sleep (no retry on 429)
    assert mock_ticker.history.call_count == 1


def test_yfinance_non_429_still_retries():
    """Non-429 exceptions use normal retry logic (3 attempts)."""
    import sys
    from unittest.mock import MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.assembled_core.data.sources.yfinance_source import _fetch_single_symbol

    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = Exception("Connection timeout")

    with patch("yfinance.Ticker", return_value=mock_ticker), patch("time.sleep"):
        result = _fetch_single_symbol("AAPL", "2024-01-01", "2024-12-31", "1d")

    assert result is None
    assert mock_ticker.history.call_count == 3  # exhausted all retries


# ---------------------------------------------------------------------------
# Alpaca fallback — graceful failure without credentials
# ---------------------------------------------------------------------------


def test_fetch_missing_alpaca_no_credentials_returns_empty(tmp_path, monkeypatch):
    """fetch_missing_alpaca must return empty DataFrame when credentials absent."""
    mod = _load_module()
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

    df = mod.fetch_missing_alpaca(["AAPL", "MSFT"], years=1)
    assert isinstance(df, __import__("pandas").DataFrame)
    assert df.empty


def test_write_failed_symbols_creates_json(tmp_path):
    """write_failed_symbols must write parseable JSON with expected keys."""
    import json

    mod = _load_module()
    out = tmp_path / "failed.json"
    mod.write_failed_symbols(["AAPL", "MSFT"], reason="yfinance_empty", path=out)

    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["symbols"] == ["AAPL", "MSFT"]
    assert data["count"] == 2
    assert data["reason"] == "yfinance_empty"
    assert "timestamp" in data


def test_write_failed_symbols_overwrites_previous(tmp_path):
    """Second call overwrites first — no stale accumulation."""
    import json

    mod = _load_module()
    out = tmp_path / "failed.json"
    mod.write_failed_symbols(["OLD"], reason="first", path=out)
    mod.write_failed_symbols(["NEW1", "NEW2"], reason="second", path=out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["symbols"] == ["NEW1", "NEW2"]
    assert data["reason"] == "second"


def test_fetch_missing_alpaca_sdk_unavailable_returns_empty(tmp_path, monkeypatch):
    """fetch_missing_alpaca must return empty DataFrame when alpaca-py not installed."""
    import sys
    from unittest.mock import patch

    mod = _load_module()
    monkeypatch.setenv("ALPACA_API_KEY", "fake_key")
    monkeypatch.setenv("ALPACA_API_SECRET", "fake_secret")

    with patch.dict(sys.modules, {"alpaca": None, "alpaca.data": None}):
        df = mod.fetch_missing_alpaca(["AAPL"], years=1)

    assert isinstance(df, __import__("pandas").DataFrame)
    assert df.empty
