"""DAT-005 — feeds distinguish a total outage from a legitimate empty window.

Covers the shared :mod:`feed_status` stamping helper plus the two reference
sources wired in this step (``fred_source`` macro, ``yfinance_source`` prices).
The remaining DAT-005 sources (newsapi / worldbank / finnhub / cboe /
altdata_loader) are a documented follow-up and are not exercised here.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assembled_core.data.feed_status import (  # noqa: E402
    FEED_EMPTY,
    FEED_ERROR,
    FEED_OK,
    get_feed_status,
    is_feed_outage,
    stamp_feed_status,
)
from src.assembled_core.data.sources import fred_source  # noqa: E402
from src.assembled_core.data.sources import yfinance_source  # noqa: E402
from src.assembled_core.data.sources.fred_source import fetch_fred_series  # noqa: E402
from src.assembled_core.data.sources.yfinance_source import (  # noqa: E402
    fetch_prices_yfinance,
)

pytestmark = pytest.mark.fast


def _fred_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02", tz="UTC")],
            "series_id": ["DGS10"],
            "value": [4.0],
        }
    )


def _price_row(symbol: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02", tz="UTC")],
            "symbol": [symbol],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [100.0],
        }
    )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


class TestFeedStatusHelper:
    def test_ok_stamp_is_recorded_and_not_outage(self) -> None:
        df = _price_row()
        out = stamp_feed_status(df, "yfinance", FEED_OK)
        st = get_feed_status(out)
        assert st == {
            "source": "yfinance",
            "status": "ok",
            "reason": None,
            "n_rows": 1,
        }
        assert is_feed_outage(out) is False

    def test_error_stamp_is_outage_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame(columns=["timestamp", "series_id", "value"])
        with caplog.at_level(logging.WARNING):
            out = stamp_feed_status(df, "fred", FEED_ERROR, reason="api_key_missing")
        st = get_feed_status(out)
        assert st is not None
        assert st["status"] == "error"
        assert st["reason"] == "api_key_missing"
        assert st["n_rows"] == 0
        assert is_feed_outage(out) is True
        assert any(
            "[DAT-005]" in r.message or "[DAT-005]" in str(r.args)
            for r in caplog.records
        )

    def test_empty_window_is_not_an_outage(self) -> None:
        df = pd.DataFrame(columns=["timestamp", "series_id", "value"])
        out = stamp_feed_status(df, "fred", FEED_EMPTY, reason="no_rows_in_window")
        assert get_feed_status(out)["status"] == "empty"
        assert is_feed_outage(out) is False

    def test_unknown_status_is_not_stamped(self) -> None:
        df = _price_row()
        out = stamp_feed_status(df, "yfinance", "bogus")
        assert get_feed_status(out) is None  # silently un-stamped, df returned

    def test_non_dataframe_returned_unchanged(self) -> None:
        assert stamp_feed_status(None, "x", FEED_ERROR) is None  # type: ignore[arg-type]
        assert get_feed_status(None) is None  # type: ignore[arg-type]
        assert is_feed_outage(None) is False  # type: ignore[arg-type]

    def test_n_rows_override(self) -> None:
        df = _price_row()
        out = stamp_feed_status(df, "yfinance", FEED_OK, n_rows=42)
        assert get_feed_status(out)["n_rows"] == 42

    def test_stamp_is_content_preserving(self) -> None:
        df = _price_row()
        before_cols = list(df.columns)
        before_len = len(df)
        out = stamp_feed_status(df, "yfinance", FEED_OK)
        assert list(out.columns) == before_cols
        assert len(out) == before_len
        assert out.empty is False


# ---------------------------------------------------------------------------
# FRED macro source
# ---------------------------------------------------------------------------


def _install_fake_fredapi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``from fredapi import Fred`` succeed without the real package."""
    fake = types.ModuleType("fredapi")
    fake.Fred = lambda **_kw: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fredapi", fake)


class TestFredFeedStatus:
    def setup_method(self) -> None:
        fred_source._FRED_CACHE.clear()

    def test_no_series_requested_is_empty(self) -> None:
        out = fetch_fred_series([], "2024-01-01", "2024-01-31")
        assert out.empty
        assert get_feed_status(out)["status"] == "empty"
        assert is_feed_outage(out) is False

    def test_missing_api_key_is_outage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fred_source, "_get_api_key", lambda: None)
        out = fetch_fred_series(["DGS10"], "2024-01-01", "2024-01-31")
        assert out.empty
        assert is_feed_outage(out) is True
        assert get_feed_status(out)["reason"] == "api_key_missing"

    def test_fredapi_not_installed_is_outage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fred_source, "_get_api_key", lambda: "k")
        # sys.modules[name] = None makes `from fredapi import Fred` raise ImportError.
        monkeypatch.setitem(sys.modules, "fredapi", None)
        out = fetch_fred_series(["DGS10"], "2024-01-01", "2024-01-31")
        assert out.empty
        assert is_feed_outage(out) is True
        assert get_feed_status(out)["reason"] == "fredapi_not_installed"

    def test_client_init_failed_is_outage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fred_source, "_get_api_key", lambda: "k")

        def _raising_fred(**_kw: object) -> object:
            raise RuntimeError("client init boom")

        fake = types.ModuleType("fredapi")
        fake.Fred = _raising_fred  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fredapi", fake)
        out = fetch_fred_series(["DGS10"], "2024-01-01", "2024-01-31")
        assert out.empty
        assert is_feed_outage(out) is True
        assert get_feed_status(out)["reason"] == "client_init_failed"

    def test_all_series_errored_is_outage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fred_source, "_get_api_key", lambda: "k")
        _install_fake_fredapi(monkeypatch)
        monkeypatch.setattr(
            fred_source, "_fetch_single_series", lambda *a, **k: (None, False)
        )
        out = fetch_fred_series(["DGS10", "VIXCLS"], "2024-01-01", "2024-01-31")
        assert out.empty
        assert is_feed_outage(out) is True
        assert get_feed_status(out)["reason"] == "all_series_errored"

    def test_empty_window_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fred_source, "_get_api_key", lambda: "k")
        _install_fake_fredapi(monkeypatch)
        empty = pd.DataFrame(columns=["timestamp", "series_id", "value"])
        monkeypatch.setattr(
            fred_source, "_fetch_single_series", lambda *a, **k: (empty, False)
        )
        out = fetch_fred_series(["DGS10"], "2024-01-01", "2024-01-31")
        assert out.empty
        assert is_feed_outage(out) is False
        assert get_feed_status(out)["status"] == "empty"

    def test_partial_outage_is_ok_with_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fred_source, "_get_api_key", lambda: "k")
        _install_fake_fredapi(monkeypatch)
        results = iter([(_fred_row(), False), (None, False)])
        monkeypatch.setattr(
            fred_source, "_fetch_single_series", lambda *a, **k: next(results)
        )
        out = fetch_fred_series(["DGS10", "VIXCLS"], "2024-01-01", "2024-01-31")
        assert not out.empty
        st = get_feed_status(out)
        assert st["status"] == "ok"
        assert st["reason"] == "partial_outage"
        assert is_feed_outage(out) is False


# ---------------------------------------------------------------------------
# yfinance price source
# ---------------------------------------------------------------------------


class TestYFinanceFeedStatus:
    def test_no_symbols_is_empty(self) -> None:
        out = fetch_prices_yfinance([], "2024-01-01", "2024-01-31")
        assert out.empty
        assert get_feed_status(out)["status"] == "empty"
        assert is_feed_outage(out) is False

    def test_all_symbols_errored_is_outage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            yfinance_source, "_fetch_single_symbol", lambda *a, **k: None
        )
        out = fetch_prices_yfinance(["AAPL", "SPY"], "2024-01-01", "2024-01-31")
        assert out.empty
        assert is_feed_outage(out) is True
        assert get_feed_status(out)["reason"] == "all_symbols_errored"

    def test_empty_window_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        empty = pd.DataFrame(columns=yfinance_source._PRICE_COLS)
        monkeypatch.setattr(
            yfinance_source, "_fetch_single_symbol", lambda *a, **k: empty
        )
        out = fetch_prices_yfinance(["AAPL"], "2024-01-01", "2024-01-31")
        assert out.empty
        assert is_feed_outage(out) is False
        assert get_feed_status(out)["status"] == "empty"

    def test_partial_outage_is_ok_with_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = iter([_price_row("AAPL"), None])
        monkeypatch.setattr(
            yfinance_source, "_fetch_single_symbol", lambda *a, **k: next(results)
        )
        out = fetch_prices_yfinance(["AAPL", "SPY"], "2024-01-01", "2024-01-31")
        assert not out.empty
        st = get_feed_status(out)
        assert st["status"] == "ok"
        assert st["reason"] == "partial_outage"
        assert is_feed_outage(out) is False

    def test_success_is_ok_no_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            yfinance_source, "_fetch_single_symbol", lambda *a, **k: _price_row("AAPL")
        )
        out = fetch_prices_yfinance(["AAPL"], "2024-01-01", "2024-01-31")
        st = get_feed_status(out)
        assert st["status"] == "ok"
        assert st["reason"] is None
        assert is_feed_outage(out) is False
