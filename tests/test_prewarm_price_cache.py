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


def test_merge_seam_guard_aborts_on_adjustment_mismatch(tmp_path):
    """E-165-Pin (2026-08-17): RAW-Bars in einen TR-Cache mergen erzeugt eine
    Naht mit riesiger Scheinrendite — merge_and_save muss fail-closed abbrechen
    statt zu schreiben. Genau dieser Fall hat am 17.08. den Live-Cache
    beschaedigt (BKNG +2444 %)."""
    import pandas as pd
    import pytest

    mod = _load_module()
    cache = tmp_path / "daily.parquet"
    days = pd.to_datetime(["2026-08-04", "2026-08-05"], utc=True)
    old = pd.DataFrame(
        {
            "timestamp": days,
            "symbol": ["BKNG", "BKNG"],
            "open": [140.0, 140.4],
            "high": [141, 141],
            "low": [139, 139],
            "close": [140.0, 140.42],
            "volume": [1e6, 1e6],
            "adj_close": [140.0, 140.42],
        }
    )
    old.to_parquet(cache, index=False)
    new = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-08-06"], utc=True),
            "symbol": ["BKNG"],
            "open": [3570.0],
            "high": [3580],
            "low": [3560],
            "close": [3572.0],
            "volume": [1e6],
            "adj_close": [3572.0],  # RAW!
        }
    )
    with pytest.raises(RuntimeError, match="seam guard"):
        mod.merge_and_save(new, cache_path=cache)
    # fail-closed: Cache unveraendert
    assert len(pd.read_parquet(cache)) == 2


def test_merge_seam_guard_passes_clean_continuation(tmp_path):
    """Gegenprobe: stetige Fortsetzung derselben Adjustierungsbasis merged."""
    import pandas as pd

    mod = _load_module()
    cache = tmp_path / "daily.parquet"
    days = pd.to_datetime(["2026-08-04", "2026-08-05"], utc=True)
    old = pd.DataFrame(
        {
            "timestamp": days,
            "symbol": ["NFLX", "NFLX"],
            "open": [73, 73.5],
            "high": [74, 74],
            "low": [72, 73],
            "close": [73.57, 74.20],
            "volume": [1e6, 1e6],
            "adj_close": [73.57, 74.20],
        }
    )
    old.to_parquet(cache, index=False)
    new = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-08-06"], utc=True),
            "symbol": ["NFLX"],
            "open": [74.1],
            "high": [75],
            "low": [74],
            "close": [74.80],
            "volume": [1e6],
            "adj_close": [74.80],
        }
    )
    n = mod.merge_and_save(new, cache_path=cache)
    assert n == 3


def test_alpaca_request_pins_adjustment_all(monkeypatch):
    """F-auditor-1a (2026-08-17): pinnt den ROOT-CAUSE-Fix des Cache-
    Zwischenfalls — StockBarsRequest MUSS adjustment=Adjustment.ALL setzen.
    Ohne diesen Pin koennte ein Refactor den Parameter still entfernen und
    der Alpaca-Default (RAW) beschaedigte den TR-Cache erneut (E-165)."""
    import sys
    import types

    import pandas as pd

    captured = {}

    class _FakeAdjustment:
        ALL = "all"
        RAW = "raw"

    class _FakeRequest:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeBars:
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "timestamp": pd.to_datetime(["2026-08-14"], utc=True),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        ).set_index(["symbol", "timestamp"])

    class _FakeClient:
        def __init__(self, **kwargs): ...
        def get_stock_bars(self, request):
            return _FakeBars()

    fake_data = types.ModuleType("alpaca.data")
    fake_data.StockHistoricalDataClient = _FakeClient
    fake_req = types.ModuleType("alpaca.data.requests")
    fake_req.StockBarsRequest = _FakeRequest
    fake_tf = types.ModuleType("alpaca.data.timeframe")
    fake_tf.TimeFrame = types.SimpleNamespace(Day="day")
    fake_enums = types.ModuleType("alpaca.data.enums")
    fake_enums.Adjustment = _FakeAdjustment
    fake_root = types.ModuleType("alpaca")
    for name, mod in (
        ("alpaca", fake_root),
        ("alpaca.data", fake_data),
        ("alpaca.data.requests", fake_req),
        ("alpaca.data.timeframe", fake_tf),
        ("alpaca.data.enums", fake_enums),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")

    mod = _load_module()
    df = mod.fetch_missing_alpaca(["AAPL"], years=1)
    assert captured.get("adjustment") == _FakeAdjustment.ALL, captured
    assert not df.empty and "adj_close" in df.columns  # Spiegel-Konvention


def test_merge_rescales_history_on_constant_overlap_ratio(tmp_path):
    """F-auditor-1b: konstante Overlap-Ratio != 1 (Corporate Action zwischen
    den Adjustierungs-Ankern) reskaliert den GESAMTEN Bestand auf den neuen
    Anker — auch die Bars VOR dem Overlap."""
    import pandas as pd

    mod = _load_module()
    cache = tmp_path / "daily.parquet"
    days = pd.bdate_range("2026-07-01", periods=10, tz="UTC")
    old = pd.DataFrame(
        {
            "timestamp": days,
            "symbol": "DIV",
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1e6,
            "adj_close": 100.0,
        }
    )
    old.to_parquet(cache, index=False)
    # Neue Quelle: Overlap auf den letzten 6 Tagen mit konstant 0.99 + 1 neuer Bar
    new_days = list(days[4:]) + [days[-1] + pd.Timedelta(days=1)]
    new = pd.DataFrame(
        {
            "timestamp": new_days,
            "symbol": "DIV",
            "open": 99.0,
            "high": 99.0,
            "low": 99.0,
            "close": 99.0,
            "volume": 1e6,
            "adj_close": 99.0,
        }
    )
    mod.merge_and_save(new, cache_path=cache)
    out = pd.read_parquet(cache).sort_values("timestamp")
    first_bar = float(out["close"].iloc[0])  # VOR dem Overlap
    assert first_bar == pytest.approx(99.0), (
        f"Bestand vor dem Overlap nicht reskaliert: {first_bar}"
    )


def test_merge_enforces_adj_close_invariant_at_write(tmp_path):
    """F-auditor-1c: new_df OHNE adj_close-Spalte (yfinance-Pfad!) darf die
    0-NaN-Invariante nicht aufreissen — Erzwingung am Schreibpunkt (E-166)."""
    import pandas as pd

    mod = _load_module()
    cache = tmp_path / "daily.parquet"
    days = pd.bdate_range("2026-08-03", periods=3, tz="UTC")
    old = pd.DataFrame(
        {
            "timestamp": days,
            "symbol": "YFI",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
            "adj_close": 10.0,
        }
    )
    old.to_parquet(cache, index=False)
    new = pd.DataFrame(
        {
            "timestamp": [days[-1] + pd.Timedelta(days=1)],
            "symbol": ["YFI"],
            "open": [10.1],
            "high": [10.2],
            "low": [10.0],
            "close": [10.15],
            "volume": [1e6],
        }
    )  # KEIN adj_close — wie fetch_prices_yfinance
    mod.merge_and_save(new, cache_path=cache)
    out = pd.read_parquet(cache)
    assert int(out["adj_close"].isna().sum()) == 0
    assert float(out.sort_values("timestamp")["adj_close"].iloc[-1]) == 10.15
