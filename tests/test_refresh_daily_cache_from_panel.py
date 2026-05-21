"""Tests for scripts/ops/refresh_daily_cache_from_panel.py.

Verifies the offline cache-bridge that keeps the paper pilot's
output/aggregates/daily.parquet fresh enough to pass the 3-day staleness
check in scripts/run_live_paper.py:_load_prices when yfinance is rate-limited.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ops"
    / "refresh_daily_cache_from_panel.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("refresh_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_cache(tmp_path: Path, latest: str) -> Path:
    """Write a tiny daily.parquet-shaped cache ending at `latest`."""
    dates = pd.date_range(end=pd.Timestamp(latest, tz="UTC"), periods=5, freq="D")
    rows = []
    for sym in ["AAPL", "MSFT"]:
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
    df = pd.DataFrame(rows)
    p = tmp_path / "daily.parquet"
    df.to_parquet(p, index=False)
    return p


def _make_panel(tmp_path: Path, latest: str, syms: list[str]) -> Path:
    """Write a master_universe_panel.parquet-shaped frame (no adj_close)."""
    dates = pd.date_range(end=pd.Timestamp(latest, tz="UTC"), periods=10, freq="D")
    rows = []
    for sym in syms:
        for d in dates:
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "open": 110.0,
                    "high": 111.0,
                    "low": 109.0,
                    "close": 110.5,
                    "volume": 2_000_000,
                }
            )
    df = pd.DataFrame(rows)
    p = tmp_path / "master_universe_panel.parquet"
    df.to_parquet(p, index=False)
    return p


def test_refresh_appends_fresher_panel_rows_and_sets_adj_close_nan(tmp_path):
    """Panel newer than cache → append new rows. F-RX-3: panel lacks adj_close,
    appended rows get NaN sentinel (NOT close) so direct-parquet consumers can
    detect-and-handle. Live-paper hot-path strips adj_close at load and is
    unaffected.
    """

    mod = _load_module()
    cache_path = _make_cache(tmp_path, latest="2026-05-14")
    panel_path = _make_panel(tmp_path, latest="2026-05-18", syms=["AAPL", "MSFT"])

    n = mod.refresh(cache_path, panel_path, dry_run=False)

    assert n > 0
    out = pd.read_parquet(cache_path)
    assert out["timestamp"].max() == pd.Timestamp("2026-05-18", tz="UTC")
    new_rows = out[out["timestamp"] > pd.Timestamp("2026-05-14", tz="UTC")]
    assert len(new_rows) > 0
    # Panel had no adj_close → set to NaN sentinel (F-RX-3 §9.12 (a))
    assert (
        new_rows["adj_close"].isna().all()
    ), "appended rows must carry NaN sentinel in adj_close, not silent close-fallback"
    # Original cache rows still have their valid adj_close (close-equivalent)
    cache_old = out[out["timestamp"] <= pd.Timestamp("2026-05-14", tz="UTC")]
    assert not cache_old["adj_close"].isna().any()
    # Cache schema preserved (no extra panel cols leaked)
    assert set(out.columns) == {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    }


def test_refresh_writes_status_json_with_payload(tmp_path, monkeypatch):
    """F-RX-5 §9.12 (c): refresh writes a status JSON sidecar for ops monitoring."""
    mod = _load_module()
    status_path = tmp_path / "ops" / "refresh_cache_status.json"
    monkeypatch.setattr(mod, "STATUS_PATH", status_path)

    cache_path = _make_cache(tmp_path, latest="2026-05-14")
    panel_path = _make_panel(tmp_path, latest="2026-05-18", syms=["AAPL"])

    mod.refresh(cache_path, panel_path, dry_run=False)

    assert status_path.exists()
    import json as _json

    payload = _json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["rc"] >= 0
    assert payload["rows_appended"] > 0
    assert "panel_latest" in payload and "cache_latest" in payload
    assert payload["error"] is None


def test_refresh_writes_status_json_on_missing_file(tmp_path, monkeypatch):
    """Status JSON reports rc=-1 + error when source files are missing."""
    mod = _load_module()
    status_path = tmp_path / "ops" / "refresh_cache_status.json"
    monkeypatch.setattr(mod, "STATUS_PATH", status_path)

    rc = mod.refresh(
        tmp_path / "nope_cache.parquet",
        tmp_path / "nope_panel.parquet",
        dry_run=False,
    )
    assert rc == -1
    assert status_path.exists()
    import json as _json

    payload = _json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["rc"] == -1
    assert payload["error"] is not None


def test_refresh_idempotent_when_panel_not_newer(tmp_path):
    """Panel same or older than cache → no-op."""
    mod = _load_module()
    cache_path = _make_cache(tmp_path, latest="2026-05-18")
    panel_path = _make_panel(tmp_path, latest="2026-05-18", syms=["AAPL", "MSFT"])

    before = pd.read_parquet(cache_path)
    n = mod.refresh(cache_path, panel_path, dry_run=False)
    after = pd.read_parquet(cache_path)

    assert n == 0
    pd.testing.assert_frame_equal(before, after)


def test_refresh_dry_run_does_not_write(tmp_path):
    mod = _load_module()
    cache_path = _make_cache(tmp_path, latest="2026-05-14")
    panel_path = _make_panel(tmp_path, latest="2026-05-18", syms=["AAPL"])

    before = pd.read_parquet(cache_path)
    n = mod.refresh(cache_path, panel_path, dry_run=True)
    after = pd.read_parquet(cache_path)

    assert n > 0  # reports what WOULD be appended
    pd.testing.assert_frame_equal(before, after)  # but didn't write


def test_refresh_drops_duplicate_symbol_timestamp_pairs(tmp_path):
    """If cache and panel both have the same (symbol, ts), keep last (panel wins)."""
    mod = _load_module()
    cache_path = _make_cache(tmp_path, latest="2026-05-18")
    # Panel overlaps with cache on the latest day
    panel_path = _make_panel(tmp_path, latest="2026-05-20", syms=["AAPL"])

    n = mod.refresh(cache_path, panel_path, dry_run=False)
    out = pd.read_parquet(cache_path)

    # New rows appended (panel was newer)
    assert n > 0
    # No duplicate (symbol, timestamp) rows
    dups = out.duplicated(subset=["symbol", "timestamp"]).sum()
    assert dups == 0


def test_refresh_missing_cache_returns_minus_one(tmp_path):
    mod = _load_module()
    panel_path = _make_panel(tmp_path, latest="2026-05-18", syms=["AAPL"])
    cache_path = tmp_path / "does_not_exist.parquet"

    rc = mod.refresh(cache_path, panel_path, dry_run=False)
    assert rc == -1


def test_refresh_missing_panel_returns_minus_one(tmp_path):
    mod = _load_module()
    cache_path = _make_cache(tmp_path, latest="2026-05-14")
    panel_path = tmp_path / "does_not_exist.parquet"

    rc = mod.refresh(cache_path, panel_path, dry_run=False)
    assert rc == -1


def test_refresh_picks_up_per_symbol_freshness_when_global_max_matches(tmp_path):
    """F-RX-2 regression guard (audit 2026-05-21).

    Even when ``panel.timestamp.max() == cache.timestamp.max()`` globally,
    individual symbols may have stale rows in cache that the panel has
    refreshed. The refresh must compare per-symbol latest, not global max.
    """
    mod = _load_module()
    # Build cache: AAPL fresh up to 2026-05-18, but MSFT only up to 2026-05-10
    cache_rows = []
    for d in pd.date_range(end="2026-05-18", periods=5, freq="D", tz="UTC"):
        cache_rows.append(
            {
                "timestamp": d,
                "symbol": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_close": 100.5,
                "volume": 1_000_000,
            }
        )
    for d in pd.date_range(end="2026-05-10", periods=5, freq="D", tz="UTC"):
        cache_rows.append(
            {
                "timestamp": d,
                "symbol": "MSFT",
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.5,
                "adj_close": 200.5,
                "volume": 2_000_000,
            }
        )
    cache_path = tmp_path / "daily.parquet"
    pd.DataFrame(cache_rows).to_parquet(cache_path, index=False)

    # Panel: both AAPL and MSFT fresh to 2026-05-18 (so panel.max == cache.max
    # globally because AAPL already at 2026-05-18 in cache).
    panel_path = _make_panel(tmp_path, latest="2026-05-18", syms=["AAPL", "MSFT"])

    n = mod.refresh(cache_path, panel_path, dry_run=False)

    assert n > 0, "must append: MSFT is stale per-symbol even though global max matches"
    out = pd.read_parquet(cache_path)
    msft_latest = out[out["symbol"] == "MSFT"]["timestamp"].max()
    assert msft_latest == pd.Timestamp("2026-05-18", tz="UTC")


def test_refresh_per_symbol_no_op_when_all_symbols_already_at_panel_max(tmp_path):
    """Same global+per-symbol latest → strictly nothing to append."""
    mod = _load_module()
    cache_path = _make_cache(tmp_path, latest="2026-05-18")
    panel_path = _make_panel(tmp_path, latest="2026-05-18", syms=["AAPL", "MSFT"])

    before = pd.read_parquet(cache_path)
    n = mod.refresh(cache_path, panel_path, dry_run=False)
    after = pd.read_parquet(cache_path)
    assert n == 0
    pd.testing.assert_frame_equal(before, after)
