"""Tests for scripts/ops/refresh_sector_etf_cache.py.

Verifies the sector-ETF + SPY freshness bridge that keeps the live
multifactor_v2 sector_rotation_bias factor computing on fresh data instead of
being neutralised by its 7-day staleness guard. yfinance is monkeypatched —
these tests never hit the network.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json as _json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ops"
    / "refresh_sector_etf_cache.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("refresh_sector_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_cache(tmp_path: Path, *, latest: str, syms: list[str]) -> Path:
    """Write a tiny daily.parquet-shaped cache (with adj_close) ending at `latest`."""
    dates = pd.date_range(end=pd.Timestamp(latest, tz="UTC"), periods=5, freq="D")
    rows = []
    for sym in syms:
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
    p = tmp_path / "daily.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def _fetch_frame(syms: list[str], date_strs: list[str]) -> pd.DataFrame:
    """yfinance-shaped frame (no adj_close) for the given symbols/dates."""
    rows = []
    for sym in syms:
        for ds in date_strs:
            rows.append(
                {
                    "timestamp": pd.Timestamp(ds, tz="UTC"),
                    "symbol": sym,
                    "open": 110.0,
                    "high": 111.0,
                    "low": 109.0,
                    "close": 110.5,
                    "volume": 2_000_000,
                }
            )
    return pd.DataFrame(rows)


def _patch_fetch(mod, frame: pd.DataFrame):
    """Replace the module's fetch_prices_yfinance with one returning `frame`
    filtered to the requested symbols (mimicking the real signature)."""

    def _fake(symbols, start_date, end_date, **kwargs):
        return frame[frame["symbol"].isin(symbols)].copy()

    mod.fetch_prices_yfinance = _fake


TODAY = _dt.date(2026, 6, 1)


def test_appends_fresh_rows_and_mirrors_adj_close(tmp_path, monkeypatch):
    """Fetched rows newer than cache -> appended with adj_close mirrored from
    close, cache schema preserved (no feed-status cols leak)."""
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    syms = mod.TARGET_SYMBOLS
    cache_path = _make_cache(tmp_path, latest="2026-05-18", syms=syms)
    _patch_fetch(mod, _fetch_frame(syms, ["2026-05-27", "2026-05-28"]))

    n = mod.refresh(cache_path, dry_run=False, today=TODAY)

    assert n > 0
    out = pd.read_parquet(cache_path)
    assert out["timestamp"].max() == pd.Timestamp("2026-05-28", tz="UTC")
    new_rows = out[out["timestamp"] > pd.Timestamp("2026-05-18", tz="UTC")]
    assert len(new_rows) == len(syms) * 2
    # CONTRACT CHANGE 2026-08-15: adj_close is mirrored from close, not NaN.
    # This fetcher calls yfinance with auto_adjust=True, so close is already
    # split- and dividend-adjusted — there is no unadjusted series the old NaN
    # "sentinel" could have been protecting against. Across the real cache,
    # every populated adj_close equalled close exactly (180,734 of 180,734).
    assert not new_rows["adj_close"].isna().any(), (
        "appended rows must not carry NaN in adj_close (auto_adjust=True)"
    )
    assert (new_rows["adj_close"] == new_rows["close"]).all(), (
        "adj_close must equal close exactly for appended rows"
    )
    old_rows = out[out["timestamp"] <= pd.Timestamp("2026-05-18", tz="UTC")]
    assert not old_rows["adj_close"].isna().any()
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


def test_excludes_today_partial_bar_pit(tmp_path, monkeypatch):
    """PIT: a bar dated == today (forming, pre-close) must NOT be appended."""
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    syms = mod.TARGET_SYMBOLS
    cache_path = _make_cache(tmp_path, latest="2026-05-18", syms=syms)
    # Fetch returns a settled bar (05-29) AND today's partial bar (06-01).
    _patch_fetch(mod, _fetch_frame(syms, ["2026-05-29", "2026-06-01"]))

    n = mod.refresh(cache_path, dry_run=False, today=TODAY)

    out = pd.read_parquet(cache_path)
    assert n == len(syms), "only the settled 05-29 bar per symbol should append"
    assert out["timestamp"].max() == pd.Timestamp("2026-05-29", tz="UTC")
    assert (out["timestamp"] == pd.Timestamp("2026-06-01", tz="UTC")).sum() == 0


def test_status_json_payload(tmp_path, monkeypatch):
    mod = _load_module()
    status_path = tmp_path / "ops" / "refresh_sector_etf_status.json"
    monkeypatch.setattr(mod, "STATUS_PATH", status_path)
    syms = mod.TARGET_SYMBOLS
    cache_path = _make_cache(tmp_path, latest="2026-05-18", syms=syms)
    _patch_fetch(mod, _fetch_frame(syms, ["2026-05-28"]))

    mod.refresh(cache_path, dry_run=False, today=TODAY)

    assert status_path.exists()
    payload = _json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["rc"] >= 0
    assert payload["rows_appended"] == len(syms)
    assert payload["symbols"] == syms
    assert payload["error"] is None
    assert "cache_latest" in payload and "fetch_latest" in payload


def test_idempotent_when_fetch_not_newer(tmp_path, monkeypatch):
    """Fetch returns only rows <= cache max → no-op, file unchanged."""
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    syms = mod.TARGET_SYMBOLS
    cache_path = _make_cache(tmp_path, latest="2026-05-18", syms=syms)
    _patch_fetch(mod, _fetch_frame(syms, ["2026-05-16", "2026-05-18"]))

    before = pd.read_parquet(cache_path)
    n = mod.refresh(cache_path, dry_run=False, today=TODAY)
    after = pd.read_parquet(cache_path)

    assert n == 0
    pd.testing.assert_frame_equal(before, after)


def test_dry_run_does_not_write(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    syms = mod.TARGET_SYMBOLS
    cache_path = _make_cache(tmp_path, latest="2026-05-18", syms=syms)
    _patch_fetch(mod, _fetch_frame(syms, ["2026-05-28"]))

    before = pd.read_parquet(cache_path)
    n = mod.refresh(cache_path, dry_run=True, today=TODAY)
    after = pd.read_parquet(cache_path)

    assert n > 0  # reports what WOULD append
    pd.testing.assert_frame_equal(before, after)  # but didn't write


def test_drops_duplicate_symbol_timestamp_pairs(tmp_path, monkeypatch):
    """Fetch overlaps cache's latest day → keep last, no dup (symbol, ts)."""
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    syms = mod.TARGET_SYMBOLS
    cache_path = _make_cache(tmp_path, latest="2026-05-18", syms=syms)
    # 05-18 overlaps cache; 05-19 is new.
    _patch_fetch(mod, _fetch_frame(syms, ["2026-05-18", "2026-05-19"]))

    n = mod.refresh(cache_path, dry_run=False, today=TODAY)
    out = pd.read_parquet(cache_path)

    assert n == len(syms)  # only the strictly-newer 05-19 rows
    assert out.duplicated(subset=["symbol", "timestamp"]).sum() == 0


def test_missing_cache_returns_minus_one(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    _patch_fetch(mod, _fetch_frame(mod.TARGET_SYMBOLS, ["2026-05-28"]))
    rc = mod.refresh(tmp_path / "nope.parquet", dry_run=False, today=TODAY)
    assert rc == -1


def test_rate_limit_with_stale_cache_flags_error(tmp_path, monkeypatch):
    """yfinance 429 UND veraltete Sektordaten → rc 0, Cache unveraendert,
    error nennt Rate-Limit UND Alter.

    VERTRAGS-UPDATE 2026-08-18 (E-189): das error-Feld wurde vorher BLIND bei
    jedem 429 gesetzt. Da alle 9 TARGET_SYMBOLS ueber prewarm/Alpaca frisch
    gehalten werden, erzeugte das einen taeglichen Degradiert-Alarm ohne
    realen Mangel (Alert-Fatigue). Jetzt entscheidet die FRISCHE — dieser
    Test deckt den Fall, in dem sie wirklich fehlt (Cache 2026-05-18 vs.
    today 2026-06-01 = 14 Tage, Guard 7 Tage)."""
    mod = _load_module()
    status_path = tmp_path / "ops" / "status.json"
    monkeypatch.setattr(mod, "STATUS_PATH", status_path)
    syms = mod.TARGET_SYMBOLS
    cache_path = _make_cache(tmp_path, latest="2026-05-18", syms=syms)

    def _raise(*a, **k):
        raise mod.YFinanceRateLimitError("429")

    mod.fetch_prices_yfinance = _raise
    before = pd.read_parquet(cache_path)

    n = mod.refresh(cache_path, dry_run=False, today=TODAY)
    after = pd.read_parquet(cache_path)

    assert n == 0
    pd.testing.assert_frame_equal(before, after)
    payload = _json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True  # rc 0 is not a hard error
    assert payload["error"] is not None
    assert "yfinance_rate_limited" in payload["error"]
    assert "stale" in payload["error"]  # nennt den echten Mangel


def test_rate_limit_with_fresh_cache_is_no_degradation(tmp_path, monkeypatch):
    """E-189-Gegenprobe: 429 bei FRISCHEN Sektordaten ist KEIN Mangel —
    error bleibt None, der Watchdog-Konsument schweigt zu Recht."""
    mod = _load_module()
    status_path = tmp_path / "ops" / "status.json"
    monkeypatch.setattr(mod, "STATUS_PATH", status_path)
    syms = mod.TARGET_SYMBOLS
    # Cache endet einen Tag vor `today` -> innerhalb des 7-Tage-Guards.
    cache_path = _make_cache(tmp_path, latest="2026-05-31", syms=syms)

    def _raise(*a, **k):
        raise mod.YFinanceRateLimitError("429")

    mod.fetch_prices_yfinance = _raise
    n = mod.refresh(cache_path, dry_run=False, today=TODAY)

    assert n == 0
    payload = _json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["error"] is None, (
        "frische Sektordaten duerfen keinen Degradiert-Alarm erzeugen "
        "(taeglicher Fehlalarm vor E-189)"
    )


def test_empty_fetch_degrades_gracefully(tmp_path, monkeypatch):
    """yfinance returns empty frame (outage/no rows) → rc 0, cache unchanged."""
    mod = _load_module()
    status_path = tmp_path / "ops" / "status.json"
    monkeypatch.setattr(mod, "STATUS_PATH", status_path)
    syms = mod.TARGET_SYMBOLS
    cache_path = _make_cache(tmp_path, latest="2026-05-18", syms=syms)
    mod.fetch_prices_yfinance = lambda *a, **k: pd.DataFrame(
        columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    )

    before = pd.read_parquet(cache_path)
    n = mod.refresh(cache_path, dry_run=False, today=TODAY)
    after = pd.read_parquet(cache_path)

    assert n == 0
    pd.testing.assert_frame_equal(before, after)
    payload = _json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["error"] == "yfinance_empty"


def test_per_symbol_freshness_when_global_max_matches(tmp_path, monkeypatch):
    """One sector ETF stale in cache while another is fresh → the stale one is
    refreshed even though the global cache max already equals the fetch max.

    FIXTURE-FIX 2026-08-17: Cache-Preise muessen zur Fetch-Fixture (110.5)
    konsistent sein — die alte 1.0-Fixture erzeugte eine +10.950%-Naht, die
    der neue guarded_merge-Naht-Guard KORREKT abbricht (E-165). Der Test
    prueft Freshness-Logik, nicht Naht-Verhalten."""
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    a, b = mod.TARGET_SYMBOLS[0], mod.TARGET_SYMBOLS[1]
    rows = []
    for d in pd.date_range(end="2026-05-28", periods=3, freq="D", tz="UTC"):
        rows.append(
            {
                "timestamp": d,
                "symbol": a,
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "close": 110.5,
                "adj_close": 110.5,
                "volume": 1,
            }
        )
    for d in pd.date_range(end="2026-05-10", periods=3, freq="D", tz="UTC"):
        rows.append(
            {
                "timestamp": d,
                "symbol": b,
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "close": 110.5,
                "adj_close": 110.5,
                "volume": 1,
            }
        )
    cache_path = tmp_path / "daily.parquet"
    pd.DataFrame(rows).to_parquet(cache_path, index=False)

    # Fetch carries both up to 2026-05-28 (global max already in cache via `a`).
    _patch_fetch(mod, _fetch_frame([a, b], ["2026-05-27", "2026-05-28"]))

    n = mod.refresh(cache_path, dry_run=False, today=TODAY)

    assert n > 0, "stale symbol must refresh even when global max matches"
    out = pd.read_parquet(cache_path)
    assert out[out["symbol"] == b]["timestamp"].max() == pd.Timestamp(
        "2026-05-28", tz="UTC"
    )


def test_absent_symbol_gets_all_fetched_rows(tmp_path, monkeypatch):
    """A target symbol absent from the cache gets all its fetched rows appended."""
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    present = mod.TARGET_SYMBOLS[0]
    absent = mod.TARGET_SYMBOLS[1]
    cache_path = _make_cache(tmp_path, latest="2026-05-18", syms=[present])
    _patch_fetch(mod, _fetch_frame([present, absent], ["2026-05-27", "2026-05-28"]))

    n = mod.refresh(cache_path, dry_run=False, today=TODAY)
    out = pd.read_parquet(cache_path)

    assert absent in set(out["symbol"])
    absent_rows = out[out["symbol"] == absent]
    assert len(absent_rows) == 2
    # See the contract-change note above: mirrored from close, never NaN.
    assert not absent_rows["adj_close"].isna().any()
    assert (absent_rows["adj_close"] == absent_rows["close"]).all()


def test_non_target_cache_rows_are_preserved(tmp_path, monkeypatch):
    """A non-sector symbol in the cache must be byte-preserved across a refresh
    (the refresh only ever touches TARGET_SYMBOLS)."""
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    syms = mod.TARGET_SYMBOLS
    cache_path = _make_cache(tmp_path, latest="2026-05-18", syms=syms)
    # Inject a non-target symbol with its own rows.
    cache = pd.read_parquet(cache_path)
    aapl_rows = [
        {
            "timestamp": d,
            "symbol": "AAPL",
            "open": 50.0,
            "high": 51.0,
            "low": 49.0,
            "close": 50.5,
            "adj_close": 50.5,
            "volume": 3_000_000,
        }
        for d in pd.date_range(end="2026-05-18", periods=5, freq="D", tz="UTC")
    ]
    pd.concat([cache, pd.DataFrame(aapl_rows)], ignore_index=True).to_parquet(
        cache_path, index=False
    )

    before_aapl = (
        pd.read_parquet(cache_path)
        .query("symbol == 'AAPL'")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    _patch_fetch(mod, _fetch_frame(syms, ["2026-05-27", "2026-05-28"]))

    n = mod.refresh(cache_path, dry_run=False, today=TODAY)
    assert n == len(syms) * 2

    after_aapl = (
        pd.read_parquet(cache_path)
        .query("symbol == 'AAPL'")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(before_aapl, after_aapl)


def test_returns_nonneg_and_main_exit_code(tmp_path, monkeypatch):
    """refresh() returning >=0 maps to a 0 process exit via main()'s rc gate."""
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    syms = mod.TARGET_SYMBOLS
    cache_path = _make_cache(tmp_path, latest="2026-05-18", syms=syms)
    monkeypatch.setattr(mod, "CACHE_PATH", cache_path)
    _patch_fetch(mod, _fetch_frame(syms, ["2026-05-28"]))
    monkeypatch.setattr(__import__("sys"), "argv", ["refresh_sector_etf_cache.py"])

    assert mod.main() == 0


def test_seam_guard_blocks_adjustment_mismatch(tmp_path, monkeypatch):
    """E-165-Pin fuer den 5. Cache-Schreiber (2026-08-17): eine
    Adjustierungs-Naht (Cache TR-adjustiert 1.0, Fetch RAW-artig 110.5)
    muss den Refresh fail-closed abbrechen — rc=-2, Cache byte-unveraendert."""
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    a = mod.TARGET_SYMBOLS[0]
    rows = [
        {
            "timestamp": d,
            "symbol": a,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adj_close": 1.0,
            "volume": 1,
        }
        for d in pd.date_range(end="2026-05-26", periods=3, freq="D", tz="UTC")
    ]
    cache_path = tmp_path / "daily.parquet"
    pd.DataFrame(rows).to_parquet(cache_path, index=False)
    before = pd.read_parquet(cache_path)

    _patch_fetch(mod, _fetch_frame([a], ["2026-05-27", "2026-05-28"]))  # 110.5!
    rc = mod.refresh(cache_path, dry_run=False, today=TODAY)
    # rc=-2 seit F-TR-1: negativ -> ok=false im Status + Exit 1 -> Operator
    # sieht den Abbruch (vorher rc=2 = ambig mit "2 rows appended", ok=true).
    assert rc == -2
    after = pd.read_parquet(cache_path)
    pd.testing.assert_frame_equal(before, after)  # nichts geschrieben
    # F-senior-7: die beiden SICHTBARKEITS-Ebenen pinnen (gelesen != getestet):
    status = _json.loads((tmp_path / "ops" / "status.json").read_text("utf-8"))
    assert status["ok"] is False
    assert "seam_guard" in (status["error"] or "")


def test_overlap_drop_lands_in_status_json(tmp_path, monkeypatch):
    """F-auditor-3-Pin (Stage 3, 2026-08-17): der Drop-Zweig der guarded_merge
    (inkonstante Overlap-Ratio -> Symbol verworfen) landet maschinenlesbar im
    Status-JSON: status['dropped_symbols'] + error-Feld. EHRLICHE GRENZE
    (E-176): das Status-JSON hat noch KEINEN Leser und der Drop-Pfad endet
    mit Exit 0 — ein Konsument/Alarm ist offener Follow-up, kein Ist-Zustand.

    VERTRAG (bewusst entschieden): ok bleibt True auf dem Drop-Pfad — der
    Refresh ist ein PARTIELLER Erfolg (uebrige Symbole geschrieben, Cache
    konsistent). ok=False ist exklusiv fuer den Naht-Abbruch (rc=-2, NICHTS
    geschrieben). Drop-Monitoring geht ueber dropped_symbols/error, nicht ok."""
    mod = _load_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "ops" / "status.json")
    bad, good = mod.TARGET_SYMBOLS[0], mod.TARGET_SYMBOLS[1]
    cache_path = _make_cache(tmp_path, latest="2026-05-26", syms=[bad, good])

    # Overlap = die 5 Cache-Tage: `bad` mit Ausreisser-Ratio (150 vs 100.5 an
    # Tag 3 -> spread >> overlap_spread_max), `good` exakt konstant 1.0;
    # dazu je 1 echter Neutag.
    dates = pd.date_range(end=pd.Timestamp("2026-05-26", tz="UTC"), periods=5, freq="D")
    rows = []
    for i, d in enumerate(dates):
        for sym, close in ((bad, 150.0 if i == 2 else 100.5), (good, 100.5)):
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": close,
                    "volume": 2_000_000,
                }
            )
    for sym in (bad, good):
        rows.append(
            {
                "timestamp": pd.Timestamp("2026-05-27", tz="UTC"),
                "symbol": sym,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 2_000_000,
            }
        )
    _patch_fetch(mod, pd.DataFrame(rows))

    rc = mod.refresh(cache_path, dry_run=False, today=TODAY)

    assert rc == 1  # nur der good-Neutag zaehlt (Drop nicht als appended, F-TR-4)
    out = pd.read_parquet(cache_path)
    newer = out[out["timestamp"] > pd.Timestamp("2026-05-26", tz="UTC")]
    assert set(newer["symbol"]) == {good}  # bad-Zeilen verworfen
    status = _json.loads((tmp_path / "ops" / "status.json").read_text("utf-8"))
    assert status["dropped_symbols"] == [bad]
    assert "overlap_ratio_not_constant" in (status["error"] or "")
    assert status["ok"] is True  # partieller Erfolg, siehe Docstring-Vertrag
