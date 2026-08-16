"""Tests for scripts/ops/refresh_daily_cache_from_eodhd.py.

This is the PRIMARY writer for ``output/aggregates/daily.parquet`` — the cache
the paper pilot reads. Until 2026-08-15 no test referenced it at all, and the
E-112 request-protocol wiring was added straight into it. Rule 30/40 asks for
targeted tests when a production risk path is touched; this closes that gap.

Two properties are pinned, stated at the precision they are actually tested:

  1. the protocol is written on every return path AFTER the request loop is
     reachable — five of ``refresh()``'s seven ``return`` statements. The two
     earliest ones (missing token, missing cache) sit before the PullLog is
     constructed and are deliberately NOT covered; saying "every return path"
     would repeat the overclaim this step already logged once.
  2. the protocol is additive with respect to the RETURN CODE: rc is identical
     with and without the PullLog import. This does NOT prove the written rows
     are identical — a row count is not a row comparison (E-148). Extending
     this to a frame comparison is a worthwhile follow-up, not a claim made
     here.

Network is never touched: ``_fetch_symbol`` is stubbed in every test.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ops"
    / "refresh_daily_cache_from_eodhd.py"
)


def _load_module(monkeypatch):
    """Import the script WITHOUT letting it load the real .env.

    The script calls ``load_dotenv(ROOT / ".env")`` at module level. Executing
    it in a test therefore pushes every real credential in that file into
    ``os.environ`` for the rest of the process — measured: 0 of 20 keys before
    the import, 20 of 20 after, and ``monkeypatch.setenv`` teardown restores the
    REAL token rather than removing it. This file never makes a network call,
    but it would set up the conditions for a LATER test file to make one with
    live credentials. That is the test-contamination class closed structurally
    on 2026-07-23, and an import side effect is not an exemption from it
    (Rule 20 + Rule 40).
    """
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    spec = importlib.util.spec_from_file_location("refresh_eodhd_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_cache(path: Path, *, latest: str, syms: list[str]) -> None:
    dates = pd.date_range(end=pd.Timestamp(latest, tz="UTC"), periods=3, freq="D")
    rows = [
        {
            "timestamp": d,
            "symbol": sym,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "adj_close": 100.5,
            "volume": 1_000.0,
        }
        for sym in syms
        for d in dates
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _eod_row(date: str) -> dict:
    return {
        "date": date,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "adjusted_close": 10.5,
        "volume": 5_000,
    }


@pytest.fixture
def mod(tmp_path, monkeypatch):
    """Module with cache, status and protocol paths redirected into tmp_path."""
    m = _load_module(monkeypatch)
    cache = tmp_path / "daily.parquet"
    _make_cache(cache, latest="2026-01-05", syms=["AAA", "BBB"])
    monkeypatch.setattr(m, "CACHE_PATH", cache)
    monkeypatch.setattr(m, "STATUS_PATH", tmp_path / "ops" / "refresh_status.json")
    monkeypatch.setenv("EODHD_API_TOKEN", "test-token-not-a-real-secret")

    import src.assembled_core.data.pull_log as pl

    monkeypatch.setattr(pl, "DEFAULT_LOG_DIR", tmp_path / "ops")
    m._tmp = tmp_path
    return m


def _protocol(tmp_path: Path) -> dict:
    files = sorted((tmp_path / "ops").glob("pull_log_eodhd_eod*.json"))
    assert files, "no protocol written"
    return json.loads(files[-1].read_text(encoding="utf-8"))


# --- protocol on every return path ---------------------------------------


def test_all_fetches_failed_still_writes_protocol(mod, monkeypatch):
    """The run the protocol exists for: provider down, rc=-1, evidence kept."""

    def _boom(tok, sym, frm):
        raise RuntimeError("HTTP Error 401: Unauthorized")

    monkeypatch.setattr(mod, "_fetch_symbol", _boom)

    rc = mod.refresh(dry_run=False)

    assert rc == -1, "all-fail must stay rc=-1 (unchanged by the protocol)"
    payload = _protocol(mod._tmp)
    assert payload["summary"]["requested"] == 2
    assert payload["summary"]["error"] == 2
    assert sorted(payload["summary"]["error_keys"]) == ["AAA", "BBB"]


def test_empty_responses_are_recorded_not_dropped(mod, monkeypatch):
    """THE E-112 case: a successful request returning nothing leaves a trace."""
    monkeypatch.setattr(mod, "_fetch_symbol", lambda tok, sym, frm: [])

    rc = mod.refresh(dry_run=False)

    assert rc == 0
    summary = _protocol(mod._tmp)["summary"]
    assert summary["requested"] == 2
    assert summary["empty"] == 2
    assert sorted(summary["empty_keys"]) == ["AAA", "BBB"]


def test_successful_fetch_is_recorded_with_row_count(mod, monkeypatch):
    monkeypatch.setattr(
        mod, "_fetch_symbol", lambda tok, sym, frm: [_eod_row("2026-01-06")]
    )

    rc = mod.refresh(dry_run=False)

    assert rc > 0
    summary = _protocol(mod._tmp)["summary"]
    assert summary["requested"] == 2
    assert summary["ok"] == 2
    assert summary["returned_rows"] == 2


def test_schema_mismatch_is_recorded_as_error(mod, monkeypatch):
    monkeypatch.setattr(
        mod, "_fetch_symbol", lambda tok, sym, frm: [{"date": "2026-01-06"}]
    )

    mod.refresh(dry_run=False)

    entries = _protocol(mod._tmp)["entries"]
    assert all(e["status"] == "error" for e in entries)
    assert all("schema mismatch" in (e["error"] or "") for e in entries)


def test_nothing_stale_still_writes_a_zero_request_protocol(mod, monkeypatch):
    """ "No protocol file" and "we asked and got nothing" must not look alike."""
    fresh = mod._tmp / "daily.parquet"
    _make_cache(
        fresh,
        latest=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
        syms=["AAA", "BBB"],
    )

    def _never(tok, sym, frm):  # pragma: no cover - must not be reached
        raise AssertionError("fetch must not run when nothing is stale")

    monkeypatch.setattr(mod, "_fetch_symbol", _never)

    rc = mod.refresh(dry_run=False)

    assert rc == 0
    assert _protocol(mod._tmp)["summary"]["requested"] == 0


# --- the protocol must be purely additive --------------------------------


def test_return_code_identical_when_pull_log_unavailable(mod, monkeypatch):
    """Bookkeeping must not change behaviour, not even by existing."""
    monkeypatch.setattr(
        mod, "_fetch_symbol", lambda tok, sym, frm: [_eod_row("2026-01-06")]
    )
    rc_with = mod.refresh(dry_run=False)

    fresh = _load_module(monkeypatch)
    cache2 = mod._tmp / "daily2.parquet"
    _make_cache(cache2, latest="2026-01-05", syms=["AAA", "BBB"])
    monkeypatch.setattr(fresh, "CACHE_PATH", cache2)
    monkeypatch.setattr(fresh, "STATUS_PATH", mod._tmp / "ops" / "status2.json")
    monkeypatch.setattr(
        fresh, "_fetch_symbol", lambda tok, sym, frm: [_eod_row("2026-01-06")]
    )

    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__
    )

    def _blocked(name, *args, **kwargs):
        if name == "src.assembled_core.data.pull_log":
            raise ImportError("simulated: pull_log unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked)
    rc_without = fresh.refresh(dry_run=False)

    assert rc_with == rc_without, (
        "return code must not depend on whether the protocol module imports"
    )
