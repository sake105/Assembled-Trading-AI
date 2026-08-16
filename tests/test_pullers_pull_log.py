"""Smoke tests for the four pullers wired to the E-112 request protocol.

WHY THIS FILE EXISTS
--------------------
A BLOCKER survived six review rounds because nothing ever ran these pullers.
The coingecko puller passed ``quote_ccy=`` to ``http_get_json``, whose signature
takes no ``**extra`` — a ``TypeError`` on the very first symbol, before any
network call. The puller had been completely dead since that change, and the
broad per-symbol ``except`` filed the author's own programming error into the
protocol as a *vendor* error. A ten-line stub test would have caught it on the
first run.

E-147 states the requirement directly: "Gegenprobe im Test: Transport gezielt
raisen lassen und pruefen, dass die Datei existiert UND den angefragten
Schluessel enthaelt." That check lived only in the EODHD writer — the one file
that did not need it most.

Every test stubs the transport. Nothing here touches the network, the real
``output/`` tree, or any credential.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parents[1]

CSV_OK = "Date,Open,High,Low,Close,Volume\n2026-01-02,1,2,0.5,1.5,100\n"


def _load(rel: str, name: str, monkeypatch=None):
    """Load a puller by path, with scripts/data on sys.path for `common.*`.

    Uses monkeypatch.syspath_prepend where available so the entry is removed at
    teardown. A bare sys.path.insert would leave `scripts/data` at the front of
    the search path for the rest of the process, making its top-level `common`
    package resolvable repo-wide — a global state change in a suite that is
    otherwise built on autouse isolation.
    """
    scripts_data = str(REPO_ROOT / "scripts" / "data")
    if monkeypatch is not None:
        monkeypatch.syspath_prepend(scripts_data)
    elif scripts_data not in sys.path:
        sys.path.insert(0, scripts_data)
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Redirect every protocol write into tmp_path."""
    import src.assembled_core.data.pull_log as pl

    target = tmp_path / "ops"
    monkeypatch.setattr(pl, "DEFAULT_LOG_DIR", target)
    return target


def _protocol(log_dir: Path, source: str) -> dict:
    files = sorted(log_dir.glob(f"pull_log_{source}*.json"))
    assert files, f"no protocol written for {source}"
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _run(mod, argv: list[str]) -> int:
    """Run main() and return the exit code (0 when it returns normally)."""
    old = sys.argv
    sys.argv = argv
    try:
        mod.main()
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = old


# --- the happy path must actually work (the BLOCKER this file exists for) ---


def test_coingecko_succeeds_and_records_rows(tmp_path, monkeypatch, log_dir):
    mod = _load("scripts/data/pull_coingecko_ohlc.py", "cg_ok", monkeypatch)
    monkeypatch.setattr(
        mod, "http_get_json", lambda url, **kw: [[1704067200000, 1.0, 2.0, 0.5, 1.5]]
    )

    rc = _run(mod, ["x", "BTC", "1", str(tmp_path / "out")])

    assert rc == 0, "a fully successful pull must exit 0"
    s = _protocol(log_dir, "coingecko_ohlc")["summary"]
    assert s["requested"] >= 1
    assert s["error"] == 0, "a working pull must not file vendor errors"


def test_stooq_succeeds_and_records_the_requested_symbol(
    tmp_path, monkeypatch, log_dir
):
    mod = _load("scripts/data/pull_stooq_eod.py", "stooq_ok", monkeypatch)
    monkeypatch.setattr(mod, "http_get_text", lambda url, **kw: CSV_OK)

    rc = _run(mod, ["x", "AAPL", str(tmp_path / "out"), ".us"])

    assert rc == 0
    entries = _protocol(log_dir, "stooq_eod")["entries"]
    keys = {e["key"] for e in entries}
    assert "AAPL.us" in keys, (
        "the protocol must name what was REQUESTED (with suffix), not the base symbol"
    )


# --- E-147: the outage must leave a trace naming the key --------------------


def test_coingecko_outage_writes_protocol_with_the_key(tmp_path, monkeypatch, log_dir):
    mod = _load("scripts/data/pull_coingecko_ohlc.py", "cg_fail", monkeypatch)

    def _boom(url, **kw):
        raise RuntimeError("HTTP Error 503: Service Unavailable")

    monkeypatch.setattr(mod, "http_get_json", _boom)

    rc = _run(mod, ["x", "BTC,ETH", "1", str(tmp_path / "out")])

    assert rc == 2, "total failure must not exit 0"
    s = _protocol(log_dir, "coingecko_ohlc")["summary"]
    assert s["error"] >= 1
    assert {"BTC", "ETH"} <= set(s["error_keys"])


def test_stooq_outage_writes_protocol_with_the_key(tmp_path, monkeypatch, log_dir):
    mod = _load("scripts/data/pull_stooq_eod.py", "stooq_fail", monkeypatch)

    def _boom(url, **kw):
        raise RuntimeError("HTTP Error 503: Service Unavailable")

    monkeypatch.setattr(mod, "http_get_text", _boom)

    rc = _run(mod, ["x", "AAPL,MSFT", str(tmp_path / "out"), ".us"])

    assert rc == 2
    s = _protocol(log_dir, "stooq_eod")["summary"]
    assert {"AAPL.us", "MSFT.us"} <= set(s["error_keys"])


def test_stooq_parse_failure_does_not_abort_remaining_symbols(
    tmp_path, monkeypatch, log_dir
):
    """A bad body for one symbol must not stop the others being requested.

    This is E-147's second stage: parse and write have to sit inside the same
    per-symbol guard as the fetch, otherwise the loop dies and the protocol
    comes out empty.
    """
    mod = _load("scripts/data/pull_stooq_eod.py", "stooq_parse", monkeypatch)
    calls: list[str] = []

    def _fetch(url, **kw):
        calls.append(url)
        return "not,a,valid,ohlc,body\n" if "nodata" in url else CSV_OK

    monkeypatch.setattr(mod, "http_get_text", _fetch)

    _run(mod, ["x", "NODATA,LATER", str(tmp_path / "out"), ".us"])

    assert len(calls) == 2, "the second symbol must still be requested"
    keys = {e["key"] for e in _protocol(log_dir, "stooq_eod")["entries"]}
    assert {"NODATA.us", "LATER.us"} <= keys


def test_ecb_unmapped_pair_is_skipped_not_empty(tmp_path, monkeypatch, log_dir):
    """A pair we never asked about must not land in empty_keys (E-112)."""
    mod = _load("scripts/data/pull_ecb_fx.py", "ecb_skip", monkeypatch)
    monkeypatch.setattr(
        mod,
        "http_get_text",
        lambda url, **kw: "Date,Open,High,Low,Close,Volume\n2026-01-02,1,1,1,1,0\n",
    )

    _run(mod, ["x", "EURUSD,NOSUCHPAIR", str(tmp_path / "out")])

    s = _protocol(log_dir, "ecb_fx")["summary"]
    assert "NOSUCHPAIR" in s["skipped_keys"]
    assert "NOSUCHPAIR" not in s["empty_keys"]
    assert "NOSUCHPAIR" not in s["error_keys"]


# --- the fourth puller: the one that was NOT covered ------------------------
#
# tests/test_pullers_pull_log.py originally covered coingecko, stooq and ecb.
# pull_alpha_vantage_intraday.py was the only one missing — and the only one
# whose indentation had been rewritten by script. It also carried a real defect
# nobody saw: a "→" in the success print, emitted AFTER to_parquet but BEFORE
# plog.record. Under Windows cp1252 that raises UnicodeEncodeError, so a fully
# successful pull was filed as a VENDOR error and exited 2 (E-151).


def test_alpha_vantage_success_is_not_filed_as_a_vendor_error(
    tmp_path, monkeypatch, log_dir
):
    import pandas as pd

    mod = _load(
        "scripts/data/pullers/pull_alpha_vantage_intraday.py", "av_ok", monkeypatch
    )

    def _fake_download(sym, **kw):
        return pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-01-02 10:00"], utc=True),
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [10],
            }
        ).set_index("Datetime")

    monkeypatch.setattr(mod.yf, "download", _fake_download)

    rc = _run(
        mod,
        [
            "x",
            "--symbols",
            "AAA",
            "--interval",
            "5m",
            "--days",
            "1",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert rc == 0, "a successful pull must exit 0, not 2"
    s = _protocol(log_dir, "yfinance_intraday")["summary"]
    assert s["error"] == 0, (
        "a successful pull must not file a vendor error — that would poison the "
        "very evidence the protocol exists to provide"
    )
    assert s["ok"] >= 1


def test_alpha_vantage_empty_is_recorded_not_dropped(tmp_path, monkeypatch, log_dir):
    import pandas as pd

    mod = _load(
        "scripts/data/pullers/pull_alpha_vantage_intraday.py", "av_empty", monkeypatch
    )
    monkeypatch.setattr(mod.yf, "download", lambda sym, **kw: pd.DataFrame())

    _run(
        mod,
        [
            "x",
            "--symbols",
            "NOSUCH",
            "--interval",
            "5m",
            "--days",
            "1",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    s = _protocol(log_dir, "yfinance_intraday")["summary"]
    assert s["empty_keys"] == ["NOSUCH"]
