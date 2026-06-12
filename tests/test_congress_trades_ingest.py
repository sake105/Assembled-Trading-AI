"""Tests for the free Congress (STOCK Act) trades ingester (offline, fixture-driven).

Two free structured sources (verified live 2026-06-09), parsed against real
saved records under tests/fixtures/congress/:
  * kadoa-org/congress-trading-monitor (BOTH chambers; ISO dates; amount range
    as numeric low/high; ``filing_date`` is the disclosure date),
  * TattooedHead/house-stock-watcher-data (House; MM/DD/YYYY dates; ``amount_mid``
    pre-computed; ``disclosure_date`` field).

PIT: ``disclosure_date`` (NOT ``transaction_date``) is the availability key. The
economic ``transaction_date``/``event_date`` is kept immutable; the +45d STOCK Act
fallback only fills a MISSING disclosure_date (E-038 discipline). No network here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.data.congress_trades_ingest import (
    amount_midpoint,
    load_congress_sample,
    normalize_congress,
    parse_house_watcher_records,
    parse_kadoa_records,
)
from src.assembled_core.data.source_latencies import CONGRESS_DAYS
from src.assembled_core.features.congress_features import add_congress_features

FIX = Path(__file__).parent / "fixtures" / "congress"


def _kadoa():
    return json.loads((FIX / "kadoa_sample.json").read_text())


def _house():
    return json.loads((FIX / "house_watcher_sample.json").read_text())


# --------------------------------------------------------------------------
# amount range -> representative scalar
# --------------------------------------------------------------------------


def test_amount_midpoint_arithmetic():
    assert amount_midpoint(1001, 15000) == 8000.5


def test_amount_midpoint_handles_missing():
    assert amount_midpoint(None, None) != amount_midpoint(None, None) or pd.isna(
        amount_midpoint(None, None)
    )


# --------------------------------------------------------------------------
# kadoa parser (both chambers)
# --------------------------------------------------------------------------


def test_kadoa_drops_rows_without_ticker():
    rows = parse_kadoa_records(_kadoa())
    # Fixture = 5 with ticker + 1 without -> 5 rows.
    assert len(rows) == 5
    assert all(r["symbol"] for r in rows)


def test_kadoa_maps_pg_record():
    rows = parse_kadoa_records(_kadoa())
    pg = next(r for r in rows if r["symbol"] == "PG")
    assert pg["transaction_type"] == "Purchase"
    assert pg["event_date"] == pd.Timestamp("2026-05-15")
    assert pg["disclosure_date"] == pd.Timestamp("2026-06-04")
    assert pg["amount"] == 8000.5  # (1001 + 15000) / 2


def test_kadoa_disclosure_date_distinct_from_transaction_date():
    rows = parse_kadoa_records(_kadoa())
    pg = next(r for r in rows if r["symbol"] == "PG")
    # PIT: availability is the filing date, never the trade date.
    assert pg["disclosure_date"] > pg["event_date"]


def test_kadoa_missing_filing_date_uses_stock_act_fallback():
    rec = copy.deepcopy(next(r for r in _kadoa() if r.get("ticker") == "PG"))
    rec["filing_date"] = None
    rows = parse_kadoa_records([rec])
    r = rows[0]
    # disclosure_date = transaction_date + CONGRESS_DAYS; transaction_date untouched.
    assert r["event_date"] == pd.Timestamp("2026-05-15")
    assert r["disclosure_date"] == pd.Timestamp("2026-05-15") + pd.Timedelta(
        days=CONGRESS_DAYS
    )


# --------------------------------------------------------------------------
# house-watcher parser (House; MM/DD/YYYY; amount_mid)
# --------------------------------------------------------------------------


def test_house_watcher_parses_mmddyyyy_and_amount_mid():
    rows = parse_house_watcher_records(_house())
    pg = next(r for r in rows if r["symbol"] == "PG")
    assert pg["event_date"] == pd.Timestamp("2026-05-15")
    assert pg["disclosure_date"] == pd.Timestamp("2026-06-04")
    assert pg["amount"] == 8000.0  # amount_mid
    assert pg["chamber"] == "House"


# --------------------------------------------------------------------------
# normalize -> typed frame
# --------------------------------------------------------------------------


def test_normalize_schema_and_dtypes():
    df = normalize_congress(parse_kadoa_records(_kadoa()))
    for col in (
        "timestamp",
        "symbol",
        "amount",
        "event_date",
        "disclosure_date",
        "available_at",
        "transaction_type",
        "member",
    ):
        assert col in df.columns
    # Resolution-agnostic (pandas 2.3.3 may yield 'us', 2.2.x 'ns'): assert tz-aware
    # UTC datetime, not the exact dtype string.
    assert (
        df["available_at"].dtype.kind == "M" and str(df["available_at"].dt.tz) == "UTC"
    )
    assert df["amount"].dtype == "float64"


def test_normalize_drops_logically_impossible_dates():
    # A PTR discloses a PAST trade -> transaction_date > disclosure_date is
    # impossible (source data-entry error). Such rows are dropped, not kept.
    rows = [
        {
            "symbol": "GOOD",
            "transaction_date": pd.Timestamp("2026-05-15"),
            "event_date": pd.Timestamp("2026-05-15"),
            "disclosure_date": pd.Timestamp("2026-06-04"),
            "amount": 100.0,
        },
        {
            "symbol": "BADFUTURE",
            "transaction_date": pd.Timestamp("2026-12-26"),
            "event_date": pd.Timestamp("2026-12-26"),
            "disclosure_date": pd.Timestamp("2026-01-21"),
            "amount": 100.0,
        },
    ]
    df = normalize_congress(rows)
    assert set(df["symbol"]) == {"GOOD"}


# --------------------------------------------------------------------------
# load_congress_sample — pipeline entry point (fail-loud + round-trip)
# --------------------------------------------------------------------------


def test_load_congress_sample_fail_loud_without_path():
    with pytest.raises(ValueError, match="allow_sample"):
        load_congress_sample()


def test_load_congress_sample_warns_commercial_use_once(tmp_path, caplog):
    import logging

    from src.assembled_core.data import congress_trades_ingest as cti

    cti._COMMERCIAL_USE_WARNED = False  # reset process-global flag for this test
    df = normalize_congress(parse_kadoa_records(_kadoa()))
    p = tmp_path / "c.parquet"
    df.to_parquet(p, index=False)
    with caplog.at_level(logging.WARNING, logger=cti.logger.name):
        load_congress_sample(path=p)
        load_congress_sample(path=p)  # second load must NOT re-warn
    hits = [r for r in caplog.records if "13107" in r.message]
    assert len(hits) == 1  # one-time §13107 commercial-use warning


def test_load_congress_sample_roundtrip_feeds_features(tmp_path):
    df = normalize_congress(parse_kadoa_records(_kadoa()))
    p = tmp_path / "congress_trades.parquet"
    df.to_parquet(p, index=False)

    events = load_congress_sample(path=p)
    assert not events.empty
    assert {"timestamp", "symbol"}.issubset(events.columns)

    prices = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-10", "2026-06-10"], utc=True),
            "symbol": ["PG", "ZZZZ"],
            "close": [160.0, 10.0],
        }
    )
    out = add_congress_features(
        prices, events, as_of=pd.Timestamp("2026-06-10", tz="UTC")
    )
    pg_row = out[out["symbol"] == "PG"].iloc[0]
    # PG had a disclosed (2026-06-04) purchase on 2026-05-15 -> within the 90d window.
    assert pg_row["congress_trade_count_90d"] >= 1
    assert pg_row["congress_total_amount_90d"] > 0


def test_normalize_emits_normalized_side_for_sign_consumers():
    # Purchase -> buy, Sale (*) -> sell, so compute_congress_net_buy_score (which
    # keys on a `type` column) cannot fail-open a Sale into a buy.
    rows = parse_kadoa_records(_kadoa())
    df = normalize_congress(rows)
    sides = dict(zip(df["transaction_type"], df["type"]))
    assert sides.get("Purchase") == "buy"
    if "Sale (Partial)" in sides:
        assert sides["Sale (Partial)"] == "sell"


# --------------------------------------------------------------------------
# Network layer — mocked (CDN fallback + DEGRADED-on-failure, no real I/O)
# --------------------------------------------------------------------------


def test_fetch_kadoa_falls_back_to_cdn(monkeypatch):
    from src.assembled_core.data import congress_trades_ingest as cti

    calls: list[str] = []

    def fake(url, ua, timeout=60.0):
        calls.append(url)
        if url == cti.KADOA_URL:
            raise OSError("primary mirror down")
        return [{"ticker": "PG"}]

    monkeypatch.setattr(cti, "_http_get_json", fake)
    out = cti.fetch_congress_trades("kadoa")
    assert calls == [cti.KADOA_URL, cti.KADOA_CDN]  # tried raw, then CDN
    assert out == [{"ticker": "PG"}]


def test_fetch_degraded_returns_empty_on_total_failure(monkeypatch):
    from src.assembled_core.data import congress_trades_ingest as cti

    def fake(url, ua, timeout=60.0):
        raise OSError("all mirrors down")

    monkeypatch.setattr(cti, "_http_get_json", fake)
    out = cti.fetch_congress_trades("kadoa")
    assert out == []  # DEGRADED -> empty (logged ERROR), never a silent crash


# --------------------------------------------------------------------------
# Cross-source dedupe (F-senior-1) + neutral-side sign (F-senior-3)
# --------------------------------------------------------------------------


def test_dedupe_collapses_same_sell_across_mirror_label_variants():
    from src.assembled_core.data.congress_trades_ingest import dedupe_congress

    # The SAME House sell appears in kadoa ("Sale (Partial)") and house_watcher
    # ("Sale"). Keying dedupe on the raw label would keep both -> double-count.
    kadoa = [
        {
            "ticker": "XOM",
            "transaction_date": "2026-05-01",
            "filing_date": "2026-05-20",
            "transaction_type": "Sale (Partial)",
            "amount_range_low": 1001,
            "amount_range_high": 15000,
            "filer_name": "Rep X",
            "chamber": "house",
        }
    ]
    house = [
        {
            "ticker": "XOM",
            "transaction_date": "05/01/2026",
            "disclosure_date": "05/20/2026",
            "type": "Sale",
            "amount_mid": 8000,
            "representative": "Rep X",
        }
    ]
    df = normalize_congress(
        parse_kadoa_records(kadoa) + parse_house_watcher_records(house)
    )
    assert len(df) == 2  # before dedupe
    deduped = dedupe_congress(df)
    assert len(deduped) == 1  # collapsed on normalized side, not raw label


def test_dedupe_keeps_buy_and_sell_on_same_day_distinct():
    from src.assembled_core.data.congress_trades_ingest import dedupe_congress

    recs = [
        {
            "ticker": "MSFT",
            "transaction_date": "2026-05-01",
            "filing_date": "2026-05-20",
            "transaction_type": "Purchase",
            "amount_range_low": 1,
            "amount_range_high": 3,
            "filer_name": "A",
            "chamber": "house",
        },
        {
            "ticker": "MSFT",
            "transaction_date": "2026-05-01",
            "filing_date": "2026-05-20",
            "transaction_type": "Sale (Full)",
            "amount_range_low": 1,
            "amount_range_high": 3,
            "filer_name": "A",
            "chamber": "house",
        },
    ]
    deduped = dedupe_congress(normalize_congress(parse_kadoa_records(recs)))
    assert len(deduped) == 2  # opposite sides -> NOT merged


def test_net_buy_score_treats_unknown_side_as_neutral():
    from src.assembled_core.features.congress_features import (
        compute_congress_net_buy_score,
    )

    trades = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "amount": [100.0, 100.0],
            "type": ["sell", None],  # one real sell + one unknown/Exchange side
            "disclosure_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        }
    )
    scores = compute_congress_net_buy_score(trades)
    # Unknown side contributes 0 (neutral), NOT -100 -> net is the sell only.
    assert scores["AAA"] == pytest.approx(-100.0)
