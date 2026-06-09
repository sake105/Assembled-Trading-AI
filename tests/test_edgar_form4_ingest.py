"""Tests for the SEC EDGAR Form 4 ingester (offline, fixture-driven).

These tests exercise the PURE parsing/classification logic against a REAL
Form 4 submission saved under tests/fixtures/edgar/. No network access.

The fixture (accession 0000950170-24-113593, issuer P10 Inc / ticker PX) is a
multi-owner joint Form 4 with exactly ONE non-derivative transaction (an
open-market sale, code 'S') plus two non-derivative HOLDINGS that must be
skipped (holdings have no transactionCode/transactionDate).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.data.edgar_form4_ingest import (
    acceptance_datetime_to_utc,
    classify_transaction_code,
    form4_rows_to_dataframe,
    parse_form4_index,
    parse_form4_submission,
)

FIXTURE = Path(__file__).parent / "fixtures" / "edgar" / "form4_full_submission.txt"


# --------------------------------------------------------------------------
# classify_transaction_code — fixes the "100% unknown type" defect
# --------------------------------------------------------------------------


def test_classify_open_market_purchase_is_P():
    assert classify_transaction_code("P") == "P"


def test_classify_open_market_sale_is_S():
    assert classify_transaction_code("S") == "S"


@pytest.mark.parametrize("code", ["A", "M", "G", "F", "C", "X", "J", ""])
def test_classify_non_ps_codes_are_unknown(code):
    # Everything that is not an open-market P or S is 'unknown' — NOT silently
    # coerced to P/S, NOT dropped. Caller surfaces a WARNING + summary.
    assert classify_transaction_code(code) == "unknown"


def test_classify_is_case_insensitive_and_strips():
    assert classify_transaction_code(" p ") == "P"
    assert classify_transaction_code("s") == "S"


# --------------------------------------------------------------------------
# acceptance_datetime_to_utc — the PIT anchor (available_at)
# --------------------------------------------------------------------------


def test_acceptance_datetime_parsed_as_eastern_to_utc():
    # 20241008181238 = 2024-10-08 18:12:38 America/New_York (EDT, UTC-4)
    #               => 2024-10-08 22:12:38 UTC
    ts = acceptance_datetime_to_utc("20241008181238")
    assert ts == pd.Timestamp("2024-10-08 22:12:38", tz="UTC")
    assert ts.tzinfo is not None


def test_acceptance_datetime_winter_is_est_offset():
    # January => EST (UTC-5): 09:30:00 ET => 14:30:00 UTC
    ts = acceptance_datetime_to_utc("20240115093000")
    assert ts == pd.Timestamp("2024-01-15 14:30:00", tz="UTC")


# --------------------------------------------------------------------------
# parse_form4_submission — full real submission .txt
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rows():
    text = FIXTURE.read_text(encoding="latin-1")
    return parse_form4_submission(text, accession="0000950170-24-113593")


def test_holdings_are_skipped_only_transactions_returned(rows):
    # 1 nonDerivativeTransaction + 2 nonDerivativeHolding => exactly 1 row.
    assert len(rows) == 1


def test_every_row_has_a_transaction_code_and_date(rows):
    # Holdings (no code/date) must never leak through as rows.
    for r in rows:
        assert r["transaction_code"]
        assert pd.notna(r["transaction_date"])


def test_transaction_fields_match_real_filing(rows):
    r = rows[0]
    assert r["symbol"] == "PX"
    assert r["issuer_cik"] == "0001841968"
    assert r["transaction_code"] == "S"
    assert r["transaction_type"] == "S"
    assert r["acquired_disposed"] == "D"
    assert r["shares"] == pytest.approx(247424.0)
    assert r["price"] == pytest.approx(11.0517)


def test_value_usd_is_gross_positive_shares_times_price(rows):
    # value_usd is the GROSS dollar amount (positive); the downstream wrapper
    # applies the +/- sign from transaction_type. So it must be positive.
    r = rows[0]
    assert r["value_usd"] == pytest.approx(247424.0 * 11.0517)
    assert r["value_usd"] > 0


def test_net_shares_signed_by_acquired_disposed(rows):
    # Disposed (D) => negative net_shares for the insider_features path.
    assert rows[0]["net_shares"] == pytest.approx(-247424.0)


def test_dates_distinguish_event_filing_and_availability(rows):
    r = rows[0]
    assert r["transaction_date"] == pd.Timestamp("2024-10-04")
    assert r["filing_date"] == pd.Timestamp("2024-10-08")
    # available_at = acceptance instant (ET->UTC), NOT filing-date midnight.
    assert r["available_at"] == pd.Timestamp("2024-10-08 22:12:38", tz="UTC")
    assert r["available_at"] != pd.Timestamp("2024-10-08", tz="UTC")


def test_primary_reporting_owner_captured(rows):
    # First reportingOwner block = 210 Capital, LLC (CIK 0001694780).
    assert rows[0]["reporting_owner_cik"] == "0001694780"


# --------------------------------------------------------------------------
# form4_rows_to_dataframe — schema serving BOTH consumer shapes
# --------------------------------------------------------------------------


def test_dataframe_has_superset_schema(rows):
    df = form4_rows_to_dataframe(rows)
    # earnings_insider_wrapper path:
    for col in ("symbol", "filing_date", "transaction_type", "value_usd"):
        assert col in df.columns
    # insider_features (time-series) path:
    for col in ("timestamp", "symbol", "net_shares", "trades_count", "role"):
        assert col in df.columns
    # audit / PIT:
    for col in ("transaction_code", "available_at", "event_date"):
        assert col in df.columns


def test_dataframe_dtypes_are_pit_safe(rows):
    df = form4_rows_to_dataframe(rows)
    assert str(df["available_at"].dtype) == "datetime64[ns, UTC]"
    assert df["value_usd"].dtype == "float64"
    assert df["trades_count"].dtype == "int64"


# --------------------------------------------------------------------------
# parse_form4_index — daily-index enumeration (offline sample)
# --------------------------------------------------------------------------

_IDX_SAMPLE = """Description:           Daily Index of EDGAR Dissemination Feed by Form Type
Last Data Received:    October 08, 2024

Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------------------------------------
4           210 Capital, LLC                                              1694780     20241008    edgar/data/1694780/0000950170-24-113593.txt
4/A         Some Amended Filer Inc                                        111222      20241008    edgar/data/111222/0000111222-24-000001.txt
8-K         Not An Insider Corp                                           999888      20241008    edgar/data/999888/0000999888-24-000009.txt
"""


def test_index_returns_only_form4_and_4a():
    entries = parse_form4_index(_IDX_SAMPLE)
    assert len(entries) == 2
    forms = {e["form_type"] for e in entries}
    assert forms == {"4", "4/A"}


def test_index_entry_fields_parsed():
    entries = parse_form4_index(_IDX_SAMPLE)
    e = entries[0]
    assert e["form_type"] == "4"
    assert e["cik"] == "1694780"
    assert e["date_filed"] == "20241008"
    assert e["filename"] == "edgar/data/1694780/0000950170-24-113593.txt"
    assert e["company"] == "210 Capital, LLC"


# --------------------------------------------------------------------------
# Network layer — mocked (SEC fair-access backoff, no real I/O)
# --------------------------------------------------------------------------


def test_http_get_retries_on_403_then_succeeds(monkeypatch):
    import urllib.error
    import urllib.request

    from src.assembled_core.data import edgar_form4_ingest as efi

    calls = {"n": 0}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"OK"

    def fake_urlopen(req, timeout=30.0):
        calls["n"] += 1
        if calls["n"] == 1:  # first attempt: SEC 403 -> must back off + retry
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(efi.time, "sleep", lambda *_: None)  # no real backoff wait

    limiter = efi._RateLimiter(min_spacing=0.0)
    data = efi._http_get(
        "https://www.sec.gov/x", "UA test", limiter=limiter, max_retries=3
    )
    assert data == b"OK"
    assert calls["n"] == 2  # one 403, one success


def test_http_get_raises_after_exhausting_retries(monkeypatch):
    import urllib.error
    import urllib.request

    import pytest

    from src.assembled_core.data import edgar_form4_ingest as efi

    def always_403(req, timeout=30.0):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", always_403)
    monkeypatch.setattr(efi.time, "sleep", lambda *_: None)

    limiter = efi._RateLimiter(min_spacing=0.0)
    with pytest.raises(urllib.error.HTTPError):
        efi._http_get("https://www.sec.gov/x", "UA", limiter=limiter, max_retries=3)
