"""Tests for the SEC EDGAR XBRL Company-Facts fundamentals ingester (offline).

These tests exercise the PURE parsing / PIT-selection / coalesce logic against
inline SEC-shaped fixtures. No network access.

The fixture (synthetic issuer ``TESTCO`` / CIK 111111) carries:
- a Q1 EarningsPerShareDiluted datapoint AND a later 10-Q/A RESTATEMENT of the
  same fiscal period (different accession, later acceptance) — the core PIT /
  restatement-versioning case;
- a duration NetIncomeLoss + two different revenue tags across two quarters
  (legacy ``Revenues`` for Q1, ASC-606 ``RevenueFromContractWithCustomer...``
  for Q2) — the ordered-coalesce case;
- a ``dei`` instant share count (different namespace);
- a ``Goodwill`` us-gaap tag that is NOT in the wanted set and must be skipped.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.data import fundamentals_xbrl_ingest as fx
from src.assembled_core.data.fundamentals_xbrl_ingest import (
    FUNDAMENTAL_TAGS,
    XBRL_COLUMNS,
    attach_available_at,
    build_accession_acceptance_map,
    coalesce_field,
    company_facts_rows_to_dataframe,
    ingest_fundamentals_xbrl,
    is_amendment,
    parse_acceptance_datetime,
    parse_company_facts,
    select_pit_rows,
    submission_page_names,
)

# ---------------------------------------------------------------------------
# Inline SEC-shaped fixtures
# ---------------------------------------------------------------------------

COMPANY_FACTS = {
    "cik": 111111,
    "entityName": "TestCo Inc.",
    "facts": {
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {
                    "shares": [
                        {
                            "end": "2023-03-31",
                            "val": 1000000,
                            "accn": "acc-q1",
                            "fy": 2023,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2023-04-20",
                        }
                    ]
                }
            }
        },
        "us-gaap": {
            "EarningsPerShareDiluted": {
                "units": {
                    "USD/shares": [
                        {
                            "start": "2023-01-01",
                            "end": "2023-03-31",
                            "val": 1.20,
                            "accn": "acc-q1",
                            "fy": 2023,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2023-04-20",
                            "frame": "CY2023Q1",
                        },
                        # RESTATEMENT of the same Q1 period via a later 10-Q/A.
                        {
                            "start": "2023-01-01",
                            "end": "2023-03-31",
                            "val": 1.25,
                            "accn": "acc-q1a",
                            "fy": 2023,
                            "fp": "Q1",
                            "form": "10-Q/A",
                            "filed": "2023-08-15",
                        },
                        {
                            "start": "2023-04-01",
                            "end": "2023-06-30",
                            "val": 1.40,
                            "accn": "acc-q2",
                            "fy": 2023,
                            "fp": "Q2",
                            "form": "10-Q",
                            "filed": "2023-07-20",
                            "frame": "CY2023Q2",
                        },
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {
                            "start": "2023-01-01",
                            "end": "2023-03-31",
                            "val": 1200000.0,
                            "accn": "acc-q1",
                            "fy": 2023,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2023-04-20",
                        }
                    ]
                }
            },
            # Legacy revenue tag ONLY for Q1 (no RevenueFromContract...).
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2023-01-01",
                            "end": "2023-03-31",
                            "val": 5000000.0,
                            "accn": "acc-q1",
                            "fy": 2023,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2023-04-20",
                        }
                    ]
                }
            },
            # ASC-606 revenue tag ONLY for Q2.
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        {
                            "start": "2023-04-01",
                            "end": "2023-06-30",
                            "val": 5500000.0,
                            "accn": "acc-q2",
                            "fy": 2023,
                            "fp": "Q2",
                            "form": "10-Q",
                            "filed": "2023-07-20",
                        }
                    ]
                }
            },
            # NOT in the wanted set -> must be skipped entirely.
            "Goodwill": {
                "units": {
                    "USD": [
                        {
                            "end": "2023-03-31",
                            "val": 9000000.0,
                            "accn": "acc-q1",
                            "fy": 2023,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2023-04-20",
                        }
                    ]
                }
            },
        },
    },
}

SUBMISSIONS = {
    "cik": "111111",
    "filings": {
        "recent": {
            "accessionNumber": ["acc-q1", "acc-q1a", "acc-q2"],
            "acceptanceDateTime": [
                "2023-04-20T18:30:00.000Z",
                "2023-08-15T17:00:00.000Z",
                "2023-07-20T18:30:00.000Z",
            ],
            "filingDate": ["2023-04-20", "2023-08-15", "2023-07-20"],
            "form": ["10-Q", "10-Q/A", "10-Q"],
        }
    },
}


def _full_frame() -> pd.DataFrame:
    """parse -> typed frame -> available_at attached (the production assembly)."""
    rows = parse_company_facts(COMPANY_FACTS, symbol="TESTCO")
    df = company_facts_rows_to_dataframe(rows)
    accn_map = build_accession_acceptance_map(SUBMISSIONS)
    return attach_available_at(df, accn_map)


# ---------------------------------------------------------------------------
# parse_acceptance_datetime — PIT anchor, ET wall-clock -> UTC (conservative)
# ---------------------------------------------------------------------------


def test_acceptance_sgml_14digit_is_eastern_to_utc():
    # 20241008181238 = 2024-10-08 18:12:38 ET (EDT, UTC-4) => 22:12:38 UTC
    assert parse_acceptance_datetime("20241008181238") == pd.Timestamp(
        "2024-10-08 22:12:38", tz="UTC"
    )


def test_acceptance_iso_z_is_utc_not_eastern():
    # The submissions-feed ISO value carries its own zone (Z = UTC) and must be
    # parsed AS UTC — NOT re-interpreted as Eastern (Phase-4 cross-check vs the
    # SGML header proved the feed is UTC; ET-treatment was 4h too late).
    assert parse_acceptance_datetime("2023-04-20T18:30:00.000Z") == pd.Timestamp(
        "2023-04-20 18:30:00", tz="UTC"
    )


def test_acceptance_iso_winter_is_utc():
    assert parse_acceptance_datetime("2024-01-15T09:30:00.000Z") == pd.Timestamp(
        "2024-01-15 09:30:00", tz="UTC"
    )


def test_acceptance_iso_naive_assumed_utc():
    assert parse_acceptance_datetime("2023-07-20T12:00:00") == pd.Timestamp(
        "2023-07-20 12:00:00", tz="UTC"
    )


def test_acceptance_unparseable_raises():
    with pytest.raises(ValueError):
        parse_acceptance_datetime("not-a-date")


# ---------------------------------------------------------------------------
# build_accession_acceptance_map
# ---------------------------------------------------------------------------


def test_build_accession_acceptance_map():
    m = build_accession_acceptance_map(SUBMISSIONS)
    # acceptanceDateTime is UTC (trailing Z) — parsed as-is, no ET shift.
    assert m["acc-q1"] == pd.Timestamp("2023-04-20 18:30:00", tz="UTC")
    assert m["acc-q1a"] == pd.Timestamp("2023-08-15 17:00:00", tz="UTC")
    assert m["acc-q2"] == pd.Timestamp("2023-07-20 18:30:00", tz="UTC")


def test_build_accession_acceptance_map_empty_recent():
    assert build_accession_acceptance_map({"filings": {"recent": {}}}) == {}


# ---------------------------------------------------------------------------
# parse_company_facts — namespace walk + wanted-set filter
# ---------------------------------------------------------------------------


def test_parse_company_facts_row_count_excludes_unwanted_tag():
    rows = parse_company_facts(COMPANY_FACTS, symbol="TESTCO")
    # dei shares(1) + eps(3) + NI(1) + Revenues(1) + RevFromContract(1) = 7.
    # Goodwill (1) is NOT in FUNDAMENTAL_TAGS -> excluded.
    assert len(rows) == 7
    assert all(r["tag"] != "Goodwill" for r in rows)


def test_parse_company_facts_dei_namespace_and_unit():
    rows = parse_company_facts(COMPANY_FACTS, symbol="TESTCO")
    dei = [r for r in rows if r["namespace"] == "dei"]
    assert len(dei) == 1
    assert dei[0]["tag"] == "EntityCommonStockSharesOutstanding"
    assert dei[0]["unit"] == "shares"
    assert dei[0]["val"] == 1000000
    # instant fact has no 'start'
    assert dei[0]["period_start"] in (None, "")


def test_parse_company_facts_eps_fields_carried():
    rows = parse_company_facts(COMPANY_FACTS, symbol="TESTCO")
    eps = [r for r in rows if r["tag"] == "EarningsPerShareDiluted"]
    assert len(eps) == 3
    q1 = next(r for r in eps if r["accession"] == "acc-q1")
    assert q1["unit"] == "USD/shares"
    assert q1["val"] == 1.20
    assert q1["fy"] == 2023
    assert q1["fp"] == "Q1"
    assert q1["period_end"] == "2023-03-31"
    assert q1["period_start"] == "2023-01-01"
    assert q1["form"] == "10-Q"
    assert q1["filed"] == "2023-04-20"
    assert q1["frame"] == "CY2023Q1"
    # the restatement carries no frame
    q1a = next(r for r in eps if r["accession"] == "acc-q1a")
    assert q1a["frame"] in (None, "")


def test_parse_company_facts_symbol_stamped():
    rows = parse_company_facts(COMPANY_FACTS, symbol="testco")
    assert all(r["symbol"] == "TESTCO" for r in rows)


def test_parse_company_facts_empty_facts():
    assert parse_company_facts({"facts": {}}, symbol="X") == []


# ---------------------------------------------------------------------------
# company_facts_rows_to_dataframe — stable typed schema
# ---------------------------------------------------------------------------


def test_rows_to_dataframe_schema_and_dtypes():
    df = company_facts_rows_to_dataframe(
        parse_company_facts(COMPANY_FACTS, symbol="TESTCO")
    )
    assert list(df.columns) == XBRL_COLUMNS
    # Resolution-agnostic (pandas 2.3.3 may yield 'us', 2.2.x 'ns'): assert kind +
    # tz, NOT the exact dtype string. period_end/filed_date naive; available_at UTC.
    assert df["period_end"].dtype.kind == "M" and df["period_end"].dt.tz is None
    assert df["filed_date"].dtype.kind == "M" and df["filed_date"].dt.tz is None
    assert (
        df["available_at"].dtype.kind == "M" and str(df["available_at"].dt.tz) == "UTC"
    )
    assert df["val"].dtype == "float64"
    assert df["is_amendment"].dtype == bool
    assert str(df["fy"].dtype) == "Int64"
    # disclosure_date defaults to filed_date (the public-availability day).
    assert (df["disclosure_date"] == df["filed_date"]).all()


def test_rows_to_dataframe_empty_is_schema_correct():
    df = company_facts_rows_to_dataframe([])
    assert list(df.columns) == XBRL_COLUMNS
    assert len(df) == 0


# ---------------------------------------------------------------------------
# is_amendment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "form,expected",
    [
        ("10-K/A", True),
        ("10-Q/A", True),
        ("10-K", False),
        ("10-Q", False),
        ("", False),
        (None, False),
    ],
)
def test_is_amendment(form, expected):
    assert is_amendment(form) is expected


# ---------------------------------------------------------------------------
# attach_available_at — join accession -> acceptance instant + is_amendment
# ---------------------------------------------------------------------------


def test_attach_available_at_maps_acceptance_and_amendment_flag():
    df = _full_frame()
    q1a = df[df["accession"] == "acc-q1a"]
    assert (q1a["available_at"] == pd.Timestamp("2023-08-15 17:00:00", tz="UTC")).all()
    assert q1a["is_amendment"].all()
    q1 = df[df["accession"] == "acc-q1"]
    assert (q1["available_at"] == pd.Timestamp("2023-04-20 18:30:00", tz="UTC")).all()
    assert (~q1["is_amendment"]).all()


def test_attach_available_at_unknown_accession_is_nat():
    rows = parse_company_facts(COMPANY_FACTS, symbol="TESTCO")
    df = company_facts_rows_to_dataframe(rows)
    df = attach_available_at(df, {})  # no acceptance resolved
    assert df["available_at"].isna().all()


# ---------------------------------------------------------------------------
# select_pit_rows — PIT gate + restatement (latest filing on-or-before as_of)
# ---------------------------------------------------------------------------


def test_select_pit_before_any_filing_is_empty():
    df = _full_frame()
    # acc-q1 accepted 2023-04-20 -> nothing available on 2023-04-01.
    out = select_pit_rows(df, "2023-04-01")
    assert len(out) == 0


def test_select_pit_returns_original_before_restatement():
    df = _full_frame()
    out = select_pit_rows(df, "2023-05-01")
    eps_q1 = out[
        (out["tag"] == "EarningsPerShareDiluted")
        & (out["period_end"] == pd.Timestamp("2023-03-31"))
    ]
    # restatement (acc-q1a, available 2023-08-15) NOT yet visible -> original 1.20.
    assert len(eps_q1) == 1
    assert eps_q1.iloc[0]["val"] == 1.20
    assert eps_q1.iloc[0]["accession"] == "acc-q1"
    # Q2 (accepted 2023-07-20) is also not yet visible on 2023-05-01.
    assert not (
        (out["tag"] == "EarningsPerShareDiluted")
        & (out["period_end"] == pd.Timestamp("2023-06-30"))
    ).any()


def test_select_pit_returns_restated_after_amendment():
    df = _full_frame()
    out = select_pit_rows(df, "2023-09-01")
    eps_q1 = out[
        (out["tag"] == "EarningsPerShareDiluted")
        & (out["period_end"] == pd.Timestamp("2023-03-31"))
    ]
    # amendment now visible & is the LATEST accepted on-or-before as_of -> 1.25.
    assert len(eps_q1) == 1
    assert eps_q1.iloc[0]["val"] == 1.25
    assert eps_q1.iloc[0]["accession"] == "acc-q1a"


def test_select_pit_nat_available_at_uses_filed_date_plus_latency():
    # Drop the acceptance for acc-q2 -> falls back to filed_date + EDGAR_DAYS(=1).
    rows = parse_company_facts(COMPANY_FACTS, symbol="TESTCO")
    df = company_facts_rows_to_dataframe(rows)
    accn_map = build_accession_acceptance_map(SUBMISSIONS)
    accn_map.pop("acc-q2")
    df = attach_available_at(df, accn_map)

    # acc-q2 filed 2023-07-20 -> effective availability 2023-07-21 (latency 1).
    out_before = select_pit_rows(df, "2023-07-20")
    assert not (out_before["accession"] == "acc-q2").any()
    out_after = select_pit_rows(df, "2023-07-21")
    assert (out_after["accession"] == "acc-q2").any()


def test_select_pit_symbol_filter():
    df = _full_frame()
    out = select_pit_rows(df, "2023-12-31", symbols=["OTHER"])
    assert len(out) == 0


# ---------------------------------------------------------------------------
# coalesce_field — ordered tag priority per (symbol, period_end)
# ---------------------------------------------------------------------------


def test_coalesce_revenue_prefers_then_falls_back():
    df = select_pit_rows(_full_frame(), "2023-12-31")
    rev = coalesce_field(df, "revenue").sort_values("period_end").reset_index(drop=True)
    # Q1: only legacy Revenues present -> used.
    q1 = rev[rev["period_end"] == pd.Timestamp("2023-03-31")].iloc[0]
    assert q1["revenue"] == 5000000.0
    assert q1["source_tag"] == "Revenues"
    # Q2: only ASC-606 tag present -> used.
    q2 = rev[rev["period_end"] == pd.Timestamp("2023-06-30")].iloc[0]
    assert q2["revenue"] == 5500000.0
    assert q2["source_tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_coalesce_eps_uses_restated_value_after_as_of():
    df = select_pit_rows(_full_frame(), "2023-09-01")
    eps = (
        coalesce_field(df, "eps_diluted")
        .sort_values("period_end")
        .reset_index(drop=True)
    )
    q1 = eps[eps["period_end"] == pd.Timestamp("2023-03-31")].iloc[0]
    assert q1["eps_diluted"] == 1.25  # restated
    q2 = eps[eps["period_end"] == pd.Timestamp("2023-06-30")].iloc[0]
    assert q2["eps_diluted"] == 1.40


def test_fundamental_tags_cover_dead_factor_fields():
    # The mapping must at least expose the fields the dead factors need.
    for field in ("eps_diluted", "net_income", "revenue", "shares_outstanding"):
        assert field in FUNDAMENTAL_TAGS
        assert len(FUNDAMENTAL_TAGS[field]) >= 1


# ---------------------------------------------------------------------------
# PIT boundary / tz / fallback hardening (Stage-1 review follow-ups)
# ---------------------------------------------------------------------------


def test_select_pit_inclusive_at_exact_availability_instant():
    # The gate is `_eff <= as_of` — a row IS visible at its exact acceptance
    # instant, and NOT one microsecond before.
    df = _full_frame()
    a = pd.Timestamp("2023-04-20 18:30:00", tz="UTC")  # acc-q1 acceptance (UTC)
    assert (select_pit_rows(df, a)["accession"] == "acc-q1").any()
    just_before = select_pit_rows(df, a - pd.Timedelta(microseconds=1))
    assert not (just_before["accession"] == "acc-q1").any()


def test_select_pit_tz_aware_as_of_matches_naive():
    df = _full_frame()
    naive = select_pit_rows(df, "2023-09-01")
    aware = select_pit_rows(df, pd.Timestamp("2023-09-01", tz="UTC"))
    assert sorted(naive["accession"].tolist()) == sorted(aware["accession"].tolist())


def test_select_pit_all_nat_available_at_uses_fallback_without_raising():
    # The single most fragile path: every available_at is NaT, so selection must
    # fall back to filed_date + EDGAR_DAYS via a tz-localized local series —
    # without raising on tz alignment and preserving the UTC dtype.
    rows = parse_company_facts(COMPANY_FACTS, symbol="TESTCO")
    df = company_facts_rows_to_dataframe(rows)
    df = attach_available_at(df, {})  # all available_at NaT
    assert df["available_at"].isna().all()
    out = select_pit_rows(df, "2023-12-31")
    assert len(out) > 0
    assert (
        out["available_at"].dtype.kind == "M"
        and str(out["available_at"].dt.tz) == "UTC"
    )
    # restatement still resolves via the filed-date fallback (acc-q1a is later).
    eps_q1 = out[
        (out["tag"] == "EarningsPerShareDiluted")
        & (out["period_end"] == pd.Timestamp("2023-03-31"))
    ]
    assert eps_q1.iloc[0]["accession"] == "acc-q1a"


def test_coalesce_prefers_higher_priority_when_both_present():
    # True contention: both the preferred ASC-606 tag AND legacy Revenues carry a
    # non-null value for the SAME period -> the higher-priority tag must win.
    df = pd.DataFrame(
        [
            {
                "symbol": "X",
                "period_end": pd.Timestamp("2023-03-31"),
                "period_start": pd.Timestamp("2023-01-01"),
                "namespace": "us-gaap",
                "tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
                "val": 100.0,
                "fp": "Q1",
                "fy": 2023,
            },
            {
                "symbol": "X",
                "period_end": pd.Timestamp("2023-03-31"),
                "period_start": pd.Timestamp("2023-01-01"),
                "namespace": "us-gaap",
                "tag": "Revenues",
                "val": 200.0,
                "fp": "Q1",
                "fy": 2023,
            },
        ]
    )
    out = coalesce_field(df, "revenue")
    assert out.iloc[0]["revenue"] == 100.0
    assert (
        out.iloc[0]["source_tag"]
        == "RevenueFromContractWithCustomerExcludingAssessedTax"
    )


def test_coalesce_falls_through_when_higher_priority_is_null():
    # Higher-priority tag present but NULL -> fall through to the lower-priority
    # non-null value (null-skip-then-fallback).
    df = pd.DataFrame(
        [
            {
                "symbol": "X",
                "period_end": pd.Timestamp("2023-03-31"),
                "period_start": pd.Timestamp("2023-01-01"),
                "namespace": "us-gaap",
                "tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
                "val": float("nan"),
                "fp": "Q1",
                "fy": 2023,
            },
            {
                "symbol": "X",
                "period_end": pd.Timestamp("2023-03-31"),
                "period_start": pd.Timestamp("2023-01-01"),
                "namespace": "us-gaap",
                "tag": "Revenues",
                "val": 200.0,
                "fp": "Q1",
                "fy": 2023,
            },
        ]
    )
    out = coalesce_field(df, "revenue")
    assert out.iloc[0]["revenue"] == 200.0
    assert out.iloc[0]["source_tag"] == "Revenues"


# ---------------------------------------------------------------------------
# Restatement-selection MAJOR fixes (review-chain): period_start in the key +
# deterministic tie-break.
# ---------------------------------------------------------------------------


def _pit_frame_from_rows(rows, accn_acceptance):
    """parse-style rows -> typed frame -> available_at attached."""
    df = company_facts_rows_to_dataframe(rows)
    return attach_available_at(df, accn_acceptance)


def test_select_pit_same_end_different_start_both_survive():
    # A single 10-K legitimately emits a Q4 quarterly fact (start Oct-1) AND an FY
    # annual fact (start Jan-1) with the SAME period_end — both must survive PIT
    # selection; keying without period_start would collapse them (the MAJOR bug).
    rows = [
        {
            "symbol": "ZCO",
            "cik": "1",
            "namespace": "us-gaap",
            "tag": "NetIncomeLoss",
            "unit": "USD",
            "val": 300000.0,
            "period_start": "2023-10-01",
            "period_end": "2023-12-31",
            "fy": 2023,
            "fp": "Q4",
            "frame": None,
            "form": "10-K",
            "accession": "acc-10k",
            "filed": "2024-02-01",
        },
        {
            "symbol": "ZCO",
            "cik": "1",
            "namespace": "us-gaap",
            "tag": "NetIncomeLoss",
            "unit": "USD",
            "val": 1000000.0,
            "period_start": "2023-01-01",
            "period_end": "2023-12-31",
            "fy": 2023,
            "fp": "FY",
            "frame": None,
            "form": "10-K",
            "accession": "acc-10k",
            "filed": "2024-02-01",
        },
    ]
    df = _pit_frame_from_rows(
        rows, {"acc-10k": pd.Timestamp("2024-02-01 22:00:00", tz="UTC")}
    )
    out = select_pit_rows(df, "2024-03-01")
    assert len(out) == 2
    assert set(out["val"]) == {300000.0, 1000000.0}
    assert set(out["period_start"]) == {
        pd.Timestamp("2023-10-01"),
        pd.Timestamp("2023-01-01"),
    }


def test_select_pit_tie_break_is_deterministic_and_amendment_wins():
    # Two filings for the SAME (symbol,tag,period_end,period_start) with an
    # IDENTICAL availability instant (the common date-only-fallback tie): the
    # amendment must win, INDEPENDENT of input row order.
    def _rows():
        return [
            {
                "symbol": "ZCO",
                "cik": "1",
                "namespace": "us-gaap",
                "tag": "EarningsPerShareDiluted",
                "unit": "USD/shares",
                "val": 1.20,
                "period_start": "2023-01-01",
                "period_end": "2023-03-31",
                "fy": 2023,
                "fp": "Q1",
                "frame": None,
                "form": "10-Q",
                "accession": "acc-orig",
                "filed": "2023-04-20",
            },
            {
                "symbol": "ZCO",
                "cik": "1",
                "namespace": "us-gaap",
                "tag": "EarningsPerShareDiluted",
                "unit": "USD/shares",
                "val": 1.25,
                "period_start": "2023-01-01",
                "period_end": "2023-03-31",
                "fy": 2023,
                "fp": "Q1",
                "frame": None,
                "form": "10-Q/A",
                "accession": "acc-amend",
                "filed": "2023-04-20",
            },
        ]

    tie = pd.Timestamp("2023-04-20 22:00:00", tz="UTC")
    accn = {"acc-orig": tie, "acc-amend": tie}
    fwd = select_pit_rows(_pit_frame_from_rows(_rows(), accn), "2023-12-31")
    rev = select_pit_rows(
        _pit_frame_from_rows(list(reversed(_rows())), accn), "2023-12-31"
    )
    assert len(fwd) == 1 and len(rev) == 1
    assert fwd.iloc[0]["accession"] == "acc-amend"
    assert rev.iloc[0]["accession"] == "acc-amend"
    assert fwd.iloc[0]["val"] == 1.25
    assert rev.iloc[0]["val"] == 1.25


def test_select_pit_tolerates_tz_aware_filed_date():
    # Defensive _effective_availability branch: a tz-aware filed_date (e.g. a
    # parquet round-trip that preserved tz) must NOT crash on the fallback path
    # and must gate identically to the naive case (filed_date + EDGAR_DAYS).
    rows = parse_company_facts(COMPANY_FACTS, symbol="TESTCO")
    df = company_facts_rows_to_dataframe(rows)
    df = attach_available_at(df, {})  # all available_at NaT -> filed_date fallback
    naive_out = select_pit_rows(df, "2023-12-31")
    df_aware = df.copy()
    df_aware["filed_date"] = df_aware["filed_date"].dt.tz_localize("UTC")
    aware_out = select_pit_rows(df_aware, "2023-12-31")
    assert len(aware_out) > 0
    assert sorted(aware_out["accession"].tolist()) == sorted(
        naive_out["accession"].tolist()
    )


# ---------------------------------------------------------------------------
# Follow-up A: submissions filings.files pagination (acceptance coverage)
# ---------------------------------------------------------------------------


def test_build_accession_acceptance_map_flat_page_shape():
    # Paginated submission pages carry the arrays at TOP LEVEL (not under
    # filings.recent) — the map builder must accept both shapes.
    page = {
        "accessionNumber": ["acc-old1", "acc-old2"],
        "acceptanceDateTime": [
            "2019-05-01T18:00:00.000Z",
            "2019-08-01T17:30:00.000Z",
        ],
    }
    m = build_accession_acceptance_map(page)
    assert m["acc-old1"] == pd.Timestamp("2019-05-01 18:00:00", tz="UTC")
    assert m["acc-old2"] == pd.Timestamp("2019-08-01 17:30:00", tz="UTC")


def test_submission_page_names():
    subs = {
        "filings": {
            "recent": {"accessionNumber": ["acc-new"]},
            "files": [
                {"name": "CIK0000320193-submissions-001.json", "filingCount": 1000},
                {"name": "CIK0000320193-submissions-002.json", "filingCount": 500},
            ],
        }
    }
    assert submission_page_names(subs) == [
        "CIK0000320193-submissions-001.json",
        "CIK0000320193-submissions-002.json",
    ]


def test_submission_page_names_empty():
    assert submission_page_names({"filings": {"recent": {}}}) == []
    assert submission_page_names({}) == []


# ---------------------------------------------------------------------------
# Follow-up C: ingest orchestration seam — submissions failure -> NaT, never now()
# ---------------------------------------------------------------------------


def test_ingest_submissions_failure_yields_nat_never_now(monkeypatch, tmp_path):
    # If the acceptance feed cannot be fetched, available_at MUST be NaT (the
    # filed_date+latency fallback then gates) — NEVER stamped with now().
    monkeypatch.setattr(
        fx, "resolve_user_agent", lambda ua=None: "Test test@example.com"
    )
    monkeypatch.setattr(fx, "fetch_cik_map", lambda ua, **k: {"TESTCO": "0000000001"})
    monkeypatch.setattr(fx, "fetch_company_facts", lambda cik, ua, **k: COMPANY_FACTS)

    def _boom(cik, ua, **k):
        raise RuntimeError("submissions feed unreachable")

    monkeypatch.setattr(fx, "fetch_submissions", _boom)

    out = tmp_path / "fundamentals_xbrl.parquet"
    df = ingest_fundamentals_xbrl(["TESTCO"], out_path=out)
    assert len(df) > 0  # facts still parsed despite acceptance-fetch failure
    assert df["available_at"].isna().all()  # NaT, not now()
    # ops ingest timestamp is set (not used for gating), but available_at stays NaT
    assert df["timestamp"].notna().all()
