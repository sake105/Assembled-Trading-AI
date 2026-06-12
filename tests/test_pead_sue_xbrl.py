"""Tests for the XBRL-fed PIT quarterly-EPS / SUE feature (features/pead_sue.py).

Builds a clean per-firm quarterly diluted-EPS series from the tall XBRL frame:
- quarterly (~3-month) durations give Q1-Q3 (and a directly-tagged Q4 if present),
- Q4 is DERIVED = FY - (Q1+Q2+Q3) when only the FY annual figure is tagged,
- EPS falls back to NetIncomeLoss / WeightedAvgDilutedShares when the EPS tag is
  absent for a quarter,
then SUE uses a TRUE (fp, fy-1) fiscal-label join (quarterly_seasonal_expected +
compute_sue_from_expected), NOT a positional shift(4) — robust to missing quarters.
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.data.fundamentals_xbrl_ingest import (
    company_facts_rows_to_dataframe,
)
from src.assembled_core.features.pead_sue import (
    build_quarterly_eps_panel,
    build_quarterly_eps_series,
    latest_sue_from_xbrl,
    quarterly_seasonal_expected,
)


def _eps(symbol, val, start, end, fy, fp, accn):
    return {
        "symbol": symbol,
        "cik": "1",
        "namespace": "us-gaap",
        "tag": "EarningsPerShareDiluted",
        "unit": "USD/shares",
        "val": val,
        "period_start": start,
        "period_end": end,
        "fy": fy,
        "fp": fp,
        "frame": None,
        "form": "10-Q",
        "accession": accn,
        "filed": end,
    }


def _fact(symbol, tag, val, start, end, fy, fp, accn, unit):
    return {
        "symbol": symbol,
        "cik": "1",
        "namespace": "us-gaap",
        "tag": tag,
        "unit": unit,
        "val": val,
        "period_start": start,
        "period_end": end,
        "fy": fy,
        "fp": fp,
        "frame": None,
        "form": "10-Q",
        "accession": accn,
        "filed": end,
    }


def _frame(rows):
    return company_facts_rows_to_dataframe(rows)


# Two complete fiscal years (calendar, Dec-31). 2023-Q2 has NO eps tag but DOES
# carry NetIncomeLoss + WeightedAvgDilutedShares -> EPS fallback = 1.4e9/1e9 = 1.4.
# Neither year tags Q4 directly -> Q4 derived = FY - (Q1+Q2+Q3).
def _two_year_rows():
    rows = [
        # FY2022 quarters
        _eps("ACME", 1.0, "2022-01-01", "2022-03-31", 2022, "Q1", "a22q1"),
        _eps("ACME", 1.1, "2022-04-01", "2022-06-30", 2022, "Q2", "a22q2"),
        _eps("ACME", 1.2, "2022-07-01", "2022-09-30", 2022, "Q3", "a22q3"),
        _eps("ACME", 4.5, "2022-01-01", "2022-12-31", 2022, "FY", "a22fy"),  # Q4=1.2
        # FY2023 quarters; Q2 eps missing -> NI/shares fallback (1.4)
        _eps("ACME", 1.3, "2023-01-01", "2023-03-31", 2023, "Q1", "a23q1"),
        _fact(
            "ACME",
            "NetIncomeLoss",
            1.4e9,
            "2023-04-01",
            "2023-06-30",
            2023,
            "Q2",
            "a23q2",
            "USD",
        ),
        _fact(
            "ACME",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            1.0e9,
            "2023-04-01",
            "2023-06-30",
            2023,
            "Q2",
            "a23q2",
            "shares",
        ),
        _eps("ACME", 1.5, "2023-07-01", "2023-09-30", 2023, "Q3", "a23q3"),
        _eps("ACME", 5.8, "2023-01-01", "2023-12-31", 2023, "FY", "a23fy"),  # Q4=1.6
    ]
    return rows


def test_quarterly_series_q4_derivation_and_ni_fallback():
    s = build_quarterly_eps_series(_frame(_two_year_rows()), "ACME")
    expected = pd.Series(
        [1.0, 1.1, 1.2, 1.2, 1.3, 1.4, 1.5, 1.6],
        index=pd.to_datetime(
            [
                "2022-03-31",
                "2022-06-30",
                "2022-09-30",
                "2022-12-31",
                "2023-03-31",
                "2023-06-30",
                "2023-09-30",
                "2023-12-31",
            ]
        ),
    )
    pd.testing.assert_series_equal(
        s.astype(float), expected, check_names=False, atol=1e-9
    )


def test_quarterly_series_no_fy_means_no_q4_derived():
    # Only Q1-Q3 tagged, no FY -> Q4 is NOT fabricated.
    rows = [
        _eps("ACME", 2.0, "2024-01-01", "2024-03-31", 2024, "Q1", "a24q1"),
        _eps("ACME", 2.1, "2024-04-01", "2024-06-30", 2024, "Q2", "a24q2"),
        _eps("ACME", 2.2, "2024-07-01", "2024-09-30", 2024, "Q3", "a24q3"),
    ]
    s = build_quarterly_eps_series(_frame(rows), "ACME")
    assert len(s) == 3
    assert pd.Timestamp("2024-12-31") not in s.index


def test_quarterly_series_direct_q4_used_not_derived():
    # When Q4 is directly tagged as a quarterly duration, it is used as-is.
    rows = [
        _eps("ACME", 1.0, "2022-01-01", "2022-03-31", 2022, "Q1", "a"),
        _eps("ACME", 1.1, "2022-04-01", "2022-06-30", 2022, "Q2", "b"),
        _eps("ACME", 1.2, "2022-07-01", "2022-09-30", 2022, "Q3", "c"),
        _eps("ACME", 1.9, "2022-10-01", "2022-12-31", 2022, "Q4", "d"),  # direct Q4
        _eps("ACME", 4.5, "2022-01-01", "2022-12-31", 2022, "FY", "e"),
    ]
    s = build_quarterly_eps_series(_frame(rows), "ACME")
    assert s.loc[pd.Timestamp("2022-12-31")] == 1.9  # direct Q4, NOT 4.5-(...)=1.2


def test_latest_sue_from_xbrl_non_null():
    sue = latest_sue_from_xbrl(_frame(_two_year_rows()), ["ACME"])
    assert "ACME" in sue.index
    assert pd.notna(sue.loc["ACME"])


def test_latest_sue_insufficient_history_is_nan():
    rows = [
        _eps("ACME", 1.0, "2022-01-01", "2022-03-31", 2022, "Q1", "a"),
        _eps("ACME", 1.1, "2022-04-01", "2022-06-30", 2022, "Q2", "b"),
    ]
    sue = latest_sue_from_xbrl(_frame(rows), ["ACME"])
    assert pd.isna(sue.loc["ACME"])


def test_latest_sue_unknown_symbol_is_nan():
    sue = latest_sue_from_xbrl(_frame(_two_year_rows()), ["NOPE"])
    assert pd.isna(sue.loc["NOPE"])


# ---------------------------------------------------------------------------
# TRUE (fp, fy-1) seasonal alignment (replaces the blind positional shift(4))
# ---------------------------------------------------------------------------


def test_panel_carries_fiscal_labels():
    p = build_quarterly_eps_panel(_frame(_two_year_rows()), "ACME")
    assert list(p.columns) == ["period_end", "fy", "fp", "eps"]
    assert len(p) == 8
    assert set(p["fp"]) == {"Q1", "Q2", "Q3", "Q4"}
    assert set(p["fy"].astype(int)) == {2022, 2023}


def test_seasonal_expected_uses_fp_fy1_not_positional():
    # A GAP (2022-Q3 missing) would make a positional shift(4) compare the WRONG
    # quarters; the (fp, fy-1) join must still pair 2023-Qn with 2022-Qn.
    panel = pd.DataFrame(
        [
            {
                "period_end": pd.Timestamp("2022-03-31"),
                "fy": 2022,
                "fp": "Q1",
                "eps": 1.0,
            },
            {
                "period_end": pd.Timestamp("2022-06-30"),
                "fy": 2022,
                "fp": "Q2",
                "eps": 1.1,
            },
            # 2022-Q3 intentionally MISSING
            {
                "period_end": pd.Timestamp("2022-12-31"),
                "fy": 2022,
                "fp": "Q4",
                "eps": 1.3,
            },
            {
                "period_end": pd.Timestamp("2023-03-31"),
                "fy": 2023,
                "fp": "Q1",
                "eps": 1.5,
            },
            {
                "period_end": pd.Timestamp("2023-06-30"),
                "fy": 2023,
                "fp": "Q2",
                "eps": 1.6,
            },
        ]
    )
    exp = quarterly_seasonal_expected(panel)
    assert (
        exp.loc[pd.Timestamp("2023-03-31")] == 1.0
    )  # 2022-Q1, NOT a positional neighbor
    assert exp.loc[pd.Timestamp("2023-06-30")] == 1.1  # 2022-Q2
    assert pd.isna(exp.loc[pd.Timestamp("2022-03-31")])  # no fy-1 -> NaN
    assert pd.isna(exp.loc[pd.Timestamp("2022-12-31")])  # 2021-Q4 absent -> NaN
