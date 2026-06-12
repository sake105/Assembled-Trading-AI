"""Tests for load_fundamentals_xbrl (PIT-safe loader in altdata_loader.py).

Writes a tall ``fundamentals_xbrl.parquet`` to a tmp ``root`` (built from the
ingester's tested pure helpers) and verifies the loader's PIT gate +
restatement selection + keep-list contract + missing-cache degradation.
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.data.altdata_loader import load_fundamentals_xbrl
from src.assembled_core.data.fundamentals_xbrl_ingest import (
    attach_available_at,
    build_accession_acceptance_map,
    company_facts_rows_to_dataframe,
    parse_company_facts,
)
from tests.test_fundamentals_xbrl_ingest import COMPANY_FACTS, SUBMISSIONS

_REQUIRED = [
    "symbol",
    "tag",
    "namespace",
    "period_end",
    "fp",
    "fy",
    "val",
    "available_at",
    "filed_date",
    "accession",
]


def _write_parquet(tmp_path):
    rows = parse_company_facts(COMPANY_FACTS, symbol="TESTCO")
    df = company_facts_rows_to_dataframe(rows)
    df = attach_available_at(df, build_accession_acceptance_map(SUBMISSIONS))
    out = tmp_path / "fundamentals_xbrl.parquet"
    df.to_parquet(out, index=False)
    return tmp_path


def test_loader_missing_cache_returns_schema_correct_empty(tmp_path):
    df = load_fundamentals_xbrl(["TESTCO"], pd.Timestamp("2023-12-31"), root=tmp_path)
    assert len(df) == 0
    for c in _REQUIRED:
        assert c in df.columns


def test_loader_pit_original_before_restatement(tmp_path):
    root = _write_parquet(tmp_path)
    df = load_fundamentals_xbrl(["TESTCO"], pd.Timestamp("2023-05-01"), root=root)
    eps_q1 = df[
        (df["tag"] == "EarningsPerShareDiluted")
        & (df["period_end"] == pd.Timestamp("2023-03-31"))
    ]
    assert len(eps_q1) == 1
    assert eps_q1.iloc[0]["val"] == 1.20
    assert eps_q1.iloc[0]["accession"] == "acc-q1"


def test_loader_pit_restated_after_amendment(tmp_path):
    root = _write_parquet(tmp_path)
    df = load_fundamentals_xbrl(["TESTCO"], pd.Timestamp("2023-09-01"), root=root)
    eps_q1 = df[
        (df["tag"] == "EarningsPerShareDiluted")
        & (df["period_end"] == pd.Timestamp("2023-03-31"))
    ]
    assert len(eps_q1) == 1
    assert eps_q1.iloc[0]["val"] == 1.25
    assert eps_q1.iloc[0]["accession"] == "acc-q1a"


def test_loader_keep_list_columns_present(tmp_path):
    root = _write_parquet(tmp_path)
    df = load_fundamentals_xbrl(["TESTCO"], pd.Timestamp("2023-12-31"), root=root)
    for c in _REQUIRED:
        assert c in df.columns


def test_loader_symbol_filter(tmp_path):
    root = _write_parquet(tmp_path)
    df = load_fundamentals_xbrl(["NOTREAL"], pd.Timestamp("2023-12-31"), root=root)
    assert len(df) == 0
