"""Repoint of altdata_loader.load_insider_filings onto the EDGAR Form 4 feed.

The live insider factor (``earnings_insider_wrapper.compute_earnings_insider_factors``)
requires ``[symbol, filing_date, transaction_type, value_usd]`` UNCONDITIONALLY
(validated even on empty frames). The legacy loader stripped to
``[symbol, filing_date, shares_delta]`` — so the factor degraded. These tests pin
that the loader now:
  * prefers ``output/insider_form4.parquet`` (the new ingester output),
  * surfaces ``transaction_type`` + ``value_usd``,
  * always returns those wrapper-required columns (even when empty),
  * derives ``shares_delta`` from the signed ``net_shares``,
  * still falls back to the legacy ``insider_trading.parquet`` for back-compat.
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.data.altdata_loader import load_insider_filings
from src.assembled_core.features.earnings_insider_wrapper import (
    compute_earnings_insider_factors,
)


def _write_form4(tmp_path, rows):
    df = pd.DataFrame(rows)
    p = tmp_path / "insider_form4.parquet"
    df.to_parquet(p, index=False)
    return tmp_path


def test_reads_insider_form4_with_wrapper_columns(tmp_path):
    root = _write_form4(
        tmp_path,
        [
            {
                "symbol": "AAPL",
                "filing_date": pd.Timestamp("2024-10-08"),
                "transaction_type": "S",
                "value_usd": 1_000_000.0,
                "net_shares": -5000.0,
            }
        ],
    )
    out = load_insider_filings(["AAPL"], pd.Timestamp("2024-10-10"), root=root)
    assert "transaction_type" in out.columns
    assert "value_usd" in out.columns
    assert len(out) == 1
    assert out.iloc[0]["transaction_type"] == "S"
    assert out.iloc[0]["value_usd"] == 1_000_000.0


def test_pit_filters_future_filings(tmp_path):
    root = _write_form4(
        tmp_path,
        [
            {
                "symbol": "AAPL",
                "filing_date": pd.Timestamp("2024-10-08"),
                "transaction_type": "S",
                "value_usd": 1.0,
                "net_shares": -1.0,
            },
            {
                "symbol": "AAPL",
                "filing_date": pd.Timestamp("2024-10-20"),
                "transaction_type": "P",
                "value_usd": 9.0,
                "net_shares": 9.0,
            },
        ],
    )
    out = load_insider_filings(["AAPL"], pd.Timestamp("2024-10-10"), root=root)
    assert len(out) == 1
    assert out.iloc[0]["transaction_type"] == "S"


def test_shares_delta_derived_from_signed_net_shares(tmp_path):
    root = _write_form4(
        tmp_path,
        [
            {
                "symbol": "AAPL",
                "filing_date": pd.Timestamp("2024-10-08"),
                "transaction_type": "S",
                "value_usd": 1.0,
                "net_shares": -5000.0,
            }
        ],
    )
    out = load_insider_filings(["AAPL"], pd.Timestamp("2024-10-10"), root=root)
    assert out.iloc[0]["shares_delta"] == -5000.0


def test_missing_file_returns_required_columns(tmp_path):
    # Nothing on disk -> empty frame that STILL carries the wrapper-required cols.
    out = load_insider_filings(["AAPL"], pd.Timestamp("2024-10-10"), root=tmp_path)
    assert out.empty
    for col in ("symbol", "filing_date", "transaction_type", "value_usd"):
        assert col in out.columns


def test_legacy_fallback_adds_value_usd(tmp_path):
    # Only the legacy file exists (no value_usd column) -> loader falls back and
    # synthesizes value_usd so the wrapper validation never crashes.
    legacy = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": pd.Timestamp("2024-10-08"),
                "transaction_type": "unknown",
                "shares": -5000.0,
            }
        ]
    )
    legacy.to_parquet(tmp_path / "insider_trading.parquet", index=False)
    out = load_insider_filings(["AAPL"], pd.Timestamp("2024-10-10"), root=tmp_path)
    assert "value_usd" in out.columns
    assert "transaction_type" in out.columns


def test_loaded_frame_passes_wrapper_validation(tmp_path):
    # End-to-end: the loader output feeds compute_earnings_insider_factors
    # without raising the "insider_df missing required columns" ValueError.
    root = _write_form4(
        tmp_path,
        [
            {
                "symbol": "AAPL",
                "filing_date": pd.Timestamp("2024-10-08"),
                "transaction_type": "S",
                "value_usd": 2_000_000.0,
                "net_shares": -5000.0,
            },
            {
                "symbol": "MSFT",
                "filing_date": pd.Timestamp("2024-10-09"),
                "transaction_type": "P",
                "value_usd": 500_000.0,
                "net_shares": 3000.0,
            },
        ],
    )
    insider_df = load_insider_filings(
        ["AAPL", "MSFT"], pd.Timestamp("2024-10-10"), root=root
    )
    earnings_empty = pd.DataFrame(
        columns=["symbol", "filing_date", "eps_actual", "eps_estimate"]
    )
    result = compute_earnings_insider_factors(
        pd.Timestamp("2024-10-10"), ["AAPL", "MSFT"], earnings_empty, insider_df
    )
    assert "insider_activity_score" in result.columns
    # AAPL (sale, -) and MSFT (purchase, +) should produce a non-degenerate score.
    assert result["insider_activity_score"].notna().any()
