# tests/test_risk_group_exposures.py
"""Tests for Group Exposure Calculation (Sprint 9).

Tests verify:
1. Toy exposures + meta -> expected gross/net weights per group
2. Deterministic ordering
3. Missing mapping raises ValueError with symbol list
4. Unknown group values allowed (string) but stable
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.risk.group_exposures import (
    compute_all_group_exposures,
    compute_group_exposures,
)


def test_toy_exposures_expected_gross_net_weights_per_group() -> None:
    """Test that group exposures compute expected gross/net weights."""
    # Create toy exposures
    exposures_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "notional": [10000.0, 8000.0, 6000.0, -4000.0],  # TSLA is short
        "weight": [0.10, 0.08, 0.06, -0.04],
    })

    # Create security metadata
    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "sector": ["Technology", "Technology", "Technology", "Consumer"],
        "region": ["US", "US", "US", "US"],
        "currency": ["USD", "USD", "USD", "USD"],
    })

    # Compute sector exposures
    sector_df, sector_summary = compute_group_exposures(
        exposures_df,
        security_meta_df,
        "sector",
    )

    # Verify: Technology sector
    tech_row = sector_df[sector_df["group_value"] == "Technology"].iloc[0]
    # Technology: AAPL (10000) + MSFT (8000) + GOOGL (6000) = 24000 net, 24000 gross
    # Equity = 10000 / 0.10 = 100000 (from AAPL)
    # Technology net_weight = 24000 / 100000 = 0.24
    # Technology gross_weight = 24000 / 100000 = 0.24
    assert abs(tech_row["net_exposure"] - 24000.0) < 1e-10
    assert abs(tech_row["gross_exposure"] - 24000.0) < 1e-10
    assert abs(tech_row["net_weight"] - 0.24) < 1e-5
    assert abs(tech_row["gross_weight"] - 0.24) < 1e-5
    assert tech_row["n_symbols"] == 3

    # Verify: Consumer sector
    consumer_row = sector_df[sector_df["group_value"] == "Consumer"].iloc[0]
    # Consumer: TSLA (-4000) = -4000 net, 4000 gross
    # Consumer net_weight = -4000 / 100000 = -0.04
    # Consumer gross_weight = 4000 / 100000 = 0.04
    assert abs(consumer_row["net_exposure"] - (-4000.0)) < 1e-10
    assert abs(consumer_row["gross_exposure"] - 4000.0) < 1e-10
    assert abs(consumer_row["net_weight"] - (-0.04)) < 1e-5
    assert abs(consumer_row["gross_weight"] - 0.04) < 1e-5
    assert consumer_row["n_symbols"] == 1


def test_deterministic_ordering() -> None:
    """Test that results are deterministically sorted."""
    exposures_df = pd.DataFrame({
        "symbol": ["MSFT", "AAPL", "GOOGL"],
        "notional": [8000.0, 10000.0, 6000.0],
        "weight": [0.08, 0.10, 0.06],
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["MSFT", "AAPL", "GOOGL"],
        "sector": ["Technology", "Technology", "Technology"],
        "region": ["US", "US", "US"],
        "currency": ["USD", "USD", "USD"],
    })

    # Run twice
    result1, _ = compute_group_exposures(exposures_df, security_meta_df, "sector")
    result2, _ = compute_group_exposures(exposures_df, security_meta_df, "sector")

    # Verify: identical results
    pd.testing.assert_frame_equal(
        result1,
        result2,
        check_dtype=False,
    )

    # Verify: sorted by group_type, group_value
    assert result1["group_type"].tolist() == ["sector"] * len(result1)
    assert result1["group_value"].tolist() == sorted(result1["group_value"].tolist())


def test_missing_mapping_raises_valueerror_with_symbol_list() -> None:
    """Test that missing mapping raises ValueError with symbol list."""
    exposures_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "INVALID"],
        "notional": [10000.0, 8000.0, 5000.0],
        "weight": [0.10, 0.08, 0.05],
    })

    # Security metadata missing INVALID
    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "sector": ["Technology", "Technology"],
    })

    # Should raise ValueError with missing symbol list
    with pytest.raises(ValueError) as exc_info:
        compute_group_exposures(exposures_df, security_meta_df, "sector")

    error_msg = str(exc_info.value)
    assert "INVALID" in error_msg or "Missing" in error_msg
    assert "sector" in error_msg


def test_unknown_group_values_allowed_but_stable() -> None:
    """Test that unknown group values are allowed (string) but stable."""
    exposures_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "UNKNOWN_SYMBOL"],
        "notional": [10000.0, 8000.0, 5000.0],
        "weight": [0.10, 0.08, 0.05],
    })

    # Security metadata with unknown sector value
    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "UNKNOWN_SYMBOL"],
        "sector": ["Technology", "Technology", "UNKNOWN_SECTOR"],
        "region": ["US", "US", "UNKNOWN_REGION"],
        "currency": ["USD", "USD", "USD"],
    })

    # Should work (unknown values are just strings)
    sector_df, _ = compute_group_exposures(exposures_df, security_meta_df, "sector")

    # Verify: UNKNOWN_SECTOR is present
    unknown_row = sector_df[sector_df["group_value"] == "UNKNOWN_SECTOR"]
    assert len(unknown_row) == 1
    assert unknown_row.iloc[0]["n_symbols"] == 1
    assert abs(unknown_row.iloc[0]["net_exposure"] - 5000.0) < 1e-10

    # Verify: deterministic (run twice, same result)
    sector_df2, _ = compute_group_exposures(exposures_df, security_meta_df, "sector")
    pd.testing.assert_frame_equal(
        sector_df.sort_values("group_value").reset_index(drop=True),
        sector_df2.sort_values("group_value").reset_index(drop=True),
        check_dtype=False,
    )


def test_compute_all_group_exposures() -> None:
    """Test that compute_all_group_exposures returns all group types."""
    exposures_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "notional": [10000.0, 8000.0],
        "weight": [0.10, 0.08],
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "sector": ["Technology", "Technology"],
        "region": ["US", "US"],
        "currency": ["USD", "USD"],
    })

    # Compute all group exposures
    result = compute_all_group_exposures(exposures_df, security_meta_df)

    # Verify: all group types present
    assert "sector" in result
    assert "region" in result
    assert "currency" in result

    # Verify: each entry is (df, summary) tuple
    for group_type, (df, summary) in result.items():
        assert isinstance(df, pd.DataFrame)
        assert hasattr(summary, "total_groups")
        assert "group_type" in df.columns
        assert "group_value" in df.columns


def test_region_exposures() -> None:
    """Test region group exposures."""
    exposures_df = pd.DataFrame({
        "symbol": ["AAPL", "TSLA", "ASML"],
        "notional": [10000.0, 8000.0, 6000.0],
        "weight": [0.10, 0.08, 0.06],
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "TSLA", "ASML"],
        "region": ["US", "US", "EU"],
    })

    region_df, region_summary = compute_group_exposures(
        exposures_df,
        security_meta_df,
        "region",
    )

    # Verify: US region
    us_row = region_df[region_df["group_value"] == "US"].iloc[0]
    assert abs(us_row["net_exposure"] - 18000.0) < 1e-10
    assert us_row["n_symbols"] == 2

    # Verify: EU region
    eu_row = region_df[region_df["group_value"] == "EU"].iloc[0]
    assert abs(eu_row["net_exposure"] - 6000.0) < 1e-10
    assert eu_row["n_symbols"] == 1


def test_currency_exposures() -> None:
    """Test currency group exposures."""
    exposures_df = pd.DataFrame({
        "symbol": ["AAPL", "SAP", "TSLA"],
        "notional": [10000.0, 8000.0, -4000.0],  # TSLA short
        "weight": [0.10, 0.08, -0.04],
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "SAP", "TSLA"],
        "currency": ["USD", "EUR", "USD"],
    })

    currency_df, currency_summary = compute_group_exposures(
        exposures_df,
        security_meta_df,
        "currency",
    )

    # Verify: USD currency (AAPL + TSLA)
    usd_row = currency_df[currency_df["group_value"] == "USD"].iloc[0]
    # USD: AAPL (10000) + TSLA (-4000) = 6000 net, 14000 gross
    assert abs(usd_row["net_exposure"] - 6000.0) < 1e-10
    assert abs(usd_row["gross_exposure"] - 14000.0) < 1e-10
    assert usd_row["n_symbols"] == 2

    # Verify: EUR currency
    eur_row = currency_df[currency_df["group_value"] == "EUR"].iloc[0]
    assert abs(eur_row["net_exposure"] - 8000.0) < 1e-10
    assert eur_row["n_symbols"] == 1


def test_invalid_group_col_raises_valueerror() -> None:
    """Test that invalid group_col raises ValueError."""
    exposures_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "notional": [10000.0],
        "weight": [0.10],
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "sector": ["Technology"],
    })

    with pytest.raises(ValueError, match="Invalid group_col"):
        compute_group_exposures(exposures_df, security_meta_df, "invalid_type")


def test_missing_required_columns_raises_valueerror() -> None:
    """Test that missing required columns raises ValueError."""
    exposures_df = pd.DataFrame({
        "symbol": ["AAPL"],
        # Missing: notional, weight
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "sector": ["Technology"],
    })

    with pytest.raises(ValueError, match="missing required columns"):
        compute_group_exposures(exposures_df, security_meta_df, "sector")


def test_empty_exposures_handled_gracefully() -> None:
    """Test that empty exposures are handled gracefully."""
    exposures_df = pd.DataFrame(columns=["symbol", "notional", "weight"])

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "sector": ["Technology"],
    })

    # Should work (empty exposures -> empty group exposures)
    sector_df, summary = compute_group_exposures(
        exposures_df,
        security_meta_df,
        "sector",
    )

    assert len(sector_df) == 0
    assert summary.total_groups == 0
    assert summary.max_gross_weight == 0.0
    assert summary.max_net_weight == 0.0
