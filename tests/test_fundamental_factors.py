"""Tests for M17: Fundamental Factor Features."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

pytest.importorskip("src.assembled_core.features.fundamental_factors")
from src.assembled_core.features.fundamental_factors import (
    FundamentalFactorResult,
    FUNDAMENTAL_COLUMNS,
    compute_single_symbol_factors,
    build_fundamental_factors,
    cross_sectional_zscore,
    clear_cache,
)

# -- Synthetic info dicts (no yfinance needed) --------------------------------

AAPL_INFO = {
    "dividendYield": 0.005,
    "bookValue": 4.0,
    "currentPrice": 180.0,
    "grossProfits": 170_000_000_000,
    "totalAssets": 352_000_000_000,
    "returnOnEquity": 1.56,
    "marketCap": 2_800_000_000_000,
    "trailingPE": 28.5,
}

MSFT_INFO = {
    "dividendYield": 0.008,
    "bookValue": 27.0,
    "currentPrice": 420.0,
    "grossProfits": 135_000_000_000,
    "totalAssets": 411_000_000_000,
    "returnOnEquity": 0.39,
    "marketCap": 3_100_000_000_000,
    "trailingPE": 35.0,
}

PARTIAL_INFO = {
    "dividendYield": 0.0,
    "currentPrice": 50.0,
    # Missing many fields
}

EMPTY_INFO: dict = {}


# -- Tests -------------------------------------------------------------------


@pytest.mark.phase12
class TestComputeSingleSymbol:
    def test_aapl_factors(self):
        factors = compute_single_symbol_factors("AAPL", AAPL_INFO)
        assert factors["carry_dividend_yield"] == pytest.approx(0.005)
        assert factors["value_book_to_market"] == pytest.approx(4.0 / 180.0, rel=0.01)
        assert factors["quality_gross_profit"] == pytest.approx(170e9 / 352e9, rel=0.01)
        assert factors["quality_roe"] == pytest.approx(1.56)
        assert factors["size_log_market_cap"] == pytest.approx(
            np.log10(2.8e12), rel=0.01
        )
        assert factors["value_earnings_yield"] == pytest.approx(1.0 / 28.5, rel=0.01)

    def test_partial_info_fills_nan(self):
        factors = compute_single_symbol_factors("XYZ", PARTIAL_INFO)
        assert factors["carry_dividend_yield"] == 0.0
        assert np.isnan(factors["value_book_to_market"])
        assert np.isnan(factors["quality_gross_profit"])

    def test_empty_info_all_nan(self):
        factors = compute_single_symbol_factors("XYZ", EMPTY_INFO)
        nan_count = sum(
            1 for v in factors.values() if isinstance(v, float) and np.isnan(v)
        )
        assert nan_count >= 4

    def test_high_dividend_yield_normalized(self):
        info = {"dividendYield": 5.0}  # 5.0 likely means 500% -> normalize
        factors = compute_single_symbol_factors("X", info)
        assert factors["carry_dividend_yield"] == pytest.approx(0.05)


@pytest.mark.phase12
class TestBuildFundamentalFactors:
    def test_basic_build(self):
        info_dict = {"AAPL": AAPL_INFO, "MSFT": MSFT_INFO}
        result = build_fundamental_factors(["AAPL", "MSFT"], info_dict=info_dict)
        assert isinstance(result, FundamentalFactorResult)
        assert len(result.factors) == 2
        assert "symbol" in result.factors.columns
        assert result.coverage > 0

    def test_empty_symbols(self):
        result = build_fundamental_factors([], info_dict={})
        assert result.factors.empty
        assert result.coverage == 0.0

    def test_mixed_coverage(self):
        info_dict = {"AAPL": AAPL_INFO, "XYZ": EMPTY_INFO}
        result = build_fundamental_factors(["AAPL", "XYZ"], info_dict=info_dict)
        assert len(result.factors) == 2
        # AAPL should have good coverage, XYZ poor

    def test_all_fundamental_columns_present(self):
        result = build_fundamental_factors(["AAPL"], info_dict={"AAPL": AAPL_INFO})
        for col in FUNDAMENTAL_COLUMNS:
            assert col in result.factors.columns


@pytest.mark.phase12
class TestCrossSectionalZscore:
    def test_basic_zscore(self):
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "carry_dividend_yield": [0.01, 0.02, 0.03, 0.04],
                "value_book_to_market": [0.5, 1.0, 1.5, 2.0],
            }
        )
        result = cross_sectional_zscore(
            df, columns=["carry_dividend_yield", "value_book_to_market"]
        )
        # Mean should be ~0, std ~1 after z-scoring
        assert result["carry_dividend_yield"].mean() == pytest.approx(0.0, abs=0.1)
        assert result["carry_dividend_yield"].std() == pytest.approx(1.0, abs=0.1)

    def test_preserves_symbol_column(self):
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "carry_dividend_yield": [0.01, 0.02, 0.03],
            }
        )
        result = cross_sectional_zscore(df)
        assert list(result["symbol"]) == ["A", "B", "C"]

    def test_nan_handling(self):
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "carry_dividend_yield": [0.01, np.nan, 0.03, 0.04],
            }
        )
        result = cross_sectional_zscore(df)
        assert result["carry_dividend_yield"].notna().sum() >= 2


@pytest.mark.phase12
class TestCache:
    def test_clear_cache(self):
        clear_cache()
        # Should not raise
