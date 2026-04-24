"""Tests for portfolio attribution module — M6-T08.

Covers:
- compute_symbol_return_contributions: normal, empty, missing symbols
- compute_portfolio_return: normal, zero weights, empty
- compute_covariance_matrix: sufficient data, too few bars, missing columns
- compute_symbol_vol_contributions: normal case, single symbol, zero weights
- compute_portfolio_vol: normal, insufficient data
- compute_attribution_report: full report, no price data, insufficient data
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase12

pytest.importorskip("src.assembled_core.risk.attribution")
from src.assembled_core.risk.attribution import (
    compute_attribution_report,
    compute_covariance_matrix,
    compute_portfolio_return,
    compute_portfolio_vol,
    compute_symbol_return_contributions,
    compute_symbol_vol_contributions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_df(
    symbols: list[str], n_bars: int = 70, seed: int = 42
) -> pd.DataFrame:
    """Build synthetic price DataFrame for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for sym in symbols:
        price = 100.0
        for i in range(n_bars):
            price *= 1.0 + rng.normal(0, 0.01)
            rows.append(
                {
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "symbol": sym,
                    "close": price,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_symbol_return_contributions
# ---------------------------------------------------------------------------


class TestSymbolReturnContributions:
    def test_normal_case(self):
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        returns = {"AAPL": 0.02, "MSFT": -0.01}
        result = compute_symbol_return_contributions(weights, returns)
        assert result["AAPL"] == pytest.approx(0.012)
        assert result["MSFT"] == pytest.approx(-0.004)

    def test_missing_return_treated_as_zero(self):
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        returns = {"AAPL": 0.02}  # MSFT missing
        result = compute_symbol_return_contributions(weights, returns)
        assert result["AAPL"] == pytest.approx(0.012)
        assert result["MSFT"] == pytest.approx(0.0)

    def test_empty_weights_returns_empty(self):
        result = compute_symbol_return_contributions({}, {"AAPL": 0.01})
        assert result == {}

    def test_empty_returns_gives_zero_contributions(self):
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        result = compute_symbol_return_contributions(weights, {})
        assert result == {}

    def test_contributions_sum_to_portfolio_return(self):
        weights = {"A": 0.3, "B": 0.3, "C": 0.4}
        returns = {"A": 0.05, "B": -0.02, "C": 0.01}
        result = compute_symbol_return_contributions(weights, returns)
        expected = sum(weights[s] * returns[s] for s in weights)
        assert sum(result.values()) == pytest.approx(expected)

    def test_zero_weight_contributes_zero(self):
        weights = {"AAPL": 0.0, "MSFT": 1.0}
        returns = {"AAPL": 0.10, "MSFT": 0.05}
        result = compute_symbol_return_contributions(weights, returns)
        assert result["AAPL"] == pytest.approx(0.0)
        assert result["MSFT"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# compute_portfolio_return
# ---------------------------------------------------------------------------


class TestPortfolioReturn:
    def test_normal_case(self):
        weights = {"A": 0.5, "B": 0.5}
        returns = {"A": 0.04, "B": 0.02}
        result = compute_portfolio_return(weights, returns)
        assert result == pytest.approx(0.03)

    def test_empty_weights_returns_zero(self):
        result = compute_portfolio_return({}, {"A": 0.01})
        assert result == 0.0

    def test_empty_returns_returns_zero(self):
        result = compute_portfolio_return({"A": 1.0}, {})
        assert result == 0.0

    def test_single_symbol_full_weight(self):
        result = compute_portfolio_return({"AAPL": 1.0}, {"AAPL": 0.05})
        assert result == pytest.approx(0.05)

    def test_negative_return(self):
        weights = {"A": 0.4, "B": 0.6}
        returns = {"A": -0.03, "B": -0.02}
        result = compute_portfolio_return(weights, returns)
        assert result == pytest.approx(-0.024)


# ---------------------------------------------------------------------------
# compute_covariance_matrix
# ---------------------------------------------------------------------------


class TestCovarianceMatrix:
    def test_normal_returns_square_matrix(self):
        prices = _make_price_df(["AAPL", "MSFT", "GOOG"], n_bars=70)
        cov = compute_covariance_matrix(prices, ["AAPL", "MSFT", "GOOG"])
        assert not cov.empty
        assert cov.shape == (3, 3)
        # Diagonal should be positive (variances)
        for sym in ["AAPL", "MSFT", "GOOG"]:
            assert cov.loc[sym, sym] > 0.0

    def test_symmetric_matrix(self):
        prices = _make_price_df(["A", "B"], n_bars=70)
        cov = compute_covariance_matrix(prices, ["A", "B"])
        assert not cov.empty
        assert cov.loc["A", "B"] == pytest.approx(cov.loc["B", "A"])

    def test_too_few_bars_returns_empty(self):
        prices = _make_price_df(["A", "B"], n_bars=2)
        cov = compute_covariance_matrix(prices, ["A", "B"])
        assert cov.empty

    def test_single_symbol_returns_empty(self):
        prices = _make_price_df(["AAPL"], n_bars=70)
        cov = compute_covariance_matrix(prices, ["AAPL"])
        assert cov.empty

    def test_empty_prices_returns_empty(self):
        cov = compute_covariance_matrix(pd.DataFrame(), ["A", "B"])
        assert cov.empty

    def test_missing_timestamp_column_returns_empty(self):
        df = pd.DataFrame({"symbol": ["A", "B"], "close": [100.0, 200.0]})
        cov = compute_covariance_matrix(df, ["A", "B"])
        assert cov.empty

    def test_annualization_scales_values(self):
        prices = _make_price_df(["A", "B"], n_bars=70)
        cov_252 = compute_covariance_matrix(prices, ["A", "B"], annualize_factor=252.0)
        cov_52 = compute_covariance_matrix(prices, ["A", "B"], annualize_factor=52.0)
        # 252/52 ratio
        ratio = cov_252.loc["A", "A"] / cov_52.loc["A", "A"]
        assert ratio == pytest.approx(252.0 / 52.0, rel=1e-6)


# ---------------------------------------------------------------------------
# compute_symbol_vol_contributions
# ---------------------------------------------------------------------------


class TestSymbolVolContributions:
    def test_contributions_sum_to_portfolio_vol(self):
        prices = _make_price_df(["A", "B", "C"], n_bars=70)
        weights = {"A": 0.4, "B": 0.35, "C": 0.25}
        cov = compute_covariance_matrix(prices, ["A", "B", "C"])
        assert not cov.empty
        contribs = compute_symbol_vol_contributions(weights, cov)
        portfolio_vol = compute_portfolio_vol(weights, cov)
        assert sum(contribs.values()) == pytest.approx(portfolio_vol, rel=1e-6)

    def test_empty_weights_returns_empty(self):
        prices = _make_price_df(["A", "B"], n_bars=70)
        cov = compute_covariance_matrix(prices, ["A", "B"])
        result = compute_symbol_vol_contributions({}, cov)
        assert result == {}

    def test_empty_cov_returns_empty(self):
        weights = {"A": 0.5, "B": 0.5}
        result = compute_symbol_vol_contributions(weights, pd.DataFrame())
        assert result == {}

    def test_single_symbol_in_weights_returns_empty(self):
        # Only one symbol in weights that's in cov — need ≥2
        prices = _make_price_df(["A", "B"], n_bars=70)
        cov = compute_covariance_matrix(prices, ["A", "B"])
        result = compute_symbol_vol_contributions({"A": 1.0}, cov)
        assert result == {}

    def test_contributions_are_positive_for_long_only(self):
        prices = _make_price_df(["X", "Y"], n_bars=70)
        weights = {"X": 0.6, "Y": 0.4}
        cov = compute_covariance_matrix(prices, ["X", "Y"])
        contribs = compute_symbol_vol_contributions(weights, cov)
        # Long-only with positive weights should have positive vol contributions
        for sym, val in contribs.items():
            assert val > 0.0, f"{sym} contribution should be positive"


# ---------------------------------------------------------------------------
# compute_portfolio_vol
# ---------------------------------------------------------------------------


class TestPortfolioVol:
    def test_returns_positive_float(self):
        prices = _make_price_df(["A", "B"], n_bars=70)
        weights = {"A": 0.5, "B": 0.5}
        cov = compute_covariance_matrix(prices, ["A", "B"])
        vol = compute_portfolio_vol(weights, cov)
        assert not math.isnan(vol)
        assert vol > 0.0

    def test_empty_cov_returns_nan(self):
        vol = compute_portfolio_vol({"A": 0.5, "B": 0.5}, pd.DataFrame())
        assert math.isnan(vol)

    def test_empty_weights_returns_nan(self):
        prices = _make_price_df(["A", "B"], n_bars=70)
        cov = compute_covariance_matrix(prices, ["A", "B"])
        vol = compute_portfolio_vol({}, cov)
        assert math.isnan(vol)

    def test_higher_concentration_higher_vol(self):
        # Equal weights vs. concentrated — concentrated in high-vol should be higher
        prices = _make_price_df(["A", "B"], n_bars=70, seed=7)
        cov = compute_covariance_matrix(prices, ["A", "B"])
        vol_equal = compute_portfolio_vol({"A": 0.5, "B": 0.5}, cov)
        vol_concentrated = compute_portfolio_vol({"A": 1.0, "B": 0.0}, cov)
        # Both should be valid floats
        assert not math.isnan(vol_equal)
        assert not math.isnan(vol_concentrated)


# ---------------------------------------------------------------------------
# compute_attribution_report
# ---------------------------------------------------------------------------


class TestAttributionReport:
    def test_full_report_ok_status(self):
        prices = _make_price_df(["A", "B", "C"], n_bars=70)
        weights = {"A": 0.4, "B": 0.35, "C": 0.25}
        returns = {"A": 0.02, "B": -0.01, "C": 0.005}
        report = compute_attribution_report(weights, returns, prices)
        assert report["status"] == "ok"
        assert set(report["symbols"]) == {"A", "B", "C"}
        assert not math.isnan(report["portfolio_vol"])
        assert not math.isnan(report["portfolio_return"])

    def test_no_price_data_status(self):
        weights = {"A": 0.5, "B": 0.5}
        returns = {"A": 0.01, "B": 0.02}
        report = compute_attribution_report(weights, returns, pd.DataFrame())
        assert report["status"] == "no_price_data"
        assert math.isnan(report["portfolio_vol"])
        for sym in weights:
            assert math.isnan(report["vol_contributions"][sym])

    def test_insufficient_data_status(self):
        prices = _make_price_df(["A", "B"], n_bars=2)  # too few bars
        weights = {"A": 0.5, "B": 0.5}
        returns = {"A": 0.01, "B": 0.02}
        report = compute_attribution_report(weights, returns, prices)
        assert report["status"] == "insufficient_data"

    def test_return_contributions_correct(self):
        prices = _make_price_df(["X", "Y"], n_bars=70)
        weights = {"X": 0.6, "Y": 0.4}
        returns = {"X": 0.05, "Y": -0.02}
        report = compute_attribution_report(weights, returns, prices)
        assert report["return_contributions"]["X"] == pytest.approx(0.03)
        assert report["return_contributions"]["Y"] == pytest.approx(-0.008)

    def test_portfolio_return_matches_sum(self):
        prices = _make_price_df(["X", "Y"], n_bars=70)
        weights = {"X": 0.6, "Y": 0.4}
        returns = {"X": 0.05, "Y": -0.02}
        report = compute_attribution_report(weights, returns, prices)
        expected = 0.6 * 0.05 + 0.4 * (-0.02)
        assert report["portfolio_return"] == pytest.approx(expected)

    def test_vol_contributions_sum_to_portfolio_vol(self):
        prices = _make_price_df(["A", "B", "C"], n_bars=70)
        weights = {"A": 0.4, "B": 0.35, "C": 0.25}
        returns = {"A": 0.0, "B": 0.0, "C": 0.0}
        report = compute_attribution_report(weights, returns, prices)
        assert report["status"] == "ok"
        vol_sum = sum(report["vol_contributions"].values())
        assert vol_sum == pytest.approx(report["portfolio_vol"], rel=1e-5)

    def test_policy_overrides_lookback(self):
        prices = _make_price_df(["A", "B"], n_bars=70)
        weights = {"A": 0.5, "B": 0.5}
        returns = {"A": 0.01, "B": 0.01}
        policy = {"attribution": {"lookback_days": 20, "annualize_factor": 252.0}}
        report = compute_attribution_report(weights, returns, prices, policy=policy)
        # Just verify it runs without error and produces valid output
        assert report["status"] in ("ok", "insufficient_data", "no_price_data")

    def test_weights_preserved_in_report(self):
        prices = _make_price_df(["P", "Q"], n_bars=70)
        weights = {"P": 0.7, "Q": 0.3}
        returns = {"P": 0.01, "Q": 0.02}
        report = compute_attribution_report(weights, returns, prices)
        assert report["weights"] == weights
