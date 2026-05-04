"""Tests for correlation guard — M6-T07.

Covers:
- compute_correlation_matrix: empty/invalid inputs, basic pivot
- detect_correlated_clusters: below threshold, above threshold, transitive
- apply_correlation_guard: disabled, no prices, single symbol, cluster over cap,
  cluster under cap, multiple clusters
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.phase12

from src.assembled_core.risk.correlation_guard import (
    apply_correlation_guard,
    compute_correlation_matrix,
    detect_correlated_clusters,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(
    data: dict[str, list[float]], start: str = "2026-01-01"
) -> pd.DataFrame:
    """Build prices DataFrame from {symbol: [close, ...]} dict."""
    n = max(len(v) for v in data.values())
    dates = pd.date_range(start, periods=n, freq="B", tz="UTC")
    rows = []
    for sym, closes in data.items():
        for i, c in enumerate(closes):
            if i < len(dates):
                rows.append({"timestamp": dates[i], "symbol": sym, "close": float(c)})
    return pd.DataFrame(rows)


def _correlated_prices(n: int = 65) -> pd.DataFrame:
    """GLD and TLT move in lockstep (correlation ≈ 1.0)."""
    vals = [100.0 * (1 + 0.01 * i) for i in range(n)]
    return _make_prices({"GLD": vals, "TLT": [v * 1.5 for v in vals]})


def _uncorrelated_prices(n: int = 65) -> pd.DataFrame:
    """GLD goes up, SHY alternates — low correlation."""
    gld = [100.0 * (1 + 0.01 * i) for i in range(n)]
    shy = [100.0 * (1 + 0.05 * (-1 if i % 2 else 1)) for i in range(n)]
    return _make_prices({"GLD": gld, "SHY": shy})


# ---------------------------------------------------------------------------
# compute_correlation_matrix
# ---------------------------------------------------------------------------


class TestComputeCorrelationMatrix:
    def test_none_prices_returns_empty(self):
        result = compute_correlation_matrix(None, ["GLD", "TLT"])  # type: ignore[arg-type]
        assert result.empty

    def test_empty_prices_returns_empty(self):
        result = compute_correlation_matrix(pd.DataFrame(), ["GLD", "TLT"])
        assert result.empty

    def test_single_symbol_returns_empty(self):
        prices = _correlated_prices()
        result = compute_correlation_matrix(prices, ["GLD"])
        assert result.empty

    def test_missing_timestamp_column_returns_empty(self):
        prices = pd.DataFrame({"symbol": ["GLD", "TLT"], "close": [100.0, 100.0]})
        result = compute_correlation_matrix(prices, ["GLD", "TLT"])
        assert result.empty

    def test_insufficient_bars_returns_empty(self):
        # Only 2 bars → less than 3 bars required after pct_change
        prices = _make_prices({"GLD": [100.0, 101.0], "TLT": [100.0, 101.0]})
        result = compute_correlation_matrix(prices, ["GLD", "TLT"])
        assert result.empty

    def test_correlated_series_high_correlation(self):
        prices = _correlated_prices()
        corr = compute_correlation_matrix(prices, ["GLD", "TLT"], lookback_days=60)
        assert not corr.empty
        assert "GLD" in corr.columns and "TLT" in corr.columns
        # Diagonal = 1.0
        assert corr.loc["GLD", "GLD"] == pytest.approx(1.0)
        # Off-diagonal: should be very high (near 1.0)
        assert corr.loc["GLD", "TLT"] > 0.95

    def test_symbols_not_in_prices_handled(self):
        prices = _correlated_prices()  # has GLD, TLT
        # Request SHY which is not in prices
        result = compute_correlation_matrix(prices, ["GLD", "SHY"])
        # SHY not present → not enough valid cols
        assert result.empty or "SHY" not in result.columns


# ---------------------------------------------------------------------------
# detect_correlated_clusters
# ---------------------------------------------------------------------------


class TestDetectCorrelatedClusters:
    def test_empty_matrix_returns_empty(self):
        result = detect_correlated_clusters(pd.DataFrame())
        assert result == []

    def test_none_matrix_returns_empty(self):
        result = detect_correlated_clusters(None)  # type: ignore[arg-type]
        assert result == []

    def test_corr_below_threshold_no_cluster(self):
        corr = pd.DataFrame(
            {"A": [1.0, 0.30], "B": [0.30, 1.0]},
            index=["A", "B"],
        )
        result = detect_correlated_clusters(corr, threshold=0.70)
        assert result == []

    def test_corr_above_threshold_forms_cluster(self):
        corr = pd.DataFrame(
            {"A": [1.0, 0.85], "B": [0.85, 1.0]},
            index=["A", "B"],
        )
        result = detect_correlated_clusters(corr, threshold=0.70)
        assert len(result) == 1
        assert sorted(result[0]) == ["A", "B"]

    def test_negative_corr_not_clustered(self):
        # Negative correlation (hedging) should NOT trigger the guard
        corr = pd.DataFrame(
            {"A": [1.0, -0.90], "B": [-0.90, 1.0]},
            index=["A", "B"],
        )
        result = detect_correlated_clusters(corr, threshold=0.70)
        assert result == []

    def test_transitive_clustering(self):
        # A-B high corr, B-C high corr → A, B, C all in one cluster
        corr = pd.DataFrame(
            {
                "A": [1.00, 0.85, 0.50],
                "B": [0.85, 1.00, 0.80],
                "C": [0.50, 0.80, 1.00],
            },
            index=["A", "B", "C"],
        )
        result = detect_correlated_clusters(corr, threshold=0.70)
        # All three should be in one cluster
        assert len(result) == 1
        assert sorted(result[0]) == ["A", "B", "C"]

    def test_two_separate_clusters(self):
        # A-B cluster, C-D cluster, no cross-cluster correlation
        corr = pd.DataFrame(
            {
                "A": [1.00, 0.90, 0.10, 0.05],
                "B": [0.90, 1.00, 0.05, 0.10],
                "C": [0.10, 0.05, 1.00, 0.85],
                "D": [0.05, 0.10, 0.85, 1.00],
            },
            index=["A", "B", "C", "D"],
        )
        result = detect_correlated_clusters(corr, threshold=0.70)
        assert len(result) == 2
        cluster_sets = [sorted(c) for c in result]
        assert sorted(["A", "B"]) in cluster_sets
        assert sorted(["C", "D"]) in cluster_sets


# ---------------------------------------------------------------------------
# apply_correlation_guard
# ---------------------------------------------------------------------------


def _policy(
    enabled: bool = True,
    threshold: float = 0.70,
    max_cluster_weight: float = 0.40,
    lookback_days: int = 60,
) -> dict:
    return {
        "correlation_guard": {
            "enabled": enabled,
            "threshold": threshold,
            "max_cluster_weight": max_cluster_weight,
            "lookback_days": lookback_days,
        }
    }


class TestApplyCorrelationGuard:
    def test_disabled_returns_original_unchanged(self):
        weights = {"GLD": 0.30, "TLT": 0.25}
        prices = _correlated_prices()
        adjusted, reasons = apply_correlation_guard(
            weights, prices, _policy(enabled=False)
        )
        assert adjusted == weights
        assert reasons == []

    def test_empty_weights_returns_empty(self):
        prices = _correlated_prices()
        adjusted, reasons = apply_correlation_guard({}, prices, _policy())
        assert adjusted == {}

    def test_single_symbol_no_guard_applied(self):
        weights = {"GLD": 0.30}
        prices = _correlated_prices()
        adjusted, reasons = apply_correlation_guard(weights, prices, _policy())
        assert adjusted == {"GLD": 0.30}
        assert reasons == []

    def test_no_prices_returns_original(self):
        weights = {"GLD": 0.30, "TLT": 0.25}
        adjusted, reasons = apply_correlation_guard(weights, None, _policy())  # type: ignore[arg-type]
        assert adjusted == weights
        assert reasons == []

    def test_cluster_within_cap_unchanged(self):
        # GLD + TLT highly correlated, but combined weight 0.20 < cap 0.40
        weights = {"GLD": 0.10, "TLT": 0.10}
        prices = _correlated_prices()
        adjusted, reasons = apply_correlation_guard(
            weights, prices, _policy(max_cluster_weight=0.40)
        )
        assert adjusted["GLD"] == pytest.approx(0.10)
        assert adjusted["TLT"] == pytest.approx(0.10)
        assert reasons == []

    def test_cluster_over_cap_scaled_down(self):
        # GLD + TLT highly correlated, combined weight 0.60 > cap 0.40
        weights = {"GLD": 0.30, "TLT": 0.30}
        prices = _correlated_prices()
        adjusted, reasons = apply_correlation_guard(
            weights, prices, _policy(max_cluster_weight=0.40)
        )
        total = adjusted["GLD"] + adjusted["TLT"]
        assert total == pytest.approx(0.40, abs=1e-6)
        assert len(reasons) == 1
        assert "correlation_guard" in reasons[0]

    def test_scaling_preserves_proportions(self):
        # GLD:TLT = 2:1, both highly correlated
        weights = {"GLD": 0.40, "TLT": 0.20}
        prices = _correlated_prices()
        adjusted, _ = apply_correlation_guard(
            weights, prices, _policy(max_cluster_weight=0.30)
        )
        # Combined = 0.60 → scale = 0.30/0.60 = 0.5
        assert adjusted["GLD"] == pytest.approx(0.20, abs=1e-6)
        assert adjusted["TLT"] == pytest.approx(0.10, abs=1e-6)

    def test_uncorrelated_symbols_not_clustered(self):
        weights = {"GLD": 0.30, "SHY": 0.30}
        prices = _uncorrelated_prices()
        adjusted, reasons = apply_correlation_guard(
            weights, prices, _policy(max_cluster_weight=0.40)
        )
        # No cluster → unchanged
        assert adjusted["GLD"] == pytest.approx(0.30)
        assert adjusted["SHY"] == pytest.approx(0.30)
        assert reasons == []

    def test_does_not_mutate_input_weights(self):
        weights = {"GLD": 0.30, "TLT": 0.30}
        original = dict(weights)
        prices = _correlated_prices()
        apply_correlation_guard(weights, prices, _policy(max_cluster_weight=0.40))
        assert weights == original
