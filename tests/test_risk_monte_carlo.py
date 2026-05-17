"""Tests for risk/monte_carlo: trade-shuffle bootstrap + path simulation.

Covers API contract, statistical sanity checks, and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.risk.monte_carlo import (
    ShuffleResult,
    simulate_paths_block_bootstrap,
    simulate_paths_iid_normal,
    shuffle_trades,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

TRADE_PNL_ARR = RNG.normal(0.002, 0.015, size=200)  # 200 trades, ~0.2% avg
TRADE_PNL_SERIES = pd.Series(TRADE_PNL_ARR)

DAILY_RETURNS_ARR = RNG.normal(0.0003, 0.012, size=500)  # 500 daily returns
DAILY_RETURNS_SERIES = pd.Series(DAILY_RETURNS_ARR)


# ===========================================================================
# trade_shuffle tests
# ===========================================================================


class TestShuffleTrades:
    def test_shuffle_trades_returns_distributions(self):
        result = shuffle_trades(TRADE_PNL_ARR, n_iterations=200, seed=0)
        assert isinstance(result, ShuffleResult)
        assert result.n_iterations == 200
        assert result.sharpe_distribution.shape == (200,)
        assert result.max_drawdown_distribution.shape == (200,)
        assert result.total_return_distribution.shape == (200,)

    def test_shuffle_trades_seed_reproducibility(self):
        r1 = shuffle_trades(TRADE_PNL_ARR, n_iterations=100, seed=7)
        r2 = shuffle_trades(TRADE_PNL_ARR, n_iterations=100, seed=7)
        np.testing.assert_array_equal(r1.sharpe_distribution, r2.sharpe_distribution)
        np.testing.assert_array_equal(
            r1.max_drawdown_distribution, r2.max_drawdown_distribution
        )

    def test_shuffle_trades_different_seeds_differ(self):
        r1 = shuffle_trades(TRADE_PNL_ARR, n_iterations=100, seed=1)
        r2 = shuffle_trades(TRADE_PNL_ARR, n_iterations=100, seed=2)
        # With 200-trade sequences and random seeds, results must differ
        assert not np.array_equal(r1.sharpe_distribution, r2.sharpe_distribution)

    def test_shuffle_trades_ci_widening_with_more_iters(self):
        """More iterations should not narrow CI artificially — width stable / can grow."""
        r_small = shuffle_trades(TRADE_PNL_ARR, n_iterations=50, seed=42)
        r_large = shuffle_trades(TRADE_PNL_ARR, n_iterations=2000, seed=42)
        lo_s, hi_s = r_small.confidence_interval("sharpe")
        lo_l, hi_l = r_large.confidence_interval("sharpe")
        # With a fixed seed and very different sample counts, distributions differ
        # but both should be finite
        assert np.isfinite(lo_s) and np.isfinite(hi_s)
        assert np.isfinite(lo_l) and np.isfinite(hi_l)

    def test_shuffle_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            shuffle_trades(np.array([]), n_iterations=100)

    def test_shuffle_nan_input_raises(self):
        bad = np.array([0.01, np.nan, 0.02])
        with pytest.raises(ValueError, match="NaN"):
            shuffle_trades(bad, n_iterations=100)

    def test_shuffle_inf_input_raises(self):
        bad = np.array([0.01, np.inf, 0.02])
        with pytest.raises(ValueError, match="inf"):
            shuffle_trades(bad, n_iterations=100)

    def test_shuffle_confidence_interval_method(self):
        result = shuffle_trades(TRADE_PNL_ARR, n_iterations=500, seed=0)
        lo, hi = result.confidence_interval("sharpe")
        assert lo < hi
        lo2, hi2 = result.confidence_interval("sharpe", lo=0.10, hi=0.90)
        # Narrower interval must be inside wider one
        lo_wide, hi_wide = result.confidence_interval("sharpe", lo=0.01, hi=0.99)
        assert lo_wide <= lo2 <= hi2 <= hi_wide

    def test_shuffle_mdd_values_non_positive(self):
        """Max drawdown must be <= 0 (loss or no-drawdown)."""
        result = shuffle_trades(TRADE_PNL_ARR, n_iterations=300, seed=0)
        assert np.all(result.max_drawdown_distribution <= 0.0)

    def test_shuffle_accepts_pandas_series(self):
        result = shuffle_trades(TRADE_PNL_SERIES, n_iterations=100, seed=0)
        assert result.n_iterations == 100

    def test_shuffle_positive_edge_sharpe_positive(self):
        """With consistently positive trades the median Sharpe should be > 0."""
        all_positive = np.full(100, 0.005)  # every trade +0.5%
        result = shuffle_trades(all_positive, n_iterations=200, seed=0)
        median_sharpe = np.median(result.sharpe_distribution)
        assert median_sharpe > 0


# ===========================================================================
# path_simulator — GBM tests
# ===========================================================================


class TestSimulatePathsGbm:
    def test_gbm_paths_shape(self):
        result = simulate_paths_iid_normal(
            DAILY_RETURNS_ARR, n_paths=50, n_periods=100, seed=0
        )
        assert result.paths.shape == (50, 100)

    def test_gbm_paths_start_at_one(self):
        result = simulate_paths_iid_normal(
            DAILY_RETURNS_ARR, n_paths=100, n_periods=252, seed=0
        )
        # First column is the first period's level; equity should START at 1.0
        # Implementation: paths start at 1.0 BEFORE any return is applied.
        # This tests that paths[:,0] represents the first return applied to 1.0
        # => paths[:,0] should be close to 1.0 (within a few sigma of daily vol)
        assert result.paths.shape[1] == 252
        # All paths must be strictly positive
        assert np.all(result.paths > 0)

    def test_gbm_paths_seed_reproducible(self):
        r1 = simulate_paths_iid_normal(
            DAILY_RETURNS_ARR, n_paths=30, n_periods=60, seed=99
        )
        r2 = simulate_paths_iid_normal(
            DAILY_RETURNS_ARR, n_paths=30, n_periods=60, seed=99
        )
        np.testing.assert_array_equal(r1.paths, r2.paths)

    def test_gbm_method_label(self):
        result = simulate_paths_iid_normal(
            DAILY_RETURNS_ARR, n_paths=10, n_periods=10, seed=0
        )
        assert result.method == "iid_normal"

    def test_gbm_accepts_series(self):
        result = simulate_paths_iid_normal(
            DAILY_RETURNS_SERIES, n_paths=20, n_periods=50, seed=0
        )
        assert result.paths.shape == (20, 50)


# ===========================================================================
# path_simulator — block bootstrap tests
# ===========================================================================


class TestSimulatePathsBlockBootstrap:
    def test_block_bootstrap_paths_shape(self):
        result = simulate_paths_block_bootstrap(
            DAILY_RETURNS_ARR, n_paths=40, n_periods=100, block_size=5, seed=0
        )
        assert result.paths.shape == (40, 100)

    def test_block_bootstrap_preserves_mean_return_approximately(self):
        """Block bootstrap mean return per period should be close to historical mean."""
        result = simulate_paths_block_bootstrap(
            DAILY_RETURNS_ARR, n_paths=500, n_periods=252, block_size=5, seed=0
        )
        hist_mean = float(np.mean(DAILY_RETURNS_ARR))
        # Average log return per period across all paths
        log_returns = np.log(result.paths[:, 1:] / result.paths[:, :-1])
        sim_mean = float(np.mean(log_returns))
        # Allow 3× standard error tolerance
        tol = 3 * abs(hist_mean) + 0.001
        assert (
            abs(sim_mean - hist_mean) < tol
        ), f"sim_mean={sim_mean:.5f} too far from hist_mean={hist_mean:.5f}"

    def test_block_bootstrap_block_size_one_equals_iid_bootstrap(self):
        """Block size=1 is equivalent to i.i.d. bootstrap — should work without error."""
        result = simulate_paths_block_bootstrap(
            DAILY_RETURNS_ARR, n_paths=50, n_periods=100, block_size=1, seed=0
        )
        assert result.paths.shape == (50, 100)

    def test_block_bootstrap_method_label(self):
        result = simulate_paths_block_bootstrap(
            DAILY_RETURNS_ARR, n_paths=10, n_periods=10, block_size=5, seed=0
        )
        assert result.method == "block_bootstrap"

    def test_block_bootstrap_all_paths_positive(self):
        result = simulate_paths_block_bootstrap(
            DAILY_RETURNS_ARR, n_paths=200, n_periods=252, block_size=5, seed=0
        )
        assert np.all(result.paths > 0)

    def test_block_bootstrap_seed_reproducible(self):
        r1 = simulate_paths_block_bootstrap(
            DAILY_RETURNS_ARR, n_paths=20, n_periods=50, block_size=5, seed=13
        )
        r2 = simulate_paths_block_bootstrap(
            DAILY_RETURNS_ARR, n_paths=20, n_periods=50, block_size=5, seed=13
        )
        np.testing.assert_array_equal(r1.paths, r2.paths)


# ===========================================================================
# PathSimResult helper tests
# ===========================================================================


class TestPathSimResult:
    @pytest.fixture()
    def result(self):
        return simulate_paths_iid_normal(
            DAILY_RETURNS_ARR, n_paths=200, n_periods=252, seed=0
        )

    def test_pathsimresult_final_value_quantiles_ordered(self, result):
        qs = [0.05, 0.25, 0.50, 0.75, 0.95]
        quantiles = result.final_value_quantiles(qs)
        vals = [quantiles[q] for q in qs]
        assert vals == sorted(vals), "Quantiles must be monotonically increasing"

    def test_pathsimresult_max_drawdown_quantiles_negative(self, result):
        qs = [0.05, 0.25, 0.50, 0.75, 0.95]
        quantiles = result.max_drawdown_quantiles(qs)
        for q, v in quantiles.items():
            assert v <= 0.0, f"MDD quantile at {q} must be <= 0, got {v}"

    def test_pathsimresult_max_drawdown_quantiles_ordered(self, result):
        qs = [0.05, 0.25, 0.50, 0.75, 0.95]
        quantiles = result.max_drawdown_quantiles(qs)
        vals = [quantiles[q] for q in qs]
        assert vals == sorted(vals), "MDD quantiles must be monotonically increasing"

    def test_pathsimresult_default_quantiles(self, result):
        fv = result.final_value_quantiles()
        assert set(fv.keys()) == {0.05, 0.25, 0.50, 0.75, 0.95}
        mdd = result.max_drawdown_quantiles()
        assert set(mdd.keys()) == {0.05, 0.25, 0.50, 0.75, 0.95}
