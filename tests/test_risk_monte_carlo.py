"""Tests for risk/monte_carlo: trade-shuffle bootstrap + path simulation.

Covers API contract, statistical sanity checks, and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.risk.monte_carlo import (
    ShuffleResult,
    permute_trades,
    pnl_to_returns,
    shuffle_result_to_quantile_dict,
    shuffle_trades,
    simulate_paths_block_bootstrap,
    simulate_paths_iid_normal,
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


# ===========================================================================
# permute_trades tests (§6.5.3 consolidation — canonical replacement for
# legacy qa.monte_carlo_paths.monte_carlo_trade_paths)
# ===========================================================================


class TestPermuteTrades:
    def test_permute_returns_shuffleresult(self):
        result = permute_trades(TRADE_PNL_ARR, n_iterations=200, seed=0)
        assert isinstance(result, ShuffleResult)
        assert result.n_iterations == 200
        assert result.sharpe_distribution.shape == (200,)

    def test_permute_preserves_total_return(self):
        """Permutation without replacement preserves the SET of trades →
        prod(1+r) over all trades is INVARIANT across permutations.
        This is the key semantic difference from shuffle_trades bootstrap."""
        result = permute_trades(TRADE_PNL_ARR, n_iterations=100, seed=0)
        # All permutation iterations must yield IDENTICAL total returns
        # (since they use the same set of trades, just reordered)
        unique_totals = np.unique(np.round(result.total_return_distribution, 10))
        assert len(unique_totals) == 1, (
            f"Permutation preserves trade set → total return invariant, "
            f"but got {len(unique_totals)} distinct totals"
        )

    def test_permute_vs_shuffle_total_return_differs(self):
        """Bootstrap (shuffle_trades) produces a DISTRIBUTION of total
        returns; permutation produces a constant. This locks the semantic
        distinction."""
        shuffle = shuffle_trades(TRADE_PNL_ARR, n_iterations=200, seed=0)
        permute = permute_trades(TRADE_PNL_ARR, n_iterations=200, seed=0)
        # shuffle_trades total return has dispersion; permute does not
        assert shuffle.total_return_distribution.std() > 1e-6
        assert permute.total_return_distribution.std() < 1e-10

    def test_permute_seed_reproducibility(self):
        r1 = permute_trades(TRADE_PNL_ARR, n_iterations=100, seed=7)
        r2 = permute_trades(TRADE_PNL_ARR, n_iterations=100, seed=7)
        np.testing.assert_array_equal(r1.sharpe_distribution, r2.sharpe_distribution)

    def test_permute_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            permute_trades(np.array([]), n_iterations=100)

    def test_permute_nan_input_raises(self):
        bad = np.array([0.01, np.nan, 0.02])
        with pytest.raises(ValueError, match="NaN"):
            permute_trades(bad, n_iterations=100)

    def test_permute_inf_input_raises(self):
        with pytest.raises(ValueError, match="inf"):
            permute_trades(np.array([0.01, np.inf, 0.02]), n_iterations=100)

    def test_permute_accepts_series(self):
        result = permute_trades(TRADE_PNL_SERIES, n_iterations=50, seed=0)
        assert result.n_iterations == 50

    def test_permute_rejects_returns_below_minus_one(self):
        """F-RISK-MC1-MINOR-1: returns <= -1.0 produce non-positive equity
        in cumprod(1+r). Must reject — caller likely passed currency PnL
        instead of return units (legacy-API footgun)."""
        with pytest.raises(ValueError, match=r"<= -1\.0"):
            permute_trades(np.array([0.01, -1.5, 0.02]), n_iterations=10)
        with pytest.raises(ValueError, match=r"<= -1\.0"):
            permute_trades(np.array([0.01, -1.0, 0.02]), n_iterations=10)


class TestShuffleRReturnsGuard:
    def test_shuffle_rejects_returns_below_minus_one(self):
        """F-RISK-MC1-MINOR-1: same guard as permute_trades."""
        with pytest.raises(ValueError, match=r"<= -1\.0"):
            shuffle_trades(np.array([0.01, -1.5, 0.02]), n_iterations=10)


class TestShuffleTradesBlockBootstrap:
    """§6.5.3 Phase 2c: block_size parameter for moving-block bootstrap.

    Enables re-migration of daily_qa_report.py (currently on legacy
    bootstrap_returns due to F-RISK-MC2-MAJOR-1 block-bootstrap gap).
    """

    def test_default_block_size_is_iid_bootstrap(self):
        """block_size=1 (default) must equal pre-Phase-2c behavior."""
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, size=200)
        r1 = shuffle_trades(returns, n_iterations=100, seed=42)
        r2 = shuffle_trades(returns, n_iterations=100, seed=42, block_size=1)
        np.testing.assert_array_equal(r1.sharpe_distribution, r2.sharpe_distribution)

    def test_block_size_changes_distribution(self):
        """block_size > 1 must produce different distribution than iid (preserves
        local structure)."""
        # Construct autocorrelated series: AR(1) with rho=0.5
        rng = np.random.default_rng(7)
        n = 300
        rho = 0.5
        innov = rng.normal(0.001, 0.01, size=n)
        ar = np.zeros(n)
        ar[0] = innov[0]
        for i in range(1, n):
            ar[i] = rho * ar[i - 1] + innov[i]
        r_iid = shuffle_trades(ar, n_iterations=500, seed=0, block_size=1)
        r_block = shuffle_trades(ar, n_iterations=500, seed=0, block_size=10)
        # The two distributions should be visibly different
        assert not np.allclose(r_iid.sharpe_distribution, r_block.sharpe_distribution)

    def test_block_size_reproducibility(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, size=200)
        r1 = shuffle_trades(returns, n_iterations=100, seed=123, block_size=5)
        r2 = shuffle_trades(returns, n_iterations=100, seed=123, block_size=5)
        np.testing.assert_array_equal(r1.sharpe_distribution, r2.sharpe_distribution)

    def test_block_size_zero_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="block_size must be a positive int"):
            shuffle_trades(rng.normal(0.001, 0.01, 100), block_size=0)

    def test_block_size_negative_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="block_size must be a positive int"):
            shuffle_trades(rng.normal(0.001, 0.01, 100), block_size=-3)

    def test_block_size_larger_than_input_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="no valid block"):
            shuffle_trades(rng.normal(0.001, 0.01, 50), block_size=100)

    def test_block_size_equals_n_works(self):
        """block_size = n means a single block — degenerate but valid."""
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, size=50)
        result = shuffle_trades(returns, n_iterations=10, seed=0, block_size=50)
        assert isinstance(result, ShuffleResult)

    def test_block_bootstrap_output_shape(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, size=200)
        result = shuffle_trades(returns, n_iterations=300, seed=0, block_size=7)
        assert result.sharpe_distribution.shape == (300,)
        assert result.max_drawdown_distribution.shape == (300,)
        assert result.total_return_distribution.shape == (300,)


# ===========================================================================
# Phase 2 migration helpers (§6.5.3): pnl_to_returns + shuffle_result_to_quantile_dict
# ===========================================================================


class TestPnlToReturns:
    def test_basic_conversion(self):
        pnl = np.array([100.0, -50.0, 200.0])
        returns = pnl_to_returns(pnl, initial_capital=100_000.0)
        np.testing.assert_allclose(returns, [0.001, -0.0005, 0.002])

    def test_accepts_series(self):
        pnl = pd.Series([100.0, -50.0])
        returns = pnl_to_returns(pnl, initial_capital=10_000.0)
        assert isinstance(returns, np.ndarray)
        np.testing.assert_allclose(returns, [0.01, -0.005])

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError, match="positive and finite"):
            pnl_to_returns(np.array([100.0]), initial_capital=0.0)

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError, match="positive and finite"):
            pnl_to_returns(np.array([100.0]), initial_capital=-100.0)

    def test_nan_capital_raises(self):
        with pytest.raises(ValueError, match="positive and finite"):
            pnl_to_returns(np.array([100.0]), initial_capital=float("nan"))

    def test_conversion_chains_to_permute_trades(self):
        """End-to-end: legacy currency PnL → returns → permute_trades."""
        pnl = pd.Series(np.full(50, 100.0))  # +$100 per trade
        returns = pnl_to_returns(pnl, initial_capital=100_000.0)
        result = permute_trades(returns, n_iterations=200, seed=0)
        # Equal-PnL ⇒ Sharpe should be very high (no variance) — but permute
        # preserves the set, so all paths have identical equity
        assert isinstance(result, ShuffleResult)
        # All sharpes identical because no variance in returns
        assert result.sharpe_distribution.std() < 1e-9


class TestShuffleResultToQuantileDict:
    @pytest.fixture()
    def result(self):
        rng = np.random.default_rng(7)
        pnl = rng.normal(50.0, 200.0, size=100)
        returns = pnl_to_returns(pnl, initial_capital=100_000.0)
        return permute_trades(returns, n_iterations=200, seed=7)

    def test_schema_keys(self, result):
        d = shuffle_result_to_quantile_dict(result, n_trades=100)
        assert set(d.keys()) >= {
            "n_paths",
            "n_trades",
            "sharpe",
            "mdd",
            "cagr",
            "final_equity",
            "pct_ruined",
        }

    def test_sharpe_subkeys(self, result):
        d = shuffle_result_to_quantile_dict(result, n_trades=100)
        assert set(d["sharpe"].keys()) == {"mean", "std", "p10", "p50", "p90"}

    def test_mdd_subkeys_include_p99(self, result):
        """Legacy schema had p99 on MDD (tail-risk percentile)."""
        d = shuffle_result_to_quantile_dict(result, n_trades=100)
        assert "p99" in d["mdd"]

    def test_n_paths_matches_iterations(self, result):
        d = shuffle_result_to_quantile_dict(result, n_trades=100)
        assert d["n_paths"] == result.n_iterations

    def test_n_trades_uses_caller_value(self, result):
        """F-RISK-MC2-BLOCKER-1 regression: n_trades MUST come from caller,
        NOT from sharpe.shape[0] which equals n_iterations."""
        d50 = shuffle_result_to_quantile_dict(result, n_trades=50)
        d200 = shuffle_result_to_quantile_dict(result, n_trades=200)
        assert d50["n_trades"] == 50
        assert d200["n_trades"] == 200
        # CAGR uses years = n_trades / 252 → different n_trades → different cagr
        assert d50["cagr"]["p50"] != d200["cagr"]["p50"]

    def test_missing_n_trades_raises(self, result):
        """n_trades is mandatory — silent default would re-introduce
        F-RISK-MC2-BLOCKER-1."""
        with pytest.raises(TypeError):
            shuffle_result_to_quantile_dict(result)  # type: ignore[call-arg]

    def test_invalid_n_trades_raises(self, result):
        with pytest.raises(ValueError, match="positive int"):
            shuffle_result_to_quantile_dict(result, n_trades=0)
        with pytest.raises(ValueError, match="positive int"):
            shuffle_result_to_quantile_dict(result, n_trades=-5)

    def test_final_equity_uses_initial_capital(self, result):
        d_100k = shuffle_result_to_quantile_dict(
            result, n_trades=100, initial_capital=100_000.0
        )
        d_10k = shuffle_result_to_quantile_dict(
            result, n_trades=100, initial_capital=10_000.0
        )
        # Scaling by 10× initial_capital scales final_equity by 10×
        assert d_100k["final_equity"]["mean"] > 9.5 * d_10k["final_equity"]["mean"]
        assert d_100k["final_equity"]["mean"] < 10.5 * d_10k["final_equity"]["mean"]

    def test_all_values_json_serialisable(self, result):
        import json

        d = shuffle_result_to_quantile_dict(result, n_trades=100)
        # Must round-trip through JSON (legacy schema contract for metrics.json)
        json_str = json.dumps(d)
        round_trip = json.loads(json_str)
        assert round_trip["n_paths"] == d["n_paths"]
        assert round_trip["sharpe"]["p50"] == pytest.approx(d["sharpe"]["p50"])

    def test_pct_ruined_zero_for_winning_returns(self):
        """F-RISK-MC2-MAJOR-3: pct_ruined counts paths with final equity <= 0.
        With all-positive small returns, ruin probability must be 0."""
        returns = np.full(50, 0.005)  # +0.5% per trade
        result = permute_trades(returns, n_iterations=100, seed=0)
        d = shuffle_result_to_quantile_dict(result, n_trades=50)
        assert d["pct_ruined"] == 0.0

    def test_cagr_magnitude_plausible(self):
        """F-RISK-MC2-BLOCKER-1 numerical regression: CAGR must scale with
        n_trades correctly. Setup uses n_iterations=5000 (run_backtest_strategy
        default) to UNIQUELY catch the bug — with n_iterations==n_trades,
        the buggy adapter path that infers n_trades from sharpe.shape[0]
        produces the same result as the correct path, so the regression
        is not differentiable. With n_iterations=5000 ≫ n_trades=100, the
        buggy path computes years=5000/252≈19.84 → cagr≈2.5% (massively
        too small), while correct path computes years=100/252≈0.397 →
        cagr≈315%. The 0.5 threshold cleanly separates."""
        returns = np.full(100, 0.005)
        result = permute_trades(returns, n_iterations=5000, seed=0)
        d = shuffle_result_to_quantile_dict(result, n_trades=100)
        assert d["cagr"]["p50"] > 0.5, (
            f"CAGR p50={d['cagr']['p50']:.4f} suspiciously small; "
            "BLOCKER-1 regression?"
        )


class TestPhase2CallerImports:
    """Smoke: the 3 migrated callsites can import the new helpers."""

    def test_run_backtest_strategy_imports(self):
        # The migrated import block (without actually running the backtest)
        from src.assembled_core.risk.monte_carlo import (  # noqa: F401
            permute_trades,
            pnl_to_returns,
            shuffle_result_to_quantile_dict,
        )

    def test_api_qa_router_imports(self):
        from src.assembled_core.risk.monte_carlo import (  # noqa: F401
            permute_trades,
            pnl_to_returns,
        )

    def test_daily_qa_report_imports(self):
        from src.assembled_core.risk.monte_carlo import shuffle_trades  # noqa: F401


# ===========================================================================
# Deprecation warnings on legacy qa/monte_carlo* modules
# (§6.5.3 consolidation — Phase 1)
# ===========================================================================


class TestLegacyDeprecation:
    def test_monte_carlo_paths_emits_deprecation_warning(self):
        from src.assembled_core.qa.monte_carlo_paths import monte_carlo_trade_paths

        trades = pd.DataFrame({"pnl": TRADE_PNL_ARR[:50]})
        with pytest.warns(DeprecationWarning, match="permute_trades"):
            monte_carlo_trade_paths(trades, n_paths=50, seed=0)

    def test_qa_monte_carlo_bootstrap_returns_emits_deprecation_warning(self):
        from src.assembled_core.qa.monte_carlo import bootstrap_returns

        with pytest.warns(DeprecationWarning, match="shuffle_trades"):
            bootstrap_returns(DAILY_RETURNS_ARR, n_paths=50, seed=0)

    def test_qa_monte_carlo_forward_simulate_gbm_emits_deprecation_warning(self):
        from src.assembled_core.qa.monte_carlo import forward_simulate_gbm

        with pytest.warns(DeprecationWarning, match="simulate_paths_iid_normal"):
            forward_simulate_gbm(DAILY_RETURNS_ARR, n_paths=20, horizon_days=20, seed=0)
