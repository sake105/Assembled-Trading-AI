"""Tests for portfolio/optimizers.py — KNOWN_ISSUES §6.5.1 closure.

Covers:
- Min-Variance (closed-form + constrained)
- Max-Sharpe / Tangency (closed-form + constrained)
- Efficient frontier tracing
- Equal Risk Contribution (Maillard 2010)
- Multivariate Fractional Kelly (Thorp 2006)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from src.assembled_core.portfolio.optimizers import (
    OptimizerResult,
    equal_risk_contribution_weights,
    max_sharpe_weights,
    mean_variance_efficient_frontier,
    min_variance_weights,
    multivariate_kelly_weights,
)


def _toy_covariance(n: int = 3, seed: int = 42) -> pd.DataFrame:
    """Generate a small synthetic PSD covariance matrix."""
    rng = np.random.default_rng(seed)
    L = rng.normal(0, 0.1, size=(n, n))
    sigma = L @ L.T + 0.0001 * np.eye(n)
    names = [f"asset_{i}" for i in range(n)]
    return pd.DataFrame(sigma, index=names, columns=names)


def _toy_expected_returns(n: int = 3, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    names = [f"asset_{i}" for i in range(n)]
    return pd.Series(rng.uniform(0.02, 0.15, n), index=names)


# ---------------------------------------------------------------------------
# min_variance_weights
# ---------------------------------------------------------------------------


def test_min_variance_returns_optimizer_result():
    cov = _toy_covariance(n=3)
    result = min_variance_weights(cov, long_only=False)
    assert isinstance(result, OptimizerResult)
    assert len(result.weights) == 3
    assert abs(result.weights.sum() - 1.0) < 1e-6
    assert result.expected_volatility > 0
    assert result.converged is True
    assert result.method == "min_variance_closed_form"


def test_min_variance_long_only_no_negatives():
    cov = _toy_covariance(n=4, seed=7)
    result = min_variance_weights(cov, long_only=True)
    assert (result.weights >= -1e-8).all()
    assert abs(result.weights.sum() - 1.0) < 1e-6


def test_min_variance_minimises_variance():
    """Compare min-var result against random portfolios — must have ≤ σ."""
    cov = _toy_covariance(n=5, seed=10)
    mv_result = min_variance_weights(cov, long_only=True)
    mv_sigma = mv_result.expected_volatility

    rng = np.random.default_rng(42)
    for _ in range(20):
        w_random = rng.uniform(0, 1, 5)
        w_random = w_random / w_random.sum()
        sigma_random = float(np.sqrt(w_random @ cov.to_numpy() @ w_random))
        assert mv_sigma <= sigma_random + 1e-6, (
            f"min_var σ={mv_sigma:.4f} should be ≤ random σ={sigma_random:.4f}"
        )


def test_min_variance_rejects_asymmetric_covariance():
    asym = pd.DataFrame([[1.0, 0.5], [0.3, 1.0]], index=["a", "b"], columns=["a", "b"])
    with pytest.raises(ValueError, match="not symmetric"):
        min_variance_weights(asym)


def test_min_variance_rejects_single_asset():
    cov = pd.DataFrame([[1.0]], index=["a"], columns=["a"])
    with pytest.raises(ValueError, match="≥2 assets"):
        min_variance_weights(cov)


# ---------------------------------------------------------------------------
# max_sharpe_weights
# ---------------------------------------------------------------------------


def test_max_sharpe_returns_optimizer_result():
    cov = _toy_covariance(n=3)
    mu = _toy_expected_returns(n=3)
    result = max_sharpe_weights(mu, cov, long_only=False)
    assert isinstance(result, OptimizerResult)
    assert abs(result.weights.sum() - 1.0) < 1e-6
    assert result.sharpe_ratio > 0
    assert result.converged is True


def test_max_sharpe_higher_sharpe_than_equal_weight():
    """Tangency portfolio should have ≥ Sharpe than equal-weight."""
    cov = _toy_covariance(n=4, seed=15)
    mu = _toy_expected_returns(n=4, seed=15)
    result = max_sharpe_weights(mu, cov, long_only=True)

    eq_w = np.full(4, 0.25)
    eq_sigma = float(np.sqrt(eq_w @ cov.to_numpy() @ eq_w))
    eq_mu = float(eq_w @ mu.to_numpy())
    eq_sharpe = eq_mu / eq_sigma

    assert result.sharpe_ratio >= eq_sharpe - 1e-6, (
        f"max_sharpe={result.sharpe_ratio:.3f} should be ≥ equal-weight Sharpe={eq_sharpe:.3f}"
    )


def test_max_sharpe_rejects_index_mismatch():
    cov = _toy_covariance(n=3)
    mu = pd.Series([0.05, 0.07], index=["wrong_x", "wrong_y"])
    with pytest.raises(ValueError, match="must match"):
        max_sharpe_weights(mu, cov)


def test_max_sharpe_long_only_no_negatives():
    cov = _toy_covariance(n=4)
    mu = _toy_expected_returns(n=4, seed=11)
    result = max_sharpe_weights(mu, cov, long_only=True)
    assert (result.weights >= -1e-8).all()


# ---------------------------------------------------------------------------
# mean_variance_efficient_frontier
# ---------------------------------------------------------------------------


def test_frontier_returns_n_points():
    cov = _toy_covariance(n=3)
    mu = _toy_expected_returns(n=3)
    frontier = mean_variance_efficient_frontier(mu, cov, n_points=10)
    assert isinstance(frontier, pd.DataFrame)
    # F-stage1-portopt-1: exactly n_points rows (non-converged kept as NaN)
    assert len(frontier) == 10
    assert set(frontier.columns) >= {
        "target_return",
        "volatility",
        "sharpe",
        "weights",
        "converged",
    }


def test_frontier_volatility_monotonic_around_min_var():
    """At returns away from the min-var return, volatility increases."""
    cov = _toy_covariance(n=4, seed=20)
    mu = _toy_expected_returns(n=4, seed=20)
    frontier = mean_variance_efficient_frontier(mu, cov, n_points=20)
    # F-stage1-portopt-1: filter to converged rows before monotonicity check
    converged = frontier[frontier["converged"]]
    # Find min-vol point
    idx_min = converged["volatility"].idxmin()
    min_ret = converged.loc[idx_min, "target_return"]
    # Points above min_ret should have non-decreasing volatility (efficient
    # frontier upper branch). Allow small numerical noise.
    above = converged[converged["target_return"] > min_ret].sort_values("target_return")
    if len(above) >= 3:
        vols = above["volatility"].to_numpy()
        # Check monotonic up to numerical noise
        diffs = np.diff(vols)
        # Allow small backward steps (1% of mean vol)
        tolerance = 0.01 * vols.mean()
        assert (diffs >= -tolerance).all(), (
            "Efficient frontier upper branch should be non-decreasing in σ"
        )


# ---------------------------------------------------------------------------
# equal_risk_contribution_weights
# ---------------------------------------------------------------------------


def test_erc_returns_optimizer_result():
    cov = _toy_covariance(n=3)
    result = equal_risk_contribution_weights(cov)
    assert isinstance(result, OptimizerResult)
    assert abs(result.weights.sum() - 1.0) < 1e-6
    assert (result.weights > 0).all()
    assert result.converged is True


def test_erc_equal_risk_contributions():
    """Core property: each asset contributes equal risk after convergence."""
    cov = _toy_covariance(n=4, seed=25)
    result = equal_risk_contribution_weights(cov, tol=1e-10)
    w = result.weights.to_numpy()
    cov_arr = cov.to_numpy()
    # Per-asset risk contribution: RC_i = w_i * (Σw)_i
    contributions = w * (cov_arr @ w)
    # All contributions should be approximately equal
    mean_contrib = contributions.mean()
    max_dev = float(np.max(np.abs(contributions - mean_contrib))) / abs(mean_contrib)
    assert max_dev < 0.05, (
        f"ERC: max deviation from mean contribution = {max_dev:.4f} (should be <5%)"
    )


def test_erc_differs_from_inverse_vol():
    """ERC accounts for covariance — should differ from naive 1/σ weighting
    when off-diagonal correlations are STRONG and asymmetric.

    Construct Σ explicitly with strongly different correlations so the
    distinction is visible. (Random toy covariances at small N can
    accidentally have near-zero correlations.)
    """
    # 3 assets: asset_0 and asset_1 strongly correlated, asset_2 independent
    sigma = pd.DataFrame(
        [
            [0.04, 0.035, 0.0],  # σ_0=0.2, ρ_01=0.875
            [0.035, 0.04, 0.0],  # σ_1=0.2
            [0.0, 0.0, 0.04],  # σ_2=0.2, independent
        ],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    result_erc = equal_risk_contribution_weights(sigma)
    # All assets have equal vol (0.2), so inverse-vol = 1/3 each
    w_inv_vol = np.full(3, 1.0 / 3.0)
    # ERC should down-weight the correlated pair (a, b) and up-weight c
    # because c is the diversifier
    max_diff = float(np.max(np.abs(result_erc.weights.to_numpy() - w_inv_vol)))
    assert max_diff > 0.05, (
        f"ERC should differ from inverse-vol when correlations matter; "
        f"got max diff {max_diff:.4f}, weights={result_erc.weights.tolist()}"
    )
    # Asset c (the diversifier) should get the largest weight
    assert result_erc.weights["c"] > result_erc.weights["a"]
    assert result_erc.weights["c"] > result_erc.weights["b"]


# ---------------------------------------------------------------------------
# multivariate_kelly_weights
# ---------------------------------------------------------------------------


def test_kelly_returns_optimizer_result():
    cov = _toy_covariance(n=3)
    excess = _toy_expected_returns(n=3) - 0.02  # risk-free 2%
    result = multivariate_kelly_weights(excess, cov, kelly_fraction=0.5)
    assert isinstance(result, OptimizerResult)
    assert result.method == "multivariate_kelly_k0.50"


def test_half_kelly_is_half_of_full_kelly():
    """Fractional Kelly with k=0.5 should give exactly half the weights of
    full Kelly — BEFORE the leverage cap activates. Pre-commit local test
    caught that with max_leverage at the edge, the cap warps the relationship
    (full gets capped, half doesn't, so they become identical). Use a very
    high cap to take it out of the picture."""
    cov = _toy_covariance(n=3)
    excess = _toy_expected_returns(n=3) - 0.02
    full = multivariate_kelly_weights(excess, cov, kelly_fraction=1.0, max_leverage=1e6)
    half = multivariate_kelly_weights(excess, cov, kelly_fraction=0.5, max_leverage=1e6)
    np.testing.assert_allclose(
        half.weights.to_numpy(), 0.5 * full.weights.to_numpy(), rtol=1e-9
    )


def test_kelly_leverage_cap_enforced():
    """sum(|w|) should be ≤ max_leverage after the cap."""
    cov = _toy_covariance(n=3)
    excess = pd.Series([0.10, 0.10, 0.10], index=cov.columns)  # large excess returns
    result = multivariate_kelly_weights(
        excess, cov, kelly_fraction=1.0, max_leverage=1.0
    )
    abs_sum = float(np.sum(np.abs(result.weights.to_numpy())))
    assert abs_sum <= 1.0 + 1e-6, f"Leverage cap violated: |w|={abs_sum:.4f}"


def test_kelly_rejects_invalid_fraction():
    cov = _toy_covariance(n=3)
    excess = _toy_expected_returns(n=3)
    with pytest.raises(ValueError, match="kelly_fraction"):
        multivariate_kelly_weights(excess, cov, kelly_fraction=0.0)
    with pytest.raises(ValueError, match="kelly_fraction"):
        multivariate_kelly_weights(excess, cov, kelly_fraction=1.5)


def test_kelly_long_only_clips_negatives():
    cov = _toy_covariance(n=3)
    # Construct excess returns where one asset has negative expected excess
    excess = pd.Series([-0.05, 0.10, 0.08], index=cov.columns)
    result = multivariate_kelly_weights(
        excess, cov, kelly_fraction=0.5, max_leverage=10.0, long_only=True
    )
    assert (result.weights >= -1e-9).all()


# ---------------------------------------------------------------------------
# F-stage1-portopt-1: frontier non-convergence rows kept with converged=False
# ---------------------------------------------------------------------------


def test_frontier_returns_exactly_n_points_with_converged_column():
    """All n_points rows must be present (non-converged kept with NaN)."""
    cov = _toy_covariance(n=3)
    mu = _toy_expected_returns(n=3)
    frontier = mean_variance_efficient_frontier(mu, cov, n_points=12)
    assert len(frontier) == 12, (
        f"Expected exactly 12 rows (one per target_return), got {len(frontier)}"
    )
    assert "converged" in frontier.columns
    assert frontier["converged"].dtype == bool


# ---------------------------------------------------------------------------
# F-stage1-portopt-2: Kelly long_only + leverage cap semantics
# ---------------------------------------------------------------------------


def test_kelly_long_only_sum_below_one_without_renormalize():
    """Kelly long-only with cap=1.0 + renormalize_to_unity=False (default)
    leaves implicit cash when natural Kelly direction sums below 1."""
    cov = _toy_covariance(n=3)
    # Small excess returns → small natural Kelly weights → cap doesn't bind
    excess = pd.Series([0.005, 0.004, 0.003], index=cov.columns)
    result = multivariate_kelly_weights(
        excess, cov, kelly_fraction=0.5, max_leverage=1.0, long_only=True
    )
    # Natural Kelly may sum well below 1; cap doesn't force full-invest.
    total = float(result.weights.sum())
    assert 0.0 < total <= 1.0 + 1e-9
    # Important: documented contract — caller sees raw sum, not forced 1.0


def test_kelly_long_only_preserves_clip_indices():
    """F-senior-portopt-2 + F-postcommit-3: under long_only=True with no
    cap binding, the clip zeros the SAME indices at full-Kelly and at
    half-Kelly (negative direction is direction-invariant under positive
    scaling). The non-zero ratios are exactly 0.5 — this locks the
    documented "clip preserves direction" property."""
    cov = _toy_covariance(n=3)
    # At least one asset gets a negative Kelly weight → clip is non-trivial
    excess = pd.Series([-0.08, 0.05, 0.04], index=cov.columns)
    full = multivariate_kelly_weights(
        excess, cov, kelly_fraction=1.0, max_leverage=1e6, long_only=True
    )
    half = multivariate_kelly_weights(
        excess, cov, kelly_fraction=0.5, max_leverage=1e6, long_only=True
    )
    full_arr = full.weights.to_numpy()
    half_arr = half.weights.to_numpy()
    # Both clip the same indices (the negative-direction positions)
    assert (full_arr == 0.0).sum() >= 1
    np.testing.assert_array_equal(full_arr == 0.0, half_arr == 0.0)
    nz = full_arr > 1e-10
    if nz.any():
        np.testing.assert_allclose(half_arr[nz] / full_arr[nz], 0.5, rtol=1e-9)


def test_kelly_long_only_cap_binds_asymmetrically():
    """F-postcommit-3: when the leverage cap binds for full-Kelly but NOT
    for half-Kelly, the scaling invariance is genuinely broken — half is
    NOT 0.5 * full because the cap intervenes only on full.

    Σ = 0.04·I, excess=[0.10, 0.08, 0.06] → w_full = [2.5, 2.0, 1.5],
    sum=6.0; w_half = [1.25, 1.0, 0.75], sum=3.0. Cap=4.0:
      - full > 4 → rescaled to sum=4
      - half < 4 → unchanged at sum=3
    """
    cov = pd.DataFrame(np.eye(3) * 0.04, index=["a", "b", "c"], columns=["a", "b", "c"])
    excess = pd.Series([0.10, 0.08, 0.06], index=["a", "b", "c"])
    full = multivariate_kelly_weights(
        excess, cov, kelly_fraction=1.0, max_leverage=4.0, long_only=True
    )
    half = multivariate_kelly_weights(
        excess, cov, kelly_fraction=0.5, max_leverage=4.0, long_only=True
    )
    full_arr = full.weights.to_numpy()
    half_arr = half.weights.to_numpy()
    full_sum = float(np.sum(np.abs(full_arr)))
    half_sum = float(np.sum(np.abs(half_arr)))
    # Full Kelly capped at 4.0 exactly; half Kelly below cap (~3.0)
    assert abs(full_sum - 4.0) < 1e-9, f"full sum={full_sum}"
    assert half_sum < 4.0 - 1e-6, f"half sum={half_sum}"
    # Ratios diverge from 0.5 since cap re-scaled full but not half
    ratios = half_arr / full_arr
    assert not np.allclose(ratios, 0.5, rtol=1e-3), (
        f"Cap binds only on full-Kelly → ratios must diverge from 0.5; "
        f"got ratios={ratios}"
    )


def test_kelly_renormalize_to_unity_forces_full_invest():
    """When renormalize_to_unity=True, sum(w)=1 exactly."""
    cov = _toy_covariance(n=3)
    excess = pd.Series([0.005, 0.004, 0.003], index=cov.columns)
    result = multivariate_kelly_weights(
        excess,
        cov,
        kelly_fraction=0.5,
        max_leverage=1.0,
        long_only=True,
        renormalize_to_unity=True,
    )
    assert abs(float(result.weights.sum()) - 1.0) < 1e-9


def test_kelly_leverage_disabled_when_max_leverage_zero():
    """max_leverage=0 disables the cap (documented contract)."""
    cov = _toy_covariance(n=3)
    excess = pd.Series([0.10, 0.10, 0.10], index=cov.columns)  # large
    capped = multivariate_kelly_weights(
        excess, cov, kelly_fraction=1.0, max_leverage=1.0
    )
    uncapped = multivariate_kelly_weights(
        excess, cov, kelly_fraction=1.0, max_leverage=0.0
    )
    # Capped result should have sum(|w|) ≤ 1.0; uncapped clearly larger
    assert float(np.sum(np.abs(uncapped.weights.to_numpy()))) > float(
        np.sum(np.abs(capped.weights.to_numpy())) + 1e-6
    )


# ---------------------------------------------------------------------------
# F-stage1-portopt-3: max_sharpe closed-form sign fallback when denom < 0
# ---------------------------------------------------------------------------


def test_max_sharpe_unconstrained_falls_back_when_all_excess_negative():
    """All μ_i < r_f → 1' Σ⁻¹ (μ-r_f) typically < 0 → closed-form is on the
    inefficient branch. The function must fall back to SLSQP rather than
    silently invert the portfolio."""
    cov = _toy_covariance(n=3)
    # All assets have expected returns BELOW the risk-free rate
    mu = pd.Series([0.01, 0.015, 0.012], index=cov.columns)
    result = max_sharpe_weights(mu, cov, risk_free_rate=0.05, long_only=False)
    # Method should reflect the fallback path (not closed-form)
    assert result.method == "max_sharpe_slsqp", (
        f"Expected SLSQP fallback for negative-denom case, got {result.method}"
    )


# ---------------------------------------------------------------------------
# F-stage1-portopt-5: non-PSD covariance must raise
# ---------------------------------------------------------------------------


def test_kelly_renormalize_with_shorts_respects_leverage_cap():
    """F-postcommit-1: with renormalize_to_unity=True and shorts present,
    the cap must apply AFTER renormalize so sum(|w|) does not blow past
    max_leverage. Empirical pre-fix repro: 3x gross exposure observed."""
    cov = pd.DataFrame(np.eye(3) * 0.04, index=["a", "b", "c"], columns=["a", "b", "c"])
    excess = pd.Series([0.5, -0.4, 0.3], index=["a", "b", "c"])
    result = multivariate_kelly_weights(
        excess,
        cov,
        kelly_fraction=1.0,
        max_leverage=1.0,
        long_only=False,
        renormalize_to_unity=True,
    )
    gross = float(np.sum(np.abs(result.weights.to_numpy())))
    assert gross <= 1.0 + 1e-6, (
        f"Cap violated under renormalize+shorts: sum(|w|)={gross}"
    )


def test_kelly_nan_max_leverage_raises():
    """F-postcommit-2: NaN max_leverage previously bypassed the cap silently
    (any NaN comparison is False) → 30x gross observed in pre-fix repro.
    Must now raise ValueError."""
    cov = _toy_covariance(n=3)
    excess = pd.Series([0.10, 0.10, 0.10], index=cov.columns)
    with pytest.raises(ValueError, match="max_leverage must be finite"):
        multivariate_kelly_weights(
            excess, cov, kelly_fraction=1.0, max_leverage=float("nan")
        )
    with pytest.raises(ValueError, match="max_leverage must be finite"):
        multivariate_kelly_weights(
            excess, cov, kelly_fraction=1.0, max_leverage=float("inf")
        )


def test_validate_covariance_rejects_index_columns_mismatch():
    """F-postcommit-5: caller may build a DataFrame where index != columns
    (e.g. typo). Validator must reject — Risk-Zone primitives never accept
    silent mislabeling."""
    bad = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.04]],
        index=["x", "y"],
        columns=["a", "b"],
    )
    with pytest.raises(ValueError, match="index and covariance.columns"):
        min_variance_weights(bad)


def test_min_variance_rejects_non_psd_covariance():
    """Σ with a negative eigenvalue must be rejected — not silently produce
    σ_p = 0 via the max(w'Σw, 0) paper-over."""
    # Hand-crafted symmetric but indefinite matrix (one negative eigenvalue)
    bad = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["a", "b"], columns=["a", "b"])
    # eigvals = {3, -1} → one negative
    with pytest.raises(ValueError, match="not PSD"):
        min_variance_weights(bad)
