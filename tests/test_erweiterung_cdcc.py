"""Aielli 2013 cDCC-GARCH variant — audit C4-072."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_correlated_returns(T: int, N: int, rho: float = 0.4, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Generate N correlated Gaussian series via Cholesky on the constant-rho matrix.
    R = np.full((N, N), rho)
    np.fill_diagonal(R, 1.0)
    L = np.linalg.cholesky(R)
    X = rng.normal(size=(T, N)) @ L.T
    # Scale to realistic daily-return magnitudes.
    X *= 0.01
    return pd.DataFrame(X, columns=[f"a{i}" for i in range(N)])


def test_cdcc_returns_valid_fit() -> None:
    from src.erweiterung.volatility.cdcc_garch import cDCCFit, fit_cdcc_garch

    df = _make_correlated_returns(200, 3, rho=0.3, seed=1)
    fit = fit_cdcc_garch(df)
    assert isinstance(fit, cDCCFit)
    assert fit.R_path.shape == (200, 3, 3)
    assert fit.sigma_path.shape == (200, 3)
    assert 0.0 <= fit.alpha <= 1.0
    assert 0.0 <= fit.beta <= 1.0
    assert fit.alpha + fit.beta < 1.0


def test_cdcc_correlation_matrices_are_psd() -> None:
    from src.erweiterung.volatility.cdcc_garch import fit_cdcc_garch

    df = _make_correlated_returns(150, 3, rho=0.5, seed=2)
    fit = fit_cdcc_garch(df)
    # Spot-check 10 time-points have PSD R_t (eigenvalues >= 0).
    for t in np.linspace(0, fit.R_path.shape[0] - 1, 10).astype(int):
        eigs = np.linalg.eigvalsh(fit.R_path[t])
        assert (eigs >= -1e-9).all()


def test_cdcc_diagonal_of_correlation_is_one() -> None:
    from src.erweiterung.volatility.cdcc_garch import fit_cdcc_garch

    df = _make_correlated_returns(120, 4, rho=0.2, seed=3)
    fit = fit_cdcc_garch(df)
    for t in range(0, fit.R_path.shape[0], 30):
        diag = np.diag(fit.R_path[t])
        assert np.allclose(diag, 1.0, atol=1e-6)


def test_cdcc_rejects_short_series() -> None:
    from src.erweiterung.volatility.cdcc_garch import fit_cdcc_garch

    df = _make_correlated_returns(20, 2)
    with pytest.raises(ValueError):
        fit_cdcc_garch(df)


def test_cdcc_covariance_at_combines_vol_and_correlation() -> None:
    from src.erweiterung.volatility.cdcc_garch import (
        cdcc_covariance_at,
        fit_cdcc_garch,
    )

    df = _make_correlated_returns(150, 3, rho=0.3, seed=4)
    fit = fit_cdcc_garch(df)
    Sigma = cdcc_covariance_at(fit, t=100)
    assert Sigma.shape == (3, 3)
    # Diagonal of Σ equals σ_i² for each asset.
    expected_var = fit.sigma_path[100] ** 2
    assert np.allclose(np.diag(Sigma), expected_var, atol=1e-9)


def test_cdcc_diverges_from_dcc_under_strong_clustering() -> None:
    """cDCC and DCC should give materially different long-run Q̄ when the
    process has strong volatility clustering — that is the whole point of
    Aielli's correction.
    """
    from src.erweiterung.volatility.cdcc_garch import fit_cdcc_garch
    from src.erweiterung.volatility.dcc_garch import fit_dcc_garch

    df = _make_correlated_returns(200, 3, rho=0.6, seed=5)
    fit_d = fit_dcc_garch(df)
    fit_c = fit_cdcc_garch(df)
    # Both targets should be positive-definite correlation-shaped, but not
    # numerically identical.
    assert not np.allclose(fit_d.Q_bar, fit_c.Q_bar, atol=1e-3)
