"""Tests for Diebold-Yilmaz Spillover Index — C4-079 closure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from src.assembled_core.qa.spillover_index import (
    SpilloverResult,
    compute_spillover_index,
    rolling_spillover_index,
)


def _synthetic_connected_returns(
    n: int = 500, n_vars: int = 3, coupling: float = 0.5, seed: int = 42
) -> pd.DataFrame:
    """Generate N variables with cross-coupling: x_i_t = coupling * x_0_{t-1} + noise."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n, n_vars))
    data[0, :] = rng.normal(0, 0.01, n_vars)
    for t in range(1, n):
        # Variable 0 is a "transmitter" driving others 1-step lagged
        data[t, 0] = 0.3 * data[t - 1, 0] + 0.01 * rng.standard_normal()
        for j in range(1, n_vars):
            data[t, j] = (
                coupling * data[t - 1, 0]
                + 0.2 * data[t - 1, j]
                + 0.01 * rng.standard_normal()
            )
    return pd.DataFrame(data, columns=[f"asset_{i}" for i in range(n_vars)])


def _independent_returns(n: int = 500, n_vars: int = 3, seed: int = 0) -> pd.DataFrame:
    """N independent return series."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 0.01, (n, n_vars))
    return pd.DataFrame(data, columns=[f"asset_{i}" for i in range(n_vars)])


# ---------------------------------------------------------------------------
# compute_spillover_index
# ---------------------------------------------------------------------------


def test_returns_spillover_result_with_all_fields():
    df = _synthetic_connected_returns(n=300)
    result = compute_spillover_index(df, lag=2, horizon=5)
    assert isinstance(result, SpilloverResult)
    assert 0.0 <= result.total_spillover_index_pct <= 100.0
    assert result.fevd_matrix.shape == (3, 3)
    assert len(result.to_others) == 3
    assert len(result.from_others) == 3
    assert len(result.net) == 3
    assert result.lag == 2
    assert result.horizon == 5
    assert result.n_obs == 300


def test_tsi_higher_for_connected_than_independent():
    """Synthetic transmission setup should produce higher TSI than independent."""
    connected = _synthetic_connected_returns(n=600, coupling=0.5, seed=1)
    independent = _independent_returns(n=600, seed=1)
    tsi_conn = compute_spillover_index(
        connected, lag=2, horizon=5
    ).total_spillover_index_pct
    tsi_indep = compute_spillover_index(
        independent, lag=2, horizon=5
    ).total_spillover_index_pct
    assert tsi_conn > tsi_indep, (
        f"Connected TSI={tsi_conn:.2f} should exceed independent TSI={tsi_indep:.2f}"
    )


def test_fevd_rows_sum_to_100():
    """After row-normalisation, each row of FEVD matrix sums to ~100%."""
    df = _synthetic_connected_returns(n=300)
    result = compute_spillover_index(df, lag=2, horizon=5)
    row_sums = result.fevd_matrix.sum(axis=1).to_numpy()
    np.testing.assert_allclose(row_sums, 100.0, atol=0.01)


def test_net_balances_to_zero():
    """Sum of net spillovers across all variables = 0 (zero-sum: what one
    transmits, others receive)."""
    df = _synthetic_connected_returns(n=400, n_vars=4)
    result = compute_spillover_index(df, lag=2, horizon=5)
    assert abs(result.net.sum()) < 0.01


def test_transmitter_has_positive_net_spillover():
    """In the synthetic setup, asset_0 is the lagged-driver → should have
    positive net spillover."""
    df = _synthetic_connected_returns(n=600, coupling=0.6, seed=3)
    result = compute_spillover_index(df, lag=2, horizon=5)
    # The transmitter (asset_0) should send more than it receives
    assert result.net["asset_0"] > 0, (
        f"Expected asset_0 to be a net transmitter, got net={result.net['asset_0']:.2f}"
    )


def test_rejects_single_variable():
    df = pd.DataFrame({"only_one": np.random.randn(100)})
    with pytest.raises(ValueError, match="≥2 variables"):
        compute_spillover_index(df)


def test_rejects_short_series():
    df = _synthetic_connected_returns(n=20)
    with pytest.raises(ValueError, match="obs"):
        compute_spillover_index(df, lag=2, horizon=5)


def test_rejects_invalid_lag():
    df = _synthetic_connected_returns(n=200)
    with pytest.raises(ValueError, match="lag"):
        compute_spillover_index(df, lag=0)


def test_rejects_invalid_horizon():
    df = _synthetic_connected_returns(n=200)
    with pytest.raises(ValueError, match="horizon"):
        compute_spillover_index(df, lag=2, horizon=0)


# ---------------------------------------------------------------------------
# rolling_spillover_index
# ---------------------------------------------------------------------------


def test_rolling_tsi_returns_dataframe():
    df = _synthetic_connected_returns(n=400)
    rolling = rolling_spillover_index(df, window=200, step=50, lag=2, horizon=5)
    assert isinstance(rolling, pd.DataFrame)
    assert "tsi_pct" in rolling.columns
    assert "end_timestamp" in rolling.columns
    assert len(rolling) > 0
    assert (rolling["tsi_pct"] >= 0).all()
    assert (rolling["tsi_pct"] <= 100).all()


def test_rolling_tsi_step_controls_count():
    df = _synthetic_connected_returns(n=400)
    rolling_fast = rolling_spillover_index(df, window=200, step=10, lag=2, horizon=5)
    rolling_slow = rolling_spillover_index(df, window=200, step=50, lag=2, horizon=5)
    assert len(rolling_fast) > len(rolling_slow)


def test_rolling_rejects_too_small_window():
    df = _synthetic_connected_returns(n=300)
    with pytest.raises(ValueError, match="window"):
        rolling_spillover_index(df, window=20, step=5, lag=2, horizon=5)


def test_rolling_uses_datetime_index_if_present():
    """If returns has a DatetimeIndex, end_timestamp must use it."""
    df = _synthetic_connected_returns(n=300)
    df.index = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    rolling = rolling_spillover_index(df, window=200, step=50, lag=2, horizon=5)
    assert len(rolling) > 0
    assert pd.api.types.is_datetime64_any_dtype(rolling["end_timestamp"])
