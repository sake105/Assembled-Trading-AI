"""Tests for C8 empirical tail-dependence sidecar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assembled_core.risk.tail_dependence import (
    classify_tail_regime,
    compute_empirical_tail_dependence,
    compute_portfolio_tail_dependence_score,
)


@pytest.mark.phase12
def test_independent_normal_returns_yield_low_score() -> None:
    rng = np.random.default_rng(11)
    n_days = 300
    symbols = [f"S{i}" for i in range(5)]
    data = rng.standard_normal((n_days, len(symbols)))
    df = pd.DataFrame(data, columns=symbols)

    tail_dep = compute_empirical_tail_dependence(df, alpha=0.05)
    score = compute_portfolio_tail_dependence_score(tail_dep)
    assert 0.0 <= score < 0.15, f"expected low tail-dep, got {score}"


@pytest.mark.phase12
def test_perfectly_correlated_returns_yield_high_score() -> None:
    rng = np.random.default_rng(11)
    n_days = 300
    base = rng.standard_normal(n_days)
    symbols = [f"S{i}" for i in range(5)]
    # All series = base + microscopic jitter (ranks stay tied to base).
    data = np.column_stack(
        [base + rng.standard_normal(n_days) * 1e-9 for _ in symbols]
    )
    df = pd.DataFrame(data, columns=symbols)

    tail_dep = compute_empirical_tail_dependence(df, alpha=0.05)
    score = compute_portfolio_tail_dependence_score(tail_dep)
    assert score > 0.4, f"expected high tail-dep, got {score}"


@pytest.mark.phase12
def test_diagonal_is_one() -> None:
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        rng.standard_normal((200, 4)),
        columns=["A", "B", "C", "D"],
    )
    tail_dep = compute_empirical_tail_dependence(df, alpha=0.1)
    diag = np.diag(tail_dep.to_numpy())
    assert np.allclose(diag, 1.0)


@pytest.mark.phase12
def test_matrix_is_symmetric() -> None:
    # Symmetric conditional probabilities require equal marginal tail
    # counts. Use an even number of observations and an alpha that lands
    # exactly on a rank boundary so every column has the same denominator.
    rng = np.random.default_rng(11)
    n_days = 200
    df = pd.DataFrame(
        rng.standard_normal((n_days, 4)),
        columns=["A", "B", "C", "D"],
    )
    tail_dep = compute_empirical_tail_dependence(df, alpha=0.1)
    arr = tail_dep.to_numpy()
    assert np.allclose(arr, arr.T, atol=1e-10)


@pytest.mark.phase12
def test_classify_tail_regime_thresholds() -> None:
    assert classify_tail_regime(0.05) == "low"
    assert classify_tail_regime(0.20) == "medium"
    assert classify_tail_regime(0.40) == "high"
    # Boundary behaviour
    assert classify_tail_regime(0.15) == "medium"
    assert classify_tail_regime(0.35) == "high"


@pytest.mark.phase12
def test_too_short_history_raises() -> None:
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        rng.standard_normal((20, 3)),
        columns=["A", "B", "C"],
    )
    with pytest.raises(ValueError, match="30 rows"):
        compute_empirical_tail_dependence(df, alpha=0.05)


@pytest.mark.phase12
def test_single_symbol_raises() -> None:
    rng = np.random.default_rng(11)
    df = pd.DataFrame(rng.standard_normal((100, 1)), columns=["A"])
    with pytest.raises(ValueError, match="at least 2 symbols"):
        compute_empirical_tail_dependence(df, alpha=0.05)


@pytest.mark.phase12
@pytest.mark.parametrize("alpha", [0.0, 0.5, -0.1, 0.6])
def test_invalid_alpha_raises(alpha: float) -> None:
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        rng.standard_normal((100, 3)),
        columns=["A", "B", "C"],
    )
    with pytest.raises(ValueError, match="alpha"):
        compute_empirical_tail_dependence(df, alpha=alpha)


@pytest.mark.phase12
def test_identical_bivariate_series_give_unit_tail_dependence() -> None:
    # Two identical return series must satisfy lambda_L = 1.0 exactly:
    # every joint tail event is, by construction, a marginal tail event.
    rng = np.random.default_rng(11)
    base = rng.standard_normal(200)
    df = pd.DataFrame({"A": base, "B": base.copy()})

    tail_dep = compute_empirical_tail_dependence(df, alpha=0.1)
    assert tail_dep.loc["A", "B"] == pytest.approx(1.0)
    assert tail_dep.loc["B", "A"] == pytest.approx(1.0)
    score = compute_portfolio_tail_dependence_score(tail_dep)
    assert score == pytest.approx(1.0)
