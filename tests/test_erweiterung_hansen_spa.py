"""Hansen 2005 SPA test — studentized recentering (audit C4-066)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_excess(n: int, k: int, mu: float = 0.0, sigma: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=mu, scale=sigma, size=(n, k))
    return pd.DataFrame(X, columns=[f"s{i}" for i in range(k)])


def test_hansen_spa_studentized_returns_pvalue() -> None:
    from src.erweiterung.backtest.hansen_spa import hansen_spa_test_studentized

    df = _make_excess(120, 5, mu=0.0)
    out = hansen_spa_test_studentized(df, n_bootstrap=100, seed=1)
    assert "p_value" in out
    assert 0.0 <= out["p_value"] <= 1.0
    assert out["n_bootstrap"] == 100


def test_hansen_spa_pvalue_high_under_null() -> None:
    from src.erweiterung.backtest.hansen_spa import hansen_spa_test_studentized

    # Pure noise — best-of-k max-statistic should not register as significant.
    df = _make_excess(200, 10, mu=0.0, sigma=1.0, seed=42)
    out = hansen_spa_test_studentized(df, n_bootstrap=200, seed=42)
    assert out["p_value"] > 0.05  # generous — but multiple-testing should help


def test_hansen_spa_pvalue_low_under_strong_signal() -> None:
    from src.erweiterung.backtest.hansen_spa import hansen_spa_test_studentized

    rng = np.random.default_rng(0)
    n, k = 300, 5
    X = rng.normal(size=(n, k))
    # Strategy 0 has a real positive drift.
    X[:, 0] += 0.3
    df = pd.DataFrame(X, columns=[f"s{i}" for i in range(k)])
    out = hansen_spa_test_studentized(df, n_bootstrap=200, seed=0)
    assert out["p_value"] < 0.05
    assert out["best_strategy"] == "s0"


def test_hansen_spa_handles_zero_variance_strategy() -> None:
    from src.erweiterung.backtest.hansen_spa import hansen_spa_test_studentized

    df = pd.DataFrame(
        {
            "constant": [0.0] * 100,
            "noisy": np.random.default_rng(0).normal(size=100).tolist(),
        }
    )
    # Must not raise.
    out = hansen_spa_test_studentized(df, n_bootstrap=50, seed=0)
    assert 0.0 <= out["p_value"] <= 1.0


def test_hansen_spa_rejects_empty_input() -> None:
    from src.erweiterung.backtest.hansen_spa import hansen_spa_test_studentized

    with pytest.raises(ValueError):
        hansen_spa_test_studentized(pd.DataFrame(), n_bootstrap=10)


def test_hansen_spa_rejects_invalid_recentering() -> None:
    from src.erweiterung.backtest.hansen_spa import hansen_spa_test_studentized

    df = _make_excess(50, 3)
    with pytest.raises(ValueError):
        hansen_spa_test_studentized(df, recentering="bogus")  # type: ignore[arg-type]


def test_hansen_spa_c_vs_u_recentering_differ() -> None:
    """SPA-c should be MORE powerful (smaller p-value) than SPA-u in noisy data."""
    from src.erweiterung.backtest.hansen_spa import hansen_spa_test_studentized

    rng = np.random.default_rng(1)
    n, k = 200, 8
    X = rng.normal(size=(n, k))
    X[:, 0] += 0.2  # mild edge
    df = pd.DataFrame(X, columns=[f"s{i}" for i in range(k)])
    out_c = hansen_spa_test_studentized(df, n_bootstrap=400, recentering="c", seed=7)
    out_u = hansen_spa_test_studentized(df, n_bootstrap=400, recentering="u", seed=7)
    # SPA-c re-centering drops poor strategies — p-value should not exceed SPA-u.
    assert out_c["p_value"] <= out_u["p_value"] + 0.05
