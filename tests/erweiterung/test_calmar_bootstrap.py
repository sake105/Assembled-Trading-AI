"""Tests für calmar_bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.backtest.calmar_bootstrap import calmar_diff_bootstrap


def _ret(
    n: int = 500, mean: float = 0.0005, vol: float = 0.01, seed: int = 0
) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(mean, vol, n),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def test_calmar_bootstrap_basic():
    a = _ret(seed=1, mean=0.0008, vol=0.008)  # higher AnnRet, lower Vol
    b = _ret(seed=2, mean=0.0003, vol=0.012)
    out = calmar_diff_bootstrap(a, b, n_bootstrap=200, avg_block_size=10)
    assert "mean_diff" in out
    assert "ci_low_2.5" in out
    assert "p_value_one_sided_greater" in out


def test_calmar_bootstrap_a_better_low_p():
    rng = np.random.default_rng(42)
    n = 1000
    # a hat höheren AnnRet und niedrigeren MDD
    a = pd.Series(
        rng.normal(0.0010, 0.008, n),
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )
    b = pd.Series(rng.normal(0.0003, 0.012, n), index=a.index)
    out = calmar_diff_bootstrap(a, b, n_bootstrap=500, avg_block_size=15)
    # a deutlich besser → mean_diff sollte positiv sein
    assert out["mean_diff"] > 0


def test_calmar_bootstrap_handles_empty():
    a = pd.Series(dtype=float)
    b = pd.Series(dtype=float)
    out = calmar_diff_bootstrap(a, b)
    assert "error" in out


def test_calmar_bootstrap_ci_contains_mean():
    a = _ret(seed=3, mean=0.0005, vol=0.01, n=600)
    b = _ret(seed=4, mean=0.0004, vol=0.01, n=600)
    out = calmar_diff_bootstrap(a, b, n_bootstrap=300)
    assert out["ci_low_2.5"] <= out["mean_diff"] <= out["ci_high_97.5"]
