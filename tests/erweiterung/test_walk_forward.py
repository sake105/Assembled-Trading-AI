"""Tests für walk_forward."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.robustness.walk_forward import (
    WalkForwardConfig,
    concat_oos_returns,
    walk_forward_threshold_search,
)


def _make_returns(n: int = 1800, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    bench = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    fac = pd.Series(rng.normal(0.0005, 0.015, n), index=idx)
    # Konstruiere einen drawdown bei index 500..600
    bench.iloc[500:600] = -0.005
    return bench, fac


def test_walk_forward_basic():
    bench, fac = _make_returns(n=1800)
    df = walk_forward_threshold_search(
        bench,
        fac,
        threshold_grid=[0.05, 0.08, 0.12],
        config=WalkForwardConfig(train_days=500, test_days=200, step_days=200),
    )
    assert not df.empty
    assert "best_threshold" in df.columns
    assert "test_ann_return" in df.columns
    assert (df["test_n_days"] > 0).all()


def test_walk_forward_threshold_search_picks_grid_value():
    bench, fac = _make_returns(n=1800)
    grid = [0.05, 0.08, 0.12]
    df = walk_forward_threshold_search(
        bench,
        fac,
        threshold_grid=grid,
        config=WalkForwardConfig(train_days=500, test_days=200, step_days=200),
    )
    # Alle gewählten Thresholds müssen aus der Grid stammen
    assert set(df["best_threshold"]).issubset(set(grid))


def test_walk_forward_skips_when_insufficient_data():
    bench, fac = _make_returns(n=400)  # short series
    df = walk_forward_threshold_search(
        bench,
        fac,
        threshold_grid=[0.08],
        config=WalkForwardConfig(train_days=500, test_days=200),
    )
    assert df.empty


def test_concat_oos_returns_returns_series():
    bench, fac = _make_returns(n=1800)
    df = walk_forward_threshold_search(
        bench,
        fac,
        threshold_grid=[0.05, 0.10],
        config=WalkForwardConfig(train_days=500, test_days=200, step_days=200),
    )
    oos = concat_oos_returns(bench, fac, df)
    assert isinstance(oos, pd.Series)
    assert not oos.empty
    # OOS-Series sollte chronologisch sein
    assert oos.index.is_monotonic_increasing


def test_walk_forward_no_lookahead():
    bench, fac = _make_returns(n=1500)
    df = walk_forward_threshold_search(
        bench,
        fac,
        threshold_grid=[0.05, 0.08],
        config=WalkForwardConfig(train_days=500, test_days=200, step_days=200),
    )
    # Train-End muss vor Test-Start sein
    assert (df["train_end"] < df["test_start"]).all()
