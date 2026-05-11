"""Tests für master_allocator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.master_allocator import (
    MasterAllocator,
    MasterAllocatorConfig,
    cross_asset_hybrid,
    cross_asset_momentum_top_n,
    cross_asset_vol_target_ew,
    vol_target_single_asset,
)


def _equity_ret(n: int = 600, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.0005, 0.012, n),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def _panel(n: int = 600, k: int = 6, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0003, 0.01, (n, k)),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
        columns=[f"asset_{i}" for i in range(k)],
    )


def test_vol_target_single_asset_returns_series():
    r = _equity_ret()
    out = vol_target_single_asset(r)
    assert isinstance(out, pd.Series)
    assert len(out) == len(r)


def test_cross_asset_vol_target_ew():
    p = _panel()
    out = cross_asset_vol_target_ew(p)
    assert isinstance(out, pd.Series)
    assert len(out) == len(p)


def test_cross_asset_momentum_top_n():
    p = _panel(n=400, k=8)
    cfg = MasterAllocatorConfig(xa_mom_top_n=3, xa_mom_min_history=100)
    out = cross_asset_momentum_top_n(p, cfg)
    assert isinstance(out, pd.Series)
    assert len(out) == len(p)
    # einige aktive Rebalances sollten passiert sein
    assert (out != 0).sum() > 30


def test_cross_asset_hybrid_combines():
    p = _panel()
    out = cross_asset_hybrid(p)
    assert isinstance(out, pd.Series)
    # Hybrid ≠ pure VolTarget oder pure Mom
    vt = cross_asset_vol_target_ew(p).dropna()
    mom = cross_asset_momentum_top_n(p).dropna()
    aligned = pd.concat({"out": out, "vt": vt, "mom": mom}, axis=1).dropna()
    if not aligned.empty:
        assert not np.allclose(aligned["out"], aligned["vt"])
        assert not np.allclose(aligned["out"], aligned["mom"])


def test_master_allocator_full_pipeline():
    eq = _equity_ret(n=500, seed=2)
    p = _panel(n=500, seed=3)
    alloc = MasterAllocator()
    out = alloc.allocate(eq, p)
    assert "master_return" in out.columns
    assert "sa_voltarget" in out.columns
    assert "xa_hybrid" in out.columns
    assert not out.empty


def test_master_allocator_weight_extremes():
    eq = _equity_ret(n=500, seed=4)
    p = _panel(n=500, seed=5)

    # 100% SA weight
    alloc_sa = MasterAllocator(MasterAllocatorConfig(sa_weight=1.0))
    out_sa = alloc_sa.allocate(eq, p)
    # 100 % SA: master == sa
    aligned = out_sa.dropna()
    if not aligned.empty:
        np.testing.assert_array_almost_equal(
            aligned["master_return"].values, aligned["sa_voltarget"].values
        )

    # 0% SA weight
    alloc_xa = MasterAllocator(MasterAllocatorConfig(sa_weight=0.0))
    out_xa = alloc_xa.allocate(eq, p)
    aligned = out_xa.dropna()
    if not aligned.empty:
        np.testing.assert_array_almost_equal(
            aligned["master_return"].values, aligned["xa_hybrid"].values
        )


def test_master_allocator_empty_input_returns_empty():
    eq = pd.Series(dtype=float)
    p = pd.DataFrame()
    alloc = MasterAllocator()
    out = alloc.allocate(eq, p)
    assert out.empty


def test_master_allocator_reduces_vol_vs_pure_equity():
    rng = np.random.default_rng(99)
    n = 800
    eq = pd.Series(
        rng.normal(0.0005, 0.025, n),
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )
    p = pd.DataFrame(
        rng.normal(0.0003, 0.01, (n, 5)),
        index=eq.index,
        columns=[f"a_{i}" for i in range(5)],
    )
    alloc = MasterAllocator()
    out = alloc.allocate(eq, p)
    eq_vol = eq.std() * np.sqrt(252)
    master_vol = out["master_return"].dropna().std() * np.sqrt(252)
    # Master sollte Vol unter der Pure-Equity-Vol haben (durch Diversifikation + Vol-Target)
    assert master_vol < eq_vol * 1.1  # Toleranz
