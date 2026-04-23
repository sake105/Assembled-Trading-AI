"""Tests for wave-28 module wiring into trading_cycle.py.

Covers:
  Step 2.18 — ml.garch_models (fit_garch)
  Step 5.12 — ops.execution_cost_meta (annotate_execution_cost)
  Step 7.65 — ops.report_retention (purge_old_dated_reports)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.garch_models import fit_garch, GARCHResult
from src.assembled_core.ops.execution_cost_meta import annotate_execution_cost
from src.assembled_core.ops.report_retention import purge_old_dated_reports


# ---------------------------------------------------------------------------
# fit_garch (Step 2.18)
# ---------------------------------------------------------------------------

def _make_returns(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, n))


def test_fit_garch_returns_none_or_result():
    rets = _make_returns(300)
    result = fit_garch(rets, symbol="SPY")
    # arch may or may not be installed — either None or GARCHResult
    assert result is None or isinstance(result, GARCHResult)


def test_fit_garch_short_series_returns_none():
    rets = _make_returns(30)  # too short
    result = fit_garch(rets, symbol="SPY")
    assert result is None


def test_fit_garch_result_fields_valid():
    pytest.importorskip("arch", reason="arch required for GARCH")
    rets = _make_returns(300)
    result = fit_garch(rets)
    assert result is not None
    assert isinstance(result.vol_1d, float)
    assert result.vol_1d >= 0.0
    assert isinstance(result.persistence, float)
    assert 0.0 <= result.persistence <= 2.0  # can be > 1 for GJR


def test_fit_garch_vol5d_geq_vol1d():
    pytest.importorskip("arch", reason="arch required for GARCH")
    rets = _make_returns(300)
    result = fit_garch(rets)
    if result is not None:
        assert result.vol_5d >= result.vol_1d - 1e-9  # 5d >= 1d (sqrt-of-time)


def test_fit_garch_converged_bool():
    pytest.importorskip("arch", reason="arch required for GARCH")
    rets = _make_returns(300)
    result = fit_garch(rets)
    if result is not None:
        assert isinstance(result.converged, bool)


def test_fit_garch_high_vol_higher_forecast():
    pytest.importorskip("arch", reason="arch required for GARCH")
    rng = np.random.default_rng(5)
    low_vol = pd.Series(rng.normal(0, 0.005, 300))
    high_vol = pd.Series(rng.normal(0, 0.05, 300))
    r_low = fit_garch(low_vol)
    r_high = fit_garch(high_vol)
    if r_low is not None and r_high is not None:
        assert r_high.vol_1d > r_low.vol_1d


# ---------------------------------------------------------------------------
# annotate_execution_cost (Step 5.12)
# ---------------------------------------------------------------------------

def _make_orders(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)],
        "side": ["BUY"] * n,
        "quantity": rng.integers(10, 100, n).astype(float),
        "price": rng.uniform(50, 200, n),
    })


def _make_prices(n_periods: int = 60, n_syms: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    ts = pd.date_range("2024-01-01", periods=n_periods, freq="B")
    for sym in [f"S{i}" for i in range(n_syms)]:
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n_periods))
        for t, p in zip(ts, prices):
            rows.append({"timestamp": t, "symbol": sym, "close": float(p), "volume": 1_000_000.0})
    return pd.DataFrame(rows)


def test_annotate_cost_returns_tuple():
    orders = _make_orders()
    prices = _make_prices()
    policy = {}
    orders_out, meta = annotate_execution_cost(orders, prices, policy)
    assert isinstance(orders_out, pd.DataFrame)
    assert isinstance(meta, dict)


def test_annotate_cost_meta_has_keys():
    orders = _make_orders()
    prices = _make_prices()
    policy = {}
    _, meta = annotate_execution_cost(orders, prices, policy)
    for key in ["enabled", "n_orders_in", "total_est_cost_bps"]:
        assert key in meta


def test_annotate_cost_empty_orders_ok():
    orders = pd.DataFrame()
    prices = _make_prices()
    policy = {}
    orders_out, meta = annotate_execution_cost(orders, prices, policy)
    assert isinstance(orders_out, pd.DataFrame)
    assert meta["n_orders_in"] == 0


def test_annotate_cost_disabled_by_default():
    orders = _make_orders()
    prices = _make_prices()
    policy = {}  # no execution.cost_meta.enabled
    _, meta = annotate_execution_cost(orders, prices, policy)
    assert meta["enabled"] is False


def test_annotate_cost_output_rows_leq_input():
    orders = _make_orders(5)
    prices = _make_prices()
    policy = {}
    orders_out, _ = annotate_execution_cost(orders, prices, policy)
    assert len(orders_out) <= len(orders)


# ---------------------------------------------------------------------------
# purge_old_dated_reports (Step 7.65)
# ---------------------------------------------------------------------------

def test_purge_nonexistent_dir_returns_zero():
    result = purge_old_dated_reports(Path("/nonexistent/path"), prefix="report_")
    assert result == 0


def test_purge_keeps_recent_files(tmp_path):
    # Create 10 files, keep_last_n=8 → should purge 2
    import time
    for i in range(10):
        f = tmp_path / f"report_{i:04d}.json"
        f.write_text("{}")
        time.sleep(0.01)  # ensure distinct mtime
    purged = purge_old_dated_reports(tmp_path, prefix="report_", suffix=".json", keep_last_n=8)
    assert purged == 2
    remaining = list(tmp_path.glob("report_*.json"))
    assert len(remaining) == 8


def test_purge_nothing_when_below_limit(tmp_path):
    for i in range(3):
        (tmp_path / f"report_{i}.json").write_text("{}")
    purged = purge_old_dated_reports(tmp_path, prefix="report_", suffix=".json", keep_last_n=10)
    assert purged == 0


def test_purge_returns_int():
    result = purge_old_dated_reports(Path("."), prefix="__nonexistent__prefix__")
    assert isinstance(result, int)


def test_purge_negative_keep_returns_zero(tmp_path):
    (tmp_path / "report_0.json").write_text("{}")
    result = purge_old_dated_reports(tmp_path, prefix="report_", keep_last_n=-1)
    assert result == 0
