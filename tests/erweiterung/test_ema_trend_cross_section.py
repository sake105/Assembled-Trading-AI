"""Tests für ema_trend_cross_section."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.ema_trend_cross_section import (
    EMATrendConfig,
    backtest_ema_trend,
    compute_ema_spread,
    cross_section_ema_signal,
)


def _make_prices(n_days: int = 200, n_symbols: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B", tz="UTC")
    rows = []
    for sym_i in range(n_symbols):
        sym = f"SYM{sym_i}"
        drift = rng.normal(0.0003, 0.0001)
        rets = rng.normal(drift, 0.012, n_days)
        prices = 100 * (1 + pd.Series(rets)).cumprod()
        for d, p, r in zip(dates, prices, rets):
            rows.append({"date": d, "symbol": sym, "close": p, "return": r})
    return pd.DataFrame(rows)


def test_compute_ema_spread_basic():
    df = _make_prices(n_days=150, n_symbols=5)
    out = compute_ema_spread(df, EMATrendConfig(ema_fast=10, ema_slow=30))
    assert "ema_spread" in out.columns
    assert out["ema_spread"].notna().any()


def test_compute_ema_spread_empty():
    out = compute_ema_spread(pd.DataFrame())
    assert out.empty


def test_cross_section_signal_returns_positions():
    df = _make_prices(n_days=200, n_symbols=10)
    sig = cross_section_ema_signal(df)
    assert "position" in sig.columns
    # Some positions should be non-zero
    assert (sig["position"] != 0).any()


def test_cross_section_long_only():
    df = _make_prices(n_days=200, n_symbols=10)
    sig = cross_section_ema_signal(df, EMATrendConfig(long_only=True))
    assert (sig["position"] >= 0).all()


def test_cross_section_long_short():
    df = _make_prices(n_days=200, n_symbols=12)
    sig = cross_section_ema_signal(df, EMATrendConfig(long_only=False))
    # Long and short positions
    assert (sig["position"] > 0).any()
    assert (sig["position"] < 0).any()


def test_backtest_ema_trend_returns_series():
    df = _make_prices(n_days=300, n_symbols=10)
    port = backtest_ema_trend(df)
    assert isinstance(port, pd.Series)
    assert not port.empty


def test_equal_weight_within_long_side():
    df = _make_prices(n_days=200, n_symbols=10)
    sig = cross_section_ema_signal(
        df, EMATrendConfig(quantile_long=0.3, long_only=True)
    )
    # On dates with positions: positions sum should equal 1.0 (within tolerance)
    sums = sig.groupby("date")["position"].sum()
    valid = sums[sums > 0]
    if not valid.empty:
        assert (abs(valid - 1.0) < 0.01).all()
