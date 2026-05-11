"""Tests für cross_section_helpers — Hochperformante Cross-Section-Ops."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.cross_section_helpers import (
    cs_long_only_wide,
    cs_long_short_wide,
    long_format_to_wide,
)


def _make_wide(n_days: int = 200, n_symbols: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_days, freq="B", tz="UTC")
    cols = [f"S{i}" for i in range(n_symbols)]
    signal = pd.DataFrame(
        rng.normal(0, 1, (n_days, n_symbols)), index=idx, columns=cols
    )
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.01, (n_days, n_symbols)), index=idx, columns=cols
    )
    return signal, rets


def test_long_format_to_wide():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"], utc=True
            ),
            "symbol": ["A", "B", "A", "B"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    wide = long_format_to_wide(df, "value")
    assert wide.shape == (2, 2)


def test_cs_long_only_wide_returns_unit_sum():
    sig, rets = _make_wide(n_days=100, n_symbols=10)
    pnl, pos = cs_long_only_wide(sig, rets, quantile=0.3)
    # Positions: top-30% von 10 Symbolen = 3 Symbole, each = 1/3
    pos_sums = pos.sum(axis=1)
    # Either 0 (warmup) or ~1.0 (equal weight)
    valid = pos_sums[pos_sums > 0]
    if not valid.empty:
        assert (abs(valid - 1.0) < 0.01).all()


def test_cs_long_only_wide_only_top_quantile_long():
    sig, rets = _make_wide(n_days=50, n_symbols=10)
    pnl, pos = cs_long_only_wide(sig, rets, quantile=0.3)
    # All positions are >= 0 (long-only)
    assert (pos.values >= 0).all()


def test_cs_long_short_wide_has_both_sides():
    sig, rets = _make_wide(n_days=100, n_symbols=15)
    pnl, pos = cs_long_short_wide(sig, rets, quantile=0.2)
    assert (pos.values > 0).any()
    assert (pos.values < 0).any()


def test_cs_lag_no_lookahead():
    """Mit lag_days=1: row 0 hat keine valid signals (NaN nach shift)."""
    sig, rets = _make_wide(n_days=20, n_symbols=5)
    pnl, pos = cs_long_only_wide(sig, rets, lag_days=1)
    # First row should be all zero (shifted NaN signal)
    assert (pos.iloc[0] == 0).all()


def test_cs_empty_input():
    pnl, pos = cs_long_only_wide(pd.DataFrame(), pd.DataFrame())
    assert pnl.empty
    assert pos.empty


def test_cs_long_only_matches_original_pandas_groupby():
    """Vektorisierte Version muss numerisch gleich der pandas-groupby-Variante sein."""
    rng = np.random.default_rng(42)
    n_days, n_symbols = 50, 8
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B", tz="UTC")
    cols = [f"S{i}" for i in range(n_symbols)]
    sig_wide = pd.DataFrame(
        rng.normal(0, 1, (n_days, n_symbols)), index=idx, columns=cols
    )
    ret_wide = pd.DataFrame(
        rng.normal(0.0003, 0.01, (n_days, n_symbols)), index=idx, columns=cols
    )

    pnl_vec, pos_vec = cs_long_only_wide(sig_wide, ret_wide, quantile=0.3, lag_days=1)

    # Reference: long-format with groupby
    long_panel = sig_wide.stack().rename("signal").reset_index()
    long_panel.columns = ["date", "symbol", "signal"]
    long_panel = long_panel.merge(
        ret_wide.stack()
        .rename("return")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "symbol"}),
        on=["date", "symbol"],
    )
    long_panel = long_panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    long_panel["sig_lag"] = long_panel.groupby("symbol")["signal"].shift(1)
    long_panel["sig_pct"] = long_panel.groupby("date")["sig_lag"].rank(pct=True)
    long_panel["pos"] = 0.0
    mask = long_panel["sig_pct"] >= 0.7
    long_panel.loc[mask, "pos"] = 1.0
    n_long = long_panel.groupby("date")["pos"].transform(lambda s: (s > 0).sum())
    long_panel.loc[mask, "pos"] = 1.0 / n_long[mask]
    long_panel["pnl"] = long_panel["pos"] * long_panel["return"]
    ref_pnl = long_panel.groupby("date")["pnl"].sum()

    # Compare (ignore warmup NaN)
    aligned = pd.concat([pnl_vec, ref_pnl], axis=1).dropna()
    if not aligned.empty:
        np.testing.assert_array_almost_equal(
            aligned.iloc[:, 0].values,
            aligned.iloc[:, 1].values,
            decimal=10,
        )
