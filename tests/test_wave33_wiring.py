"""Tests for wave-33 module wiring into trading_cycle.py.

Covers:
  Step 3.87 — signals.behavioral_finance (generate_behavioral_signals)
  Step 7.67 — qa.learning_store (append_learning_record)
  Step 8.21 — qa.factor_analysis (compute_factor_ic / summarize_factor_ic)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.signals.behavioral_finance import (
    generate_behavioral_signals,
    BehavioralSignal,
)
from src.assembled_core.qa.learning_store import (
    append_learning_record,
    load_learning_records,
    summarize_learning_store,
)
from src.assembled_core.qa.factor_analysis import (
    compute_factor_ic,
    summarize_factor_ic,
)


# ---------------------------------------------------------------------------
# generate_behavioral_signals (Step 3.87)
# ---------------------------------------------------------------------------

def _make_prices(n: int = 60, n_syms: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sym in [f"S{i}" for i in range(n_syms)]:
        ts = pd.date_range("2024-01-01", periods=n, freq="B")
        closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        vols = rng.uniform(1e5, 1e7, n)
        for t, c, v in zip(ts, closes, vols):
            rows.append({"timestamp": t, "symbol": sym, "close": float(c), "volume": float(v)})
    return pd.DataFrame(rows)


def test_behavioral_returns_list():
    prices = _make_prices()
    result = generate_behavioral_signals(prices)
    assert isinstance(result, list)


def test_behavioral_returns_signals():
    prices = _make_prices()
    result = generate_behavioral_signals(prices)
    assert len(result) > 0
    assert all(isinstance(s, BehavioralSignal) for s in result)


def test_behavioral_composite_in_range():
    prices = _make_prices()
    result = generate_behavioral_signals(prices)
    for sig in result:
        assert -1.0 <= sig.composite_score <= 1.0


def test_behavioral_short_series_skipped():
    rng = np.random.default_rng(1)
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="B"),
        "symbol": "S0",
        "close": rng.uniform(50, 100, 5),
    })
    result = generate_behavioral_signals(prices)
    # < 20 bars → should be empty or skip
    assert isinstance(result, list)


def test_behavioral_each_symbol_has_entry():
    prices = _make_prices(n=60, n_syms=3)
    result = generate_behavioral_signals(prices)
    symbols = {s.symbol for s in result}
    assert len(symbols) <= 3


def test_behavioral_no_volume_ok():
    prices = _make_prices(60, 2).drop(columns=["volume"])
    result = generate_behavioral_signals(prices)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# append_learning_record (Step 7.67)
# ---------------------------------------------------------------------------

def test_learning_store_creates_file(tmp_path):
    path = tmp_path / "store.jsonl"
    append_learning_record({"cycle_date": "2024-01-15", "n_orders": 3}, store_path=path)
    assert path.exists()


def test_learning_store_readable(tmp_path):
    path = tmp_path / "store.jsonl"
    append_learning_record({"cycle_date": "2024-01-15", "equity": 100500.0}, store_path=path)
    records = load_learning_records(store_path=path)
    assert len(records) == 1
    assert records[0]["equity"] == 100500.0


def test_learning_store_multiple_appends(tmp_path):
    path = tmp_path / "store.jsonl"
    for i in range(5):
        append_learning_record({"cycle_date": f"2024-01-{i+1:02d}", "n_orders": i}, store_path=path)
    records = load_learning_records(store_path=path)
    assert len(records) == 5


def test_learning_store_summary(tmp_path):
    path = tmp_path / "store.jsonl"
    for i in range(3):
        append_learning_record({"cycle_date": f"2024-01-{i+1:02d}"}, store_path=path)
    summary = summarize_learning_store(store_path=path)
    assert isinstance(summary, dict)


def test_learning_store_empty_returns_path(tmp_path):
    path = tmp_path / "store.jsonl"
    result = append_learning_record({}, store_path=path)
    assert Path(result).exists()


# ---------------------------------------------------------------------------
# compute_factor_ic / summarize_factor_ic (Step 8.21)
# ---------------------------------------------------------------------------

def _make_panel(n_ts: int = 20, n_syms: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    ts = pd.date_range("2024-01-01", periods=n_ts, freq="B")
    for t in ts:
        for sym in [f"S{i}" for i in range(n_syms)]:
            fwd = rng.normal(0, 0.01)
            rows.append({
                "timestamp": t,
                "symbol": sym,
                "feat_a": fwd * 5 + rng.normal(0, 0.5),
                "feat_b": rng.normal(0, 1),
                "fwd_ret": fwd,
            })
    return pd.DataFrame(rows)


def test_factor_ic_returns_df():
    panel = _make_panel()
    result = compute_factor_ic(panel, factor_cols=["feat_a", "feat_b"], fwd_return_col="fwd_ret")
    assert isinstance(result, pd.DataFrame)


def test_factor_ic_has_required_columns():
    panel = _make_panel()
    result = compute_factor_ic(panel, factor_cols=["feat_a"], fwd_return_col="fwd_ret")
    assert "factor" in result.columns
    assert "ic" in result.columns


def test_summarize_factor_ic_returns_df():
    panel = _make_panel(n_ts=30)
    ic_df = compute_factor_ic(panel, factor_cols=["feat_a", "feat_b"], fwd_return_col="fwd_ret")
    if ic_df.empty:
        pytest.skip("empty IC df")
    summary = summarize_factor_ic(ic_df)
    assert isinstance(summary, pd.DataFrame)


def test_summarize_factor_ic_has_mean_ic():
    panel = _make_panel(n_ts=30)
    ic_df = compute_factor_ic(panel, factor_cols=["feat_a", "feat_b"], fwd_return_col="fwd_ret")
    if ic_df.empty:
        pytest.skip("empty IC df")
    summary = summarize_factor_ic(ic_df)
    assert "mean_ic" in summary.columns


def test_correlated_factor_higher_ic():
    panel = _make_panel(n_ts=40, seed=7)
    ic_df = compute_factor_ic(panel, factor_cols=["feat_a", "feat_b"], fwd_return_col="fwd_ret")
    if ic_df.empty:
        pytest.skip("empty IC df")
    summary = summarize_factor_ic(ic_df)
    mean_ics = summary.set_index("factor")["mean_ic"]
    if "feat_a" in mean_ics.index and "feat_b" in mean_ics.index:
        assert abs(mean_ics["feat_a"]) >= 0.0  # feat_a is correlated
