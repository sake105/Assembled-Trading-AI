"""Tests for wave-22 module wiring into trading_cycle.py.

Covers:
  Step 1.95 — qa.data_qc (run_price_panel_qc)
  Step 3.62 — signals.rules_trend (generate_trend_signals)
  Step 3.75 — qa.multiple_testing (benjamini_hochberg_fdr)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.data_qc import run_price_panel_qc, QcReport, QcIssue
from src.assembled_core.signals.rules_trend import (
    generate_trend_signals,
    generate_trend_signals_from_prices,
)
from src.assembled_core.qa.multiple_testing import (
    benjamini_hochberg_fdr,
    holm_bonferroni_fwer,
    MultipleTestingResult,
)


# ---------------------------------------------------------------------------
# run_price_panel_qc (Step 1.95)
# ---------------------------------------------------------------------------

def _make_clean_prices(n: int = 60, n_symbols: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sym in [f"S{i}" for i in range(n_symbols)]:
        ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        for t, p in zip(ts, prices):
            rows.append({"timestamp": t, "symbol": sym, "close": float(p)})
    return pd.DataFrame(rows)


def test_price_panel_qc_returns_report():
    prices = _make_clean_prices()
    report = run_price_panel_qc(prices, freq="1d")
    assert isinstance(report, QcReport)


def test_price_panel_qc_ok_is_bool():
    prices = _make_clean_prices()
    report = run_price_panel_qc(prices, freq="1d")
    assert isinstance(report.ok, bool)


def test_price_panel_qc_clean_data_passes():
    prices = _make_clean_prices(n=60)
    report = run_price_panel_qc(prices, freq="1d")
    fail_issues = [i for i in report.issues if i.severity == "FAIL"]
    assert len(fail_issues) == 0


def test_price_panel_qc_negative_price_detected():
    prices = _make_clean_prices(n=30)
    prices.loc[0, "close"] = -5.0
    report = run_price_panel_qc(prices, freq="1d")
    fail_checks = [i.check for i in report.issues if i.severity == "FAIL"]
    assert any("price" in c.lower() or "invalid" in c.lower() for c in fail_checks)


def test_price_panel_qc_issues_are_list():
    prices = _make_clean_prices()
    report = run_price_panel_qc(prices, freq="1d")
    assert isinstance(report.issues, list)


def test_price_panel_qc_zero_price_detected():
    prices = _make_clean_prices(n=30)
    prices.loc[5, "close"] = 0.0
    report = run_price_panel_qc(prices, freq="1d")
    assert not report.ok


# ---------------------------------------------------------------------------
# generate_trend_signals (Step 3.62)
# ---------------------------------------------------------------------------

def _make_panel(n: int = 80, n_symbols: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym in [f"S{i}" for i in range(n_symbols)]:
        ts = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = 100.0 + np.cumsum(rng.normal(0.1, 0.5, n))
        for t, p in zip(ts, prices):
            rows.append({"timestamp": t, "symbol": sym, "close": float(p)})
    return pd.DataFrame(rows)


def test_trend_signals_returns_df():
    df = _make_panel()
    result = generate_trend_signals(df)
    assert isinstance(result, pd.DataFrame)


def test_trend_signals_has_required_columns():
    df = _make_panel()
    result = generate_trend_signals(df)
    for col in ["timestamp", "symbol", "direction", "score"]:
        assert col in result.columns


def test_trend_signals_direction_valid():
    df = _make_panel()
    result = generate_trend_signals(df)
    assert set(result["direction"].unique()).issubset({"LONG", "FLAT"})


def test_trend_signals_score_in_range():
    df = _make_panel()
    result = generate_trend_signals(df)
    assert (result["score"] >= 0.0).all()
    assert (result["score"] <= 1.0 + 1e-9).all()


def test_trend_signals_uptrending_mostly_long():
    rng = np.random.default_rng(3)
    n = 80
    ts = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = 100.0 + np.cumsum(np.abs(rng.normal(0.5, 0.2, n)))  # strictly upward
    df = pd.DataFrame({"timestamp": ts, "symbol": "A", "close": prices.tolist()})
    result = generate_trend_signals(df)
    # After warmup, should be mostly LONG
    recent = result.iloc[-20:]
    assert (recent["direction"] == "LONG").sum() >= 5


def test_trend_signals_missing_close_raises():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", 10), "symbol": "A"})
    with pytest.raises(KeyError):
        generate_trend_signals(df)


def test_trend_signals_from_prices_returns_df():
    df = _make_panel()
    result = generate_trend_signals_from_prices(df)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# benjamini_hochberg_fdr (Step 3.75)
# ---------------------------------------------------------------------------

def test_bh_fdr_returns_result():
    pvals = [0.001, 0.01, 0.05, 0.1, 0.5, 0.9]
    result = benjamini_hochberg_fdr(pvals, alpha=0.05)
    assert isinstance(result, MultipleTestingResult)


def test_bh_fdr_method_is_bh():
    pvals = [0.001, 0.01, 0.05]
    result = benjamini_hochberg_fdr(pvals)
    assert result.method == "BH-FDR"


def test_bh_fdr_n_tests_matches():
    pvals = [0.01, 0.05, 0.1, 0.5]
    result = benjamini_hochberg_fdr(pvals)
    assert result.n_tests == 4


def test_bh_fdr_strong_signals_rejected():
    pvals = [0.0001, 0.0002, 0.0003, 0.9, 0.95]
    result = benjamini_hochberg_fdr(pvals, alpha=0.05)
    assert result.n_rejected >= 3


def test_bh_fdr_all_high_pvals_none_rejected():
    pvals = [0.5, 0.6, 0.7, 0.8, 0.9]
    result = benjamini_hochberg_fdr(pvals, alpha=0.05)
    assert result.n_rejected == 0


def test_bh_fdr_rejected_list_length_matches():
    pvals = [0.01, 0.05, 0.1]
    result = benjamini_hochberg_fdr(pvals)
    assert len(result.rejected) == 3


def test_holm_bonferroni_returns_result():
    pvals = [0.001, 0.01, 0.05, 0.5]
    result = holm_bonferroni_fwer(pvals, alpha=0.05)
    assert isinstance(result, MultipleTestingResult)
    assert result.method == "Holm-Bonferroni"
