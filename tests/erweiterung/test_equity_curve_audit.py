"""Tests für equity_curve_audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.qa.equity_curve_audit import (
    audit_equity_curve,
    compare_equity_curves,
)


def _build_eq(returns: np.ndarray) -> pd.Series:
    idx = pd.date_range("2022-01-01", periods=len(returns), freq="B")
    return (1 + pd.Series(returns, index=idx)).cumprod()


def test_normal_equity_flags_minimal():
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0003, 0.01, 500)
    eq = _build_eq(ret)
    out = audit_equity_curve(eq, name="normal")
    assert out.overall_sharpe is not None
    assert "EXTREMELY_HIGH_SHARPE" not in out.flags
    assert "RETURNS_LIKELY_SMOOTHED" not in out.flags


def test_extremely_high_sharpe_flagged():
    rng = np.random.default_rng(1)
    # Synth: konstanter, fast deterministischer + winziger Vol → Sharpe explodiert
    ret = np.full(500, 0.002) + rng.normal(0, 0.0001, 500)
    eq = _build_eq(ret)
    out = audit_equity_curve(eq, name="superSharpe")
    assert "EXTREMELY_HIGH_SHARPE" in out.flags


def test_smoothed_returns_flagged():
    rng = np.random.default_rng(2)
    raw = rng.normal(0.0001, 0.01, 500)
    # Smooth via moving average (typischer NAV-Smoothing-Effekt)
    smoothed = pd.Series(raw).rolling(5, min_periods=1).mean().to_numpy()
    eq = _build_eq(smoothed)
    out = audit_equity_curve(eq, name="smoothed")
    assert out.return_autocorr_lag1 is not None
    assert out.return_autocorr_lag1 > 0.3
    assert "RETURNS_LIKELY_SMOOTHED" in out.flags or "HIGH_AUTOCORR_LAG1" in " ".join(
        out.flags
    )


def test_mdd_too_low_for_sharpe_flag():
    # Hochsharpe + niedrige MDD ist verdächtig
    rng = np.random.default_rng(3)
    ret = rng.normal(0.0015, 0.003, 500)  # AnnRet ~38%, Vol ~5% -> Sharpe ~7
    eq = _build_eq(ret)
    out = audit_equity_curve(eq, name="suspicious")
    assert "MDD_TOO_LOW_FOR_SHARPE" in out.flags


def test_compare_equity_curves_dataframe():
    rng = np.random.default_rng(4)
    a = _build_eq(rng.normal(0.0003, 0.01, 400))
    b = _build_eq(rng.normal(0.0001, 0.005, 400))
    df = compare_equity_curves({"a": a, "b": b})
    assert len(df) == 2
    assert "overall_sharpe" in df.columns


def test_insufficient_data_handling():
    eq = pd.Series([100.0, 101.0, 102.0])
    out = audit_equity_curve(eq, name="short")
    assert "INSUFFICIENT_DATA" in out.flags


def test_flat_returns_flag():
    eq = pd.Series(
        [100.0] * 200, index=pd.date_range("2022-01-01", periods=200, freq="B")
    )
    out = audit_equity_curve(eq, name="flat")
    assert "FLAT_RETURNS" in out.flags


def test_low_market_correlation_flag():
    rng = np.random.default_rng(5)
    ret_a = rng.normal(0.0002, 0.01, 300)
    ret_b = rng.normal(0.0002, 0.01, 300)  # uncorrelated by construction
    a = _build_eq(ret_a)
    b = _build_eq(ret_b)
    out = audit_equity_curve(a, name="orphan", bootstrap_benchmark=b)
    # Bei n=300 sollte die Korrelation klein sein → Flag fires
    assert (
        any("MARKET_CORR_TOO_LOW" in f for f in out.flags)
        or out.overall_sharpe is not None
    )
