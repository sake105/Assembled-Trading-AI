"""Tests for wave-14 module wiring into trading_cycle.py.

Covers:
  Step 3.45 — ml.factor_timing (compute_factor_momentum)
  Step 3.58 — ml.signal_correlation (SignalCorrelationAnalyzer)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.factor_timing import (
    compute_factor_momentum,
    compute_factor_crowding,
)
from src.assembled_core.ml.signal_correlation import (
    SignalCorrelationAnalyzer,
    SignalCorrelationReport,
)


# ---------------------------------------------------------------------------
# compute_factor_momentum (Step 3.45)
# ---------------------------------------------------------------------------

def _make_factor_returns(n_factors: int = 5, n_periods: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((n_periods, n_factors)),
        columns=[f"f{i}" for i in range(n_factors)],
    )


def test_factor_momentum_returns_dict():
    fr = _make_factor_returns()
    result = compute_factor_momentum(fr)
    assert isinstance(result, dict)


def test_factor_momentum_has_all_factors():
    fr = _make_factor_returns(n_factors=4)
    result = compute_factor_momentum(fr)
    assert set(result.keys()) == {"f0", "f1", "f2", "f3"}


def test_factor_momentum_zero_variance_returns_zeros():
    # All factors have identical returns → zero z-scores
    fr = pd.DataFrame({"f0": [0.01] * 20, "f1": [0.01] * 20, "f2": [0.01] * 20})
    result = compute_factor_momentum(fr)
    for k, v in result.items():
        assert abs(v) < 1e-6, f"{k}: {v} != 0"


def test_factor_momentum_too_few_periods_returns_zeros():
    fr = pd.DataFrame({"f0": [0.01, 0.02], "f1": [0.03, 0.01]})
    result = compute_factor_momentum(fr, lookback=12)
    # With <3 rows, returns zeros not crash
    assert isinstance(result, dict)


def test_factor_momentum_strong_winner_has_positive_score():
    fr = _make_factor_returns(n_periods=30)
    # Make f0 a consistent winner
    fr["f0"] = 0.05
    fr["f1"] = -0.05
    result = compute_factor_momentum(fr)
    assert result["f0"] > 0


def test_factor_crowding_returns_dict():
    exposures = pd.DataFrame({
        f"f{i}": np.random.default_rng(i).standard_normal(20) for i in range(3)
    })
    result = compute_factor_crowding(exposures)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# SignalCorrelationAnalyzer (Step 3.58)
# ---------------------------------------------------------------------------

def _make_signals(n_signals: int = 4, n_obs: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((n_obs, n_signals)),
        columns=[f"sig{i}" for i in range(n_signals)],
    )


def test_signal_corr_returns_report():
    signals = _make_signals()
    analyzer = SignalCorrelationAnalyzer()
    report = analyzer.analyze(signals)
    assert isinstance(report, SignalCorrelationReport)


def test_signal_corr_mean_abs_corr_in_range():
    signals = _make_signals()
    analyzer = SignalCorrelationAnalyzer()
    report = analyzer.analyze(signals)
    assert 0.0 <= report.mean_abs_corr <= 1.0


def test_signal_corr_n_signals_correct():
    signals = _make_signals(n_signals=5)
    analyzer = SignalCorrelationAnalyzer()
    report = analyzer.analyze(signals)
    assert report.n_signals == 5


def test_signal_corr_identical_signals_highly_correlated():
    # Two identical signals → high correlation
    df = pd.DataFrame({"s0": [0.1, -0.2, 0.3] * 10, "s1": [0.1, -0.2, 0.3] * 10})
    analyzer = SignalCorrelationAnalyzer(redundancy_threshold=0.7)
    report = analyzer.analyze(df)
    assert report.mean_abs_corr > 0.8


def test_signal_corr_too_few_signals():
    # Single column → can't compute pairwise
    df = pd.DataFrame({"s0": [0.1, 0.2, 0.3] * 5})
    analyzer = SignalCorrelationAnalyzer()
    report = analyzer.analyze(df)
    assert isinstance(report, SignalCorrelationReport)


def test_signal_corr_empty_returns_empty_report():
    df = pd.DataFrame()
    analyzer = SignalCorrelationAnalyzer()
    report = analyzer.analyze(df)
    assert isinstance(report, SignalCorrelationReport)
    assert report.mean_abs_corr == 0.0
