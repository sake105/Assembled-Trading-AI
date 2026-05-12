"""Syrupy snapshot tests for qa/metrics deterministic outputs (audit E-008).

These tests pin the *numerical* output of pure-function metrics on a
fixed deterministic input. Numerical drift caused by an accidental
refactor or a dependency upgrade is caught immediately and surfaces in
the diff — much louder than a silent change to a Sharpe number.

To regenerate the snapshots after an *intentional* numerical change::

    pytest tests/test_snapshot_metrics.py --snapshot-update

The snapshots live alongside this file in ``__snapshots__/`` and are
git-tracked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("syrupy")


def _round_numerics(d: dict[str, object], digits: int = 6) -> dict[str, object]:
    """Round floats so snapshot diffs don't churn on machine-epsilon drift."""
    out: dict[str, object] = {}
    for k, v in d.items():
        if isinstance(v, float) and not (np.isnan(v) or np.isinf(v)):
            out[k] = round(v, digits)
        else:
            out[k] = v
    return out


def _build_fixed_equity_curve(n: int = 252) -> pd.DataFrame:
    """Deterministic synthetic equity curve — same every run."""
    rng = np.random.default_rng(seed=20260512)
    returns = 0.0003 + 0.01 * rng.standard_normal(n)
    equity = 100_000.0 * np.cumprod(1.0 + returns)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"timestamp": idx, "equity": equity})


def test_compute_equity_metrics_snapshot(snapshot) -> None:
    from src.assembled_core.qa.metrics import compute_equity_metrics

    equity_df = _build_fixed_equity_curve()
    metrics = compute_equity_metrics(equity_df, start_capital=100_000.0, freq="1d")

    payload = {
        "final_pf": metrics.final_pf,
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "volatility": metrics.volatility,
        "var_95": metrics.var_95,
        "periods": metrics.periods,
    }
    assert _round_numerics(payload) == snapshot


def test_deflated_sharpe_snapshot(snapshot) -> None:
    pytest.importorskip("scipy")
    from src.assembled_core.qa.metrics import deflated_sharpe_ratio

    # Three points along the Sharpe x n_tests parameter space.
    samples = {
        "sr_1_0__n_252__tests_1": deflated_sharpe_ratio(
            sharpe_annual=1.0, n_obs=252, n_tests=1
        ),
        "sr_1_0__n_252__tests_100": deflated_sharpe_ratio(
            sharpe_annual=1.0, n_obs=252, n_tests=100
        ),
        "sr_2_5__n_504__tests_1000": deflated_sharpe_ratio(
            sharpe_annual=2.5, n_obs=504, n_tests=1000
        ),
    }
    assert _round_numerics(samples) == snapshot


def test_har_rv_forecast_snapshot(snapshot) -> None:
    from src.assembled_core.features.volatility_estimators import har_rv_forecast

    rng = np.random.default_rng(seed=20260512)
    rv = pd.Series(0.0001 + 0.00005 * np.abs(rng.standard_normal(300).cumsum() / 25))
    pred = har_rv_forecast(rv, horizon=1, min_samples=80)
    # Pin first valid prediction + summary stats — captures fit drift
    # without snapshotting 300 floats.
    valid = pred.dropna()
    payload = {
        "n_predictions": int(len(valid)),
        "first_valid_index": int(valid.index[0]),
        "first_valid_value_e6": round(float(valid.iloc[0]) * 1e6, 3),
        "mean_value_e6": round(float(valid.mean()) * 1e6, 3),
        "min_value_e6": round(float(valid.min()) * 1e6, 3),
        "max_value_e6": round(float(valid.max()) * 1e6, 3),
    }
    assert payload == snapshot
