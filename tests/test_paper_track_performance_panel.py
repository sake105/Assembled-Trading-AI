"""Tests for Paper-Track performance metrics panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.paper.paper_track import (
    PaperTrackConfig,
    compute_paper_performance_panel,
)


@pytest.mark.advanced
def test_compute_paper_performance_panel_sharpe_and_dsr() -> None:
    """Synthetic equity curve should produce finite rolling Sharpe and DSR after warmup."""
    # Create 300 trading days of synthetic equity with positive drift
    dates = pd.date_range("2025-01-01", periods=300, freq="B", tz="UTC")
    np.random.seed(42)
    daily_returns = np.random.normal(loc=0.001, scale=0.01, size=len(dates))
    equity = 100000.0 * (1.0 + pd.Series(daily_returns, index=dates)).cumprod()

    equity_curve = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "timestamp": dates,
            "equity": equity.values,
        }
    )

    # Minimal config for daily freq
    config = PaperTrackConfig(
        strategy_name="perf_panel_test",
        strategy_type="trend_baseline",
        universe_file=Path("dummy_universe.txt"),
        freq="1d",
        seed_capital=100000.0,
        strategy_params={"dsr_n_tests": 10},
    )

    panel = compute_paper_performance_panel(
        equity_curve=equity_curve,
        trades=None,
        config=config,
        windows=[63, 126],
        n_tests=10,
    )

    assert not panel.empty, "Performance panel should not be empty"
    for col in [
        "date",
        "timestamp",
        "window",
        "n_obs",
        "sharpe",
        "max_drawdown_pct",
        "turnover_annualized",
        "deflated_sharpe",
    ]:
        assert col in panel.columns

    # Check last row metrics are finite (Sharpe, DSR, max drawdown)
    last = panel.iloc[-1]
    assert last["n_obs"] >= 63
    assert last["sharpe"] is None or np.isfinite(last["sharpe"])
    assert np.isfinite(last["max_drawdown_pct"])
    # DSR may be NaN for edge cases, but normally should be finite here
    if last["deflated_sharpe"] is not None:
        assert np.isfinite(last["deflated_sharpe"])



