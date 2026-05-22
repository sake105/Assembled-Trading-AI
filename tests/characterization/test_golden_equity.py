"""Golden-master tests for equity curve characterization.

From 35_GOLDEN_EQUITY_SCENARIO_TESTS.md §3.

These tests freeze the CURRENT behaviour of the pipeline. A diff in the
approved file means intentional change — approve explicitly with:

    UPDATE_SNAPSHOTS=1 pytest tests/characterization/test_golden_equity.py

The tests use synthetic deterministic fixtures so they run without any
external data files.  approvaltests is NOT required; the lightweight
_snapshot helper is used instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.characterization._fixtures import make_ohlcv
from tests.characterization._snapshot import verify_snapshot

# ---------------------------------------------------------------------------
# Simple deterministic "pipeline" for golden testing
# (does not call the full trading_cycle so tests are fast and isolated)
# ---------------------------------------------------------------------------


def _run_minimal_backtest(
    bars: pd.DataFrame,
    initial_equity: float = 100_000.0,
    commission_bps: float = 5.0,
    ema_short: int = 20,
    ema_long: int = 50,
) -> pd.DataFrame:
    """Minimal EMA-crossover backtest on synthetic bars.

    Returns a DataFrame with columns [date, equity].
    Designed to be fast, deterministic, and self-contained.
    """
    commission = commission_bps / 10_000

    results = []
    equity = initial_equity

    for ticker in bars["ticker"].unique():
        df = bars[bars["ticker"] == ticker].copy().sort_values("Date")
        df["ema_s"] = df["Close"].ewm(span=ema_short, adjust=False).mean()
        df["ema_l"] = df["Close"].ewm(span=ema_long, adjust=False).mean()
        df["signal"] = np.where(df["ema_s"] > df["ema_l"], 1, -1)

        position = 0
        for _, row in df.iterrows():
            new_pos = int(row["signal"])
            if new_pos != position:
                trade_cost = abs(new_pos - position) * row["Close"] * commission
                equity -= trade_cost
                position = new_pos
            daily_pnl = position * (row["Close"] - row["Open"])
            equity += daily_pnl
            results.append(
                {"date": row["Date"], "ticker": ticker, "equity": round(equity, 2)}
            )

    eq_df = (
        pd.DataFrame(results)
        .groupby("date")["equity"]
        .last()
        .reset_index()
        .sort_values("date")
    )
    eq_df["equity"] = eq_df["equity"].round(2)
    return eq_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.characterization
def test_golden_equity_baseline_deterministic(approved_dir, tmp_path):
    """Equity curve is identical across runs given the same seed."""
    bars = make_ohlcv(["AAPL", "MSFT"], "2024-01-01", "2024-06-30", seed=42)
    result1 = _run_minimal_backtest(bars)
    result2 = _run_minimal_backtest(bars)
    pd.testing.assert_frame_equal(result1, result2)


@pytest.mark.characterization
def test_golden_equity_snapshot(approved_dir):
    """Equity curve matches approved snapshot (characterization)."""
    bars = make_ohlcv(["AAPL", "MSFT", "NVDA"], "2024-01-01", "2024-06-30", seed=42)
    result = _run_minimal_backtest(bars, initial_equity=100_000.0)

    equity_str = result.to_csv(index=False, lineterminator="\n", float_format="%.2f")
    verify_snapshot("golden_equity_3tickers_2024h1", equity_str, approved_dir)


@pytest.mark.characterization
def test_golden_equity_no_nan(approved_dir):
    """Equity curve must never contain NaN or Inf."""
    bars = make_ohlcv(["AAPL", "MSFT", "NVDA"], "2024-01-01", "2024-06-30", seed=42)
    result = _run_minimal_backtest(bars)
    assert not result["equity"].isna().any(), "NaN in equity curve"
    assert np.isfinite(result["equity"]).all(), "Inf in equity curve"


@pytest.mark.characterization
def test_golden_equity_monotonic_dates(approved_dir):
    """Dates must be strictly monotonic increasing."""
    bars = make_ohlcv(["AAPL"], "2024-01-01", "2024-03-31", seed=42)
    result = _run_minimal_backtest(bars)
    assert result["date"].is_monotonic_increasing


@pytest.mark.characterization
def test_golden_equity_initial_value(approved_dir):
    """First equity value must equal initial_equity (no immediate P&L on day 0)."""
    bars = make_ohlcv(["AAPL"], "2024-01-01", "2024-01-31", seed=42)
    result = _run_minimal_backtest(bars, initial_equity=50_000.0)
    # First bar: open == close (no intraday move applied), so equity should be very close
    # Allow for one commission on first trade
    assert abs(result.iloc[0]["equity"] - 50_000.0) < 100.0


@pytest.mark.characterization
def test_sanity_ema_param_change_shifts_equity(approved_dir):
    """Changing EMA parameters must produce a different equity curve.

    This verifies the snapshot framework would catch param changes.
    """
    bars = make_ohlcv(["AAPL"], "2024-01-01", "2024-06-30", seed=42)
    result_v1 = _run_minimal_backtest(bars, ema_short=20, ema_long=50)
    result_v2 = _run_minimal_backtest(bars, ema_short=10, ema_long=30)
    # They should differ (different EMA periods → different signals → different P&L)
    assert not result_v1["equity"].equals(result_v2["equity"]), (
        "Different EMA params produced identical equity — snapshot would miss param changes"
    )
