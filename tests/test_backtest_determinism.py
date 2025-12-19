"""Determinism tests for backtests with explicit seeding (B1).

These tests verify that using the central seed utilities yields
reproducible backtest results for a given seed and that different
seeds can produce different but individually deterministic outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase4


def _build_prices_with_seed(seed: int) -> pd.DataFrame:
    """Create a small synthetic price panel whose values depend on a seed.

    The randomness is purely for testing determinism of the overall
    pipeline; production backtests use real historical data.
    """
    from src.assembled_core.utils import set_global_seed

    set_global_seed(seed)

    dates = pd.date_range(
        start="2024-01-01", end="2024-01-15", freq="B", tz="UTC"
    )
    symbols = ["AAPL", "MSFT"]

    rows: list[dict[str, object]] = []
    for symbol in symbols:
        base_price = 150.0 if symbol == "AAPL" else 200.0
        n_days = len(dates)
        # Seeded Gaussian noise -> deterministic per seed
        returns = np.random.normal(0.0005, 0.01, n_days)
        prices = base_price * np.exp(np.cumsum(returns))

        for i, ts in enumerate(dates):
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "close": float(prices[i]),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


def _simple_signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Very simple deterministic signal function.

    Long if price above a static baseline, otherwise flat.
    """
    signals = []
    for _, row in prices_df.iterrows():
        baseline = 150.0 if row["symbol"] == "AAPL" else 200.0
        direction = "LONG" if row["close"] > baseline else "FLAT"
        signals.append(
            {
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "direction": direction,
                "score": 1.0 if direction == "LONG" else 0.0,
            }
        )
    return pd.DataFrame(signals)


def _simple_position_sizing_fn(
    signals_df: pd.DataFrame, capital: float
) -> pd.DataFrame:
    """Equal-weight sizing across all LONG signals."""
    long_signals = signals_df[signals_df["direction"] == "LONG"]
    if long_signals.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])

    symbols = sorted(long_signals["symbol"].unique())
    n = len(symbols)
    targets = []
    for symbol in symbols:
        targets.append(
            {
                "symbol": symbol,
                "target_weight": 1.0 / n if n > 0 else 0.0,
                "target_qty": (capital / n) / 150.0,
            }
        )
    return pd.DataFrame(targets)


@pytest.mark.advanced
def test_backtest_deterministic_for_fixed_seed() -> None:
    """Backtest results must be identical for the same seed and config."""
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

    def _run_once(seed: int):
        prices = _build_prices_with_seed(seed)
        result = run_portfolio_backtest(
            prices=prices,
            signal_fn=_simple_signal_fn,
            position_sizing_fn=_simple_position_sizing_fn,
            start_capital=10000.0,
            include_costs=False,
            include_trades=True,
        )
        return result

    result_1 = _run_once(seed=123)
    result_2 = _run_once(seed=123)

    # Equity curves identical
    pd.testing.assert_frame_equal(
        result_1.equity.sort_values("timestamp").reset_index(drop=True),
        result_2.equity.sort_values("timestamp").reset_index(drop=True),
        rtol=1e-12,
        atol=1e-12,
    )

    # Trades identical
    assert result_1.trades is not None
    assert result_2.trades is not None
    pd.testing.assert_frame_equal(
        result_1.trades.sort_values(
            ["timestamp", "symbol", "side", "qty"]
        ).reset_index(drop=True),
        result_2.trades.sort_values(
            ["timestamp", "symbol", "side", "qty"]
        ).reset_index(drop=True),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.advanced
def test_backtest_separate_seeds_individually_deterministic() -> None:
    """Different seeds may change paths, but each seed is deterministic."""
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

    def _run_once(seed: int):
        prices = _build_prices_with_seed(seed)
        result = run_portfolio_backtest(
            prices=prices,
            signal_fn=_simple_signal_fn,
            position_sizing_fn=_simple_position_sizing_fn,
            start_capital=10000.0,
            include_costs=False,
            include_trades=True,
        )
        return result

    # Deterministic per seed
    a1 = _run_once(seed=111)
    a2 = _run_once(seed=111)
    b1 = _run_once(seed=222)
    b2 = _run_once(seed=222)

    pd.testing.assert_frame_equal(
        a1.equity.reset_index(drop=True),
        a2.equity.reset_index(drop=True),
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        b1.equity.reset_index(drop=True),
        b2.equity.reset_index(drop=True),
        rtol=1e-12,
        atol=1e-12,
    )

    # With different seeds, the random-walk prices (and thus equity) should differ
    final_a = float(a1.equity["equity"].iloc[-1])
    final_b = float(b1.equity["equity"].iloc[-1])
    assert final_a != pytest.approx(final_b, rel=1e-6, abs=1e-6)


