"""Pytest configuration and shared fixtures for backtest regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow bare `from assembled_core.X import Y` in test files (no `src.` prefix).
# The package is not installed as an editable package, so we expose `src/` on
# sys.path so both import styles work side-by-side.
_src_path = str(Path(__file__).parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import pandas as pd
import pytest

from src.assembled_core.portfolio.position_sizing import compute_target_positions

# ---------------------------------------------------------------------------
# P0 A9 (Deep Run v2, 2026-04-18) — marker consolidation.
#
# Eight legacy phase-markers are aliased onto the four canonical markers
# (fast / integration / regression / smoke) so old selections like
# `-m phase12` keep working while the CI can already target `-m fast`.
# Sunset schedule: docs/tech_debt/markers_migration.md (2026-07-01).
# ---------------------------------------------------------------------------

_LEGACY_MARKER_ALIASES: dict[str, str] = {
    "phase4": "fast",
    "phase6": "fast",
    "phase7": "fast",
    "phase8": "fast",
    "phase9": "fast",
    "phase10": "fast",
    "phase11": "fast",
    "phase12": "fast",
    "phase13": "fast",
    "phase_zero": "regression",
    "phase_speed": "regression",
    "phase_realism": "regression",
    "phase_depth": "regression",
}


def pytest_collection_modifyitems(config, items) -> None:  # pragma: no cover - wiring
    for item in items:
        legacy_markers = [
            m.name for m in item.iter_markers() if m.name in _LEGACY_MARKER_ALIASES
        ]
        for legacy in legacy_markers:
            canonical = _LEGACY_MARKER_ALIASES[legacy]
            if canonical not in {m.name for m in item.iter_markers()}:
                item.add_marker(canonical)


@pytest.fixture
def golden_mini_backtest_data():
    """Golden mini backtest fixture: 2-3 symbols, 5-10 days, deterministic signals.

    This fixture provides a small, deterministic dataset for regression testing.
    It ensures that optimizations (vectorization, Numba) don't change the logic.

    Returns:
        Dictionary with:
        - prices: DataFrame with columns: timestamp, symbol, close
        - signals: DataFrame with columns: timestamp, symbol, direction, score
        - expected_orders_count: Expected number of orders
        - expected_equity_start: Expected starting equity
        - expected_equity_end: Expected ending equity (approximate)
    """
    # Create deterministic price data: 3 symbols, 10 days
    symbols = ["AAPL", "MSFT", "GOOGL"]
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D", tz="UTC")

    prices_data = []
    for date in dates:
        for i, symbol in enumerate(symbols):
            # Deterministic prices: base price + day offset + symbol offset
            base_price = 100.0 + (i * 10.0)  # AAPL=100, MSFT=110, GOOGL=120
            day_offset = float((date - dates[0]).days) * 0.5  # Small daily increment
            price = base_price + day_offset
            prices_data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "close": price,
                }
            )

    prices = pd.DataFrame(prices_data)

    # Create deterministic signals: simple trend-following
    # Day 1-3: All LONG
    # Day 4-6: AAPL LONG, others NEUTRAL
    # Day 7-10: All NEUTRAL
    signals_data = []
    for date in dates:
        day_idx = (date - dates[0]).days
        for symbol in symbols:
            if day_idx < 3:
                direction = "LONG"
                score = 1.0
            elif day_idx < 6 and symbol == "AAPL":
                direction = "LONG"
                score = 0.8
            else:
                direction = "NEUTRAL"
                score = 0.0

            signals_data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "direction": direction,
                    "score": score,
                }
            )

    signals = pd.DataFrame(signals_data)

    # Expected values (computed from original implementation)
    # These will be validated against actual results
    expected_orders_count = 6  # Approximate: 3 buys on day 1, 3 sells on day 4, etc.
    expected_equity_start = 10000.0
    expected_equity_end = 10000.0  # Approximate, will be validated

    return {
        "prices": prices,
        "signals": signals,
        "symbols": symbols,
        "dates": dates,
        "expected_orders_count": expected_orders_count,
        "expected_equity_start": expected_equity_start,
        "expected_equity_end": expected_equity_end,
    }


@pytest.fixture
def position_sizing_fn():
    """Position sizing function for golden backtest."""

    def sizing_fn(signals_df, capital):
        return compute_target_positions(
            signals_df, total_capital=capital, equal_weight=True, top_n=3
        )

    return sizing_fn
