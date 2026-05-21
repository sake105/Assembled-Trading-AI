"""§9.6 (a) ADV universe filter regression guards.

select_top_adv_symbols returns top-N symbols ranked by trailing dollar-volume
(close × volume mean over lookback_days). Motivated by backtest evidence that
restricting to liquid names removes transaction-cost noise from the long tail.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.data.universe import select_top_adv_symbols


def _make_panel(per_sym_close_vol: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Build a 30-day panel with constant close/volume per symbol."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for sym, (close, vol) in per_sym_close_vol.items():
        for d in dates:
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "close": close,
                    "volume": vol,
                }
            )
    return pd.DataFrame(rows)


def test_select_top_adv_symbols_orders_by_dollar_volume():
    # AAA: 100 × 1M = $100M ADV
    # BBB: 200 × 0.3M = $60M ADV
    # CCC: 50 × 5M = $250M ADV
    # Expected order: CCC, AAA, BBB
    panel = _make_panel(
        {"AAA": (100, 1_000_000), "BBB": (200, 300_000), "CCC": (50, 5_000_000)}
    )
    top = select_top_adv_symbols(panel, top_n=10)
    assert top == ["CCC", "AAA", "BBB"]


def test_select_top_adv_symbols_truncates_at_top_n():
    panel = _make_panel(
        {"A": (100, 100), "B": (100, 200), "C": (100, 300), "D": (100, 400)}
    )
    top = select_top_adv_symbols(panel, top_n=2)
    assert top == ["D", "C"]  # highest two volumes


def test_select_top_adv_symbols_top_n_larger_than_universe():
    panel = _make_panel({"A": (100, 100), "B": (100, 200)})
    top = select_top_adv_symbols(panel, top_n=10)
    assert set(top) == {"A", "B"}
    assert len(top) == 2


def test_select_top_adv_symbols_zero_or_negative_returns_empty():
    panel = _make_panel({"A": (100, 100)})
    assert select_top_adv_symbols(panel, top_n=0) == []
    assert select_top_adv_symbols(panel, top_n=-5) == []


def test_select_top_adv_symbols_empty_panel_returns_empty():
    empty = pd.DataFrame(columns=["timestamp", "symbol", "close", "volume"])
    assert select_top_adv_symbols(empty, top_n=10) == []


def test_select_top_adv_symbols_missing_required_columns_returns_empty():
    # Missing 'volume'
    panel = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
            "symbol": ["AAA"],
            "close": [100.0],
        }
    )
    assert select_top_adv_symbols(panel, top_n=10) == []


def test_select_top_adv_symbols_respects_lookback_window():
    """Lookback should restrict ADV computation to recent rows only.

    Symbol A: low volume historically (last 30d=200), high recent (last 5d=1000)
    Symbol B: steady volume (1000 throughout)
    With lookback=5: A average = 1000, B average = 1000 → tie
    With lookback=30: A average = ~333 (mostly 200 + 5×1000), B = 1000 → B wins
    """
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for i, d in enumerate(dates):
        # A: 200 for first 25 days, 1000 for last 5 days
        a_vol = 200 if i < 25 else 1000
        rows.append({"timestamp": d, "symbol": "A", "close": 100.0, "volume": a_vol})
        rows.append({"timestamp": d, "symbol": "B", "close": 100.0, "volume": 1000})
    panel = pd.DataFrame(rows)

    # Short lookback: A's recent spike makes it competitive
    top_short = select_top_adv_symbols(panel, top_n=2, lookback_days=5)
    assert set(top_short) == {"A", "B"}

    # Long lookback: A's historical drag pulls its average below B
    top_long = select_top_adv_symbols(panel, top_n=1, lookback_days=30)
    assert top_long == ["B"]


def test_select_top_adv_symbols_handles_nan_values():
    """NaN close/volume rows are dropped before ADV computation."""
    panel = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")] * 4,
            "symbol": ["A", "A", "B", "B"],
            "close": [100.0, None, 50.0, 50.0],
            "volume": [1000.0, 1000.0, 5000.0, 5000.0],
        }
    )
    top = select_top_adv_symbols(panel, top_n=2)
    # A has one valid row × 100 × 1000 = 100k; B has 2 × 50 × 5000 = 250k average
    assert top == ["B", "A"]
