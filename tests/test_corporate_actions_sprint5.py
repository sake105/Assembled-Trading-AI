"""Tests for Sprint 5 / C2 corporate actions: delisting + spinoff.

Covers:
  - Delisting forced exit at last available price
  - Delisting with as_of filter
  - Delisting for non-held symbols (no-op)
  - Spinoff position split into parent + child
  - Spinoff with zero ratio (no-op)
  - Edge cases: empty inputs, missing columns
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.corporate_actions import (  # noqa: E402
    apply_delisting_exits,
    apply_spinoff,
)

pytestmark = [pytest.mark.phase12, pytest.mark.unwired_code]


# ---------------------------------------------------------------------------
# Delisting tests
# ---------------------------------------------------------------------------


def test_delisting_generates_exit_event() -> None:
    """Delisted symbol with open position generates DELIST_EXIT."""
    positions = pd.DataFrame({"symbol": ["AAPL", "LEH"], "qty": [100.0, 50.0]})
    actions = pd.DataFrame(
        {
            "symbol": ["LEH"],
            "action_type": ["DELISTING"],
            "effective_date": [pd.Timestamp("2008-09-15", tz="UTC")],
        }
    )
    prices = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2008-09-12", tz="UTC"),
                pd.Timestamp("2008-09-15", tz="UTC"),
            ]
            * 2,
            "symbol": ["LEH", "LEH", "AAPL", "AAPL"],
            "close": [3.65, 0.21, 150.0, 148.0],
        }
    )

    result = apply_delisting_exits(positions, actions, prices)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["symbol"] == "LEH"
    assert row["exit_type"] == "DELIST_EXIT"
    assert row["qty"] == 50.0
    assert row["exit_price"] == pytest.approx(0.21)


def test_delisting_uses_last_price_before_date() -> None:
    """If no price on delisting date, use last available before it."""
    positions = pd.DataFrame({"symbol": ["XYZ"], "qty": [200.0]})
    actions = pd.DataFrame(
        {
            "symbol": ["XYZ"],
            "action_type": ["DELISTING"],
            "effective_date": [pd.Timestamp("2024-06-15", tz="UTC")],
        }
    )
    # Price data ends before delisting
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-06-01", periods=10, freq="B", tz="UTC"),
            "symbol": ["XYZ"] * 10,
            "close": [50.0 + i for i in range(10)],
        }
    )

    result = apply_delisting_exits(positions, actions, prices)
    assert len(result) == 1
    # Should use closest price before June 15
    assert result.iloc[0]["exit_price"] > 0


def test_delisting_as_of_filter() -> None:
    """Delisting after as_of is not processed."""
    positions = pd.DataFrame({"symbol": ["XYZ"], "qty": [100.0]})
    actions = pd.DataFrame(
        {
            "symbol": ["XYZ"],
            "action_type": ["DELISTING"],
            "effective_date": [pd.Timestamp("2024-12-01", tz="UTC")],
        }
    )
    prices = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-11-01", tz="UTC")],
            "symbol": ["XYZ"],
            "close": [10.0],
        }
    )

    result = apply_delisting_exits(
        positions,
        actions,
        prices,
        as_of=pd.Timestamp("2024-06-01", tz="UTC"),
    )
    assert result.empty


def test_delisting_no_position() -> None:
    """Delisting for symbol not held is a no-op."""
    positions = pd.DataFrame({"symbol": ["AAPL"], "qty": [100.0]})
    actions = pd.DataFrame(
        {
            "symbol": ["LEH"],
            "action_type": ["DELISTING"],
            "effective_date": [pd.Timestamp("2008-09-15", tz="UTC")],
        }
    )
    prices = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2008-09-12", tz="UTC")],
            "symbol": ["LEH"],
            "close": [3.65],
        }
    )

    result = apply_delisting_exits(positions, actions, prices)
    assert result.empty


def test_delisting_empty_inputs() -> None:
    """Empty positions or actions returns empty result."""
    empty_pos = pd.DataFrame(columns=["symbol", "qty"])
    empty_act = pd.DataFrame(columns=["symbol", "action_type", "effective_date"])
    prices = pd.DataFrame(columns=["timestamp", "symbol", "close"])

    assert apply_delisting_exits(empty_pos, empty_act, prices).empty
    assert apply_delisting_exits(None, empty_act, prices).empty


# ---------------------------------------------------------------------------
# Spinoff tests
# ---------------------------------------------------------------------------


def test_spinoff_creates_child_position() -> None:
    """Spinoff adds child position based on ratio."""
    positions = pd.DataFrame(
        {
            "symbol": ["PARENT"],
            "qty": [100.0],
            "avg_price": [50.0],
        }
    )
    actions = pd.DataFrame(
        {
            "symbol": ["PARENT"],
            "action_type": ["SPINOFF"],
            "effective_date": [pd.Timestamp("2024-06-01", tz="UTC")],
            "child_symbol": ["CHILD"],
            "spinoff_ratio": [0.25],  # 0.25 child shares per parent share
        }
    )

    result = apply_spinoff(positions, actions)
    assert len(result) == 2  # parent + child
    parent = result[result["symbol"] == "PARENT"].iloc[0]
    child = result[result["symbol"] == "CHILD"].iloc[0]
    assert parent["qty"] == 100.0  # unchanged
    assert child["qty"] == pytest.approx(25.0)  # 100 * 0.25
    assert child["avg_price"] == 0.0  # cost basis TBD


def test_spinoff_zero_ratio() -> None:
    """Spinoff with ratio=0 is a no-op."""
    positions = pd.DataFrame({"symbol": ["X"], "qty": [100.0]})
    actions = pd.DataFrame(
        {
            "symbol": ["X"],
            "action_type": ["SPINOFF"],
            "effective_date": [pd.Timestamp("2024-01-01", tz="UTC")],
            "child_symbol": ["Y"],
            "spinoff_ratio": [0.0],
        }
    )

    result = apply_spinoff(positions, actions)
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "X"


def test_spinoff_no_matching_parent() -> None:
    """Spinoff for non-held parent is a no-op."""
    positions = pd.DataFrame({"symbol": ["AAPL"], "qty": [100.0]})
    actions = pd.DataFrame(
        {
            "symbol": ["MSFT"],
            "action_type": ["SPINOFF"],
            "effective_date": [pd.Timestamp("2024-01-01", tz="UTC")],
            "child_symbol": ["CHILD"],
            "spinoff_ratio": [0.5],
        }
    )

    result = apply_spinoff(positions, actions)
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "AAPL"


def test_spinoff_empty_inputs() -> None:
    """Empty positions or actions returns copy."""
    empty_pos = pd.DataFrame(columns=["symbol", "qty"])
    empty_act = pd.DataFrame(
        columns=[
            "symbol",
            "action_type",
            "effective_date",
            "child_symbol",
            "spinoff_ratio",
        ]
    )

    assert apply_spinoff(empty_pos, empty_act).empty
    assert apply_spinoff(None, empty_act).empty


def test_spinoff_multiple() -> None:
    """Multiple spinoffs from different parents."""
    positions = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "qty": [100.0, 200.0],
            "avg_price": [10.0, 20.0],
        }
    )
    actions = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "action_type": ["SPINOFF", "SPINOFF"],
            "effective_date": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-02-01", tz="UTC"),
            ],
            "child_symbol": ["A_CHILD", "B_CHILD"],
            "spinoff_ratio": [0.1, 0.5],
        }
    )

    result = apply_spinoff(positions, actions)
    assert len(result) == 4  # 2 parents + 2 children
    a_child = result[result["symbol"] == "A_CHILD"].iloc[0]
    b_child = result[result["symbol"] == "B_CHILD"].iloc[0]
    assert a_child["qty"] == pytest.approx(10.0)  # 100 * 0.1
    assert b_child["qty"] == pytest.approx(100.0)  # 200 * 0.5
