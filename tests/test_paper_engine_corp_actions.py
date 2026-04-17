"""Phase 5 regression tests for corporate-actions hook.

Covers:

* ``enable_corporate_actions=False`` (default) → prices/positions unchanged
* 2:1 split on held symbol → position qty doubled, cost basis halved
* Split on symbol not held → noop
* Dividend on held long → cash credited by qty * dividend_cash
* Dividend on short position → cash debited by |qty| * dividend_cash
* Action effective on a different date → noop
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


def _make_engine(
    tmp_path: Path,
    *,
    enable_corporate_actions: bool,
    ca_path: Path | None = None,
) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        enable_corporate_actions=enable_corporate_actions,
        corporate_actions_path=ca_path,
        run_id="ca_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


def _write_actions(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "actions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_corp_actions_disabled_is_noop(tmp_path: Path) -> None:
    ca_path = _write_actions(
        tmp_path,
        [
            {
                "symbol": "AAA",
                "action_type": "SPLIT",
                "effective_date": "2025-01-15",
                "split_ratio": 2.0,
            }
        ],
    )
    eng = _make_engine(tmp_path, enable_corporate_actions=False, ca_path=ca_path)
    eng._state["positions"]["AAA"] = 100.0
    eng._state["cost_basis"]["AAA"] = 50.0
    prices = pd.DataFrame(
        [{"symbol": "AAA", "timestamp": "2025-01-15", "close": 100.0}]
    )
    result = eng._apply_corporate_actions("2025-01-15", prices)
    assert eng._state["positions"]["AAA"] == 100.0
    assert eng._state["cost_basis"]["AAA"] == 50.0
    # returns prices unchanged
    pd.testing.assert_frame_equal(result, prices)


def test_two_for_one_split_doubles_qty_and_halves_cost_basis(tmp_path: Path) -> None:
    ca_path = _write_actions(
        tmp_path,
        [
            {
                "symbol": "AAA",
                "action_type": "SPLIT",
                "effective_date": "2025-01-15",
                "split_ratio": 2.0,
            }
        ],
    )
    eng = _make_engine(tmp_path, enable_corporate_actions=True, ca_path=ca_path)
    eng._state["positions"]["AAA"] = 100.0
    eng._state["cost_basis"]["AAA"] = 50.0
    prices = pd.DataFrame(
        [{"symbol": "AAA", "timestamp": "2025-01-15", "close": 100.0}]
    )
    eng._apply_corporate_actions("2025-01-15", prices)
    assert eng._state["positions"]["AAA"] == pytest.approx(200.0)
    assert eng._state["cost_basis"]["AAA"] == pytest.approx(25.0)


def test_split_on_unheld_symbol_is_noop(tmp_path: Path) -> None:
    ca_path = _write_actions(
        tmp_path,
        [
            {
                "symbol": "BBB",
                "action_type": "SPLIT",
                "effective_date": "2025-01-15",
                "split_ratio": 2.0,
            }
        ],
    )
    eng = _make_engine(tmp_path, enable_corporate_actions=True, ca_path=ca_path)
    eng._state["positions"]["AAA"] = 100.0
    prices = pd.DataFrame(
        [{"symbol": "AAA", "timestamp": "2025-01-15", "close": 100.0}]
    )
    eng._apply_corporate_actions("2025-01-15", prices)
    assert eng._state["positions"]["AAA"] == 100.0


def test_dividend_on_long_credits_cash(tmp_path: Path) -> None:
    ca_path = _write_actions(
        tmp_path,
        [
            {
                "symbol": "AAA",
                "action_type": "DIVIDEND",
                "effective_date": "2025-01-15",
                "dividend_cash": 1.50,
            }
        ],
    )
    eng = _make_engine(tmp_path, enable_corporate_actions=True, ca_path=ca_path)
    eng._state["positions"]["AAA"] = 100.0
    eng._apply_corporate_actions("2025-01-15", prices=None)
    assert eng._state["cash"] == pytest.approx(1_000_000.0 + 100.0 * 1.50)


def test_dividend_on_short_debits_cash(tmp_path: Path) -> None:
    """A short holder pays the dividend when ex-date hits."""
    ca_path = _write_actions(
        tmp_path,
        [
            {
                "symbol": "AAA",
                "action_type": "DIVIDEND",
                "effective_date": "2025-01-15",
                "dividend_cash": 1.50,
            }
        ],
    )
    eng = _make_engine(tmp_path, enable_corporate_actions=True, ca_path=ca_path)
    eng._state["positions"]["AAA"] = -100.0
    eng._apply_corporate_actions("2025-01-15", prices=None)
    # qty * per_share = -100 * 1.50 = -150 → cash decreases by 150
    assert eng._state["cash"] == pytest.approx(1_000_000.0 - 150.0)


def test_action_on_different_date_is_noop(tmp_path: Path) -> None:
    ca_path = _write_actions(
        tmp_path,
        [
            {
                "symbol": "AAA",
                "action_type": "DIVIDEND",
                "effective_date": "2025-01-16",
                "dividend_cash": 1.50,
            }
        ],
    )
    eng = _make_engine(tmp_path, enable_corporate_actions=True, ca_path=ca_path)
    eng._state["positions"]["AAA"] = 100.0
    eng._apply_corporate_actions("2025-01-15", prices=None)
    assert eng._state["cash"] == 1_000_000.0
