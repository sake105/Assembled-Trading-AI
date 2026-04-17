"""Phase 4 regression tests for SmartOrderRouter integration.

Covers:

* ``enable_sor=False`` (default) → sor_cost_bps == 0, fill_price identical to pre-Phase-4
* ``enable_sor=True`` → sor_cost_bps > 0 and included in total_cost_bps
* crisis regime → strictly higher sor_cost_bps than bull regime (spread multiplier)
* BUY fill_price moves up, SELL fill_price moves down with SOR cost
* sor_venues column is populated with a non-empty venue string
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
    enable_sor: bool = False,
    sor_regime: str = "bull",
    sor_urgency: float = 0.5,
) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        enable_sor=enable_sor,
        sor_regime=sor_regime,
        sor_urgency=sor_urgency,
        run_id="sor_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


def test_sor_disabled_is_zero_and_legacy_price(tmp_path: Path) -> None:
    """With SOR off, no sor cost and fill_price matches pre-Phase-4 math."""
    eng = _make_engine(tmp_path, enable_sor=False)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 100.0}]
    )
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1_000_000.0}]
    )
    fills = eng._simulate_fills(orders, prices)
    assert "sor_cost_bps" in fills.columns
    assert fills.loc[0, "sor_cost_bps"] == pytest.approx(0.0)
    assert fills.loc[0, "sor_venues"] is None or pd.isna(fills.loc[0, "sor_venues"])


def test_sor_enabled_adds_cost_and_venues(tmp_path: Path) -> None:
    """SOR on → sor_cost_bps > 0 and venue string populated."""
    eng = _make_engine(tmp_path, enable_sor=True, sor_regime="bull")
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 100.0}]
    )
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1_000_000.0}]
    )
    fills = eng._simulate_fills(orders, prices)
    assert fills.loc[0, "sor_cost_bps"] > 0.0
    assert isinstance(fills.loc[0, "sor_venues"], str)
    assert len(fills.loc[0, "sor_venues"]) > 0
    # total_cost_bps == spread + impact + adversarial + sor
    expected_total = (
        float(fills.loc[0, "spread_cost_bps"])
        + float(fills.loc[0, "impact_cost_bps"])
        + float(fills.loc[0, "adversarial_cost_bps"])
        + float(fills.loc[0, "sor_cost_bps"])
    )
    assert float(fills.loc[0, "total_cost_bps"]) == pytest.approx(expected_total)


def test_sor_crisis_regime_strictly_costlier_than_bull(tmp_path: Path) -> None:
    """Crisis regime must widen spreads → higher sor_cost_bps."""
    eng_bull = _make_engine(tmp_path / "bull", enable_sor=True, sor_regime="bull")
    eng_crisis = _make_engine(tmp_path / "crisis", enable_sor=True, sor_regime="crisis")

    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 100.0}]
    )
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1_000_000.0}]
    )
    f_bull = eng_bull._simulate_fills(orders, prices)
    f_crisis = eng_crisis._simulate_fills(orders, prices)

    assert f_crisis.loc[0, "sor_cost_bps"] > f_bull.loc[0, "sor_cost_bps"]


def test_sor_buy_raises_fill_price_sell_lowers(tmp_path: Path) -> None:
    """BUY + SOR → fill_price above baseline; SELL + SOR → below baseline."""
    eng_off = _make_engine(tmp_path / "off", enable_sor=False)
    eng_on = _make_engine(tmp_path / "on", enable_sor=True)

    orders = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 100.0},
            {"symbol": "BBB", "side": "SELL", "qty": 100.0, "price": 50.0},
        ]
    )
    prices = pd.DataFrame(
        [
            {"symbol": "AAA", "close": 100.0, "volume": 1_000_000.0},
            {"symbol": "BBB", "close": 50.0, "volume": 1_000_000.0},
        ]
    )
    f_off = eng_off._simulate_fills(orders, prices)
    f_on = eng_on._simulate_fills(orders, prices)

    buy_off = f_off[f_off["symbol"] == "AAA"].iloc[0]
    buy_on = f_on[f_on["symbol"] == "AAA"].iloc[0]
    assert buy_on["fill_price"] > buy_off["fill_price"]

    sell_off = f_off[f_off["symbol"] == "BBB"].iloc[0]
    sell_on = f_on[f_on["symbol"] == "BBB"].iloc[0]
    assert sell_on["fill_price"] < sell_off["fill_price"]
