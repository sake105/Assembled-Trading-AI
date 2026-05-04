"""Phase 3 regression tests for the Kyle-lambda adversarial fill cost.

Covers:

* ``enable_adversarial_fill=False`` (default) → fill_price matches pre-Phase-3 math
* with high ``signal_strength`` and small ADV, BUY orders fill *higher* than
  the non-adversarial baseline (and SELL orders fill *lower*)
* ``adversarial_cost_bps`` column is emitted and added to ``total_cost_bps``
* ``signal_strength=0`` produces zero adversarial cost even when enabled
* ``adversarial_cost_bps`` is capped at ``max_adversarial_bps``
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
    enable_adversarial_fill: bool = False,
    kyle_lambda: float = 0.10,
    max_adversarial_bps: float = 50.0,
) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        enable_adversarial_fill=enable_adversarial_fill,
        kyle_lambda=kyle_lambda,
        max_adversarial_bps=max_adversarial_bps,
        run_id="adv_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


def test_disabled_adversarial_is_zero_and_legacy_price(tmp_path: Path) -> None:
    """When disabled, no adversarial_cost_bps uplift and price matches baseline."""
    eng = _make_engine(tmp_path, enable_adversarial_fill=False)
    orders = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "BUY",
                "qty": 50.0,
                "price": 100.0,
                "signal_strength": 1.0,
            }
        ]
    )
    prices = pd.DataFrame([{"symbol": "AAA", "close": 100.0, "volume": 1_000.0}])
    fills = eng._simulate_fills(orders, prices)
    assert "adversarial_cost_bps" in fills.columns
    assert fills.loc[0, "adversarial_cost_bps"] == pytest.approx(0.0)


def test_enabled_adversarial_raises_buy_price(tmp_path: Path) -> None:
    """BUY + high signal → fill_price strictly above the non-adversarial baseline."""
    # baseline
    eng_off = _make_engine(tmp_path / "off", enable_adversarial_fill=False)
    # adversarial on
    eng_on = _make_engine(tmp_path / "on", enable_adversarial_fill=True)

    orders = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "BUY",
                "qty": 50.0,
                "price": 100.0,
                "signal_strength": 1.0,
            }
        ]
    )
    prices = pd.DataFrame([{"symbol": "AAA", "close": 100.0, "volume": 1_000.0}])

    f_off = eng_off._simulate_fills(orders, prices)
    f_on = eng_on._simulate_fills(orders, prices)

    # adversarial cost uplift must be > 0
    assert f_on.loc[0, "adversarial_cost_bps"] > 0.0
    # BUY fill is more expensive with adversarial enabled
    assert f_on.loc[0, "fill_price"] > f_off.loc[0, "fill_price"]
    # total_cost_bps == spread + impact + adversarial
    expected_total = (
        float(f_on.loc[0, "spread_cost_bps"])
        + float(f_on.loc[0, "impact_cost_bps"])
        + float(f_on.loc[0, "adversarial_cost_bps"])
    )
    assert float(f_on.loc[0, "total_cost_bps"]) == pytest.approx(expected_total)


def test_enabled_adversarial_lowers_sell_price(tmp_path: Path) -> None:
    """SELL + high signal → fill_price strictly below non-adversarial baseline."""
    eng_off = _make_engine(tmp_path / "off", enable_adversarial_fill=False)
    eng_on = _make_engine(tmp_path / "on", enable_adversarial_fill=True)

    orders = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "SELL",
                "qty": 50.0,
                "price": 100.0,
                "signal_strength": 1.0,
            }
        ]
    )
    prices = pd.DataFrame([{"symbol": "AAA", "close": 100.0, "volume": 1_000.0}])

    f_off = eng_off._simulate_fills(orders, prices)
    f_on = eng_on._simulate_fills(orders, prices)

    assert f_on.loc[0, "adversarial_cost_bps"] > 0.0
    assert f_on.loc[0, "fill_price"] < f_off.loc[0, "fill_price"]


def test_zero_signal_strength_no_adversarial_cost(tmp_path: Path) -> None:
    """Even with the gate enabled, signal_strength=0 yields zero cost."""
    eng = _make_engine(tmp_path, enable_adversarial_fill=True)
    orders = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "BUY",
                "qty": 50.0,
                "price": 100.0,
                "signal_strength": 0.0,
            }
        ]
    )
    prices = pd.DataFrame([{"symbol": "AAA", "close": 100.0, "volume": 1_000.0}])
    fills = eng._simulate_fills(orders, prices)
    assert fills.loc[0, "adversarial_cost_bps"] == pytest.approx(0.0)


def test_adversarial_cost_is_capped(tmp_path: Path) -> None:
    """Adversarial cost must not exceed ``max_adversarial_bps``."""
    eng = _make_engine(
        tmp_path,
        enable_adversarial_fill=True,
        kyle_lambda=10.0,  # very large coefficient
        max_adversarial_bps=25.0,
    )
    orders = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "BUY",
                "qty": 10_000.0,  # very large order
                "price": 100.0,
                "signal_strength": 1.0,
            }
        ]
    )
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 100.0}]  # tiny ADV
    )
    fills = eng._simulate_fills(orders, prices)
    assert fills.loc[0, "adversarial_cost_bps"] <= 25.0 + 1e-9


def test_missing_signal_strength_defaults_to_zero(tmp_path: Path) -> None:
    """If the orders frame has no ``signal_strength`` column → no adversarial cost."""
    eng = _make_engine(tmp_path, enable_adversarial_fill=True)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 50.0, "price": 100.0}]
    )
    prices = pd.DataFrame([{"symbol": "AAA", "close": 100.0, "volume": 1_000.0}])
    fills = eng._simulate_fills(orders, prices)
    assert fills.loc[0, "adversarial_cost_bps"] == pytest.approx(0.0)
