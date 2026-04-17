"""Phase 7 regression tests for engine TCA artifacts.

Covers:

* ``enable_tca=False`` → no artifacts written
* Normal fills produce per-order CSV and aggregate JSON
* Aggregate reports fill_rate, slippage percentiles, cost breakdowns
* Arrival slippage uses side sign (BUY: fp>ap → positive, SELL: fp<ap → positive)
* Empty orders + empty fills → noop (no files)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


def _make_engine(
    tmp_path: Path,
    *,
    enable_tca: bool = True,
) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        tca_dir=tmp_path / "tca",
        enable_tca=enable_tca,
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        run_id="tca_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


def test_tca_disabled_writes_nothing(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, enable_tca=False)
    orders = pd.DataFrame([{"symbol": "AAA", "side": "BUY", "qty": 100.0}])
    fills = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 100.0, "fill_qty": 100.0,
          "fill_price": 100.5, "arrival_price": 100.0, "total_cost_bps": 5.0,
          "spread_cost_bps": 2.0, "impact_cost_bps": 1.0,
          "adversarial_cost_bps": 1.0, "sor_cost_bps": 1.0, "status": "filled"}]
    )
    out = eng._write_tca_artifacts("2025-01-15", orders, fills)
    assert out is None
    assert not (tmp_path / "tca").exists()


def test_tca_empty_run_is_noop(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    orders = pd.DataFrame()
    fills = pd.DataFrame()
    out = eng._write_tca_artifacts("2025-01-15", orders, fills)
    assert out is None


def test_tca_writes_per_order_and_aggregate(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    orders = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 100.0},
            {"symbol": "BBB", "side": "SELL", "qty": 50.0},
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "symbol": "AAA", "side": "BUY", "qty": 100.0, "fill_qty": 100.0,
                "fill_price": 100.5, "arrival_price": 100.0,
                "total_cost_bps": 5.0,
                "spread_cost_bps": 2.0, "impact_cost_bps": 1.0,
                "adversarial_cost_bps": 1.0, "sor_cost_bps": 1.0,
                "status": "filled",
            },
            {
                "symbol": "BBB", "side": "SELL", "qty": 50.0, "fill_qty": 50.0,
                "fill_price": 49.75, "arrival_price": 50.0,
                "total_cost_bps": 5.0,
                "spread_cost_bps": 2.0, "impact_cost_bps": 1.0,
                "adversarial_cost_bps": 1.0, "sor_cost_bps": 1.0,
                "status": "filled",
            },
        ]
    )
    out = eng._write_tca_artifacts("2025-01-15", orders, fills)
    assert out is not None
    csv_path, json_path = out
    assert csv_path.exists()
    assert json_path.exists()

    per_order = pd.read_csv(csv_path)
    assert len(per_order) == 2
    assert set(per_order["symbol"]) == {"AAA", "BBB"}
    # Arrival slippage: BUY AAA at 100.5 vs 100.0 → +50 bps; SELL BBB at 49.75 vs 50.0 → +50 bps
    buy = per_order[per_order["symbol"] == "AAA"].iloc[0]
    sell = per_order[per_order["symbol"] == "BBB"].iloc[0]
    assert abs(buy["arrival_slippage_bps"] - 50.0) < 1e-6
    assert abs(sell["arrival_slippage_bps"] - 50.0) < 1e-6

    agg = json.loads(json_path.read_text())
    assert agg["n_orders"] == 2
    assert agg["n_fills"] == 2
    assert agg["fill_rate"] == 1.0
    assert agg["slippage_bps"]["p50"] > 0
    assert agg["cost_bps_avg"]["total"] == 5.0


def test_tca_partial_fill_and_reject(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    orders = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 100.0},
            {"symbol": "BBB", "side": "SELL", "qty": 100.0},
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "symbol": "AAA", "side": "BUY", "qty": 100.0, "fill_qty": 60.0,
                "fill_price": 100.2, "arrival_price": 100.0,
                "total_cost_bps": 3.0, "spread_cost_bps": 2.0,
                "impact_cost_bps": 0.5, "adversarial_cost_bps": 0.3,
                "sor_cost_bps": 0.2, "status": "partial",
            },
            {
                "symbol": "BBB", "side": "SELL", "qty": 100.0, "fill_qty": 0.0,
                "fill_price": 0.0, "arrival_price": 50.0,
                "total_cost_bps": 0.0, "spread_cost_bps": 0.0,
                "impact_cost_bps": 0.0, "adversarial_cost_bps": 0.0,
                "sor_cost_bps": 0.0, "status": "rejected",
            },
        ]
    )
    out = eng._write_tca_artifacts("2025-01-15", orders, fills)
    assert out is not None
    _, json_path = out
    agg = json.loads(json_path.read_text())
    assert agg["n_orders"] == 2
    # Partial counts as a fill for fill-rate purposes (non-rejected).
    assert agg["n_fills"] == 1
    assert agg["fill_rate"] == 0.5
