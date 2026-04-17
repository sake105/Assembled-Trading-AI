"""Phase 9 tests for attribution drilldowns.

Covers:

* ``compute_cost_attribution`` — per-symbol and aggregate bps/cash with the
  notional-weighting identity: total_cost_cash ≈ sum of component cash.
* ``compute_regime_attribution`` — maps fills.date through regime_history and
  aggregates per regime; unknown dates fall back to ``regime="unknown"``.
* ``compute_factor_attribution`` — returns empty frame when the dominant-factor
  column is missing; groups by factor when present.
* Engine integration — ``_write_attribution_artifacts`` writes CSV + JSON into
  ``<run_id>/attribution_<date>.(csv|json)`` next to the TCA dir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.assembled_core.accounting.attribution import (
    compute_cost_attribution,
    compute_factor_attribution,
    compute_regime_attribution,
)
from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


# --- unit: compute_cost_attribution ----------------------------------------


def _sample_fills() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-01-15",
                "symbol": "AAA",
                "side": "BUY",
                "qty": 100.0,
                "fill_qty": 100.0,
                "fill_price": 100.0,
                "spread_cost_bps": 2.0,
                "impact_cost_bps": 1.0,
                "adversarial_cost_bps": 1.0,
                "sor_cost_bps": 1.0,
                "total_cost_bps": 5.0,
                "status": "filled",
            },
            {
                "date": "2025-01-15",
                "symbol": "AAA",
                "side": "SELL",
                "qty": 50.0,
                "fill_qty": 50.0,
                "fill_price": 101.0,
                "spread_cost_bps": 2.0,
                "impact_cost_bps": 2.0,
                "adversarial_cost_bps": 0.0,
                "sor_cost_bps": 0.0,
                "total_cost_bps": 4.0,
                "status": "filled",
            },
            {
                "date": "2025-01-16",
                "symbol": "BBB",
                "side": "BUY",
                "qty": 200.0,
                "fill_qty": 200.0,
                "fill_price": 50.0,
                "spread_cost_bps": 3.0,
                "impact_cost_bps": 1.0,
                "adversarial_cost_bps": 0.0,
                "sor_cost_bps": 2.0,
                "total_cost_bps": 6.0,
                "status": "filled",
            },
        ]
    )


def test_cost_attribution_empty() -> None:
    out = compute_cost_attribution(pd.DataFrame())
    assert out["per_symbol"].empty
    assert out["total"] == {"notional": 0.0, "n_fills": 0}


def test_cost_attribution_per_symbol_and_total() -> None:
    fills = _sample_fills()
    out = compute_cost_attribution(fills)
    per = out["per_symbol"]
    tot = out["total"]

    # Two symbols, sorted alphabetically
    assert list(per["symbol"]) == ["AAA", "BBB"]

    aaa = per[per["symbol"] == "AAA"].iloc[0]
    # Notional AAA: 100*100 + 50*101 = 10000 + 5050 = 15050
    assert abs(aaa["notional"] - 15050.0) < 1e-6
    # Spread cash AAA: 5bps * 10000 + 2bps * 5050 = 5 + 1.01 = 6.01  (bps 2, 2)
    # In cash: (2/10000)*10000 + (2/10000)*5050 = 2 + 1.01 = 3.01
    assert abs(aaa["spread_cost_cash"] - 3.01) < 1e-6
    # Notional-weighted spread bps for AAA: 3.01 / 15050 * 10_000 ≈ 2.0
    assert abs(aaa["spread_cost_bps"] - 2.0) < 1e-6

    # Totals: identity — sum of component cash equals total cost cash
    component_cash_sum = (
        tot["spread_cost_cash"]
        + tot["impact_cost_cash"]
        + tot["adversarial_cost_cash"]
        + tot["sor_cost_cash"]
    )
    assert abs(component_cash_sum - tot["total_cost_cash"]) < 1e-6
    assert tot["n_fills"] == 3


# --- unit: compute_regime_attribution --------------------------------------


def test_regime_attribution_maps_dates() -> None:
    fills = _sample_fills()
    history = [
        {"date": "2025-01-15", "regime": "normal"},
        {"date": "2025-01-16", "regime": "crisis"},
    ]
    out = compute_regime_attribution(fills, history)
    assert list(out["regime"]) == ["crisis", "normal"]

    normal = out[out["regime"] == "normal"].iloc[0]
    assert normal["n_fills"] == 2  # two fills on 2025-01-15
    crisis = out[out["regime"] == "crisis"].iloc[0]
    assert crisis["n_fills"] == 1


def test_regime_attribution_unknown_falls_back() -> None:
    fills = pd.DataFrame(
        [
            {"date": "2025-01-15", "symbol": "AAA", "fill_qty": 10,
             "fill_price": 100, "total_cost_bps": 5.0},
        ]
    )
    out = compute_regime_attribution(fills, [])  # no mappings
    assert list(out["regime"]) == ["unknown"]


def test_regime_attribution_empty() -> None:
    out = compute_regime_attribution(pd.DataFrame(), [])
    assert out.empty


# --- unit: compute_factor_attribution --------------------------------------


def test_factor_attribution_missing_column_returns_empty() -> None:
    fills = _sample_fills()  # no dominant_factor column
    out = compute_factor_attribution(fills)
    assert out.empty


def test_factor_attribution_groups_by_dominant_factor() -> None:
    fills = _sample_fills()
    fills = fills.copy()
    fills["dominant_factor"] = ["momentum", "momentum", "value"]
    out = compute_factor_attribution(fills)
    assert set(out["factor"]) == {"momentum", "value"}
    mom = out[out["factor"] == "momentum"].iloc[0]
    # Momentum sees both AAA fills: notional 10000 + 5050 = 15050
    assert abs(mom["notional"] - 15050.0) < 1e-6


# --- engine integration ----------------------------------------------------


def _make_engine(tmp_path: Path, *, enable_attribution: bool = True) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        tca_dir=tmp_path / "tca",
        attribution_dir=tmp_path / "attribution",
        enable_attribution=enable_attribution,
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        run_id="attr_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


def test_engine_attribution_disabled_writes_nothing(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, enable_attribution=False)
    fills = _sample_fills()
    out = eng._write_attribution_artifacts("2025-01-15", fills)
    assert out is None


def test_engine_attribution_empty_fills_is_noop(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    out = eng._write_attribution_artifacts("2025-01-15", pd.DataFrame())
    assert out is None


def test_engine_attribution_writes_csv_and_json(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    fills = _sample_fills()
    out = eng._write_attribution_artifacts("2025-01-15", fills)
    assert out is not None
    csv_path, json_path = out
    assert csv_path.exists()
    assert json_path.exists()

    per_symbol = pd.read_csv(csv_path)
    assert set(per_symbol["symbol"]) == {"AAA", "BBB"}

    payload = json.loads(json_path.read_text())
    assert "total" in payload
    assert payload["total"]["n_fills"] == 3
    # Regime attribution is present but empty when no history supplied
    assert "regime" in payload
    assert "factor" in payload
