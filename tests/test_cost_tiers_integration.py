"""E0.2 — cost-tier activation tests.

Covers both layers:

* the :func:`get_tier_for_symbol` / :func:`get_tier_costs` helpers
* the ``UnifiedPaperEngine._simulate_fills`` integration with
  ``enable_cost_tiers=True`` — the important behavioural claim is that
  a mega-cap and a micro-cap pay meaningfully different per-fill costs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.costs import (
    get_tier_costs,
    get_tier_costs_for_symbol,
    get_tier_for_symbol,
)
from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)

pytestmark = pytest.mark.phase_zero


# --- helper layer ------------------------------------------------------------


def test_tier_classification_for_mega_cap() -> None:
    # AAPL-style ADV (~$5B) → mega_cap
    assert get_tier_for_symbol("AAPL", 5_000_000_000.0) == "mega_cap"


def test_tier_classification_for_micro_cap() -> None:
    # Obscure ticker with $500k ADV → micro_cap
    assert get_tier_for_symbol("XYZ", 500_000.0) == "micro_cap"


def test_tier_classification_for_unknown_adv() -> None:
    # None/0 ADV → default tier
    default = get_tier_for_symbol("???", None)
    assert default in {"mid_cap", "small_cap", "micro_cap", "large_cap", "mega_cap"}


def test_tier_costs_monotonic() -> None:
    """Half-spread must not decrease as liquidity falls."""
    mega = get_tier_costs("mega_cap")
    large = get_tier_costs("large_cap")
    mid = get_tier_costs("mid_cap")
    small = get_tier_costs("small_cap")
    micro = get_tier_costs("micro_cap")
    halves = [
        mega["half_spread_bps"],
        large["half_spread_bps"],
        mid["half_spread_bps"],
        small["half_spread_bps"],
        micro["half_spread_bps"],
    ]
    assert halves == sorted(halves), f"tier half-spreads not monotone: {halves}"


def test_aapl_vs_xyz_total_cost_delta_exceeds_plan_floor() -> None:
    """Plan E0-Exit: AAPL vs XYZ total-cost Δ ≥ 18 bps (incl. empirical slippage)."""
    _, aapl = get_tier_costs_for_symbol("AAPL", 5_000_000_000.0)
    _, xyz = get_tier_costs_for_symbol("XYZ", 500_000.0)
    aapl_total = aapl["half_spread_bps"] + aapl["commission_bps"] + aapl["slippage_bps"]
    xyz_total = xyz["half_spread_bps"] + xyz["commission_bps"] + xyz["slippage_bps"]
    delta = xyz_total - aapl_total
    assert delta >= 18.0, f"tier delta {delta:.2f} bps < 18 bps plan floor"


# --- engine integration ------------------------------------------------------


def _make_engine(tmp_path, *, enable_cost_tiers: bool) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        enable_cost_tiers=enable_cost_tiers,
        enable_partial_fills=False,
        enable_borrow_costs=False,
        impact_coefficient=0.0,  # silence impact so we read tier delta cleanly
    )
    return UnifiedPaperEngine(cfg)


def _orders_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 100,
                "price": 180.0,
                "order_id": "o1",
            },
            {
                "symbol": "XYZ",
                "side": "BUY",
                "qty": 100,
                "price": 5.0,
                "order_id": "o2",
            },
        ]
    )


def _prices_frame() -> pd.DataFrame:
    # ADV in shares; ADV-in-USD = adv * close.
    # AAPL: 100M shares × $180 = $18B → mega_cap
    # XYZ:  50k  shares × $5   = $250k → micro_cap
    return pd.DataFrame(
        [
            {"symbol": "AAPL", "close": 180.0, "adv": 100_000_000.0},
            {"symbol": "XYZ", "close": 5.0, "adv": 50_000.0},
        ]
    )


def test_fills_with_tiers_have_tier_annotation(tmp_path) -> None:
    eng = _make_engine(tmp_path, enable_cost_tiers=True)
    fills = eng._simulate_fills(_orders_frame(), _prices_frame())
    assert {"AAPL", "XYZ"} == set(fills["symbol"])
    tiers = dict(zip(fills["symbol"], fills["tier"]))
    assert tiers["AAPL"] == "mega_cap"
    assert tiers["XYZ"] == "micro_cap"


def test_mega_cap_cheaper_than_micro_cap_when_tiers_on(tmp_path) -> None:
    eng = _make_engine(tmp_path, enable_cost_tiers=True)
    fills = eng._simulate_fills(_orders_frame(), _prices_frame())
    totals = dict(zip(fills["symbol"], fills["total_cost_bps"]))
    delta = totals["XYZ"] - totals["AAPL"]
    # With impact_coefficient=0, delta is driven entirely by
    # (half_spread + commission) difference. Micro-cap = 8+1.5 = 9.5 bps;
    # Mega-cap = 1+0.2 = 1.2 bps; delta = 8.3 bps.
    assert delta >= 8.0, f"tier fill-cost delta {delta:.2f} bps too small"


def test_tier_disabled_preserves_flat_cost(tmp_path) -> None:
    eng = _make_engine(tmp_path, enable_cost_tiers=False)
    fills = eng._simulate_fills(_orders_frame(), _prices_frame())
    # When tiers are off, commission_bps is 0 and tier is None.
    assert set(fills["commission_bps"]) == {0.0}
    assert set(fills["tier"]) == {None}
    # Both symbols should observe the same half-spread cost
    # (impact is zero because impact_coefficient=0).
    spreads = set(round(x, 6) for x in fills["spread_cost_bps"])
    assert len(spreads) == 1, (
        f"legacy path should produce flat half-spread, got {spreads}"
    )


def test_reject_unknown_adv(tmp_path) -> None:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        enable_cost_tiers=True,
        reject_unknown_adv=True,
        enable_partial_fills=False,
        enable_borrow_costs=False,
    )
    eng = UnifiedPaperEngine(cfg)
    orders = pd.DataFrame(
        [{"symbol": "UNK", "side": "BUY", "qty": 10, "price": 5.0, "order_id": "x"}]
    )
    prices = pd.DataFrame([{"symbol": "UNK", "close": 5.0}])  # no adv column
    fills = eng._simulate_fills(orders, prices)
    assert fills["status"].iloc[0] == "rejected"
    assert fills["reject_reason"].iloc[0] == "UNKNOWN_ADV"


def test_borrow_costs_default_enabled() -> None:
    cfg = UnifiedPaperConfig()
    assert cfg.enable_borrow_costs is True


def test_default_adv_lowered() -> None:
    cfg = UnifiedPaperConfig()
    assert cfg.default_adv == 100_000.0
