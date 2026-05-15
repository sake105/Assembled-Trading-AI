"""Regression test for F-A3-2: pre_trade_checks recomputes exposures between
sector/region group checks.

R3 audit (F-A3-2): _ptc_check_group_exposures computed exposures_df ONCE
before the group-check loop. sector → region → FX each consumed the snapshot,
but the sector loop (_apply_group_scale) mutated filtered_orders (qty
reductions). Region/FX therefore made decisions against PRE-scaling exposures.

R4 fix (2af9227): factor out _recompute_exposures() closure; call after
sector loop and after region loop.

R6 test backfill (F-C4-N-5).
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

pytestmark = [pytest.mark.unit]


def _make_inputs():
    """Build minimal inputs for _ptc_check_group_exposures with both
    sector AND region caps configured (the configuration that exercises
    the recompute path)."""
    from src.assembled_core.execution.pre_trade_checks import PreTradeConfig

    config = PreTradeConfig(
        max_sector_exposure=0.30,  # 30% per sector
        max_region_exposure=0.40,  # 40% per region
        max_fx_exposure=None,
        missing_security_meta="warn",
    )

    filtered_orders = pd.DataFrame(
        [
            {"symbol": "AAPL", "side": "BUY", "qty": 100.0, "price": 150.0},
            {"symbol": "MSFT", "side": "BUY", "qty": 50.0, "price": 300.0},
        ]
    )
    orders_with_notional = filtered_orders.copy()

    current_positions = pd.DataFrame(columns=["symbol", "target_qty"])
    prices_latest = pd.DataFrame(
        [
            {"symbol": "AAPL", "close": 150.0},
            {"symbol": "MSFT", "close": 300.0},
        ]
    )
    equity = 100000.0
    security_meta_df = pd.DataFrame(
        [
            {"symbol": "AAPL", "sector": "Tech", "region": "US"},
            {"symbol": "MSFT", "sector": "Tech", "region": "US"},
        ]
    )

    return (
        config,
        filtered_orders,
        orders_with_notional,
        current_positions,
        prices_latest,
        equity,
        security_meta_df,
    )


def test_compute_exposures_called_at_least_3_times_F_A3_2() -> None:
    """With both sector AND region caps configured, exposures_df must be
    recomputed 3 times: initial + after sector + after region. The
    original bug computed it only ONCE.
    """
    from src.assembled_core.execution import pre_trade_checks as ptc

    (
        config,
        filtered_orders,
        orders_with_notional,
        current_positions,
        prices_latest,
        equity,
        security_meta_df,
    ) = _make_inputs()

    fake_exposures = pd.DataFrame(
        [
            {"symbol": "AAPL", "target_weight": 0.5, "gross_weight": 0.5},
            {"symbol": "MSFT", "target_weight": 0.3, "gross_weight": 0.3},
        ]
    )
    fake_group_df = pd.DataFrame(
        [{"group_value": "Tech", "gross_weight": 0.8}]
    )  # > both 0.30 sector and 0.40 region caps → both trigger scaling

    call_count = {"compute_exposures": 0}

    def spy_compute_exposures(*args, **kwargs):
        call_count["compute_exposures"] += 1
        return fake_exposures, None

    blocked_reasons: list = []
    reduced_orders: list = []
    summary: dict = {}

    with (
        (
            patch.object(
                ptc,
                "compute_exposures",
                side_effect=spy_compute_exposures,
                create=True,
            )
            if hasattr(ptc, "compute_exposures")
            else patch(
                "src.assembled_core.risk.exposure_engine.compute_exposures",
                side_effect=spy_compute_exposures,
            )
        ),
        patch(
            "src.assembled_core.risk.exposure_engine.compute_target_positions",
            return_value=pd.DataFrame(),
        ),
        patch(
            "src.assembled_core.risk.group_exposures.compute_group_exposures",
            return_value=(fake_group_df, None),
        ),
    ):
        ptc._ptc_check_group_exposures(
            filtered_orders=filtered_orders,
            orders_with_notional=orders_with_notional,
            blocked_reasons=blocked_reasons,
            reduced_orders=reduced_orders,
            summary=summary,
            config=config,
            current_positions=current_positions,
            prices_latest=prices_latest,
            equity=equity,
            security_meta_df=security_meta_df,
        )

    # F-A3-2 regression: exposures must be recomputed AT LEAST 3x
    # (initial + after sector loop + after region loop). Pre-fix: only 1.
    assert call_count["compute_exposures"] >= 3, (
        f"F-A3-2 regression: compute_exposures called only "
        f"{call_count['compute_exposures']}x — expected >=3 for "
        f"initial + post-sector + post-region recomputes"
    )


def test_no_recompute_when_no_caps_configured() -> None:
    """Optimization check: if no caps are configured at all, the function
    early-returns without computing exposures. Ensures we don't pay
    recompute cost unnecessarily."""
    from src.assembled_core.execution import pre_trade_checks as ptc
    from src.assembled_core.execution.pre_trade_checks import PreTradeConfig

    config = PreTradeConfig(
        max_sector_exposure=None,
        max_region_exposure=None,
        max_fx_exposure=None,
    )
    filtered_orders = pd.DataFrame([{"symbol": "AAPL", "side": "BUY", "qty": 100.0}])

    call_count = {"compute_exposures": 0}

    def spy(*args, **kwargs):
        call_count["compute_exposures"] += 1
        return pd.DataFrame(), None

    with patch(
        "src.assembled_core.risk.exposure_engine.compute_exposures",
        side_effect=spy,
    ):
        ptc._ptc_check_group_exposures(
            filtered_orders=filtered_orders,
            orders_with_notional=filtered_orders.copy(),
            blocked_reasons=[],
            reduced_orders=[],
            summary={},
            config=config,
            current_positions=None,
            prices_latest=None,
            equity=None,
            security_meta_df=None,
        )

    assert (
        call_count["compute_exposures"] == 0
    ), "Optimization: no caps configured → no compute_exposures calls expected"
