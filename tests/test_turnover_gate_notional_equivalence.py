"""Equivalence/unit-mix guard for the turnover budget gate (RISK path).

Pins the NOTIONAL contract of ``apply_turnover_gate``'s gated output: after a
cap-firing scale, the scaled ``target_qty`` (NOTIONAL dollars, == weight*capital
in the live ``_tc_sizing`` flow) must equal ``target_weight * portfolio_value``
element-wise.

Discrimination: prices are deliberately != 1.0 (137.0) and capital is 100_000 so
that the historical wrong-by-units bug — blending current SHARES (``cq``) with the
incoming NOTIONAL ``target_qty`` instead of blending current NOTIONAL
(``cq * price``) — produces a value off by a factor of ``price`` on the current
leg. With a non-trivial existing position the test FAILS against the buggy code
and PASSES once ``cq`` is converted to notional before blending.

Live consumption (why this is safety-critical, not cosmetic):
size_positions() -> route_orders() -> _generate_orders_default() reads the
``target_qty`` column directly (trading_cycle_shared.py) and
generate_orders_from_targets() computes target_shares = target_qty / price
(execution/order_generation.py). So a mixed-unit ``target_qty`` reaches live
orders on cap-firing days.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.risk.turnover_budget import (
    apply_turnover_gate,
    estimate_turnover,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]

# Non-trivial price (NOT 1.0) so a missing *price on the current leg is exposed.
PRICE = 137.0
# Non-trivial capital so target_qty (notional) != target_weight numerically.
CAPITAL = 100_000.0


def _notional_target(symbols: list[str], weights: list[float]) -> pd.DataFrame:
    """Build a target frame using the LIVE notional contract: target_qty = w*capital."""
    return pd.DataFrame(
        {
            "symbol": symbols,
            "target_weight": weights,
            "target_qty": [w * CAPITAL for w in weights],
        }
    )


def test_gated_target_qty_equals_weight_times_capital_on_cap_firing() -> None:
    """After a cap-firing scale, target_qty must == target_weight * portfolio_value.

    Uses an existing (non-zero SHARES) current position so the current leg is the
    discriminating term: the bug blends SHARES into a NOTIONAL quantity.
    """
    target = _notional_target(["A", "B"], [0.60, 0.40])

    # Current holds SHARES (live broker-sync convention). A is already partly held.
    current = pd.DataFrame({"symbol": ["A", "B"], "qty": [100.0, 0.0]})
    prices = pd.DataFrame({"symbol": ["A", "B"], "close": [PRICE, PRICE]})

    estimated = estimate_turnover(current, target, prices, portfolio_value=CAPITAL)
    cap = 0.15  # << estimated, so the gate fires and scales
    assert estimated > cap, "fixture must drive the cap-firing branch"

    gated, scale = apply_turnover_gate(
        target,
        current,
        cap=cap,
        estimated_turnover=estimated,
        behavior="scale",
        prices=prices,
        portfolio_value=CAPITAL,
    )
    assert 0.0 < scale < 1.0, "cap must actually fire (partial scale)"

    # The notional contract: target_qty == target_weight * portfolio_value, elementwise.
    # This FAILS on the buggy shares-blend (cq used as notional) and PASSES after fix.
    expected_qty = gated["target_weight"] * CAPITAL
    pd.testing.assert_series_equal(
        gated["target_qty"].reset_index(drop=True),
        expected_qty.reset_index(drop=True),
        check_names=False,
        rtol=1e-9,
        atol=1e-6,
    )


def test_gated_target_qty_notional_consistent_multi_symbol() -> None:
    """Same contract across several symbols with mixed existing holdings."""
    symbols = ["A", "B", "C"]
    weights = [0.50, 0.30, 0.20]
    target = _notional_target(symbols, weights)
    current = pd.DataFrame({"symbol": symbols, "qty": [200.0, 50.0, 0.0]})
    prices = pd.DataFrame({"symbol": symbols, "close": [PRICE, PRICE, PRICE]})

    estimated = estimate_turnover(current, target, prices, portfolio_value=CAPITAL)
    cap = 0.10
    assert estimated > cap

    gated, scale = apply_turnover_gate(
        target,
        current,
        cap=cap,
        estimated_turnover=estimated,
        behavior="scale",
        prices=prices,
        portfolio_value=CAPITAL,
    )
    assert 0.0 < scale < 1.0

    expected_qty = gated["target_weight"] * CAPITAL
    pd.testing.assert_series_equal(
        gated["target_qty"].reset_index(drop=True),
        expected_qty.reset_index(drop=True),
        check_names=False,
        rtol=1e-9,
        atol=1e-6,
    )


def test_non_cap_firing_path_byte_identical() -> None:
    """When estimated turnover <= effective cap the gate must not fire (no mutation)."""
    target = _notional_target(["A", "B"], [0.05, 0.05])
    current = pd.DataFrame({"symbol": ["A", "B"], "qty": [0.0, 0.0]})
    prices = pd.DataFrame({"symbol": ["A", "B"], "close": [PRICE, PRICE]})

    estimated = estimate_turnover(current, target, prices, portfolio_value=CAPITAL)
    cap = 0.15
    assert estimated <= cap, "fixture must stay below cap"

    gated, scale = apply_turnover_gate(
        target,
        current,
        cap=cap,
        estimated_turnover=estimated,
        behavior="scale",
        prices=prices,
        portfolio_value=CAPITAL,
    )
    assert scale == pytest.approx(1.0)
    # Untouched: target_qty equals the original notional input exactly.
    pd.testing.assert_frame_equal(
        gated.reset_index(drop=True), target.reset_index(drop=True)
    )
