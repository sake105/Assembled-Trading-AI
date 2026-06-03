"""Batch-B9b: target_notional honesty + notional->shares sizing on the
position_sizing -> consumer path (Diagnostik §portfolio MAJOR a).

position_sizing.compute_target_positions* cannot produce shares (no prices), so it
emits a NOTIONAL column (= target_weight * total_capital). Historically that column
was named only `target_qty`, with a comment calling it a "placeholder" / "in units",
which invited downstream code to mistake it for a share count (the documented B1
failure mode). These tests lock in:

  1. The honest `target_notional` column is now emitted (= weight * capital).
  2. `target_qty` is retained as a value-identical ALIAS (notional, NOT shares).
  3. The order-generation consumer converts notional -> shares correctly
     (shares = target_notional / price), guarding price <= 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.execution.order_generation import (
    generate_orders_from_targets,
)
from src.assembled_core.portfolio.position_sizing import (
    compute_kelly_weights,
    compute_risk_parity_weights,
    compute_target_positions,
    compute_target_positions_from_trend_signals,
    compute_vol_scaled_weights,
)

pytestmark = pytest.mark.fast


@pytest.fixture
def long_signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "direction": ["LONG", "LONG", "LONG"],
            "score": [0.9, 0.6, 0.3],
        }
    )


# ---------------------------------------------------------------------------
# 1. position_sizing emits honest target_notional == weight * capital
# ---------------------------------------------------------------------------


class TestTargetNotionalColumn:
    def test_compute_target_positions_emits_target_notional(self, long_signals):
        cap = 10_000.0
        res = compute_target_positions(long_signals, total_capital=cap)
        assert "target_notional" in res.columns
        # notional == weight * capital (value unchanged from the old formula)
        assert np.allclose(res["target_notional"], res["target_weight"] * cap)

    def test_target_qty_is_value_identical_alias(self, long_signals):
        """target_qty is retained but must equal target_notional (NOT shares)."""
        res = compute_target_positions(long_signals, total_capital=50_000.0)
        assert "target_qty" in res.columns
        assert np.allclose(res["target_qty"], res["target_notional"])

    def test_notional_default_capital_equals_weight(self, long_signals):
        """With default total_capital=1.0 notional == weight (documented behaviour)."""
        res = compute_target_positions(long_signals)  # total_capital=1.0
        assert np.allclose(res["target_notional"], res["target_weight"])

    def test_empty_frame_has_target_notional_column(self):
        empty = pd.DataFrame(columns=["symbol", "direction"])
        res = compute_target_positions(empty)
        assert res.empty
        assert "target_notional" in res.columns
        assert "target_qty" in res.columns

    def test_trend_wrapper_emits_target_notional(self):
        sig = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "direction": ["LONG", "LONG"],
                "score": [0.8, 0.4],
            }
        )
        res = compute_target_positions_from_trend_signals(sig, total_capital=20_000.0)
        assert "target_notional" in res.columns
        assert np.allclose(res["target_notional"], res["target_qty"])
        assert res["target_notional"].sum() == pytest.approx(20_000.0)

    def test_kelly_emits_aligned_notional_alias(self, long_signals):
        res = compute_kelly_weights(long_signals, total_capital=10_000.0)
        assert "target_notional" in res.columns
        assert np.allclose(res["target_notional"], res["target_weight"] * 10_000.0)
        assert np.allclose(res["target_qty"], res["target_notional"])

    def test_risk_parity_emits_aligned_notional_alias(self, long_signals):
        vols = {"AAPL": 0.20, "MSFT": 0.30, "GOOGL": 0.40}
        res = compute_risk_parity_weights(long_signals, vols, total_capital=10_000.0)
        assert "target_notional" in res.columns
        assert np.allclose(res["target_notional"], res["target_weight"] * 10_000.0)
        assert np.allclose(res["target_qty"], res["target_notional"])

    def test_vol_scaled_emits_aligned_notional_alias(self, long_signals):
        vols = {"AAPL": 0.20, "MSFT": 0.30, "GOOGL": 0.40}
        res = compute_vol_scaled_weights(long_signals, vols, total_capital=10_000.0)
        assert "target_notional" in res.columns
        assert np.allclose(res["target_notional"], res["target_weight"] * 10_000.0)
        assert np.allclose(res["target_qty"], res["target_notional"])


# ---------------------------------------------------------------------------
# 2. order_generation converts notional -> shares correctly
#    (this is the safety-critical part: an order's qty must be in SHARES)
# ---------------------------------------------------------------------------


class TestNotionalToSharesConversion:
    def test_orders_qty_is_shares_not_notional(self):
        """A $10,000 notional at $100/share must produce 100 shares, not 10,000.

        This test FAILS against the pre-fix "placeholder-as-shares" interpretation:
        if order generation treated target_qty (=10,000 notional) as a share count,
        it would emit qty=10,000. The correct value is 10,000 / 100 = 100 shares.
        """
        target = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "target_weight": [1.0],
                "target_qty": [10_000.0],  # notional alias
                "target_notional": [10_000.0],
            }
        )
        prices = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "timestamp": [pd.Timestamp("2025-01-15", tz="UTC")],
                "close": [100.0],
            }
        )
        orders = generate_orders_from_targets(
            target_positions=target,
            current_positions=pd.DataFrame(columns=["symbol", "qty"]),
            timestamp=pd.Timestamp("2025-01-15", tz="UTC"),
            prices=prices,
        )
        assert len(orders) == 1
        assert orders["side"].iloc[0] == "BUY"
        # 10_000 notional / 100 price == 100 shares (NOT 10_000)
        assert orders["qty"].iloc[0] == pytest.approx(100.0)
        assert orders["qty"].iloc[0] != pytest.approx(10_000.0)
        assert orders.attrs.get("qty_unit") == "shares"

    def test_conversion_with_existing_position_delta_in_shares(self):
        """Delta must be computed in shares: target 100sh, hold 30sh -> BUY 70sh."""
        target = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "target_weight": [1.0],
                "target_qty": [10_000.0],
                "target_notional": [10_000.0],
            }
        )
        current = pd.DataFrame({"symbol": ["AAA"], "qty": [30.0]})
        prices = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "timestamp": [pd.Timestamp("2025-01-15", tz="UTC")],
                "close": [100.0],
            }
        )
        orders = generate_orders_from_targets(
            target_positions=target,
            current_positions=current,
            timestamp=pd.Timestamp("2025-01-15", tz="UTC"),
            prices=prices,
        )
        assert len(orders) == 1
        assert orders["side"].iloc[0] == "BUY"
        # target 100 shares - current 30 shares == 70 shares
        assert orders["qty"].iloc[0] == pytest.approx(70.0)

    def test_multi_symbol_per_price_conversion(self):
        target = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "target_weight": [0.5, 0.5],
                "target_qty": [5_000.0, 5_000.0],
                "target_notional": [5_000.0, 5_000.0],
            }
        )
        prices = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "timestamp": [pd.Timestamp("2025-01-15", tz="UTC")] * 2,
                "close": [50.0, 250.0],
            }
        )
        orders = generate_orders_from_targets(
            target_positions=target,
            current_positions=pd.DataFrame(columns=["symbol", "qty"]),
            timestamp=pd.Timestamp("2025-01-15", tz="UTC"),
            prices=prices,
        )
        qty_by_sym = dict(zip(orders["symbol"], orders["qty"]))
        assert qty_by_sym["AAA"] == pytest.approx(100.0)  # 5000 / 50
        assert qty_by_sym["BBB"] == pytest.approx(20.0)  # 5000 / 250


# ---------------------------------------------------------------------------
# 3. Guard: price <= 0 must not divide-by-zero / explode the size
# ---------------------------------------------------------------------------


class TestPriceGuard:
    def test_zero_price_produces_no_order_not_division_error(self):
        target = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "target_weight": [1.0],
                "target_qty": [10_000.0],
                "target_notional": [10_000.0],
            }
        )
        prices = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "timestamp": [pd.Timestamp("2025-01-15", tz="UTC")],
                "close": [0.0],  # bad price
            }
        )
        # Must not raise; price<=0 -> shares treated as 0 (no spurious order).
        orders = generate_orders_from_targets(
            target_positions=target,
            current_positions=pd.DataFrame(columns=["symbol", "qty"]),
            timestamp=pd.Timestamp("2025-01-15", tz="UTC"),
            prices=prices,
        )
        assert orders.empty or (orders["symbol"] == "AAA").sum() == 0

    def test_missing_price_produces_no_order(self):
        target = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "target_weight": [1.0],
                "target_qty": [10_000.0],
                "target_notional": [10_000.0],
            }
        )
        # No prices at all -> price defaults to 0.0 -> safe (no order, no crash)
        orders = generate_orders_from_targets(
            target_positions=target,
            current_positions=pd.DataFrame(columns=["symbol", "qty"]),
            timestamp=pd.Timestamp("2025-01-15", tz="UTC"),
            prices=None,
        )
        assert orders.empty or (orders["symbol"] == "AAA").sum() == 0


# ---------------------------------------------------------------------------
# 4. End-to-end: position_sizing output -> order_generation -> shares
# ---------------------------------------------------------------------------


def test_end_to_end_sizing_to_shares(long_signals):
    """compute_target_positions notional flows through to share-denominated orders."""
    targets = compute_target_positions(
        long_signals, total_capital=30_000.0, equal_weight=True
    )
    # 3 LONG names, equal weight -> 10,000 notional each
    assert np.allclose(targets["target_notional"], 10_000.0)

    prices = pd.DataFrame(
        {
            "symbol": ["AAPL", "GOOGL", "MSFT"],
            "timestamp": [pd.Timestamp("2025-01-15", tz="UTC")] * 3,
            "close": [200.0, 100.0, 400.0],
        }
    )
    orders = generate_orders_from_targets(
        target_positions=targets,
        current_positions=pd.DataFrame(columns=["symbol", "qty"]),
        timestamp=pd.Timestamp("2025-01-15", tz="UTC"),
        prices=prices,
    )
    qty_by_sym = dict(zip(orders["symbol"], orders["qty"]))
    assert qty_by_sym["AAPL"] == pytest.approx(50.0)  # 10000 / 200
    assert qty_by_sym["GOOGL"] == pytest.approx(100.0)  # 10000 / 100
    assert qty_by_sym["MSFT"] == pytest.approx(25.0)  # 10000 / 400
    assert orders.attrs.get("qty_unit") == "shares"
