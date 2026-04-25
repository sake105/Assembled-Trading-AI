"""Tests for turnover budget module — M6 coverage (module already implemented in M3/M5).

Covers estimate_turnover and apply_turnover_gate directly (not via pre_trade_checks).

Covers:
- estimate_turnover: empty targets, no prices (→ inf), flat positions, delta computation
- apply_turnover_gate: below cap (no-op), above cap scale, above cap block
- scale behavior: proportional delta reduction
- block behavior: targets set to current
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

pytestmark = pytest.mark.phase12

from src.assembled_core.risk.turnover_budget import (
    apply_turnover_gate,
    estimate_turnover,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prices(data: dict[str, float]) -> pd.DataFrame:
    """Build minimal prices DataFrame: symbol -> latest close."""
    rows = [
        {
            "timestamp": pd.Timestamp("2026-03-30", tz="UTC"),
            "symbol": sym,
            "close": price,
        }
        for sym, price in data.items()
    ]
    return pd.DataFrame(rows)


def _targets(*pairs: tuple[str, float]) -> pd.DataFrame:
    """Build target positions DataFrame: (symbol, target_weight) pairs."""
    rows = [{"symbol": sym, "target_weight": w} for sym, w in pairs]
    return pd.DataFrame(rows)


def _current(*pairs: tuple[str, float]) -> pd.DataFrame:
    """Build current positions DataFrame: (symbol, qty) pairs."""
    rows = [{"symbol": sym, "qty": qty} for sym, qty in pairs]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# estimate_turnover
# ---------------------------------------------------------------------------


class TestEstimateTurnover:
    def test_empty_target_returns_zero(self):
        result = estimate_turnover(None, pd.DataFrame(), None)
        assert result == pytest.approx(0.0)

    def test_no_symbol_column_returns_zero(self):
        result = estimate_turnover(None, pd.DataFrame({"target_weight": [0.1]}), None)
        assert result == pytest.approx(0.0)

    def test_no_prices_returns_inf(self):
        # No prices → cannot compute current weights → inf to force full scaling
        targets = _targets(("GLD", 0.20))
        current = _current(("GLD", 10.0))  # has qty but no price
        result = estimate_turnover(current, targets, prices=None)
        assert math.isinf(result)

    def test_empty_prices_returns_inf(self):
        targets = _targets(("GLD", 0.20))
        current = _current(("GLD", 10.0))
        result = estimate_turnover(current, targets, prices=pd.DataFrame())
        assert math.isinf(result)

    def test_flat_transition_zero_turnover(self):
        # Current weight = target weight → no delta → zero turnover
        # GLD: qty=10, price=100 → weight=10*100/1000=1.0; target_weight=1.0
        prices = _prices({"GLD": 100.0})
        targets = _targets(("GLD", 1.0))
        current = _current(("GLD", 10.0))
        result = estimate_turnover(current, targets, prices, portfolio_value=1000.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_full_buy_from_cash_turnover(self):
        # Current: empty (all cash), target: GLD=0.30 → delta=0.30, turnover=0.15
        prices = _prices({"GLD": 100.0})
        targets = _targets(("GLD", 0.30))
        result = estimate_turnover(None, targets, prices, portfolio_value=10000.0)
        assert result == pytest.approx(0.30 / 2.0, rel=1e-6)

    def test_partial_delta_turnover(self):
        # Current: GLD weight = 0.10, target = 0.30 → delta = 0.20, turnover = 0.10
        prices = _prices({"GLD": 100.0})
        targets = _targets(("GLD", 0.30))
        # qty * price / pv = current_weight: qty=10, price=100, pv=10000 → weight=0.10
        current = _current(("GLD", 10.0))
        result = estimate_turnover(current, targets, prices, portfolio_value=10000.0)
        # delta = 0.30 - 0.10 = 0.20; turnover = 0.20/2 = 0.10
        assert result == pytest.approx(0.10, rel=1e-6)

    def test_multi_symbol_turnover_sums(self):
        prices = _prices({"GLD": 100.0, "TLT": 200.0})
        # GLD: current=0.0, target=0.20 → delta=0.20
        # TLT: current=0.0, target=0.10 → delta=0.10
        # turnover = (0.20 + 0.10) / 2 = 0.15
        targets = _targets(("GLD", 0.20), ("TLT", 0.10))
        result = estimate_turnover(None, targets, prices, portfolio_value=1000.0)
        assert result == pytest.approx(0.15, rel=1e-6)


# ---------------------------------------------------------------------------
# apply_turnover_gate
# ---------------------------------------------------------------------------


class TestApplyTurnoverGate:
    def test_empty_target_returns_unchanged(self):
        out, scale = apply_turnover_gate(
            pd.DataFrame(), None, cap=0.10, estimated_turnover=0.20
        )
        assert out.empty
        assert scale == pytest.approx(1.0)

    def test_zero_cap_no_scaling(self):
        targets = _targets(("GLD", 0.30))
        out, scale = apply_turnover_gate(
            targets, None, cap=0.0, estimated_turnover=0.50
        )
        assert scale == pytest.approx(1.0)

    def test_turnover_below_cap_returns_copy_unchanged(self):
        targets = _targets(("GLD", 0.30), ("TLT", 0.20))
        out, scale = apply_turnover_gate(
            targets, None, cap=0.50, estimated_turnover=0.10
        )
        assert scale == pytest.approx(1.0)
        assert out["target_weight"].tolist() == pytest.approx([0.30, 0.20])

    def test_turnover_above_cap_scales_weights(self):
        # estimated=0.30, cap=0.15 → scale=0.50
        prices = _prices({"GLD": 100.0, "TLT": 100.0})
        current = _current(("GLD", 0.0), ("TLT", 0.0))
        targets = _targets(("GLD", 0.30), ("TLT", 0.20))
        out, scale = apply_turnover_gate(
            targets,
            current,
            cap=0.15,
            estimated_turnover=0.30,
            behavior="scale",
            prices=prices,
            portfolio_value=1000.0,
        )
        assert scale == pytest.approx(0.15 / 0.30, rel=1e-6)
        # Scaled weights: from 0 + scale * (target - 0)
        assert out.loc[out["symbol"] == "GLD", "target_weight"].iloc[
            0
        ] == pytest.approx(0.30 * scale, rel=1e-6)
        assert out.loc[out["symbol"] == "TLT", "target_weight"].iloc[
            0
        ] == pytest.approx(0.20 * scale, rel=1e-6)

    def test_block_behavior_sets_targets_to_current(self):
        # Block: targets replaced with current weights
        prices = _prices({"GLD": 100.0})
        # Current: GLD qty=10, price=100, pv=1000 → weight=1.0
        current = _current(("GLD", 10.0))
        targets = _targets(("GLD", 0.30))
        out, scale = apply_turnover_gate(
            targets,
            current,
            cap=0.05,
            estimated_turnover=0.30,
            behavior="block",
            prices=prices,
            portfolio_value=1000.0,
        )
        assert scale == pytest.approx(0.0)
        # Target weight should be set to current weight (1.0)
        gld_weight = out.loc[out["symbol"] == "GLD", "target_weight"].iloc[0]
        assert gld_weight == pytest.approx(1.0, rel=1e-6)

    def test_scale_from_nonzero_current(self):
        # Current weight=0.10, target=0.30, cap=0.05, estimated_turnover=0.10 → scale=0.5
        # New target = 0.10 + 0.5 * (0.30 - 0.10) = 0.10 + 0.10 = 0.20
        prices = _prices({"GLD": 100.0})
        current = _current(("GLD", 10.0))  # 10 * 100 / 10000 = 0.10 weight
        targets = _targets(("GLD", 0.30))
        out, scale = apply_turnover_gate(
            targets,
            current,
            cap=0.05,
            estimated_turnover=0.10,
            behavior="scale",
            prices=prices,
            portfolio_value=10000.0,
        )
        assert scale == pytest.approx(0.5, rel=1e-6)
        expected_weight = 0.10 + 0.5 * (0.30 - 0.10)
        gld_weight = out.loc[out["symbol"] == "GLD", "target_weight"].iloc[0]
        assert gld_weight == pytest.approx(expected_weight, rel=1e-6)
