"""Tests for turnover budget gate (INT-6.1): cap turnover, scale targets."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.risk.turnover_budget import (
    apply_turnover_gate,
    estimate_turnover,
)


pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def test_turnover_gate_no_change_when_below_cap() -> None:
    """When estimated turnover <= cap, targets unchanged, scale_factor 1.0."""
    target = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "target_weight": [0.5, 0.5],
            "target_qty": [50.0, 50.0],
        }
    )
    current = pd.DataFrame({"symbol": ["A", "B"], "qty": [40.0, 60.0]})
    prices = pd.DataFrame({"symbol": ["A", "B"], "close": [10.0, 10.0]})
    cap = 0.15
    portfolio_value = 1000.0
    estimated = estimate_turnover(current, target, prices, portfolio_value)
    # current weight A=0.4, B=0.6; target 0.5, 0.5; delta A=+0.1, B=-0.1; turnover = (0.1+0.1)/2 = 0.1
    assert estimated == pytest.approx(0.1)
    assert estimated <= cap
    new_targets, scale = apply_turnover_gate(
        target,
        current,
        cap=cap,
        estimated_turnover=estimated,
        behavior="scale",
        prices=prices,
        portfolio_value=portfolio_value,
    )
    assert scale == pytest.approx(1.0)
    assert new_targets["target_weight"].tolist() == [0.5, 0.5]


def test_turnover_gate_scales_when_above_cap() -> None:
    """When estimated turnover > cap, target deltas scaled down."""
    target = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "target_weight": [0.6, 0.4],
            "target_qty": [60.0, 40.0],
        }
    )
    current = pd.DataFrame({"symbol": ["A", "B"], "qty": [0.0, 0.0]})
    prices = pd.DataFrame({"symbol": ["A", "B"], "close": [10.0, 10.0]})
    portfolio_value = 1000.0
    estimated = estimate_turnover(current, target, prices, portfolio_value)
    # current weight 0,0; target 0.6, 0.4; delta 0.6, 0.4; turnover = (0.6+0.4)/2 = 0.5
    assert estimated == pytest.approx(0.5)
    cap = 0.15
    new_targets, scale = apply_turnover_gate(
        target,
        current,
        cap=cap,
        estimated_turnover=estimated,
        behavior="scale",
        prices=prices,
        portfolio_value=portfolio_value,
    )
    assert scale == pytest.approx(0.15 / 0.5)
    # new_target = 0 + scale * (0.6 - 0) = 0.18, 0 + scale * (0.4 - 0) = 0.12
    assert new_targets["target_weight"].iloc[0] == pytest.approx(0.18)
    assert new_targets["target_weight"].iloc[1] == pytest.approx(0.12)


def test_scale_factor_correct() -> None:
    """scale_factor = cap / estimated_turnover when above cap."""
    target = pd.DataFrame(
        {
            "symbol": ["X"],
            "target_weight": [1.0],
            "target_qty": [100.0],
        }
    )
    current = pd.DataFrame({"symbol": ["X"], "qty": [0.0]})
    prices = pd.DataFrame({"symbol": ["X"], "close": [10.0]})
    estimated = estimate_turnover(current, target, prices, portfolio_value=1000.0)
    assert estimated == pytest.approx(0.5)
    cap = 0.2
    _, scale = apply_turnover_gate(
        target,
        current,
        cap=cap,
        estimated_turnover=estimated,
        behavior="scale",
        prices=prices,
        portfolio_value=1000.0,
    )
    assert scale == pytest.approx(0.2 / 0.5)
