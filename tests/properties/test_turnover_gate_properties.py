"""Property-based tests for turnover budget gate (C20).

Invariants:
1. Monotonicity: larger cap → scale_factor >= smaller cap
2. scale_factor always in [0, 1]
3. Zero turnover → scale_factor = 1.0
4. estimate_turnover always >= 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.risk.turnover_budget import (  # noqa: E402
    apply_turnover_gate,
    estimate_turnover,
)

pytestmark = pytest.mark.phase12


def _make_target(symbols: list[str], weights: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"symbol": symbols, "target_weight": weights})


@given(
    w1=st.floats(min_value=0.01, max_value=0.5),
    w2=st.floats(min_value=0.01, max_value=0.5),
)
@settings(max_examples=50)
def test_estimate_turnover_nonneg(w1: float, w2: float) -> None:
    """estimate_turnover is always >= 0."""
    targets = _make_target(["A", "B"], [w1, w2])
    result = estimate_turnover(None, targets, None, 1.0)
    assert result >= 0.0


@given(
    turnover=st.floats(min_value=0.01, max_value=2.0),
    cap=st.floats(min_value=0.01, max_value=2.0),
)
@settings(max_examples=50)
def test_scale_factor_bounded(turnover: float, cap: float) -> None:
    """scale_factor is always in [0, 1]."""
    targets = _make_target(["A"], [0.5])
    _, sf = apply_turnover_gate(
        targets,
        None,
        cap=cap,
        estimated_turnover=turnover,
    )
    assert 0.0 <= sf <= 1.0


@given(
    cap_small=st.floats(min_value=0.01, max_value=1.0),
    cap_delta=st.floats(min_value=0.0, max_value=1.0),
    turnover=st.floats(min_value=0.01, max_value=2.0),
)
@settings(max_examples=50)
def test_monotonicity_larger_cap(
    cap_small: float, cap_delta: float, turnover: float
) -> None:
    """Larger cap → scale_factor >= smaller cap's scale_factor."""
    cap_large = cap_small + cap_delta
    targets = _make_target(["A"], [0.5])
    _, sf_small = apply_turnover_gate(
        targets,
        None,
        cap=cap_small,
        estimated_turnover=turnover,
    )
    _, sf_large = apply_turnover_gate(
        targets,
        None,
        cap=cap_large,
        estimated_turnover=turnover,
    )
    assert sf_large >= sf_small - 1e-9  # tolerance for float precision


def test_zero_turnover_no_scaling() -> None:
    """Zero estimated turnover → scale_factor = 1.0."""
    targets = _make_target(["A"], [0.5])
    _, sf = apply_turnover_gate(
        targets,
        None,
        cap=0.1,
        estimated_turnover=0.0,
    )
    assert sf == 1.0
