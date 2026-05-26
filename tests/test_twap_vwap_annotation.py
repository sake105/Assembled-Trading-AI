"""Tests for Phase 17.86 TWAP/VWAP execution annotation in route_orders().

Verifies:
- algo_type and algo_n_slices columns added when policy enabled + mode in (live, paper)
- no annotation when disabled
- no annotation when mode is backtest
- VWAP algo_type annotation when policy specifies algo=VWAP
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd


def _make_orders(qty: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-02")],
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [qty],
            "price": [150.0],
        }
    )


def _make_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "target_weight": [0.5],
            "target_qty": [100.0],
        }
    )


def _make_ctx(mode: str = "paper") -> MagicMock:
    ctx = MagicMock()
    ctx.mode = mode
    ctx._policy_cache = None
    return ctx


def _route(policy: dict, mode: str = "paper", qty: float = 100.0) -> pd.DataFrame:
    """Run route_orders with full mocking of everything except Phase 17.86."""
    from src.assembled_core.pipeline._tc_execution import route_orders

    ctx = _make_ctx(mode=mode)
    ctx._policy_cache = policy

    targets = _make_targets()
    orders_out = _make_orders(qty=qty)

    with patch(
        "src.assembled_core.pipeline._tc_execution._generate_orders_default",
        return_value=orders_out,
    ):
        result = route_orders(targets, ctx, do_rebal=True)
    return result


def _route_multi(policy: dict, qtys: list, mode: str = "paper") -> pd.DataFrame:
    """Route with a multi-row orders DataFrame."""
    from src.assembled_core.pipeline._tc_execution import route_orders

    ctx = _make_ctx(mode=mode)
    ctx._policy_cache = policy

    targets = _make_targets()
    orders_out = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-02")] * len(qtys),
            "symbol": [f"SYM{i}" for i in range(len(qtys))],
            "side": ["BUY"] * len(qtys),
            "qty": qtys,
            "price": [100.0] * len(qtys),
        }
    )

    with patch(
        "src.assembled_core.pipeline._tc_execution._generate_orders_default",
        return_value=orders_out,
    ):
        result = route_orders(targets, ctx, do_rebal=True)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_route_orders_adds_algo_annotations_when_enabled():
    """Policy enabled + mode=paper → algo_type and algo_n_slices columns present."""
    policy = {
        "execution": {
            "algo_execution": {
                "enabled": True,
                "algo": "TWAP",
                "n_slices": 10,
                "participation_rate": 0.10,
            }
        }
    }
    result = _route(policy, mode="paper")

    assert "algo_type" in result.columns, "algo_type column missing"
    assert "algo_n_slices" in result.columns, "algo_n_slices column missing"
    assert result["algo_type"].iloc[0] == "TWAP"
    assert result["algo_n_slices"].iloc[0] == 10
    # Core columns must be unmodified
    assert result["qty"].iloc[0] == 100.0
    assert result["price"].iloc[0] == 150.0
    assert result["side"].iloc[0] == "BUY"


def test_route_orders_no_annotation_when_disabled():
    """Policy disabled → no algo columns added."""
    policy = {
        "execution": {
            "algo_execution": {
                "enabled": False,
                "algo": "TWAP",
                "n_slices": 10,
            }
        }
    }
    result = _route(policy, mode="paper")

    assert "algo_type" not in result.columns
    assert "algo_n_slices" not in result.columns


def test_route_orders_no_annotation_when_mode_backtest():
    """Policy enabled but mode=backtest → no annotation (live/paper only)."""
    policy = {
        "execution": {
            "algo_execution": {
                "enabled": True,
                "algo": "TWAP",
                "n_slices": 10,
            }
        }
    }
    result = _route(policy, mode="backtest")

    assert "algo_type" not in result.columns
    assert "algo_n_slices" not in result.columns


def test_vwap_annotation_type():
    """Policy with algo=VWAP → algo_type == 'VWAP'."""
    policy = {
        "execution": {
            "algo_execution": {
                "enabled": True,
                "algo": "VWAP",
                "n_slices": 5,
                "participation_rate": 0.15,
            }
        }
    }
    result = _route(policy, mode="live")

    assert "algo_type" in result.columns
    assert result["algo_type"].iloc[0] == "VWAP"
    assert result["algo_n_slices"].iloc[0] == 5


def test_algo_n_slices_clamped_to_qty():
    """qty < n_slices: effective slices clamped to int(qty), not n_slices."""
    policy = {
        "execution": {
            "algo_execution": {"enabled": True, "algo": "TWAP", "n_slices": 10}
        }
    }
    result = _route(policy, mode="paper", qty=3.0)

    assert result["algo_n_slices"].iloc[0] == 3


def test_nan_qty_falls_back_to_one_slice():
    """NaN qty → treated as qty=1 → n_slices clamped to 1."""
    policy = {
        "execution": {
            "algo_execution": {"enabled": True, "algo": "TWAP", "n_slices": 10}
        }
    }
    result = _route(policy, mode="paper", qty=float("nan"))

    assert "algo_n_slices" in result.columns
    assert result["algo_n_slices"].iloc[0] == 1


def test_inf_qty_does_not_crash():
    """inf qty → annotation survives; inf treated as unknown → falls back to 1 slice."""
    policy = {
        "execution": {
            "algo_execution": {"enabled": True, "algo": "TWAP", "n_slices": 10}
        }
    }
    result = _route(policy, mode="paper", qty=float("inf"))

    assert "algo_n_slices" in result.columns
    # inf → replace → NaN → fillna(1.0) → clamped to 1 slice (safe default)
    assert result["algo_n_slices"].iloc[0] == 1


def test_per_row_slice_count_with_mixed_qtys():
    """Multi-row: each row gets its own clamped slice count.

    qty=100 → 10, qty=3 → 3, qty=NaN → 1 (fallback), qty=-50 → 10 (abs).
    qty=inf → 1 (inf treated as unknown, same as NaN).
    """
    policy = {
        "execution": {
            "algo_execution": {"enabled": True, "algo": "TWAP", "n_slices": 10}
        }
    }
    result = _route_multi(policy, qtys=[100.0, 3.0, float("nan"), -50.0, float("inf")])

    assert list(result["algo_n_slices"]) == [10, 3, 1, 10, 1]
