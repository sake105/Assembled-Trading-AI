"""Test that reject_reason is never empty when status=rejected."""

from __future__ import annotations

import pytest

import pandas as pd

from src.assembled_core.execution.fill_model import (
    REJECT_UNKNOWN,
    ensure_fill_schema,
)
from src.assembled_core.execution.fill_model_pipeline import apply_fill_model_pipeline


@pytest.mark.unit
def test_ensure_fill_schema_sets_unknown_when_rejected_and_reason_empty():
    """ensure_fill_schema: rejected row with empty reject_reason gets UNKNOWN."""
    ts = pd.Timestamp("2025-01-15 16:00", tz="UTC")
    trades = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 10.0,
                "price": 100.0,
            },
        ]
    )
    trades["fill_qty"] = 0.0
    trades["fill_price"] = 100.0
    trades["status"] = "rejected"
    trades["remaining_qty"] = 10.0
    trades["reject_reason"] = ""

    out = ensure_fill_schema(trades, default_full_fill=False)

    assert out["status"].iloc[0] == "rejected"
    assert out["reject_reason"].iloc[0] == REJECT_UNKNOWN


@pytest.mark.unit
def test_ensure_fill_schema_rejected_rows_have_non_empty_ascii_reason():
    """After ensure_fill_schema, every rejected row has non-empty ASCII reject_reason."""
    ts = pd.Timestamp("2025-01-15 16:00", tz="UTC")
    trades = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "A",
                "side": "BUY",
                "qty": 10.0,
                "price": 100.0,
            },
            {"timestamp": ts, "symbol": "B", "side": "BUY", "qty": 5.0, "price": 50.0},
        ]
    )
    trades["fill_qty"] = [0.0, 5.0]
    trades["fill_price"] = [100.0, 50.0]
    trades["status"] = ["rejected", "filled"]
    trades["remaining_qty"] = [10.0, 0.0]
    trades["reject_reason"] = ["", ""]

    out = ensure_fill_schema(trades, default_full_fill=False)

    rejected = out[out["status"] == "rejected"]
    assert len(rejected) == 1
    reason = rejected["reject_reason"].iloc[0]
    assert reason is not None and str(reason).strip() != ""
    assert reason.isascii()


@pytest.mark.unit
def test_pipeline_output_rejected_rows_have_non_empty_reason():
    """After fill pipeline, any rejected row has non-empty ASCII reject_reason."""
    dates = pd.date_range(start="2025-01-01", end="2025-01-10", freq="B", tz="UTC")
    prices = pd.DataFrame(
        [
            {
                "timestamp": d,
                "symbol": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1e6,
            }
            for d in dates
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "timestamp": dates[0],
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 1000.0,
                "price": 100.0,
            },
        ]
    )
    # available_cash too low -> cash gate will reject
    fills = apply_fill_model_pipeline(
        orders,
        prices=prices,
        freq="1d",
        available_cash=100.0,
        strict_session_gate=False,
    )
    rejected = fills[fills["status"] == "rejected"]
    for _, row in rejected.iterrows():
        rr = row.get("reject_reason", "")
        assert (
            rr is not None and str(rr).strip() != ""
        ), "rejected row must have non-empty reject_reason"
        assert str(rr).isascii()
