"""Tier-1 wiring — verify accounting.currency FX converter is consumed by
``build_positions_from_ledger`` when a ``currency_map`` is supplied.

Passive addition: when ``currency_map`` is not provided the output schema
is unchanged. When provided, ``currency`` and ``usd_notional`` columns
appear and honour the FX rates.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = [pytest.mark.phase12]

from src.assembled_core.accounting.ledger import events_from_trades  # noqa: E402
from src.assembled_core.accounting.position_engine import (  # noqa: E402
    build_positions_from_ledger,
)


def _events() -> pd.DataFrame:
    trades = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-04-01 14:30:00", "2026-04-01 14:30:00"], utc=True
            ),
            "symbol": ["AAPL", "SAP.DE"],
            "side": ["BUY", "BUY"],
            "qty": [10.0, 20.0],
            "price": [150.0, 100.0],
            "fill_qty": [10.0, 20.0],
            "fill_price": [150.0, 100.0],
        }
    )
    return events_from_trades(trades, run_id="unit-test", source="test")


def test_currency_map_absent_preserves_legacy_schema() -> None:
    result = build_positions_from_ledger(_events())
    cols = set(result["positions_df"].columns)
    assert "usd_notional" not in cols
    assert "currency" not in cols


def test_currency_map_present_enriches_usd_notional() -> None:
    events = _events()
    prices = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-04-01T16:00:00", tz="UTC"),
             "symbol": "AAPL", "close": 150.0},
            {"timestamp": pd.Timestamp("2026-04-01T16:00:00", tz="UTC"),
             "symbol": "SAP.DE", "close": 100.0},
        ]
    )
    result = build_positions_from_ledger(
        events,
        prices_df=prices,
        currency_map={"AAPL": "USD", "SAP.DE": "EUR"},
        fx_rates={"USD": 1.0, "EUR": 1.10},
    )
    pos = result["positions_df"].set_index("symbol")

    assert "usd_notional" in pos.columns
    assert "currency" in pos.columns
    assert pos.loc["AAPL", "currency"] == "USD"
    assert pos.loc["SAP.DE", "currency"] == "EUR"

    # AAPL notional is in USD already (no FX shift)
    assert pos.loc["AAPL", "usd_notional"] == pytest.approx(
        pos.loc["AAPL", "notional"], rel=1e-9
    )
    # SAP.DE notional converted EUR → USD via 1.10x
    assert pos.loc["SAP.DE", "usd_notional"] == pytest.approx(
        pos.loc["SAP.DE", "notional"] * 1.10, rel=1e-9
    )
