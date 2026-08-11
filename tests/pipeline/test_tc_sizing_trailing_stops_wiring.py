"""Regression: Trailing-Stops-pos_map erreicht compute_trailing_stops (2026-08-11).

Der DEGRADED-Pfad in ``_sp_apply_trailing_stops`` verschluckte seit jeher
einen AttributeError: Unterstrich-Spaltennamen (``_sym``/``_entry``) werden
von ``itertuples`` positional umbenannt — ``row._sym`` warf IMMER, die
Trailing-Stops wurden in JEDER Runde uebersprungen (beide pandas-
Versionen; Fund via CI-Log nach gh-Auth). Dieser Test stellt sicher, dass
die pos_map real gebaut und uebergeben wird — und dass der DEGRADED-Pfad
fuer diesen Fall nie wieder still zuschnappt.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

import src.assembled_core.pipeline._tc_sizing as tc_sizing
import src.assembled_core.risk.trailing_stops as ts_mod
from src.assembled_core.pipeline.trading_cycle_shared import TradingContext

pytestmark = pytest.mark.fast


def test_pos_map_erreicht_compute_trailing_stops(monkeypatch):
    ts = pd.Timestamp("2025-06-26", tz="UTC")
    prices = pd.DataFrame(
        {
            "timestamp": [ts] * 2,
            "symbol": ["AAPL", "MSFT"],
            "close": [100.0, 200.0],
            "high": [101.0, 202.0],
            "low": [99.0, 198.0],
        }
    )
    ctx = TradingContext(prices=prices, as_of=ts, write_outputs=False)
    ctx.current_positions = pd.DataFrame(
        {
            "symbol": ["aapl", "MSFT"],
            "qty": [10.0, 5.0],
            "entry_price": [90.0, 210.0],
        }
    )
    ctx.market_stress = None

    erhalten: dict = {}

    class _Res:
        triggered_symbols: list = []
        reduction_symbols: list = []

    def _fake_compute(pos_map, *a, **k):
        erhalten.update(pos_map)
        return _Res()

    monkeypatch.setattr(ts_mod, "compute_trailing_stops", _fake_compute)
    targets = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "target_weight": [0.5, 0.5]})
    meta: dict = {}

    log = logging.getLogger("test_ts_wiring")
    out = tc_sizing._sp_apply_trailing_stops(
        targets,
        ctx,
        prices,
        {"trailing_stops": {"enabled": True}},
        meta,
        log,
    )
    # Level-unabhaengig statt Log-Capture (F-senior-8): der DEGRADED-Pfad
    # hinterlaesst strukturell einen Eintrag in meta["degraded_steps"] —
    # genau der darf fuer trailing_stops nie wieder auftauchen.
    degraded = [
        d for d in (meta.get("degraded_steps") or []) if "trailing" in str(d).lower()
    ]
    assert not degraded, f"Trailing-Stops-Pfad degradiert: {degraded}"
    assert erhalten, "pos_map hat compute_trailing_stops nie erreicht"
    assert erhalten["AAPL"]["entry_price"] == pytest.approx(90.0)
    assert erhalten["AAPL"]["qty"] == pytest.approx(10.0)
    assert erhalten["MSFT"]["entry_price"] == pytest.approx(210.0)
    assert isinstance(out, pd.DataFrame)
