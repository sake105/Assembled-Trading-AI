"""entry_ts schema for the zombie-killer hold-time check (E-059 #1 follow-up).

The 2026-07-27 operator flip (policy zombie_killer.enabled: true) was a silent
no-op at first: pilot ledger positions carried no ``entry_ts``, so
``check_zombie_position`` returned ``(False, "")`` for every position without
any log line — indistinguishable from "no zombies found".

Pinned here:
1. ledger lifecycle: entry_ts is set on position open (and side flips),
   preserved on same-side adds/partials, and gone after close+reopen cycles
   produce a fresh one;
2. _norm_position round-trips entry_ts (load/save);
3. pilot-realistic records WITHOUT entry_ts stay unflagged AND produce a loud
   warn-once (never a silent skip);
4. _prd_load_paper_state forwards entry_ts/entry_price/current_price.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

import src.assembled_core.risk.zombie_killer as zk_module
from src.assembled_core.ops.paper_ledger import (
    _norm_position,
    apply_fills_to_ledger,
)
from src.assembled_core.risk.zombie_killer import get_zombie_positions

pytestmark = pytest.mark.regression

_POLICY = {
    "zombie_killer": {
        "enabled": True,
        "shadow_only": True,
        "max_hold_days": 30,
    }
}


def _state(positions: dict | None = None, cash: float = 100_000.0) -> dict:
    return {
        "schema_version": 1,
        "updated_utc": None,
        "cash": cash,
        "positions": positions or {},
        "equity_curve": [],
    }


def _buy(symbol: str, qty: float, price: float, ts: str | None = None) -> dict:
    f: dict = {"symbol": symbol, "side": "BUY", "qty": qty, "price": price}
    if ts:
        f["timestamp"] = ts
    return f


def _sell(symbol: str, qty: float, price: float) -> dict:
    return {"symbol": symbol, "side": "SELL", "qty": qty, "price": price}


class TestLedgerEntryTsLifecycle:
    def test_open_sets_entry_ts_from_fill_timestamp(self) -> None:
        out = apply_fills_to_ledger(
            _state(), [_buy("AAPL", 10, 100.0, ts="2026-07-01T15:00:00+00:00")]
        )
        assert out["positions"]["AAPL"]["entry_ts"] == "2026-07-01T15:00:00+00:00"

    def test_open_without_fill_timestamp_uses_apply_time(self) -> None:
        out = apply_fills_to_ledger(_state(), [_buy("AAPL", 10, 100.0)])
        entry = out["positions"]["AAPL"].get("entry_ts")
        assert entry, "open-from-zero must always stamp an entry_ts"
        # parseable ISO UTC
        assert pd.Timestamp(entry).tzinfo is not None

    def test_add_preserves_entry_ts(self) -> None:
        s1 = apply_fills_to_ledger(
            _state(), [_buy("AAPL", 10, 100.0, ts="2026-07-01T15:00:00+00:00")]
        )
        s2 = apply_fills_to_ledger(s1, [_buy("AAPL", 5, 120.0)])
        assert s2["positions"]["AAPL"]["entry_ts"] == "2026-07-01T15:00:00+00:00"

    def test_partial_sell_preserves_entry_ts(self) -> None:
        s1 = apply_fills_to_ledger(
            _state(), [_buy("AAPL", 10, 100.0, ts="2026-07-01T15:00:00+00:00")]
        )
        s2 = apply_fills_to_ledger(s1, [_sell("AAPL", 4, 110.0)])
        assert s2["positions"]["AAPL"]["entry_ts"] == "2026-07-01T15:00:00+00:00"

    def test_close_and_reopen_gets_fresh_entry_ts(self) -> None:
        s1 = apply_fills_to_ledger(
            _state(), [_buy("AAPL", 10, 100.0, ts="2026-07-01T15:00:00+00:00")]
        )
        s2 = apply_fills_to_ledger(s1, [_sell("AAPL", 10, 110.0)])
        assert "AAPL" not in s2["positions"]
        s3 = apply_fills_to_ledger(
            s2, [_buy("AAPL", 3, 105.0, ts="2026-07-20T15:00:00+00:00")]
        )
        assert s3["positions"]["AAPL"]["entry_ts"] == "2026-07-20T15:00:00+00:00"

    def test_flip_long_to_short_stamps_new_entry_ts(self) -> None:
        s1 = apply_fills_to_ledger(
            _state(), [_buy("AAPL", 10, 100.0, ts="2026-07-01T15:00:00+00:00")]
        )
        f = _sell("AAPL", 15, 110.0)
        f["timestamp"] = "2026-07-10T15:00:00+00:00"
        s2 = apply_fills_to_ledger(s1, [f])
        pos = s2["positions"]["AAPL"]
        assert pos["qty"] == -5
        assert pos["entry_ts"] == "2026-07-10T15:00:00+00:00"

    def test_norm_position_round_trips_entry_ts(self) -> None:
        norm = _norm_position(
            {"qty": 10, "avg_price": 100.0, "entry_ts": "2026-07-01T15:00:00+00:00"}
        )
        assert norm["entry_ts"] == "2026-07-01T15:00:00+00:00"
        # legacy shape without entry_ts stays without one (never invented)
        assert "entry_ts" not in _norm_position({"qty": 10, "avg_price": 100.0})


class TestLegacyPositionsLoudSkip:
    def test_pilot_realistic_records_stay_unflagged_but_warn_once(self, caplog) -> None:
        """Pilot-shape records ({symbol, qty, target_qty} — no entry_ts) must
        not be flagged AND must not be a SILENT no-op (review H1/M1)."""
        zk_module._MISSING_ENTRY_TS_WARNED = False
        now = pd.Timestamp("2026-07-27", tz="UTC").to_pydatetime()
        positions = [
            {"symbol": "GLD", "qty": 10, "target_qty": 10},
            {"symbol": "TLT", "qty": 20, "target_qty": 20},
        ]

        with caplog.at_level(logging.WARNING):
            zombies = get_zombie_positions(positions, now, _POLICY)

        assert zombies == []
        warns = [r for r in caplog.records if "entry_ts" in r.message]
        assert len(warns) == 1, "missing entry_ts must be warned exactly once"
        assert "2 position(s)" in warns[0].message

        # second call: DEBUG only, no second WARN
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            zombies2 = get_zombie_positions(positions, now, _POLICY)
        assert zombies2 == []
        assert [r for r in caplog.records if "entry_ts" in r.message] == []

    def test_position_with_entry_ts_still_flagged(self) -> None:
        """The warn path must not swallow real zombies in the same batch."""
        zk_module._MISSING_ENTRY_TS_WARNED = False
        now = pd.Timestamp("2026-07-27", tz="UTC").to_pydatetime()
        positions = [
            {"symbol": "GLD", "qty": 10, "target_qty": 10},  # legacy, skipped
            {
                "symbol": "OLD",
                "qty": 5,
                "entry_ts": "2026-01-01T00:00:00+00:00",  # ~207 days > 30
            },
        ]
        zombies = get_zombie_positions(positions, now, _POLICY)
        assert [p["symbol"] for p, _ in zombies] == ["OLD"]


class TestPaperStateForwardsZombieInputs:
    def test_prd_load_paper_state_records_carry_zombie_fields(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.assembled_core.ops.paper_runner import _prd_load_paper_state

        # Seed a ledger with one post-schema position (entry_ts) and one legacy.
        ledger_dir = tmp_path / "output" / "runs" / "_paper_ledger"
        ledger_dir.mkdir(parents=True)
        import json

        (ledger_dir / "ledger_state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_utc": None,
                    "cash": 50_000.0,
                    "positions": {
                        "AAPL": {
                            "qty": 10.0,
                            "avg_price": 100.0,
                            "hwm": 120.0,
                            "entry_ts": "2026-07-01T15:00:00+00:00",
                        },
                        "GLD": {"qty": 5.0, "avg_price": 300.0, "hwm": 310.0},
                    },
                    "equity_curve": [],
                }
            ),
            encoding="utf-8",
        )
        as_of = pd.Timestamp("2026-07-27", tz="UTC")
        prices = pd.DataFrame(
            {
                "timestamp": [as_of - pd.Timedelta(days=1)] * 2,
                "symbol": ["AAPL", "GLD"],
                "close": [111.0, 305.0],
            }
        )

        _state_out, _path, _eq, pos_df, _es, _eci = _prd_load_paper_state(
            "paper", {"paper_runner": {}}, prices, as_of, tmp_path, 100_000.0
        )

        recs = {r["symbol"]: r for r in pos_df.to_dict("records")}
        assert recs["AAPL"]["entry_ts"] == "2026-07-01T15:00:00+00:00"
        assert recs["AAPL"]["entry_price"] == 100.0
        assert recs["AAPL"]["current_price"] == 111.0
        # legacy position: entry_ts absent/NaN, prices still forwarded
        assert (
            not isinstance(recs["GLD"].get("entry_ts"), str)
            or not recs["GLD"]["entry_ts"]
        )
        assert recs["GLD"]["entry_price"] == 300.0
        assert recs["GLD"]["current_price"] == 305.0
