"""E-059 #5: QuestDB write-through (Step 7.70) — real tick_store module API.

Until 2026-07-27 the write-through block imported a non-existent ``TickStore``
class from ``src.assembled_core.data.tick_store`` (the module exposes only
functions: ``ping``/``ensure_table``/``write_ticks``/...). The ImportError was
swallowed by the enclosing ``except`` — the write-through NEVER ran, even with
``questdb.write_through.enabled: true``.

Pinned here:

* (a) enabled + fills + ping OK → ``ensure_table`` then ``write_ticks`` is
  called with correctly populated ``OHLCVTick`` objects (price → o/h/l/c,
  |qty| → volume, symbol, tz-aware UTC ts);
* (b) ping False → no ensure_table/write_ticks;
* (c) disabled (or absent) policy → tick_store untouched;
* (d) zero/missing prices are skipped; no ticks → no write;
* (e) ensure_table False → no write_ticks (graceful skip);
* (f) policy ``url`` is not supported by tick_store (env-configured) → WARN
  logged, write-through still runs via env-configured module API.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

import src.assembled_core.data.tick_store as tick_store
from src.assembled_core.data.tick_store import OHLCVTick
from src.assembled_core.pipeline._tc_execution import book_fills
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    TradingCycleResult,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Helpers (ctx/result construction mirrors test_tc_execution_mode_wiring.py)
# --------------------------------------------------------------------------- #
def _signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    return signals


def _make_ctx(tmp_path: Path, policy: dict | None = None) -> TradingContext:
    ts = pd.Timestamp("2024-04-22", tz="UTC")
    prices = pd.DataFrame({"timestamp": [ts], "symbol": ["AAPL"], "close": [150.0]})
    ctx = TradingContext(
        prices=prices,
        as_of=ts,
        mode="backtest",
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        use_factor_store=False,
        write_outputs=False,  # keep artifact writers quiet; Step 7.70 is not gated on it
        capital=100_000.0,
        intel_sim_applied=True,
    )
    ctx.output_dir = tmp_path / "output"
    # book_fills reads the policy from ctx._policy_cache before load_policy()
    ctx._policy_cache = policy if policy is not None else {}
    return ctx


def _result_with_orders(rows: list[dict] | None = None) -> TradingCycleResult:
    ts = pd.Timestamp("2024-04-22", tz="UTC")
    if rows is None:
        rows = [
            {"symbol": "AAPL", "side": "buy", "qty": 100.0, "price": 150.0},
            {"symbol": "MSFT", "side": "sell", "qty": -20.0, "price": 410.5},
        ]
    r = TradingCycleResult(
        run_id=None, timestamp=pd.Timestamp.now("UTC"), status="success"
    )
    r.orders_filtered = pd.DataFrame(
        {
            "timestamp": [ts] * len(rows),
            "symbol": [x["symbol"] for x in rows],
            "side": [x["side"] for x in rows],
            "qty": [x["qty"] for x in rows],
            "price": [x["price"] for x in rows],
        }
    )
    r.orders = r.orders_filtered.copy()
    r.signals = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    return r


_ENABLED_POLICY: dict = {"questdb": {"write_through": {"enabled": True}}}


class _Spy:
    """Monkeypatch harness for the tick_store module functions."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        ping: bool = True,
        ensure_table: bool = True,
    ) -> None:
        self.ping_calls = 0
        self.ensure_calls = 0
        self.write_calls: list[list[OHLCVTick]] = []

        def _ping() -> bool:
            self.ping_calls += 1
            return ping

        def _ensure_table() -> bool:
            self.ensure_calls += 1
            return ensure_table

        def _write_ticks(ticks: list[OHLCVTick]) -> int:
            self.write_calls.append(ticks)
            return len(ticks)

        monkeypatch.setattr(tick_store, "ping", _ping)
        monkeypatch.setattr(tick_store, "ensure_table", _ensure_table)
        monkeypatch.setattr(tick_store, "write_ticks", _write_ticks)


# --------------------------------------------------------------------------- #
# (a) enabled + ping OK → write_ticks with correctly built OHLCVTick objects
# --------------------------------------------------------------------------- #
def test_enabled_writes_fill_ticks(monkeypatch, tmp_path) -> None:
    spy = _Spy(monkeypatch)
    ctx = _make_ctx(tmp_path, policy=_ENABLED_POLICY)
    result = _result_with_orders()

    out = book_fills(result, ctx)

    assert out.status == "success"
    assert spy.ping_calls == 1
    assert spy.ensure_calls == 1
    assert len(spy.write_calls) == 1
    ticks = spy.write_calls[0]
    assert [t.symbol for t in ticks] == ["AAPL", "MSFT"]
    aapl, msft = ticks
    assert isinstance(aapl, OHLCVTick)
    # fill price fans out to all four OHLC fields
    assert (aapl.open, aapl.high, aapl.low, aapl.close) == (150.0,) * 4
    assert (msft.open, msft.high, msft.low, msft.close) == (410.5,) * 4
    # volume = |qty| (sell qty was negative)
    assert aapl.volume == 100.0
    assert msft.volume == 20.0
    # tz-aware UTC timestamp
    for t in ticks:
        assert t.ts.tzinfo is not None
        assert (
            str(t.ts.tzinfo) in ("UTC", "utc") or t.ts.utcoffset().total_seconds() == 0
        )


# --------------------------------------------------------------------------- #
# (b) ping False → no ensure_table / write_ticks
# --------------------------------------------------------------------------- #
def test_ping_false_skips_write(monkeypatch, tmp_path) -> None:
    spy = _Spy(monkeypatch, ping=False)
    ctx = _make_ctx(tmp_path, policy=_ENABLED_POLICY)

    out = book_fills(_result_with_orders(), ctx)

    assert out.status == "success"
    assert spy.ping_calls == 1
    assert spy.ensure_calls == 0
    assert spy.write_calls == []


# --------------------------------------------------------------------------- #
# (c) disabled / absent policy → tick_store never touched
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"questdb": {}},
        {"questdb": {"write_through": {"enabled": False}}},
    ],
    ids=["no-policy", "no-write-through", "explicit-disabled"],
)
def test_disabled_touches_nothing(monkeypatch, tmp_path, policy) -> None:
    spy = _Spy(monkeypatch)
    ctx = _make_ctx(tmp_path, policy=policy)

    out = book_fills(_result_with_orders(), ctx)

    assert out.status == "success"
    assert spy.ping_calls == 0
    assert spy.ensure_calls == 0
    assert spy.write_calls == []


# --------------------------------------------------------------------------- #
# (d) zero/invalid prices are dropped; all-invalid → no write at all
# --------------------------------------------------------------------------- #
def test_zero_price_rows_are_skipped(monkeypatch, tmp_path) -> None:
    spy = _Spy(monkeypatch)
    ctx = _make_ctx(tmp_path, policy=_ENABLED_POLICY)
    result = _result_with_orders(
        rows=[
            {"symbol": "AAPL", "side": "buy", "qty": 10.0, "price": 150.0},
            {"symbol": "ZERO", "side": "buy", "qty": 10.0, "price": 0.0},
        ]
    )

    out = book_fills(result, ctx)

    assert out.status == "success"
    assert len(spy.write_calls) == 1
    assert [t.symbol for t in spy.write_calls[0]] == ["AAPL"]


def test_all_invalid_prices_no_write(monkeypatch, tmp_path) -> None:
    spy = _Spy(monkeypatch)
    ctx = _make_ctx(tmp_path, policy=_ENABLED_POLICY)
    result = _result_with_orders(
        rows=[{"symbol": "ZERO", "side": "buy", "qty": 10.0, "price": 0.0}]
    )

    out = book_fills(result, ctx)

    assert out.status == "success"
    assert spy.ping_calls == 1
    assert spy.ensure_calls == 0
    assert spy.write_calls == []


# --------------------------------------------------------------------------- #
# (e) ensure_table False → graceful skip, no write_ticks
# --------------------------------------------------------------------------- #
def test_ensure_table_false_skips_write(monkeypatch, tmp_path) -> None:
    spy = _Spy(monkeypatch, ensure_table=False)
    ctx = _make_ctx(tmp_path, policy=_ENABLED_POLICY)

    out = book_fills(_result_with_orders(), ctx)

    assert out.status == "success"
    assert spy.ensure_calls == 1
    assert spy.write_calls == []


# --------------------------------------------------------------------------- #
# (f) policy url is unsupported → WARN, but write-through still runs
# --------------------------------------------------------------------------- #
def test_policy_url_warns_but_still_writes(monkeypatch, tmp_path, caplog) -> None:
    spy = _Spy(monkeypatch)
    policy = {
        "questdb": {
            "write_through": {"enabled": True, "url": "postgres://qdb:8812/qdb"}
        }
    }
    ctx = _make_ctx(tmp_path, policy=policy)

    with caplog.at_level(logging.WARNING):
        out = book_fills(_result_with_orders(), ctx)

    assert out.status == "success"
    warn_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("write_through.url" in m and "ignored" in m for m in warn_msgs), (
        f"expected url-ignored WARN, got: {warn_msgs}"
    )
    # env-configured module API is still used despite the unusable url
    assert len(spy.write_calls) == 1
