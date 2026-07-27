"""E-059 #2: ``execution_mode`` wiring — TradingContext field + _tc_execution consumers.

Until 2026-07-27 ``TradingContext`` had NO ``execution_mode`` field and nothing
set it — the five consumer sites in ``_tc_execution.py`` (KPI artifact, run
manifest, run index ``ctx.equity``, trade journal ``signal_context``, heartbeat
``details``) raised AttributeError and died silently inside their enclosing
``except`` blocks. The trade-journal AttributeError even killed the whole
``append_trade_journal_entries`` call (audit-trail gap).

Pinned here:

* (a) the dataclass now carries ``execution_mode`` with default ``"sim"``;
* (b) the trade-journal block runs again and forwards ``execution_mode`` in
  ``signal_context`` (no ``trade_journal`` degraded step);
* (c) the heartbeat details contain ``execution_mode`` (mode != backtest);
* (d) regression: dynamically overwriting ``ctx.execution_mode`` (as older
  tests do) still wins over the default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.assembled_core.ops.trade_journal as trade_journal
from src.assembled_core.pipeline._tc_execution import book_fills
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    TradingCycleResult,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Helpers (ctx/result construction mirrors tests/test_batchB5_pipeline.py)
# --------------------------------------------------------------------------- #
def _signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    return signals


def _make_ctx(
    tmp_path: Path, mode: str = "backtest", as_of: pd.Timestamp | None = None
) -> TradingContext:
    ts = as_of if as_of is not None else pd.Timestamp("2024-04-22", tz="UTC")
    prices = pd.DataFrame({"timestamp": [ts], "symbol": ["AAPL"], "close": [150.0]})
    ctx = TradingContext(
        prices=prices,
        as_of=ts,
        mode=mode,
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        use_factor_store=False,
        write_outputs=True,
        capital=100_000.0,
        intel_sim_applied=True,
    )
    ctx.output_dir = tmp_path / "output"
    return ctx


# The heartbeat is a liveness signal: it is only emitted when the cycle
# processes the current wall-clock UTC date (see current-date guard in
# _tc_execution.py, review MAJOR-1). Heartbeat tests therefore need a
# current as_of; the historical default exercises the suppression path.
_NOW_UTC = pd.Timestamp.now("UTC")


def _result_with_orders() -> TradingCycleResult:
    ts = pd.Timestamp("2024-04-22", tz="UTC")
    r = TradingCycleResult(
        run_id=None, timestamp=pd.Timestamp.now("UTC"), status="success"
    )
    r.orders_filtered = pd.DataFrame(
        {
            "timestamp": [ts],
            "symbol": ["AAPL"],
            "side": ["buy"],
            "qty": [100.0],
            "price": [150.0],
        }
    )
    r.orders = r.orders_filtered.copy()
    r.signals = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    return r


def _capture_journal_calls(monkeypatch) -> list[dict]:
    """Replace append_trade_journal_entries with a recorder; returns the call list."""
    calls: list[dict] = []

    def _recorder(fills, signal_context=None, **kwargs):
        calls.append({"fills": fills, "signal_context": signal_context, **kwargs})
        return []

    monkeypatch.setattr(trade_journal, "append_trade_journal_entries", _recorder)
    return calls


# --------------------------------------------------------------------------- #
# (a) dataclass field + default
# --------------------------------------------------------------------------- #
def test_trading_context_default_execution_mode_is_sim(tmp_path) -> None:
    ctx = _make_ctx(tmp_path)
    assert ctx.execution_mode == "sim"


def test_trading_context_accepts_execution_mode_kwarg() -> None:
    ts = pd.Timestamp("2024-04-22", tz="UTC")
    prices = pd.DataFrame({"timestamp": [ts], "symbol": ["AAPL"], "close": [150.0]})
    ctx = TradingContext(prices=prices, as_of=ts, execution_mode="broker")
    assert ctx.execution_mode == "broker"


# --------------------------------------------------------------------------- #
# (b) trade-journal block no longer dies; signal_context carries execution_mode
# --------------------------------------------------------------------------- #
def test_trade_journal_signal_context_gets_default_execution_mode(
    monkeypatch, tmp_path
) -> None:
    """With fills present, the journal write must run (no AttributeError-kill,
    E-059) and forward the DEFAULT execution_mode without any dynamic injection."""
    calls = _capture_journal_calls(monkeypatch)
    ctx = _make_ctx(tmp_path)
    result = _result_with_orders()

    out = book_fills(result, ctx)

    assert out.status != "error"
    assert len(calls) == 1, "append_trade_journal_entries must be called exactly once"
    assert calls[0]["signal_context"]["execution_mode"] == "sim"
    degraded = [
        s for s in out.meta.get("degraded_steps", []) if s["step"] == "trade_journal"
    ]
    assert degraded == [], "trade_journal must not be degraded anymore (E-059)"


# --------------------------------------------------------------------------- #
# (c) heartbeat details contain execution_mode (mode != backtest)
# --------------------------------------------------------------------------- #
def test_heartbeat_details_contain_execution_mode(monkeypatch, tmp_path) -> None:
    _capture_journal_calls(monkeypatch)  # keep journal write off the real FS
    ctx = _make_ctx(tmp_path, mode="live", as_of=_NOW_UTC)
    ctx.write_outputs = False  # skip csv/KPI/manifest; heartbeat is mode-gated only
    result = _result_with_orders()

    out = book_fills(result, ctx)

    assert out.status != "error"
    hb_path = Path(ctx.output_dir) / "state" / "heartbeat.json"
    assert hb_path.exists(), "live book_fills must write state/heartbeat.json"
    payload = json.loads(hb_path.read_text(encoding="utf-8"))
    assert payload["details"]["execution_mode"] == "sim"


# --------------------------------------------------------------------------- #
# (d) regression: dynamic override (pre-field test pattern) still wins
# --------------------------------------------------------------------------- #
def test_dynamic_execution_mode_override_still_wins(monkeypatch, tmp_path) -> None:
    """Older tests set ctx.execution_mode dynamically after construction
    (tests/test_batchB5_pipeline.py, tests/test_qual_zensus1_execution_degraded.py).
    The override must reach both the journal signal_context and the heartbeat."""
    calls = _capture_journal_calls(monkeypatch)
    ctx = _make_ctx(tmp_path, mode="live", as_of=_NOW_UTC)
    ctx.execution_mode = "broker"
    result = _result_with_orders()

    out = book_fills(result, ctx)

    assert out.status != "error"
    assert ctx.execution_mode == "broker"
    assert calls and calls[0]["signal_context"]["execution_mode"] == "broker"
    hb_path = Path(ctx.output_dir) / "state" / "heartbeat.json"
    assert hb_path.exists()
    payload = json.loads(hb_path.read_text(encoding="utf-8"))
    assert payload["details"]["execution_mode"] == "broker"


# --------------------------------------------------------------------------- #
# (e) DMS guard: historical backfill must NOT refresh the live heartbeat
# --------------------------------------------------------------------------- #
def test_historical_as_of_does_not_write_heartbeat(monkeypatch, tmp_path) -> None:
    """A range backfill / experiment loop runs mode="eod" with a PAST as_of.
    It must not emit the liveness heartbeat — that would mask a real
    scheduler stall for the dead-man switch (review MAJOR-1)."""
    _capture_journal_calls(monkeypatch)
    ctx = _make_ctx(tmp_path, mode="eod")  # default as_of = 2024-04-22 (past)
    ctx.write_outputs = False
    result = _result_with_orders()

    out = book_fills(result, ctx)

    assert out.status != "error"
    hb_path = Path(ctx.output_dir) / "state" / "heartbeat.json"
    assert not hb_path.exists(), "historical cycle must not claim liveness"


def test_recent_as_of_within_grace_window_writes_heartbeat(
    monkeypatch, tmp_path
) -> None:
    """Midnight-straddle protection (review F-senior-1): a live cycle whose
    as_of was captured up to 2h ago must still emit its liveness beat even
    if the UTC date flipped in between. Deterministic proxy: as_of = now-90min
    is covered either by date-equality or by the grace window."""
    _capture_journal_calls(monkeypatch)
    ctx = _make_ctx(
        tmp_path, mode="live", as_of=pd.Timestamp.now("UTC") - pd.Timedelta(minutes=90)
    )
    ctx.write_outputs = False
    result = _result_with_orders()

    out = book_fills(result, ctx)

    assert out.status != "error"
    hb_path = Path(ctx.output_dir) / "state" / "heartbeat.json"
    assert hb_path.exists(), "live cycle within grace window must emit liveness"


# --------------------------------------------------------------------------- #
# (f) run index: no fake final_equity — column empty without a real producer
# --------------------------------------------------------------------------- #
def test_run_index_has_no_fake_final_equity(monkeypatch, tmp_path) -> None:
    """ctx.current_equity has no producer yet; the index must leave
    final_equity EMPTY instead of silently writing start capital under that
    label (review MAJOR-2)."""
    _capture_journal_calls(monkeypatch)
    ctx = _make_ctx(tmp_path, mode="eod")
    result = _result_with_orders()

    out = book_fills(result, ctx)

    assert out.status != "error"
    index_path = Path(ctx.output_dir) / "manifests" / "index.csv"
    assert index_path.exists(), "run index must be written (write_outputs=True)"
    df = pd.read_csv(index_path)
    assert df["n_fills"].iloc[0] == 1
    assert (
        df["final_equity"].isna().all()
        or (df["final_equity"].astype(str).str.strip() == "").all()
    ), "final_equity must stay empty without a real equity producer"


def test_run_index_uses_real_current_equity_when_present(monkeypatch, tmp_path) -> None:
    """A real (dynamically injected) current_equity lands in final_equity —
    including the valid 0.0 edge case (explicit None-check, no `or`)."""
    _capture_journal_calls(monkeypatch)
    ctx = _make_ctx(tmp_path, mode="eod")
    ctx.current_equity = 0.0  # total loss is a VALID equity value
    result = _result_with_orders()

    out = book_fills(result, ctx)

    assert out.status != "error"
    df = pd.read_csv(Path(ctx.output_dir) / "manifests" / "index.csv")
    assert float(df["final_equity"].iloc[0]) == 0.0
