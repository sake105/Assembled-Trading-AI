"""QUAL/Zensus-1 sub-commit C1: _tc_execution audit-trail swallows become OBSERVABLE.

``book_fills`` in ``_tc_execution.py`` wrote its trade-journal and order-lifecycle
audit entries inside ``try: ... except Exception: log.debug("... skipped")``. At prod
log level a DEBUG swallow is invisible — a trade-journal write that silently failed
(QUAL-19, an AUDIT-TRAIL GAP) looks identical to one that succeeded.

This pins that a real failure of the journal write (or the nested FILLED-lifecycle
hook) routes through ``_record_degraded_step`` (WARN + structured
``result.meta['degraded_steps']`` trail) instead of a DEBUG swallow — WITHOUT
changing book_fills' fire-and-forget contract (the artifacts are best-effort; the
cycle result still returns). Two sites are exercised: the outer trade_journal
swallow and the nested order_lifecycle_filled swallow.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import src.assembled_core.ops.trade_journal as trade_journal
from src.assembled_core.pipeline._tc_execution import book_fills
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    TradingCycleResult,
)


def _signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    return signals


def _make_ctx(tmp_path: Path) -> TradingContext:
    ts = pd.Timestamp("2024-04-22", tz="UTC")
    prices = pd.DataFrame({"timestamp": [ts], "symbol": ["AAPL"], "close": [150.0]})
    ctx = TradingContext(
        prices=prices,
        as_of=ts,
        mode="backtest",
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        use_factor_store=False,
        write_outputs=True,
        capital=100_000.0,
        intel_sim_applied=True,
    )
    ctx.output_dir = tmp_path / "output"
    # execution_mode is a dynamic attr set by the orchestrator in production;
    # set it here so the journal block reaches the (monkeypatched) write call.
    ctx.execution_mode = "backtest"
    return ctx


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
    return r


def test_trade_journal_failure_recorded_not_blocking(
    monkeypatch, caplog, tmp_path
) -> None:
    """Force the trade-journal write to raise. The failure must land in
    result.meta['degraded_steps'] + a WARNING — and book_fills must still
    return the result (fire-and-forget artifact, not order-blocking)."""

    def _boom(*a, **k):
        raise RuntimeError("journal boom")

    monkeypatch.setattr(trade_journal, "append_trade_journal_entries", _boom)

    ctx = _make_ctx(tmp_path)
    result = _result_with_orders()

    with caplog.at_level(logging.WARNING):
        out = book_fills(result, ctx)

    tj_entries = [
        s for s in out.meta.get("degraded_steps", []) if s["step"] == "trade_journal"
    ]
    assert len(tj_entries) == 1
    assert "journal boom" in tj_entries[0]["error"]
    assert any(
        "[DEGRADED]" in r.message and "trade_journal" in str(r.args)
        for r in caplog.records
    )
    # fire-and-forget: result still returned, status untouched
    assert out is result
    assert out.status == "success"


def test_lifecycle_filled_failure_recorded_not_blocking(
    monkeypatch, caplog, tmp_path
) -> None:
    """The journal write succeeds, but the nested FILLED-lifecycle hook raises.
    It must be recorded as 'order_lifecycle_filled' + WARNING, and the
    trade_journal step must NOT be marked degraded (only the inner hook failed)."""

    monkeypatch.setattr(
        trade_journal, "append_trade_journal_entries", lambda *a, **k: None
    )

    import src.assembled_core.ops.order_lifecycle_log as oll

    def _boom(*a, **k):
        raise RuntimeError("lifecycle boom")

    monkeypatch.setattr(oll, "append_lifecycle_event", _boom)

    ctx = _make_ctx(tmp_path)
    result = _result_with_orders()

    with caplog.at_level(logging.WARNING):
        out = book_fills(result, ctx)

    steps = [s["step"] for s in out.meta.get("degraded_steps", [])]
    assert "order_lifecycle_filled" in steps
    assert "trade_journal" not in steps
    assert any(
        "[DEGRADED]" in r.message and "order_lifecycle_filled" in str(r.args)
        for r in caplog.records
    )
    assert out is result
