"""QUAL/Zensus-1 regression: silently-degraded pipeline steps become OBSERVABLE.

The ``_tc_*.py`` stages wrap optional / protective overlays in
``try: <step> ... except Exception: log.debug("... skipped")``. At the prod log
level a DEBUG-only swallow is invisible — a protective overlay that no-ops looks
identical to one that ran. ``_record_degraded_step`` promotes such a swallow to
WARN AND records a structured entry in ``result.meta['degraded_steps']`` so a
per-cycle QA consumer can see which overlays failed.

This pins two things:
1. the helper itself (WARN + structured trail; log-only when no meta);
2. that the soft tail-risk overlays in ``_tc_risk.check_risk`` route a real
   failure through the helper instead of swallowing it at DEBUG — WITHOUT
   blocking orders (these are EXTRA reductions on top of the independently
   fail-closed hard VaR/DD/CB gates, so fail-soft-but-visible is correct).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.assembled_core.pipeline._tc_risk as tc_risk  # noqa: E402
from src.assembled_core.pipeline._tc_risk import check_risk  # noqa: E402
from src.assembled_core.pipeline.trading_cycle_shared import (  # noqa: E402
    TradingContext,
    TradingCycleResult,
    _record_degraded_step,
)

SYMBOLS = ["AAPL", "MSFT", "GOOG"]


# --------------------------------------------------------------------------- #
# 1. _record_degraded_step — unit
# --------------------------------------------------------------------------- #


def test_record_warns_and_appends_structured_entry(caplog) -> None:
    meta: dict = {}
    with caplog.at_level(logging.WARNING):
        _record_degraded_step("copula_tail_risk", ValueError("bad fit"), meta=meta)
    # observable at WARNING (not DEBUG)
    assert any(
        "[DEGRADED]" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    )
    # structured trail
    assert meta["degraded_steps"] == [
        {"step": "copula_tail_risk", "error": "ValueError: bad fit"}
    ]


def test_record_accumulates_multiple_steps() -> None:
    meta: dict = {}
    _record_degraded_step("risk_shared_returns_pivot", ValueError("a"), meta=meta)
    _record_degraded_step("copula_tail_risk", KeyError("b"), meta=meta)
    assert [s["step"] for s in meta["degraded_steps"]] == [
        "risk_shared_returns_pivot",
        "copula_tail_risk",
    ]


def test_record_log_only_when_meta_none(caplog) -> None:
    # No meta in scope → must still WARN, must not raise.
    with caplog.at_level(logging.WARNING):
        _record_degraded_step("anti_churn_filters", RuntimeError("x"), meta=None)
    assert any("anti_churn_filters" in r.message for r in caplog.records)


def test_record_detail_in_message(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _record_degraded_step(
            "halt_check", RuntimeError("x"), meta={}, detail="3 symbols dropped"
        )
    assert any("3 symbols dropped" in r.getMessage() for r in caplog.records)


def test_record_does_not_clobber_existing_trail() -> None:
    meta: dict = {"degraded_steps": [{"step": "pre_existing", "error": "X"}]}
    _record_degraded_step("copula_tail_risk", ValueError("a"), meta=meta)
    assert [s["step"] for s in meta["degraded_steps"]] == [
        "pre_existing",
        "copula_tail_risk",
    ]


# --------------------------------------------------------------------------- #
# 2. check_risk integration: a soft overlay failure is recorded, not swallowed
# --------------------------------------------------------------------------- #


def _signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])


def _sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])


def _make_prices(n_days: int = 80) -> pd.DataFrame:
    # >= 60 return rows so the copula tail-risk overlay actually executes
    # (it requires >= 60 rows + 2..30 symbols).
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B", tz="UTC")
    rows = []
    for j, sym in enumerate(SYMBOLS):
        for i, ts in enumerate(dates):
            rows.append({"timestamp": ts, "symbol": sym, "close": 100.0 + i + j})
    return pd.DataFrame(rows)


def _make_ctx() -> TradingContext:
    prices = _make_prices()
    ctx = TradingContext(
        prices=prices,
        as_of=prices["timestamp"].max(),
        mode="backtest",
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        use_factor_store=False,
        write_outputs=False,
        capital=100_000.0,
        intel_sim_applied=True,
    )
    ctx.qa_block_trading = False
    ctx._policy_cache = {}  # empty policy → policy-gated steps skip
    return ctx


def _orders() -> pd.DataFrame:
    ts = pd.Timestamp("2024-04-22", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [ts] * 3,
            "symbol": SYMBOLS,
            "side": ["buy", "buy", "buy"],
            "qty": [100.0, 80.0, 60.0],
            "price": [150.0, 300.0, 140.0],
        }
    )


def _result() -> TradingCycleResult:
    return TradingCycleResult(
        run_id=None, timestamp=pd.Timestamp.now("UTC"), status="success"
    )


def _patch_gates_passthrough(monkeypatch) -> None:
    monkeypatch.setattr(
        tc_risk, "_apply_risk_controls_default", lambda ctx, orders: orders.copy()
    )
    monkeypatch.setattr(tc_risk, "_evaluate_var_gate", lambda *a, **k: None)
    monkeypatch.setattr(tc_risk, "_evaluate_auto_dd_kill_switch", lambda *a, **k: None)
    monkeypatch.setattr(tc_risk, "_evaluate_circuit_breaker", lambda *a, **k: None)


def test_copula_overlay_failure_recorded_not_blocking(monkeypatch, caplog) -> None:
    """Force the copula tail-risk overlay to raise. The failure must be
    recorded in meta['degraded_steps'] + logged at WARNING — and orders must
    NOT be blocked (soft overlay, not a hard gate)."""
    _patch_gates_passthrough(monkeypatch)

    import src.assembled_core.ml.copula_models as copula_models

    def _boom(*a, **k):
        raise RuntimeError("copula blew up")

    monkeypatch.setattr(copula_models, "compute_portfolio_tail_risk", _boom)

    with caplog.at_level(logging.WARNING):
        out = check_risk(_orders(), _result(), _make_ctx())

    # exactly the copula overlay degraded, carrying the forced error string
    # (membership alone would silently tolerate an unrelated step degrading too)
    copula_entries = [
        s for s in out.meta.get("degraded_steps", []) if s["step"] == "copula_tail_risk"
    ]
    assert len(copula_entries) == 1
    assert "copula blew up" in copula_entries[0]["error"]
    assert any(
        "[DEGRADED]" in r.message and "copula_tail_risk" in str(r.args)
        for r in caplog.records
    )
    # soft overlay → orders survive (not emptied), no hard-gate error flag
    assert len(out.orders_filtered) == 3
    assert out.meta.get("risk_gate_error") is not True
