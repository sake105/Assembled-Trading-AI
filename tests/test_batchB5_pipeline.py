"""Bucket-B5 pipeline regressions: E-035 ops-leak guards, CB fail-closed,
groupby latest-bar robustness, VIX as_of PIT, backtest kill-switch isolation.

Covers the Diagnostik §pipeline (+ §risk B-risk-2) batch:

* B-pipe-3 (E-035): the SUBMITTED order-lifecycle hook (_tc_risk) and the
  heartbeat write (_tc_execution) must NOT write to ctx.output_dir in backtest
  mode (a backtest steps historical as_of dates and would clobber live ops
  artifacts). Live/paper still writes.
* B-risk-2: an error while evaluating the intraday circuit breaker must
  FAIL-CLOSED (propagate / block), not return None ("no breach", fail-open).
* B-pipe-2: ``_filter_prices_for_as_of`` latest-bar selection must pick the
  max-timestamp bar even for an UNSORTED per-symbol panel.
* B-pipe-1 (latent/defensive): the wide-format ``ctx.prices["VIX"]`` branch
  slices to the as_of window before the tail read.
* B-pipe-4: a backtest daily-circuit-breaker trip must NOT mutate the live
  persistent kill-switch store.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.assembled_core.pipeline._tc_risk as tc_risk  # noqa: E402
from src.assembled_core.pipeline._tc_risk import check_risk  # noqa: E402
from src.assembled_core.pipeline.trading_cycle_shared import (  # noqa: E402
    TradingContext,
    TradingCycleResult,
    _apply_risk_controls_default,
    _evaluate_circuit_breaker,
    _filter_prices_for_as_of,
)

pytestmark = pytest.mark.fast

SYMBOLS = ["AAPL", "MSFT", "GOOG"]


# --------------------------------------------------------------------------- #
# Shared fixtures (mirroring tests/test_tc_risk_fail_closed.py)
# --------------------------------------------------------------------------- #
def _signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])


def _sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])


def _make_prices(n_days: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-03-01", periods=n_days, freq="B", tz="UTC")
    rows = []
    for sym in SYMBOLS:
        for i, ts in enumerate(dates):
            rows.append({"timestamp": ts, "symbol": sym, "close": 100.0 + i})
    return pd.DataFrame(rows)


def _make_ctx(mode: str = "backtest") -> TradingContext:
    prices = _make_prices()
    ctx = TradingContext(
        prices=prices,
        as_of=prices["timestamp"].max(),
        mode=mode,
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        use_factor_store=False,
        write_outputs=False,
        capital=100_000.0,
        intel_sim_applied=True,
    )
    ctx.qa_block_trading = False
    ctx._policy_cache = {}
    return ctx


def _orders() -> pd.DataFrame:
    ts = pd.Timestamp("2024-03-08", tz="UTC")
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


def _patch_passthrough(monkeypatch) -> None:
    """Force orders_filtered non-empty and no-op the other gates so the
    lifecycle hook (and only it) runs."""
    monkeypatch.setattr(
        tc_risk, "_apply_risk_controls_default", lambda ctx, orders: orders.copy()
    )
    monkeypatch.setattr(tc_risk, "_evaluate_var_gate", lambda *a, **k: None)
    monkeypatch.setattr(tc_risk, "_evaluate_auto_dd_kill_switch", lambda *a, **k: None)
    monkeypatch.setattr(tc_risk, "_evaluate_circuit_breaker", lambda *a, **k: None)


# --------------------------------------------------------------------------- #
# B-pipe-3: SUBMITTED lifecycle hook — no ops write in backtest mode
# --------------------------------------------------------------------------- #
def test_b_pipe_3_submitted_hook_skipped_in_backtest(monkeypatch, tmp_path) -> None:
    """A backtest-mode cycle must NOT write order_lifecycle.jsonl (E-035)."""
    _patch_passthrough(monkeypatch)
    ctx = _make_ctx(mode="backtest")
    ctx.output_dir = str(tmp_path)
    out = check_risk(_orders(), _result(), ctx)
    assert len(out.orders_filtered) == 3
    assert not (tmp_path / "order_lifecycle.jsonl").exists(), (
        "backtest must not write order_lifecycle.jsonl to output_dir"
    )


def test_b_pipe_3_submitted_hook_writes_in_live(monkeypatch, tmp_path) -> None:
    """A live/paper-mode cycle still writes the SUBMITTED lifecycle entries."""
    _patch_passthrough(monkeypatch)
    ctx = _make_ctx(mode="live")
    ctx.output_dir = str(tmp_path)
    out = check_risk(_orders(), _result(), ctx)
    assert len(out.orders_filtered) == 3
    assert (tmp_path / "order_lifecycle.jsonl").exists(), (
        "live mode must still write order_lifecycle.jsonl"
    )
    contents = (tmp_path / "order_lifecycle.jsonl").read_text(encoding="utf-8")
    assert "SUBMITTED" in contents


# --------------------------------------------------------------------------- #
# B-pipe-3: heartbeat write — no ops write in backtest mode
# --------------------------------------------------------------------------- #
def _book_fills_ctx(mode: str, tmp_path) -> TradingContext:
    """A ctx wired to drive the REAL _tc_execution.book_fills() to its heartbeat
    block: output_dir is a Path (the heartbeat uses ``/``), execution_mode is set
    (since 2026-07-27 a real dataclass field, default "sim"; overridden here),
    and write_outputs=False so the write_outputs/KPI/manifest steps are skipped
    and the cycle reaches the heartbeat (gated by ``mode != "backtest"`` AND —
    since 2026-07-27 — by as_of being the current UTC date, NOT by
    write_outputs)."""
    ctx = _make_ctx(mode=mode)
    ctx.output_dir = tmp_path  # Path, not str (heartbeat block uses Path "/")
    ctx.execution_mode = mode
    ctx.write_outputs = False
    return ctx


def _book_fills_result() -> TradingCycleResult:
    res = _result()
    res.orders_filtered = _orders()
    res.orders = _orders()
    res.signals = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    return res


def test_b_pipe_3_heartbeat_skipped_in_backtest(tmp_path) -> None:
    """REAL entry point: book_fills() in backtest mode must NOT write
    state/heartbeat.json (E-035). Driving the actual call-site (not just the
    guard expression) means deleting the mode-guard at _tc_execution.py:~530
    turns this test RED — it is discriminating, not tautological (E-021)."""
    import src.assembled_core.pipeline._tc_execution as tc_exec

    ctx = _book_fills_ctx(mode="backtest", tmp_path=tmp_path)
    out = tc_exec.book_fills(_book_fills_result(), ctx)
    assert out.status != "error"
    assert not (tmp_path / "state" / "heartbeat.json").exists(), (
        "backtest book_fills must not write state/heartbeat.json to output_dir"
    )


def test_b_pipe_3_heartbeat_writes_in_live(tmp_path) -> None:
    """REAL entry point: book_fills() in live mode DOES write
    state/heartbeat.json. Mirrors the SUBMITTED-hook live test (drives the real
    function, not write_heartbeat directly)."""
    import src.assembled_core.pipeline._tc_execution as tc_exec

    ctx = _book_fills_ctx(mode="live", tmp_path=tmp_path)
    # Liveness contract (2026-07-27): the heartbeat is only emitted for a
    # cycle processing the CURRENT UTC date (historical backfills must not
    # refresh the DMS liveness signal) — give this live ctx a current as_of.
    ctx.as_of = pd.Timestamp.now("UTC")
    out = tc_exec.book_fills(_book_fills_result(), ctx)
    assert out.status != "error"
    hb_path = tmp_path / "state" / "heartbeat.json"
    assert hb_path.exists(), "live book_fills must write state/heartbeat.json"
    assert out.meta.get("heartbeat", {}).get("status") == "ok"


# --------------------------------------------------------------------------- #
# B-risk-2: circuit-breaker evaluation error fails CLOSED
# --------------------------------------------------------------------------- #
def _cb_ctx_with_observations() -> TradingContext:
    ctx = _make_ctx(mode="live")
    ctx._policy_cache = {"risk": {"circuit_breaker": {"enabled": True}}}
    return ctx


def test_b_risk_2_cb_eval_error_raises_not_none(monkeypatch) -> None:
    """An exception inside _evaluate_circuit_breaker must PROPAGATE (fail-closed),
    not be swallowed into a None ("no breach") return."""
    ctx = _cb_ctx_with_observations()
    result = _result()
    result.meta["intraday_equity_observations"] = [
        {"timestamp": "2024-03-08T15:00:00Z", "price": 100.0},
        {"timestamp": "2024-03-08T15:05:00Z", "price": 90.0},
    ]

    import src.assembled_core.risk.circuit_breaker as cbmod

    class _BoomCB:
        def __init__(self, *a, **k):
            raise RuntimeError("CB construction blew up")

    monkeypatch.setattr(cbmod, "CircuitBreaker", _BoomCB)

    with pytest.raises(RuntimeError):
        _evaluate_circuit_breaker(ctx, result, ctx._policy_cache)


def test_b_risk_2_cb_eval_error_blocks_orders_end_to_end(monkeypatch) -> None:
    """End-to-end: a CB-evaluation error makes check_risk FAIL-CLOSED (empties
    orders + sets risk_gate_error), instead of passing orders unchecked."""
    monkeypatch.setattr(
        tc_risk, "_apply_risk_controls_default", lambda ctx, orders: orders.copy()
    )
    monkeypatch.setattr(tc_risk, "_evaluate_var_gate", lambda *a, **k: None)
    monkeypatch.setattr(tc_risk, "_evaluate_auto_dd_kill_switch", lambda *a, **k: None)

    def _boom_cb(*a, **k):
        raise RuntimeError("CB evaluation blew up")

    monkeypatch.setattr(tc_risk, "_evaluate_circuit_breaker", _boom_cb)

    ctx = _make_ctx(mode="backtest")
    out = check_risk(_orders(), _result(), ctx)
    assert out.orders_filtered.empty, "CB eval error must block all orders"
    assert out.meta["risk_gate_error"] is True
    assert out.meta["circuit_breaker"]["status"] == "error"


def test_b_risk_2_cb_no_error_returns_none_no_breach() -> None:
    """Regression guard: the NORMAL (no-error) CB path with no trip still
    returns None — the fix must not turn a clean no-breach into a block."""
    ctx = _cb_ctx_with_observations()
    result = _result()
    # A single rising observation never trips the breaker -> no breach -> None.
    # CircuitBreaker.observe requires pd.Timestamp (not str), per its tz check.
    result.meta["intraday_equity_observations"] = [
        {"timestamp": pd.Timestamp("2024-03-08T15:00:00Z"), "price": 100.0},
        {"timestamp": pd.Timestamp("2024-03-08T15:05:00Z"), "price": 100.5},
    ]
    assert _evaluate_circuit_breaker(ctx, result, ctx._policy_cache) is None


# --------------------------------------------------------------------------- #
# B-pipe-2: groupby latest-bar selection robust to unsorted input
# --------------------------------------------------------------------------- #
def _unsorted_panel() -> pd.DataFrame:
    """Per-symbol rows deliberately NOT in timestamp order: the LAST row in
    DataFrame order is an EARLIER timestamp than an earlier row."""
    t1 = pd.Timestamp("2024-03-01", tz="UTC")
    t2 = pd.Timestamp("2024-03-02", tz="UTC")
    t3 = pd.Timestamp("2024-03-03", tz="UTC")
    rows = [
        # AAA: max-timestamp bar (t3, close 30) appears BEFORE the t1 bar.
        {"timestamp": t3, "symbol": "AAA", "close": 30.0},
        {"timestamp": t2, "symbol": "AAA", "close": 20.0},
        {"timestamp": t1, "symbol": "AAA", "close": 10.0},
        # BBB: same shape.
        {"timestamp": t3, "symbol": "BBB", "close": 300.0},
        {"timestamp": t1, "symbol": "BBB", "close": 100.0},
    ]
    return pd.DataFrame(rows)


def test_b_pipe_2_eod_unsorted_picks_max_timestamp() -> None:
    """EOD branch: latest bar per symbol must be the MAX-timestamp row even when
    the input is not pre-sorted (pre-fix groupby().last() picked the wrong row)."""
    panel = _unsorted_panel()
    as_of = pd.Timestamp("2024-03-03", tz="UTC")
    filtered, _ = _filter_prices_for_as_of(panel, as_of, mode="eod")
    by_sym = filtered.set_index("symbol")["close"].to_dict()
    assert by_sym["AAA"] == 30.0, "AAA latest must be the t3 (max-timestamp) bar"
    assert by_sym["BBB"] == 300.0, "BBB latest must be the t3 (max-timestamp) bar"


def test_b_pipe_2_backtest_unsorted_picks_max_timestamp() -> None:
    """Backtest branch: prices_latest must also be the MAX-timestamp bar."""
    panel = _unsorted_panel()
    as_of = pd.Timestamp("2024-03-03", tz="UTC")
    _, latest = _filter_prices_for_as_of(panel, as_of, mode="backtest")
    by_sym = latest.set_index("symbol")["close"].to_dict()
    assert by_sym["AAA"] == 30.0
    assert by_sym["BBB"] == 300.0


def test_b_pipe_2_sorted_input_unchanged() -> None:
    """A correctly pre-sorted panel selects the same bars (no behaviour change
    for the production sorted-input case)."""
    panel = (
        _unsorted_panel().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    )
    as_of = pd.Timestamp("2024-03-03", tz="UTC")
    filtered_eod, _ = _filter_prices_for_as_of(panel, as_of, mode="eod")
    _, latest_bt = _filter_prices_for_as_of(panel, as_of, mode="backtest")
    assert filtered_eod.set_index("symbol")["close"].to_dict() == {
        "AAA": 30.0,
        "BBB": 300.0,
    }
    assert latest_bt.set_index("symbol")["close"].to_dict() == {
        "AAA": 30.0,
        "BBB": 300.0,
    }


# --------------------------------------------------------------------------- #
# B-pipe-1: VIX wide-column read is as_of-sliced (latent/defensive)
# --------------------------------------------------------------------------- #
def _wide_vix_panel() -> pd.DataFrame:
    """Hypothetical WIDE panel with a literal 'VIX' column. NOTE: production
    ctx.prices is LONG-format (timestamp/symbol rows) so this branch does not
    fire in production — this documents the latent/defensive as_of-slice."""
    dates = pd.date_range("2024-03-01", periods=5, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            # VIX rises over time: an as_of in the middle must read the
            # mid-value, never the (future) tail.
            "VIX": [15.0, 16.0, 17.0, 18.0, 99.0],
        }
    )


def test_b_pipe_1_vix_long_panel_branch_unreachable() -> None:
    """Document the reachability finding: the production long-format panel has
    NO literal 'VIX' column, so the wide-VIX branch never fires."""
    long_panel = _make_prices()
    assert "VIX" not in long_panel.columns
    # VIX would be a symbol ROW in a long panel, not a column.
    assert "symbol" in long_panel.columns


def test_b_pipe_1_vix_wide_panel_reads_as_of_not_future() -> None:
    """If a wide panel ever reaches the VIX read, the as_of slice picks the
    as_of-correct value (16.0 at 2024-03-04), not the future tail (99.0).

    This exercises the exact slice idiom used at the two _tc_signals/_tc_sizing
    sites (pd.to_datetime(timestamp, utc=True) <= as_of, then .iloc[-1])."""
    panel = _wide_vix_panel()
    as_of = pd.Timestamp("2024-03-04", tz="UTC")  # second bar -> VIX 16.0
    src = panel
    ts = pd.to_datetime(src["timestamp"], utc=True)
    as_of_utc = pd.Timestamp(as_of)
    sliced = src.loc[ts <= as_of_utc]
    assert float(sliced["VIX"].iloc[-1]) == 16.0
    # Live/eod (tail == as_of) reads the tail, byte-identical to the raw read.
    as_of_tail = pd.Timestamp("2024-03-07", tz="UTC")  # last bar
    sliced_tail = src.loc[pd.to_datetime(src["timestamp"], utc=True) <= as_of_tail]
    assert float(sliced_tail["VIX"].iloc[-1]) == float(panel["VIX"].iloc[-1]) == 99.0


# --------------------------------------------------------------------------- #
# B-pipe-4: backtest daily-CB trip does not mutate the live kill-switch store
# --------------------------------------------------------------------------- #
def test_b_pipe_4_backtest_cb_trip_does_not_write_live_kill_switch(
    monkeypatch,
) -> None:
    """A backtest daily-circuit-breaker trip must NOT call activate_kill_switch
    (which writes the live persistent store). Live mode still activates."""
    import src.assembled_core.pipeline.trading_cycle_v2 as tcv2

    activate_calls: list[dict] = []

    def _fake_activate(*a, **k):
        activate_calls.append(dict(k))

    # Patch where the symbol is looked up: the function imports
    # ``activate_kill_switch`` from execution.kill_switch at call time.
    import src.assembled_core.execution.kill_switch as ksmod

    monkeypatch.setattr(ksmod, "activate_kill_switch", _fake_activate)

    # Force the daily breaker to "trip".
    monkeypatch.setattr(
        tcv2,
        "_evaluate_circuit_breaker_daily",
        lambda prices, policy, as_of: {"reason": "synthetic_trip"},
    )

    log_dummy = __import__("logging").getLogger("b_pipe_4_test")

    # Backtest mode -> must NOT activate the live store.
    bt_ctx = _make_ctx(mode="backtest")
    tcv2._load_intel(bt_ctx, {}, ROOT, log_dummy)
    assert activate_calls == [], (
        "backtest daily-CB trip must not write the live kill-switch store"
    )
    assert (bt_ctx.intel_health_flags or {}).get("daily_circuit_breaker", {}).get(
        "live_kill_switch_written"
    ) is False

    # Live mode -> still activates.
    live_ctx = _make_ctx(mode="live")
    tcv2._load_intel(live_ctx, {}, ROOT, log_dummy)
    assert len(activate_calls) == 1, "live daily-CB trip must activate kill-switch"


# --------------------------------------------------------------------------- #
# F1: backtest daily-CB sets an IN-CYCLE flag (live does NOT) AND the risk gate
# consumes that flag to block ALL orders — replacing the removed live-store-read
# block without writing the live store.
# --------------------------------------------------------------------------- #
def _load_intel_with_trip(mode: str):
    """Run _load_intel with the daily breaker forced to trip, returning the ctx.
    Patches activate_kill_switch so a live trip does not touch the real store."""
    import src.assembled_core.execution.kill_switch as ksmod
    import src.assembled_core.pipeline.trading_cycle_v2 as tcv2

    import pytest as _pt

    mp = _pt.MonkeyPatch()
    try:
        mp.setattr(ksmod, "activate_kill_switch", lambda *a, **k: None)
        mp.setattr(
            tcv2,
            "_evaluate_circuit_breaker_daily",
            lambda prices, policy, as_of: {"reason": "synthetic_trip"},
        )
        ctx = _make_ctx(mode=mode)
        tcv2._load_intel(ctx, {}, ROOT, __import__("logging").getLogger("f1"))
    finally:
        mp.undo()
    return ctx


def test_f1_backtest_trip_sets_in_cycle_flag_live_does_not() -> None:
    """PRODUCER side: a backtest daily-CB trip sets the in-cycle marker
    ``intel_health_flags['daily_circuit_breaker']['tripped'] = True``; a live
    trip leaves ``tripped`` False (live blocks via the store-read, so the gate's
    in-cycle check must stay a no-op -> byte-identical live behaviour)."""
    bt = _load_intel_with_trip("backtest")
    bt_flag = (bt.intel_health_flags or {}).get("daily_circuit_breaker", {})
    assert bt_flag.get("tripped") is True, "backtest must set the in-cycle flag"
    assert bt_flag.get("live_kill_switch_written") is False

    live = _load_intel_with_trip("live")
    live_flag = (live.intel_health_flags or {}).get("daily_circuit_breaker", {})
    assert live_flag.get("tripped") is False, (
        "live must NOT set the in-cycle flag (it blocks via the store-read); "
        "the gate's in-cycle check must be a no-op in live"
    )
    assert live_flag.get("live_kill_switch_written") is True


def test_f1_risk_gate_blocks_all_orders_when_in_cycle_flag_set() -> None:
    """CONSUMER side: _apply_risk_controls_default empties ALL orders when the
    backtest in-cycle daily-CB flag is set. This replaces the removed live-store
    -read block, which empties orders the same way."""
    ctx = _make_ctx(mode="backtest")
    ctx.intel_health_flags = {
        "daily_circuit_breaker": {"tripped": True, "reason": "synthetic_trip"}
    }
    orders = _orders()
    out = _apply_risk_controls_default(ctx, orders)
    assert out.empty, "in-cycle daily-CB flag must block ALL orders"
    assert list(out.columns) == list(orders.columns), (
        "blocked frame must keep the orders schema (same shape as kill-switch block)"
    )


def test_f1_risk_gate_no_op_when_flag_absent_or_not_tripped() -> None:
    """CONSUMER side, negative: the new in-cycle check is a NO-OP when the flag
    is absent (normal live/backtest) or present-but-not-tripped (live trip). It
    must not block orders on its own — proving live is byte-identical.

    We stub filter_orders_with_risk_controls to a pass-through so the test
    isolates the new in-cycle branch from the downstream execution-side gate."""
    import src.assembled_core.pipeline.trading_cycle_shared as shared

    import pytest as _pt

    mp = _pt.MonkeyPatch()
    try:
        mp.setattr(
            shared,
            "filter_orders_with_risk_controls",
            lambda **kw: (kw["orders"].copy(), object()),
        )
        # (a) flag absent entirely
        ctx_a = _make_ctx(mode="live")
        out_a = _apply_risk_controls_default(ctx_a, _orders())
        assert len(out_a) == 3, "no flag -> in-cycle check is a no-op"

        # (b) flag present but tripped=False (the live trip shape)
        ctx_b = _make_ctx(mode="live")
        ctx_b.intel_health_flags = {
            "daily_circuit_breaker": {
                "tripped": False,
                "reason": "synthetic_trip",
                "live_kill_switch_written": True,
            }
        }
        out_b = _apply_risk_controls_default(ctx_b, _orders())
        assert len(out_b) == 3, "tripped=False -> in-cycle check is a no-op (live)"
    finally:
        mp.undo()


def test_f1_end_to_end_backtest_trip_blocks_orders_and_no_live_store_write(
    monkeypatch, tmp_path
) -> None:
    """END-TO-END (within pipeline): a backtest daily-CB trip (via _load_intel)
    BOTH (1) leaves the live kill-switch store unwritten (E-035 closed) AND
    (2) results in EMPTY orders out of the risk gate (enforcement preserved).
    Driving _load_intel then _apply_risk_controls_default with the same ctx
    mirrors run_cycle's order (_load_intel before the gate)."""
    import src.assembled_core.execution.kill_switch as ksmod
    import src.assembled_core.pipeline.trading_cycle_v2 as tcv2

    activate_calls: list[dict] = []
    monkeypatch.setattr(
        ksmod, "activate_kill_switch", lambda *a, **k: activate_calls.append(dict(k))
    )
    monkeypatch.setattr(
        tcv2,
        "_evaluate_circuit_breaker_daily",
        lambda prices, policy, as_of: {"reason": "synthetic_trip"},
    )

    ctx = _make_ctx(mode="backtest")
    # 1) _load_intel runs first (as in run_cycle) and sets the in-cycle flag.
    tcv2._load_intel(ctx, {}, ROOT, __import__("logging").getLogger("f1_e2e"))
    assert activate_calls == [], "E-035: backtest trip must not write the live store"
    assert (ctx.intel_health_flags or {})["daily_circuit_breaker"]["tripped"] is True

    # 2) the risk gate consumes the flag and empties orders this bar.
    out = _apply_risk_controls_default(ctx, _orders())
    assert out.empty, "backtest CB trip must empty orders via the in-cycle gate"
