"""FU2 — pipeline + risk follow-up regression tests.

Four small safety/observability fixes, each independently scoped:

* **FIX 1** trading_cycle_shared._evaluate_circuit_breaker: a CB module IMPORT
  failure while CB is ENABLED must FAIL-CLOSED (raise → consumer blocks orders),
  not fail-OPEN (return None → "no breach"). CB DISABLED stays unchanged (None).
* **FIX 2** trading_cycle_v2._load_intel daily circuit breaker: on a NON-trip bar
  the in-cycle ``daily_circuit_breaker["tripped"]`` flag is explicitly reset to
  False, so a caller reusing one ctx across bars cannot carry a stale tripped=True
  from an earlier trip bar into a non-trip bar (which would wrongly block orders).
* **FIX 3** risk.circuit_breaker.VolCircuitBreaker.check_returns: accepts an
  optional ``timestamp`` and stamps ``_tripped_at`` with it (replay PIT-correct);
  default None → wall-clock now() (byte-identical live).
* **FIX 4** orchestrator reconcile-blocked manifest: carries the on-disk reconcile
  report path + a specific ``failure_reason`` (so the CLI message is not "unknown")
  even though ledger_result is None on that path; a healthy run is unchanged.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_LOG = logging.getLogger("test_fu2_pipeline_risk")

import src.assembled_core.pipeline.trading_cycle_shared as tcs  # noqa: E402
from src.assembled_core.pipeline import trading_cycle_v2 as tc_v2  # noqa: E402
from src.assembled_core.pipeline.orchestrator import (  # noqa: E402
    _eo_build_manifest,
)
from src.assembled_core.pipeline.trading_cycle_shared import (  # noqa: E402
    TradingContext,
    TradingCycleResult,
)

pytestmark = pytest.mark.fast


# ===========================================================================
# FIX 1 — CB IMPORT failure fails CLOSED when CB is ENABLED
# ===========================================================================


def _signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])


def _sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])


def _cb_ctx() -> TradingContext:
    ts = pd.Timestamp("2024-03-08", tz="UTC")
    ctx = TradingContext(
        prices=pd.DataFrame({"timestamp": [ts], "symbol": ["SPY"], "close": [100.0]}),
        as_of=ts,
        mode="backtest",
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        use_factor_store=False,
        write_outputs=False,
        capital=100_000.0,
    )
    # Provide intraday observations so the evaluator reaches the import/construct
    # path (it early-returns None when there are no observations).
    ctx.intraday_equity_observations = [
        {"timestamp": ts, "price": 100.0},
        {"timestamp": ts + pd.Timedelta(minutes=1), "price": 90.0},
    ]
    return ctx


def _result() -> TradingCycleResult:
    return TradingCycleResult(
        run_id=None, timestamp=pd.Timestamp.now("UTC"), status="success"
    )


def test_fix1_cb_import_failure_enabled_fails_closed(monkeypatch) -> None:
    """CB ENABLED + module unimportable → must RAISE (fail-closed), not return
    None. The consumer (_tc_risk) only blocks on a raised exception; returning
    None here would silently disable a required breaker (fail-OPEN)."""
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__
    )

    def _boom_import(name, *args, **kwargs):
        if name == "src.assembled_core.risk.circuit_breaker":
            raise ImportError("simulated CB module unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _boom_import)

    policy = {"risk": {"circuit_breaker": {"enabled": True}}}
    with pytest.raises(ImportError):
        tcs._evaluate_circuit_breaker(_cb_ctx(), _result(), policy)


def test_fix1_cb_import_failure_disabled_returns_none(monkeypatch) -> None:
    """CB DISABLED → unchanged: returns None at the enabled-gate BEFORE the
    import is even attempted (no raise even if the module were unimportable)."""
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__
    )

    def _boom_import(name, *args, **kwargs):
        if name == "src.assembled_core.risk.circuit_breaker":
            raise ImportError("simulated CB module unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _boom_import)

    policy = {"risk": {"circuit_breaker": {"enabled": False}}}
    assert tcs._evaluate_circuit_breaker(_cb_ctx(), _result(), policy) is None


def test_fix1_cb_enabled_import_ok_no_trip_returns_none() -> None:
    """Sanity: CB ENABLED, module imports fine, no drop in observations → no
    breach → None (the import-failure fix does not alter the happy path)."""
    ts = pd.Timestamp("2024-03-08", tz="UTC")
    ctx = _cb_ctx()
    # Flat observations → no trip.
    ctx.intraday_equity_observations = [
        {"timestamp": ts, "price": 100.0},
        {"timestamp": ts + pd.Timedelta(minutes=1), "price": 100.0},
    ]
    policy = {"risk": {"circuit_breaker": {"enabled": True}}}
    assert tcs._evaluate_circuit_breaker(ctx, _result(), policy) is None


# ===========================================================================
# FIX 2 — daily_circuit_breaker in-cycle flag resets on a non-trip bar
# ===========================================================================


def _intel_ctx(closes: list[float]) -> TradingContext:
    dates = pd.date_range("2025-01-01", periods=len(closes), freq="D", tz="UTC")
    prices = pd.DataFrame(
        {"timestamp": dates, "symbol": ["SPY"] * len(closes), "close": closes}
    )
    return TradingContext(
        prices=prices,
        as_of=prices["timestamp"].max(),
        mode="backtest",
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        use_factor_store=False,
        write_outputs=False,
        capital=100_000.0,
    )


_CB_POLICY = {
    "circuit_breaker": {
        "enabled": True,
        "reference_symbol": "SPY",
        "drop_threshold_pct": 3.0,
    }
}


def test_fix2_trip_bar_sets_tripped_true(tmp_path: Path) -> None:
    """Baseline: a -4% drop in backtest sets the in-cycle tripped flag True."""
    ctx = _intel_ctx([100.0, 96.0])  # -4%
    tc_v2._load_intel(ctx, _CB_POLICY, tmp_path, _LOG)
    icb = ctx.intel_health_flags["daily_circuit_breaker"]
    assert icb["tripped"] is True


def test_fix2_nontrip_bar_resets_stale_tripped_on_reused_ctx(tmp_path: Path) -> None:
    """Same-ctx reuse across a TRIP bar then a NON-trip bar: the flag must be
    False on the non-trip bar (no stale block from the earlier trip)."""
    # Bar 1: trip.
    ctx = _intel_ctx([100.0, 96.0])  # -4%
    tc_v2._load_intel(ctx, _CB_POLICY, tmp_path, _LOG)
    assert ctx.intel_health_flags["daily_circuit_breaker"]["tripped"] is True

    # Bar 2: REUSE the same ctx (do NOT clear intel_health_flags), swap in a
    # calm price slice (-1%). Without the fix the stale tripped=True would carry.
    ctx.prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-02-01", periods=2, freq="D", tz="UTC"),
            "symbol": ["SPY", "SPY"],
            "close": [100.0, 99.0],  # -1%, no trip
        }
    )
    ctx.as_of = ctx.prices["timestamp"].max()
    tc_v2._load_intel(ctx, _CB_POLICY, tmp_path, _LOG)
    assert ctx.intel_health_flags["daily_circuit_breaker"]["tripped"] is False, (
        "non-trip bar must clear the stale tripped flag on a reused ctx"
    )


def test_fix2_fresh_ctx_nontrip_is_false(tmp_path: Path) -> None:
    """Canonical fresh-ctx path: a non-trip bar leaves tripped False (unchanged
    — a fresh ctx never carried a stale flag, and now it is explicit)."""
    ctx = _intel_ctx([100.0, 99.5])  # -0.5%
    tc_v2._load_intel(ctx, _CB_POLICY, tmp_path, _LOG)
    icb = ctx.intel_health_flags["daily_circuit_breaker"]
    assert icb["tripped"] is False


# ===========================================================================
# FIX 3 — VolCircuitBreaker.check_returns honours an explicit timestamp
# ===========================================================================


def _spiking_series() -> list[float]:
    calm = [0.001, -0.001] * 30  # 60 bars
    spike = [0.05, -0.06, 0.055, -0.058, 0.052]
    return calm + spike


def test_fix3_explicit_timestamp_stamps_tripped_at() -> None:
    """check_returns(..., timestamp=historical) stamps _tripped_at at THAT time
    (replay PIT-correct) — not wall-clock now()."""
    from src.assembled_core.risk.circuit_breaker import VolCircuitBreaker

    historical = datetime(2022, 3, 7, 14, 30, tzinfo=timezone.utc)
    vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.0)
    assert vcb.check_returns(_spiking_series(), timestamp=historical) is True
    assert vcb._tripped_at == historical
    # The PIT cooldown read agrees: as_of just after the trip is still in cooldown.
    assert vcb.is_tripped_at(historical + timedelta(minutes=1)) is True
    # As_of well past cooldown is no longer tripped (PIT-correct replay).
    assert vcb.is_tripped_at(historical + timedelta(hours=2)) is False


def test_fix3_naive_timestamp_assumed_utc() -> None:
    """A tz-naive explicit timestamp is treated as UTC (mirrors observe())."""
    from src.assembled_core.risk.circuit_breaker import VolCircuitBreaker

    naive = datetime(2022, 3, 7, 14, 30)  # no tzinfo
    vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.0)
    assert vcb.check_returns(_spiking_series(), timestamp=naive) is True
    assert vcb._tripped_at == naive.replace(tzinfo=timezone.utc)


def test_fix3_default_uses_wallclock_now_unchanged() -> None:
    """Default (no timestamp) → _tripped_at ~ now(UTC): byte-identical legacy
    behaviour (the param defaults to None)."""
    from src.assembled_core.risk.circuit_breaker import VolCircuitBreaker

    before = datetime.now(timezone.utc)
    vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.0)
    assert vcb.check_returns(_spiking_series()) is True
    after = datetime.now(timezone.utc)
    assert before <= vcb._tripped_at <= after


def test_fix3_no_trip_no_stamp_with_timestamp() -> None:
    """No trip → _tripped_at stays None even when a timestamp is passed."""
    from src.assembled_core.risk.circuit_breaker import VolCircuitBreaker

    calm = [0.001, -0.001] * 50  # 100 calm bars, ratio ~ 1.0
    vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.0)
    assert (
        vcb.check_returns(calm, timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc))
        is False
    )
    assert vcb._tripped_at is None


# ===========================================================================
# FIX 4 — reconcile-blocked manifest points at the report + specific reason
# ===========================================================================


def _now() -> datetime:
    return datetime(2025, 1, 15, 21, 30, 0, tzinfo=timezone.utc)


def test_fix4_blocked_manifest_carries_report_path_and_reason() -> None:
    """Reconcile-blocked manifest (ledger_result=None) carries the on-disk
    reconcile-report path + a specific failure_reason (not "unknown")."""
    manifest = _eo_build_manifest(
        freq="1d",
        start_capital=10000.0,
        data_snapshot_id="snap",
        completed_steps=["prices", "signals", "portfolio"],  # NO "ledger"
        qa={},
        ledger_result=None,
        started_at=_now(),
        finished_at=_now(),
        failure_flag=True,
        reconciliation_blocked=True,
        reconcile_report_path_blocked="reconcile_report_run_20250115_213000",
        failure_reason="reconciliation_blocked: SLO FAIL violations=[...]",
        base=Path("/tmp"),
    )
    assert manifest["reconciliation_blocked"] is True
    assert manifest["failure"] is True
    # The report path now points at the on-disk artifact (not None).
    assert manifest["reconcile_report_path"] is not None
    assert "reconcile_report_run_20250115_213000" in manifest["reconcile_report_path"]
    # The failure reason is specific — resolves to a concrete CLI message.
    assert manifest["failure_reason"] == (
        "reconciliation_blocked: SLO FAIL violations=[...]"
    )
    # Mirror the CLI resolution (run_eod_pipeline.py): failed_steps OR
    # failure_reason OR "unknown" — must NOT fall through to "unknown".
    resolved = (
        manifest.get("failed_steps") or manifest.get("failure_reason") or "unknown"
    )
    assert resolved != "unknown"


def test_fix4_healthy_manifest_unchanged() -> None:
    """A healthy run leaves reconcile fields at their existing values and
    failure_reason None (key present, value None) — no behaviour change."""
    manifest = _eo_build_manifest(
        freq="1d",
        start_capital=10000.0,
        data_snapshot_id="snap",
        completed_steps=["ledger"],
        qa={},
        ledger_result={"reconciliation_ok": True},
        started_at=_now(),
        finished_at=_now(),
        failure_flag=False,
        base=Path("/tmp"),
    )
    assert manifest["reconciliation_blocked"] is False
    assert manifest["failure"] is False
    assert manifest["failure_reason"] is None
    # No blocked report path threaded → reconcile_report_path resolves via the
    # ledger_result branch (None here, since the healthy stub omits the key).
    assert manifest["reconcile_report_path"] is None
