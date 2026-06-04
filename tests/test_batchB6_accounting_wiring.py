"""Batch B6 — accounting fail-closed *wiring* regression tests.

Batch B2 (test_batchB2_accounting_failclosed.py) proved the accounting layer
contract: build_ledger_from_trades escalates fail-closed (raises
ReconciliationError) on a live/paper SLO "fail" only when is_backtest=False,
and stays side-effect-free in backtest (is_backtest=True). That escalation was
DORMANT because the live EOD caller never passed is_backtest=False, and the
orchestrator step wrapped the call in a broad except that downgraded the error
to a soft {failed:True} flag.

This batch covers the WIRING that arms + propagates that mechanism:

* **B6-1** orchestrator._eo_step_ledger passes is_backtest=False (live/paper EOD).
* **B6-2** A ReconciliationError from build_ledger_from_trades is NOT swallowed
  into a soft {failed:True}: the step returns a HARD failure carrying
  reconciliation_blocked=True (completed=False, failed=True).
* **B6-3** A healthy live ledger step completes normally (unchanged behaviour).
* **B6-4** _eo_build_manifest surfaces reconciliation_blocked + failure in the
  run manifest, so the block is durably recorded.
* **B6-5** (E-035) the BACKTEST path (qa.backtest_engine._pb_build_ledger) on a
  diverging historical reconcile does NOT raise, does NOT fire an alert, and
  does NOT write the live ops reconcile audit — is_backtest=True end-to-end.

Scope honesty: this makes a live EOD reconcile SLO-fail a HARD, RECORDED,
ALERTED failure. It does NOT by itself block the next trading cycle's order
generation (kill-switch / reconcile-blocked pre-trade gate is a deliberate
operator decision, out of scope here).
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.accounting.broker_snapshot_store import (
    store_broker_snapshot_json,
    store_broker_snapshot_parquet,
)
from src.assembled_core.errors import ReconciliationError
from src.assembled_core.pipeline import orchestrator as orch_mod
from src.assembled_core.pipeline.orchestrator import (
    _eo_build_manifest,
    _eo_step_ledger,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_recon_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate_reconcile_slo appends to a JSONL audit path; redirect to tmp."""
    monkeypatch.setenv("ASSEMBLED_RECONCILE_AUDIT", str(tmp_path / "recon_audit.jsonl"))


def _minimal_trades() -> pd.DataFrame:
    base = datetime(2025, 1, 15, 10, 0, 0)
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(base, tz="UTC"),
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 5.0,
                "price": 150.0,
                "fill_qty": 5.0,
                "fill_price": 150.0,
                "status": "filled",
                "total_cost_cash": 0.0,
            }
        ]
    )


def _seed_diverging_snapshot(
    output_dir: Path, snapshot_run_id: str, as_of: pd.Timestamp
) -> None:
    """Store a broker snapshot that diverges hard from the paper view, so BOTH
    the cash and position SLO fail thresholds are breached (mirrors B2)."""
    positions = pd.DataFrame({"symbol": ["AAPL"], "qty": [50.0]})
    store_broker_snapshot_json(
        cash=1000.0,
        positions_df=positions,
        output_dir=output_dir,
        run_id=snapshot_run_id,
        as_of_date=as_of,
    )
    store_broker_snapshot_parquet(
        positions_df=positions,
        output_dir=output_dir,
        run_id=snapshot_run_id,
        as_of_date=as_of,
    )


def _call_eo_step_ledger(base: Path, snapshot_run_id: str | None = None) -> dict:
    """Invoke the orchestrator ledger step with minimal live/paper inputs."""
    return _eo_step_ledger(
        freq="1d",
        base=base,
        started_at=datetime(2025, 1, 15, 21, 30, 0, tzinfo=timezone.utc),
        start_capital=10000.0,
        portfolio_trades_df=_minimal_trades(),
        broker_snapshot_policy="prefer" if snapshot_run_id else "ignore",
        broker_snapshot_file=None,
        broker_snapshot_date=None,
        broker_snapshot_run_id=snapshot_run_id,
        write_evidence_pack=False,
        write_paper_broker_snapshot=False,
    )


# ---------------------------------------------------------------------------
# B6-1 — orchestrator arms the live escalation (is_backtest=False)
# ---------------------------------------------------------------------------


def test_orchestrator_passes_is_backtest_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live/paper EOD ledger step must call build_ledger_from_trades with
    is_backtest=False — this ARMS the fail-closed raise for a real drift."""
    captured: dict[str, object] = {}

    def _spy_build(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {
            "ledger_pack_path": str(tmp_path / "pack"),
            "reconciliation_ok": True,
            "reconciliation_severity": "ok",
            "reconciliation_blocked": False,
        }

    monkeypatch.setattr(
        "src.assembled_core.accounting.ledger_integration.build_ledger_from_trades",
        _spy_build,
    )

    result = _call_eo_step_ledger(tmp_path)

    assert captured.get("is_backtest") is False, (
        "live/paper EOD ledger step must pass is_backtest=False"
    )
    assert result["completed"] is True
    assert result["failed"] is False


# ---------------------------------------------------------------------------
# B6-2 — ReconciliationError is NOT swallowed into a soft flag
# ---------------------------------------------------------------------------


def test_reconciliation_error_is_hard_block_not_soft_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A ReconciliationError raised by build_ledger_from_trades must surface as a
    HARD step failure (completed=False, failed=True, reconciliation_blocked=True)
    and be logged at ERROR — never downgraded to the broad-except soft flag."""

    def _raise_recon(**kwargs):  # noqa: ANN003
        raise ReconciliationError("synthetic live reconcile drift")

    monkeypatch.setattr(
        "src.assembled_core.accounting.ledger_integration.build_ledger_from_trades",
        _raise_recon,
    )

    with caplog.at_level(
        logging.ERROR, logger="src.assembled_core.pipeline.orchestrator"
    ):
        result = _call_eo_step_ledger(tmp_path)

    # HARD block, not soft {failed:True} only.
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["reconciliation_blocked"] is True
    assert "synthetic live reconcile drift" in str(result["reconciliation_error"])
    assert result["ledger_result"] is None

    # The block is logged at ERROR with the reconcile-blocked marker.
    blocked_logs = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "BLOCKED" in r.getMessage()
    ]
    assert blocked_logs, "reconcile block must be ERROR-logged"


def test_generic_exception_still_soft_flag_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NON-reconcile exception must keep the existing soft-flag behaviour
    (completed=False, failed=True, NO reconciliation_blocked key)."""

    def _raise_other(**kwargs):  # noqa: ANN003
        raise RuntimeError("unrelated ledger failure")

    monkeypatch.setattr(
        "src.assembled_core.accounting.ledger_integration.build_ledger_from_trades",
        _raise_other,
    )

    result = _call_eo_step_ledger(tmp_path)

    assert result["completed"] is False
    assert result["failed"] is True
    # Not a reconcile block — the generic path must not claim one.
    assert result.get("reconciliation_blocked") is not True


# ---------------------------------------------------------------------------
# B6-3 — healthy live reconcile completes normally (unchanged)
# ---------------------------------------------------------------------------


def test_healthy_live_reconcile_completes(tmp_path: Path) -> None:
    """A live/paper EOD ledger step with a paper-view reconcile (policy='ignore' →
    paper==paper) completes normally: no raise, no block, completed=True. Per
    B-acct-3 the reconcile is UNVERIFIED (no independent broker snapshot), so
    reconciliation_ok is None — NOT a healthy True pass — but it does NOT block
    trading (there is no real drift signal to act on)."""
    result = _call_eo_step_ledger(tmp_path, snapshot_run_id=None)

    assert result["completed"] is True
    assert result["failed"] is False
    assert result.get("reconciliation_blocked") is not True
    assert result["ledger_result"] is not None
    assert result["ledger_result"]["reconciliation_ok"] is None


# ---------------------------------------------------------------------------
# B6-4 — manifest surfaces reconciliation_blocked durably
# ---------------------------------------------------------------------------


def test_manifest_surfaces_reconciliation_blocked() -> None:
    """_eo_build_manifest records reconciliation_blocked=True + failure=True so a
    reconcile-blocked run is durable even though ledger_result is None."""
    now = datetime(2025, 1, 15, 21, 30, 0, tzinfo=timezone.utc)
    manifest = _eo_build_manifest(
        freq="1d",
        start_capital=10000.0,
        data_snapshot_id="snap",
        completed_steps=["prices", "signals", "portfolio"],  # NO "ledger"
        qa={},
        ledger_result=None,
        started_at=now,
        finished_at=now,
        failure_flag=True,
        reconciliation_blocked=True,
        base=Path("/tmp"),
    )
    assert manifest["reconciliation_blocked"] is True
    assert manifest["failure"] is True
    assert "ledger" not in manifest["completed_steps"]


def test_manifest_default_reconciliation_blocked_false() -> None:
    """Default (healthy) path leaves reconciliation_blocked False."""
    now = datetime(2025, 1, 15, 21, 30, 0, tzinfo=timezone.utc)
    manifest = _eo_build_manifest(
        freq="1d",
        start_capital=10000.0,
        data_snapshot_id="snap",
        completed_steps=["ledger"],
        qa={},
        ledger_result={"reconciliation_ok": True},
        started_at=now,
        finished_at=now,
        failure_flag=False,
        base=Path("/tmp"),
    )
    assert manifest["reconciliation_blocked"] is False
    assert manifest["failure"] is False


# ---------------------------------------------------------------------------
# B6-5 (E-035) — backtest path stays side-effect-free on a diverging reconcile
# ---------------------------------------------------------------------------


def test_backtest_ledger_step_no_raise_no_alert_no_ops_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qa.backtest_engine._pb_build_ledger on a hard-diverging historical
    reconcile must NOT raise, NOT fire AlertManager, and NOT write the live ops
    reconcile audit — proving is_backtest=True is threaded through (E-035)."""
    from src.assembled_core.qa.backtest_engine import _pb_build_ledger

    import src.assembled_core.accounting.reconciliation as recon_mod
    import src.assembled_core.ops.alerting as alerting_mod

    fire_calls: list[tuple] = []
    audit_calls: list[dict] = []

    monkeypatch.setattr(
        recon_mod, "_append_recon_audit", lambda event: audit_calls.append(event)
    )

    def _spy_fire(self, rule_name, payload):  # noqa: ANN001
        fire_calls.append((rule_name, payload))

    monkeypatch.setattr(alerting_mod.AlertManager, "fire", _spy_fire, raising=True)

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run_id = "snap_bt_wiring"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    _seed_diverging_snapshot(output_dir, snapshot_run_id, as_of)

    # No pytest.raises wrapper: a backtest historical drift must NOT raise.
    ledger_result = _pb_build_ledger(
        orders_df=_minimal_trades(),
        trades_df=_minimal_trades(),
        prices=pd.DataFrame(),
        start_capital=10000.0,
        include_ledger=True,
        run_id="run_bt_wiring",
        output_dir=output_dir,
        broker_snapshot_policy="prefer",
        write_broker_snapshot=False,
        broker_snapshot_run_id=snapshot_run_id,
        broker_snapshot_file=None,
        broker_snapshot_date=None,
        include_costs=True,
        write_evidence_pack=False,
    )

    # The drift was classified (fail surfaced) but never escalated/alerted.
    assert ledger_result is not None
    assert ledger_result.get("reconciliation_severity") == "fail"
    assert ledger_result.get("reconciliation_blocked") is False
    assert fire_calls == [], "backtest must NOT fire AlertManager on a drift"
    assert audit_calls == [], "backtest must NOT write the live ops reconcile audit"


def test_backtest_caller_passes_is_backtest_true_literal() -> None:
    """Static guard: the backtest ledger caller passes is_backtest=True
    explicitly (defends against a future refactor silently dropping it)."""
    from src.assembled_core.qa import backtest_engine as bt_mod

    src = inspect.getsource(bt_mod._pb_build_ledger)
    assert "is_backtest=True" in src, (
        "_pb_build_ledger must pass is_backtest=True explicitly (E-035)"
    )


def test_orchestrator_caller_passes_is_backtest_false_literal() -> None:
    """Static guard: the live orchestrator ledger caller passes is_backtest=False
    explicitly (defends the armed live escalation against a silent revert)."""
    src = inspect.getsource(orch_mod._eo_step_ledger)
    assert "is_backtest=False" in src, (
        "_eo_step_ledger must pass is_backtest=False explicitly (arms fail-closed)"
    )
