"""Batch B2 — accounting fail-closed regression tests.

Covers two safety fixes in src/assembled_core/accounting:

* **B-acct-2** (tax_lots.close_lots): a FIFO over-close (qty_to_close exceeds the
  sum of open lots) must emit a LOUD WARNING naming the unmatched remainder and
  the matched portion, while still persisting the matched part. A normal close
  (sufficient lots) must NOT emit that warning.

* **B-acct-1** (ledger_integration.build_ledger_from_trades): a reconciliation
  drift that breaches the ReconcileSLO fail threshold must, in LIVE/PAPER mode
  (is_backtest=False), ESCALATE fail-closed (raise ReconciliationError) AFTER the
  reconciliation report is written; the SAME failing drift in BACKTEST mode
  (is_backtest=True, the safe default) must NOT escalate but still surface
  severity=="fail" + ERROR log. A passing reconciliation escalates in neither
  mode. E-035: escalation can never fire in backtest.

These tests build inputs against the REAL evaluate_reconcile_slo + reconcile
result shapes; only the broker/ledger frames are minimal synthetic data.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.accounting.broker_snapshot_store import (
    store_broker_snapshot_json,
    store_broker_snapshot_parquet,
)
from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades
from src.assembled_core.accounting.tax_lots import TaxLot, TaxLotStore
from src.assembled_core.errors import ReconciliationError

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Shared fixture: keep the reconciliation audit log out of the real output/ dir
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_recon_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate_reconcile_slo appends to a JSONL audit path; redirect to tmp."""
    monkeypatch.setenv("ASSEMBLED_RECONCILE_AUDIT", str(tmp_path / "recon_audit.jsonl"))


# ---------------------------------------------------------------------------
# B-acct-2 — FIFO over-close drops qty_remaining (now LOUD)
# ---------------------------------------------------------------------------


def _seed_store_with_one_lot(tmp_path: Path, symbol: str, qty: float) -> TaxLotStore:
    store = TaxLotStore(db_path=tmp_path / "tax_lots.db")
    store.add_lot(
        TaxLot.open_lot(
            symbol=symbol,
            qty=qty,
            price_usd=100.0,
            usd_eur_rate=1.0,
            trade_date=date(2025, 1, 2),
            trade_timestamp=datetime(2025, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )
    )
    return store


def test_overclose_emits_warning_and_persists_matched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Closing MORE shares than open lots → WARNING naming remainder + matched,
    and the matched portion is still persisted (realized P&L recorded)."""
    store = _seed_store_with_one_lot(tmp_path, "AAPL", qty=10.0)

    with caplog.at_level(
        logging.WARNING, logger="src.assembled_core.accounting.tax_lots"
    ):
        result = store.close_lots(
            symbol="AAPL",
            qty_to_close=15.0,  # 5 more than available
            exit_price_usd=110.0,
            usd_eur_rate=1.0,
            exit_date=date(2025, 6, 1),
        )

    # FIFOCloseResult unchanged: matched 10, 5 unmatched.
    assert result.qty_remaining == pytest.approx(5.0)
    assert sum(c["qty"] for c in result.lots_closed) == pytest.approx(10.0)

    over_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "over-close" in r.getMessage()
    ]
    assert len(over_warnings) == 1, "exactly one over-close WARNING expected"
    msg = over_warnings[0].getMessage()
    assert "AAPL" in msg
    # Names the unmatched remainder AND the matched portion.
    assert "qty_remaining=5.0" in msg.replace("5.000000", "5.0")
    assert "5.000000" in msg  # remainder
    assert "10.000000" in msg  # matched portion
    assert "UNDER-REPORTED" in msg

    # Matched portion persisted: realized P&L for the closed lot was booked.
    assert store.realized_pnl_for_year(2025) == pytest.approx(
        result.total_pnl_eur, abs=1e-6
    )
    assert result.total_pnl_eur != 0.0


def test_normal_close_emits_no_overclose_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A close with sufficient open lots emits NO over-close warning."""
    store = _seed_store_with_one_lot(tmp_path, "MSFT", qty=10.0)

    with caplog.at_level(
        logging.WARNING, logger="src.assembled_core.accounting.tax_lots"
    ):
        result = store.close_lots(
            symbol="MSFT",
            qty_to_close=8.0,  # within available
            exit_price_usd=120.0,
            usd_eur_rate=1.0,
            exit_date=date(2025, 6, 1),
        )

    assert result.qty_remaining == pytest.approx(0.0)
    over_warnings = [r for r in caplog.records if "over-close" in r.getMessage()]
    assert over_warnings == [], "no over-close warning on a sufficient close"


# ---------------------------------------------------------------------------
# B-acct-1 — reconciliation fail-closed escalation (live/paper vs backtest)
# ---------------------------------------------------------------------------


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
    """Store a broker snapshot that diverges hard from the paper view.

    Paper view after a BUY of 5 @ 150 from start_cash=10000 holds ~9250 cash and
    5 AAPL. We store cash=1000 (huge cash_diff_bps) and 50 AAPL (45-share diff)
    so BOTH the cash and position SLO fail thresholds are breached.
    """
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


def _reconcile_report_exists(output_dir: Path) -> bool:
    """True iff at least one reconciliation report artifact was written."""
    hits = list(output_dir.rglob("reconcile*"))
    return len(hits) > 0


def test_failclosed_raises_in_live_and_writes_report_first(tmp_path: Path) -> None:
    """Live/paper (is_backtest=False) + SLO fail → ReconciliationError raised,
    AND the reconciliation report was written before the raise."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run_id = "snap_live"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    _seed_diverging_snapshot(output_dir, snapshot_run_id, as_of)

    with pytest.raises(ReconciliationError):
        build_ledger_from_trades(
            orders_df=_minimal_trades(),
            trades_df=_minimal_trades(),
            run_id="run_live",
            output_dir=output_dir,
            as_of_date=as_of,
            start_cash=10000.0,
            broker_snapshot_policy="prefer",
            broker_snapshot_run_id=snapshot_run_id,
            is_backtest=False,
        )

    # Report-before-raise: the failure must be recorded in the EOD pack.
    assert _reconcile_report_exists(output_dir), (
        "reconciliation report must be written BEFORE the fail-closed raise"
    )


def test_failclosed_backtest_does_not_raise_but_surfaces_fail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Backtest (is_backtest=True, default) + same SLO fail → NO raise, but
    severity=='fail' surfaced in the result and an ERROR was logged."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run_id = "snap_bt"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    _seed_diverging_snapshot(output_dir, snapshot_run_id, as_of)

    with caplog.at_level(
        logging.ERROR, logger="src.assembled_core.accounting.ledger_integration"
    ):
        result = build_ledger_from_trades(
            orders_df=_minimal_trades(),
            trades_df=_minimal_trades(),
            run_id="run_bt",
            output_dir=output_dir,
            as_of_date=as_of,
            start_cash=10000.0,
            broker_snapshot_policy="prefer",
            broker_snapshot_run_id=snapshot_run_id,
            is_backtest=True,  # explicit; also the safe default
        )

    # No raise, but the failure is fully surfaced.
    assert result["reconciliation_severity"] == "fail"
    assert result["reconciliation_ok"] is False
    assert result["reconciliation_blocked"] is False
    assert len(result["reconciliation_violations"]) >= 1
    error_records = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "SLO FAIL" in r.getMessage()
    ]
    assert error_records, "an ERROR-level SLO FAIL log is required in backtest"
    # Escalation log must NOT appear in backtest.
    assert not any("Escalating fail-closed" in r.getMessage() for r in caplog.records)
    assert _reconcile_report_exists(output_dir)


def test_default_mode_is_backtest_safe_no_raise(tmp_path: Path) -> None:
    """Unspecified mode (default) must behave as backtest — never escalate."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run_id = "snap_default"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    _seed_diverging_snapshot(output_dir, snapshot_run_id, as_of)

    # No is_backtest passed → default True → no raise even on a hard fail.
    result = build_ledger_from_trades(
        orders_df=_minimal_trades(),
        trades_df=_minimal_trades(),
        run_id="run_default",
        output_dir=output_dir,
        as_of_date=as_of,
        start_cash=10000.0,
        broker_snapshot_policy="prefer",
        broker_snapshot_run_id=snapshot_run_id,
    )
    assert result["reconciliation_severity"] == "fail"
    assert result["reconciliation_blocked"] is False


def test_passing_reconciliation_does_not_escalate_either_mode(tmp_path: Path) -> None:
    """A clean reconciliation (paper==paper via policy='ignore') → severity 'ok',
    no raise and no block, in BOTH live and backtest mode."""
    for is_bt in (True, False):
        output_dir = tmp_path / f"out_{is_bt}"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = build_ledger_from_trades(
            orders_df=_minimal_trades(),
            trades_df=_minimal_trades(),
            run_id=f"run_ok_{is_bt}",
            output_dir=output_dir,
            as_of_date=pd.Timestamp("2025-01-15", tz="UTC"),
            start_cash=10000.0,
            broker_snapshot_policy="ignore",  # paper == paper → ok
            is_backtest=is_bt,
        )
        assert result["reconciliation_severity"] == "ok", (
            f"clean recon should be ok (is_backtest={is_bt})"
        )
        assert result["reconciliation_blocked"] is False
        assert result["reconciliation_ok"] is True


# ---------------------------------------------------------------------------
# F1 (E-035) — backtest must NOT fire alerts nor write the live ops audit
# ---------------------------------------------------------------------------


def test_backtest_fail_fires_no_alert_and_writes_no_ops_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BACKTEST reconcile with a fail-severity drift must NOT call
    AlertManager.fire and must NOT append to the reconciliation ops-audit JSONL.

    Proves suppress_side_effects=True is passed through from build_ledger_from_trades
    in backtest mode — WITHOUT this test redirecting the audit path itself. We spy
    directly on both side-effect sinks in the reconciliation module.
    """
    import src.assembled_core.accounting.reconciliation as recon_mod

    fire_calls: list[tuple] = []
    audit_calls: list[dict] = []

    # Spy on the audit-append sink used by evaluate_reconcile_slo.
    def _spy_audit(event: dict) -> None:
        audit_calls.append(event)

    monkeypatch.setattr(recon_mod, "_append_recon_audit", _spy_audit)

    # Spy on AlertManager.fire (patch the class method on the source module so the
    # lazy `from ... import AlertManager` inside evaluate_reconcile_slo picks it up).
    import src.assembled_core.ops.alerting as alerting_mod

    def _spy_fire(self, rule_name, payload):  # noqa: ANN001
        fire_calls.append((rule_name, payload))

    monkeypatch.setattr(alerting_mod.AlertManager, "fire", _spy_fire, raising=True)

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run_id = "snap_bt_noalert"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    _seed_diverging_snapshot(output_dir, snapshot_run_id, as_of)

    result = build_ledger_from_trades(
        orders_df=_minimal_trades(),
        trades_df=_minimal_trades(),
        run_id="run_bt_noalert",
        output_dir=output_dir,
        as_of_date=as_of,
        start_cash=10000.0,
        broker_snapshot_policy="prefer",
        broker_snapshot_run_id=snapshot_run_id,
        is_backtest=True,
    )

    # Classification still happened (fail surfaced) ...
    assert result["reconciliation_severity"] == "fail"
    # ... but NO production side effects fired in backtest.
    assert fire_calls == [], "backtest must NOT fire AlertManager"
    assert audit_calls == [], "backtest must NOT write the live ops reconcile audit"


def test_live_fail_fires_alert_and_writes_ops_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A LIVE/PAPER reconcile with a fail-severity drift MUST still fire the alert
    and append to the ops-audit JSONL (existing behaviour, suppress=False)."""
    import src.assembled_core.accounting.reconciliation as recon_mod

    fire_calls: list[tuple] = []
    audit_calls: list[dict] = []

    monkeypatch.setattr(
        recon_mod, "_append_recon_audit", lambda event: audit_calls.append(event)
    )

    import src.assembled_core.ops.alerting as alerting_mod

    def _spy_fire(self, rule_name, payload):  # noqa: ANN001
        fire_calls.append((rule_name, payload))

    monkeypatch.setattr(alerting_mod.AlertManager, "fire", _spy_fire, raising=True)

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run_id = "snap_live_alert"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    _seed_diverging_snapshot(output_dir, snapshot_run_id, as_of)

    with pytest.raises(ReconciliationError):
        build_ledger_from_trades(
            orders_df=_minimal_trades(),
            trades_df=_minimal_trades(),
            run_id="run_live_alert",
            output_dir=output_dir,
            as_of_date=as_of,
            start_cash=10000.0,
            broker_snapshot_policy="prefer",
            broker_snapshot_run_id=snapshot_run_id,
            is_backtest=False,
        )

    # Live keeps both side effects (the classifier ran before the deferred raise).
    assert any(r == "reconciliation_fail" for r, _ in fire_calls), (
        "live fail must fire the reconciliation_fail alert"
    )
    assert audit_calls, "live fail must append to the reconcile ops-audit"


# ---------------------------------------------------------------------------
# F2 — classifier crash must fail CLOSED in live, continue in backtest
# ---------------------------------------------------------------------------


def test_classifier_crash_fails_closed_in_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If evaluate_reconcile_slo itself raises, live/paper mode must ESCALATE
    (raise ReconciliationError) — never swallow into a silent fail-open — and the
    reconciliation report must still have been written before the raise."""
    import src.assembled_core.accounting.ledger_integration as li_mod

    def _boom(**kwargs):  # noqa: ANN003
        raise RuntimeError("synthetic classifier crash")

    monkeypatch.setattr(li_mod, "evaluate_reconcile_slo", _boom)

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run_id = "snap_crash_live"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    _seed_diverging_snapshot(output_dir, snapshot_run_id, as_of)

    with pytest.raises(ReconciliationError) as exc_info:
        build_ledger_from_trades(
            orders_df=_minimal_trades(),
            trades_df=_minimal_trades(),
            run_id="run_crash_live",
            output_dir=output_dir,
            as_of_date=as_of,
            start_cash=10000.0,
            broker_snapshot_policy="prefer",
            broker_snapshot_run_id=snapshot_run_id,
            is_backtest=False,
        )

    assert "classifier failed" in str(exc_info.value)
    # Report still written before the fail-closed raise.
    assert _reconcile_report_exists(output_dir)


def test_classifier_crash_does_not_raise_in_backtest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If evaluate_reconcile_slo raises in BACKTEST mode, build_ledger_from_trades
    must NOT raise (E-035) — it ERROR-logs and continues; the report is written and
    the result is returned with reconciliation_ok False, no block."""
    import src.assembled_core.accounting.ledger_integration as li_mod

    def _boom(**kwargs):  # noqa: ANN003
        raise RuntimeError("synthetic classifier crash")

    monkeypatch.setattr(li_mod, "evaluate_reconcile_slo", _boom)

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run_id = "snap_crash_bt"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    _seed_diverging_snapshot(output_dir, snapshot_run_id, as_of)

    with caplog.at_level(
        logging.ERROR, logger="src.assembled_core.accounting.ledger_integration"
    ):
        result = build_ledger_from_trades(
            orders_df=_minimal_trades(),
            trades_df=_minimal_trades(),
            run_id="run_crash_bt",
            output_dir=output_dir,
            as_of_date=as_of,
            start_cash=10000.0,
            broker_snapshot_policy="prefer",
            broker_snapshot_run_id=snapshot_run_id,
            is_backtest=True,
        )

    # No raise; classifier crash surfaced as not-ok but never escalated.
    assert result["reconciliation_ok"] is False
    assert result["reconciliation_blocked"] is False
    assert _reconcile_report_exists(output_dir)
    crash_logs = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR
        and "classifier crashed in backtest" in r.getMessage()
    ]
    assert crash_logs, "backtest classifier crash must be ERROR-logged"
    assert not any("Escalating fail-closed" in r.getMessage() for r in caplog.records)
