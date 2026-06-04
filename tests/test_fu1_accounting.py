"""FU-1 accounting follow-ups — reconciliation visibility regression tests.

Covers two follow-up fixes in ``src/assembled_core/accounting`` (Diagnostik.md
§accounting):

* **B-acct-3** (ledger_integration.build_ledger_from_trades): a reconciliation
  that ran against the PAPER VIEW (no independent broker snapshot — policy
  "ignore", or "prefer" falling back) compares the ledger to a copy of itself.
  That is NOT a healthy pass: it MUST surface ``reconciliation_ok=None`` +
  ``reconciliation_severity="unverified"`` + a WARNING, and the B-acct-1
  fail-closed SLO escalation MUST be skipped (no real drift signal). The
  REAL-broker-snapshot path (``broker_view_source=="stored_snapshot"``) is left
  exactly as B-acct-1 left it: a genuine drift still escalates fail-closed in
  live/paper, a healthy stored-snapshot reconcile still passes.

* **B-acct-4** (reconciliation.reconcile_daily_pnl + currency.FXConverter):
  - a price-feed gap on a HELD symbol is surfaced in ``skipped_symbols`` AND
    forces ``ok=False`` + ``degraded=True`` (no silent pass);
  - a (near-)zero start price is guarded by tolerance — no divide-by-zero;
  - the hard-coded ``DEFAULT_FX_RATES`` fallback emits a one-time WARNING.

These tests build inputs against the REAL functions; only the broker/ledger
frames are minimal synthetic data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.accounting.accounting_report import (
    write_accounting_report_csv,
    write_accounting_report_json,
)
from src.assembled_core.accounting.broker_snapshot_store import (
    store_broker_snapshot_json,
    store_broker_snapshot_parquet,
)
from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades
from src.assembled_core.accounting.reconciliation import reconcile_daily_pnl
from src.assembled_core.errors import ReconciliationError

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Shared fixtures (mirror test_batchB2_accounting_failclosed.py)
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
    """Store a broker snapshot that diverges hard from the paper view (cash + qty)."""
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


def _seed_matching_snapshot(
    output_dir: Path, snapshot_run_id: str, as_of: pd.Timestamp
) -> None:
    """Store a broker snapshot that MATCHES the paper view (clean reconcile).

    Paper view after BUY 5 AAPL @ 150 from start_cash=10000 holds cash≈9250 and
    5 AAPL (total_cost_cash=0 in the minimal trade).
    """
    positions = pd.DataFrame({"symbol": ["AAPL"], "qty": [5.0]})
    store_broker_snapshot_json(
        cash=9250.0,
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
    return len(list(output_dir.rglob("reconcile*"))) > 0


# ===========================================================================
# B-acct-3 — paper_view reconcile is UNVERIFIED, not a healthy pass
# ===========================================================================


def test_paper_view_reconcile_is_unverified_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No stored snapshot (policy='prefer' → paper_view fallback): the reconcile
    is UNVERIFIED, NOT a healthy pass.

    reconciliation_ok is None (not True), severity is 'unverified', no
    violations, no block, no raise — and a WARNING explicitly states the book was
    NOT reconciled against an independent broker snapshot.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    with caplog.at_level(
        logging.WARNING, logger="src.assembled_core.accounting.ledger_integration"
    ):
        result = build_ledger_from_trades(
            orders_df=_minimal_trades(),
            trades_df=_minimal_trades(),
            run_id="run_paper",
            output_dir=output_dir,
            as_of_date=as_of,
            start_cash=10000.0,
            broker_snapshot_policy="prefer",  # no snapshot exists → paper_view
            broker_snapshot_run_id="snap_missing",
            is_backtest=False,  # even live/paper: paper_view must NOT escalate
        )

    assert result["reconciliation_ok"] is None, (
        "paper_view must NOT report a healthy True pass"
    )
    assert result["reconciliation_severity"] == "unverified"
    assert result["reconciliation_violations"] == []
    assert result["reconciliation_blocked"] is False
    assert result["broker_meta"]["broker_view_source"] == "paper_view"

    unverified_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "UNVERIFIED" in r.getMessage()
    ]
    assert unverified_warnings, (
        "paper_view reconcile must emit an explicit UNVERIFIED WARNING"
    )
    warn_msg = unverified_warnings[0].getMessage().lower()
    assert "not reconciled" in warn_msg
    assert "broker" in warn_msg


def test_paper_view_does_not_run_slo_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the paper_view path the SLO classifier must be SKIPPED entirely —
    comparing a frame to itself is meaningless, so evaluate_reconcile_slo must not
    even be called (and certainly cannot escalate)."""
    import src.assembled_core.accounting.ledger_integration as li_mod

    calls: list[dict] = []

    def _spy_slo(**kwargs):  # noqa: ANN003
        calls.append(kwargs)
        raise AssertionError("SLO classifier must NOT run on paper_view")

    monkeypatch.setattr(li_mod, "evaluate_reconcile_slo", _spy_slo)

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = build_ledger_from_trades(
        orders_df=_minimal_trades(),
        trades_df=_minimal_trades(),
        run_id="run_paper_noslo",
        output_dir=output_dir,
        as_of_date=pd.Timestamp("2025-01-15", tz="UTC"),
        start_cash=10000.0,
        broker_snapshot_policy="ignore",  # always paper_view
        is_backtest=False,
    )

    assert calls == [], "evaluate_reconcile_slo must not be called on paper_view"
    assert result["reconciliation_ok"] is None
    assert result["reconciliation_severity"] == "unverified"


def test_stored_snapshot_drift_still_escalates_failclosed(tmp_path: Path) -> None:
    """B-acct-1 path UNCHANGED: a REAL stored broker snapshot with a genuine drift
    still escalates fail-closed (raises ReconciliationError) in live/paper, and the
    report is written before the raise."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run_id = "snap_drift"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    _seed_diverging_snapshot(output_dir, snapshot_run_id, as_of)

    with pytest.raises(ReconciliationError):
        build_ledger_from_trades(
            orders_df=_minimal_trades(),
            trades_df=_minimal_trades(),
            run_id="run_drift",
            output_dir=output_dir,
            as_of_date=as_of,
            start_cash=10000.0,
            broker_snapshot_policy="prefer",
            broker_snapshot_run_id=snapshot_run_id,
            is_backtest=False,
        )

    assert _reconcile_report_exists(output_dir)


def test_stored_snapshot_healthy_reconcile_passes(tmp_path: Path) -> None:
    """B-acct-1 path UNCHANGED: a REAL stored broker snapshot that MATCHES the
    ledger reconciles cleanly: reconciliation_ok True, severity 'ok',
    broker_view_source 'stored_snapshot', no block, no raise."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run_id = "snap_clean"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")
    _seed_matching_snapshot(output_dir, snapshot_run_id, as_of)

    result = build_ledger_from_trades(
        orders_df=_minimal_trades(),
        trades_df=_minimal_trades(),
        run_id="run_clean",
        output_dir=output_dir,
        as_of_date=as_of,
        start_cash=10000.0,
        broker_snapshot_policy="prefer",
        broker_snapshot_run_id=snapshot_run_id,
        is_backtest=False,
    )

    assert result["broker_meta"]["broker_view_source"] == "stored_snapshot"
    assert result["reconciliation_ok"] is True
    assert result["reconciliation_severity"] == "ok"
    assert result["reconciliation_blocked"] is False


# ===========================================================================
# B-acct-4 — reconcile_daily_pnl skip visibility + zero-start guard
# ===========================================================================


def test_feed_gap_symbol_is_surfaced_and_downgrades_ok() -> None:
    """A HELD symbol with a missing end price is surfaced in skipped_symbols AND
    forces ok=False + degraded=True (a feed gap is never a silent pass)."""
    positions = {"AAPL": 0.5, "MSFT": 0.5}
    prices_start = {"AAPL": 100.0, "MSFT": 200.0}
    # MSFT has NO end price → feed gap on a held position.
    prices_end = {"AAPL": 102.0}

    # Reported return matches only the explained (AAPL) part — numerically "fine".
    result = reconcile_daily_pnl(positions, prices_start, prices_end, 0.01)

    assert "MSFT" in result["skipped_symbols"], "feed-gap symbol must be surfaced"
    assert result["degraded"] is True
    assert result["ok"] is False, "a feed gap on a held symbol must not be a pass"
    assert "DEGRADED" in result["break_reason"]


def test_no_feed_gap_still_passes_cleanly() -> None:
    """Full price coverage with a matching return is still ok=True / degraded=False
    (no behaviour change when there is no feed gap)."""
    positions = {"AAPL": 0.5, "MSFT": 0.5}
    prices_start = {"AAPL": 100.0, "MSFT": 200.0}
    prices_end = {"AAPL": 102.0, "MSFT": 198.0}
    # 0.5*2% + 0.5*(-1%) = 0.5%
    result = reconcile_daily_pnl(positions, prices_start, prices_end, 0.005)

    assert result["skipped_symbols"] == []
    assert result["degraded"] is False
    assert result["ok"] is True


def test_zero_start_price_does_not_divide() -> None:
    """A (near-)zero start price on a held symbol is guarded by tolerance — no
    divide-by-zero — and the symbol is surfaced as a skip/degraded."""
    positions = {"AAPL": 1.0}
    prices_start = {"AAPL": 0.0}  # zero start price
    prices_end = {"AAPL": 100.0}

    # Must NOT raise ZeroDivisionError / produce inf.
    result = reconcile_daily_pnl(positions, prices_start, prices_end, 0.0)

    assert "AAPL" in result["skipped_symbols"]
    assert result["degraded"] is True
    assert result["ok"] is False
    # Contribution booked as 0, not inf/nan.
    assert result["position_contributions"]["AAPL"] == 0.0


def test_tiny_residual_start_price_is_treated_as_zero() -> None:
    """A start price below the float tolerance (e.g. 1e-15) is treated as a
    zero-start skip, not a near-infinite division."""
    positions = {"AAPL": 1.0}
    prices_start = {"AAPL": 1e-15}
    prices_end = {"AAPL": 100.0}

    result = reconcile_daily_pnl(positions, prices_start, prices_end, 0.0)

    assert "AAPL" in result["skipped_symbols"]
    assert result["position_contributions"]["AAPL"] == 0.0


# ===========================================================================
# B-acct-4 — DEFAULT_FX_RATES fallback emits a one-time WARNING
# ===========================================================================


def test_default_fx_rates_fallback_warns(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Using FXConverter() with NO operator-supplied rates (the hard-coded
    DEFAULT_FX_RATES fallback, no as_of/freshness check) emits a one-time WARNING
    on first conversion — so a silent stale-FX conversion is observable."""
    import src.assembled_core.accounting.currency as ccy_mod

    # Reset the one-time latch so this test is order-independent.
    monkeypatch.setattr(ccy_mod, "_DEFAULT_FX_WARNED", False)

    fx = ccy_mod.FXConverter()  # no rates → default fallback

    with caplog.at_level(
        logging.WARNING, logger="src.assembled_core.accounting.currency"
    ):
        usd1 = fx.to_usd(100.0, "EUR")
        usd2 = fx.to_usd(100.0, "GBP")  # second call must NOT warn again

    assert usd1 == pytest.approx(100.0 * ccy_mod.DEFAULT_FX_RATES["EUR"])
    assert usd2 == pytest.approx(100.0 * ccy_mod.DEFAULT_FX_RATES["GBP"])

    fx_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "DEFAULT_FX_RATES" in r.getMessage()
    ]
    assert len(fx_warnings) == 1, (
        "DEFAULT_FX_RATES fallback must warn exactly once per process"
    )


def test_operator_supplied_fx_rates_do_not_warn(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the operator supplies explicit rates, NO stale-FX fallback WARNING
    fires (the rates are intentional, not the default fallback)."""
    import src.assembled_core.accounting.currency as ccy_mod

    monkeypatch.setattr(ccy_mod, "_DEFAULT_FX_WARNED", False)

    fx = ccy_mod.FXConverter(rates={"USD": 1.0, "EUR": 1.10})

    with caplog.at_level(
        logging.WARNING, logger="src.assembled_core.accounting.currency"
    ):
        fx.to_usd(100.0, "EUR")

    fx_warnings = [r for r in caplog.records if "DEFAULT_FX_RATES" in r.getMessage()]
    assert fx_warnings == [], (
        "operator-supplied rates must NOT trigger the fallback warning"
    )


# ===========================================================================
# B-acct-3 (artifact level) — accounting report SUMMARY must NOT record the
# trivial paper-vs-paper self-comparison True. It must reflect the GRADED
# reconciliation_ok: None/"unverified" on a paper_view run, True ONLY on a
# healthy stored_snapshot reconcile, False on a real drift.
#
# Before the fix the report writers derived reconciliation_ok solely from
# reconciliation_result["ok"] (the self-comparison True on paper_view), which
# re-introduced the masked pass B-acct-3 removed at the function/return level.
# These tests assert the WRITTEN CSV/JSON artifact, and FAIL against the
# pre-fix writers on the paper_view case (which recorded True).
# ===========================================================================


def _report_positions_result() -> dict:
    """A minimal positions_result for the report writers (mirrors the
    broker_meta-report tests' shape)."""
    positions_df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [10.0],
            "avg_price": [100.0],
            "realized_pnl": [5.0],
            "unrealized_pnl": [2.0],
            "notional": [1000.0],
            "last_price": [102.0],
        }
    )
    summary = {
        "total_realized_pnl": 5.0,
        "total_unrealized_pnl": 2.0,
        "total_pnl": 7.0,
        "n_positions": 1,
        "gross_exposure": 1000.0,
        "net_exposure": 1000.0,
    }
    return {
        "positions_df": positions_df,
        "cash_balance": 10007.0,
        "summary": summary,
    }


# On a paper_view run the reconcile compares the ledger to a copy of itself, so
# the RAW result is the trivial self-comparison ok=True — exactly what must NOT
# leak into the artifact. The GRADED value the caller threads in is None.
_PAPER_VIEW_RESULT = {"ok": True, "cash_match": True, "cash_diff": 0.0}
_PAPER_VIEW_BROKER_META = {
    "broker_view_source": "paper_view",
    "broker_snapshot_run_id": "",
    "broker_snapshot_date": "",
    "broker_snapshot_path": "",
}
_STORED_BROKER_META = {
    "broker_view_source": "stored_snapshot",
    "broker_snapshot_run_id": "snap_clean",
    "broker_snapshot_date": "2025-01-15",
    "broker_snapshot_path": "broker_snapshot_snap_clean/snapshot_2025-01-15.json",
}


def _csv_summary_reconciliation_ok(csv_path: Path):
    """Return the raw SUMMARY reconciliation_ok cell from a written report CSV."""
    df = pd.read_csv(csv_path)
    return df[df["section"] == "SUMMARY"].iloc[0]["reconciliation_ok"]


def _json_reconciliation_ok(json_path: Path):
    """Return reconciliation.ok from a written report JSON (None if absent)."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("reconciliation", {}).get("ok", "MISSING")


def test_report_csv_paper_view_records_unverified_not_true(tmp_path: Path) -> None:
    """CSV SUMMARY reconciliation_ok on a paper_view run must be empty/None
    (graded "unverified"), NOT the trivial self-comparison True.

    This FAILS against the pre-fix writer (which wrote reconciliation_result["ok"]
    == True on paper_view)."""
    csv_path = write_accounting_report_csv(
        positions_result=_report_positions_result(),
        output_dir=tmp_path / "out",
        run_id="paper_csv",
        as_of=pd.Timestamp("2025-01-15", tz="UTC"),
        start_cash=10000.0,
        reconciliation_result=_PAPER_VIEW_RESULT,
        reconciliation_ok=None,  # GRADED unverified on paper_view
        broker_meta=_PAPER_VIEW_BROKER_META,
    )
    cell = _csv_summary_reconciliation_ok(csv_path)
    # None is written as empty → read back as NaN/"".
    assert pd.isna(cell) or cell == "", (
        f"paper_view CSV must not record a True pass; got {cell!r}"
    )

    # broker_view_source is still recorded for traceability.
    df = pd.read_csv(csv_path)
    assert df[df["section"] == "SUMMARY"].iloc[0]["broker_view_source"] == "paper_view"


def test_report_json_paper_view_records_unverified_not_true(tmp_path: Path) -> None:
    """JSON reconciliation.ok on a paper_view run must be None (graded
    unverified), NOT the trivial self-comparison True.

    This FAILS against the pre-fix writer (which wrote reconciliation_result["ok"]
    == True on paper_view)."""
    json_path = write_accounting_report_json(
        positions_result=_report_positions_result(),
        output_dir=tmp_path / "out",
        run_id="paper_json",
        as_of=pd.Timestamp("2025-01-15", tz="UTC"),
        start_cash=10000.0,
        reconciliation_result=_PAPER_VIEW_RESULT,
        reconciliation_ok=None,
        broker_meta=_PAPER_VIEW_BROKER_META,
    )
    assert _json_reconciliation_ok(json_path) is None, (
        "paper_view JSON must record reconciliation.ok=None, not True"
    )
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["broker_meta"]["broker_view_source"] == "paper_view"


def test_report_stored_snapshot_healthy_records_true(tmp_path: Path) -> None:
    """A healthy stored_snapshot reconcile (graded True) still records True in
    BOTH the CSV SUMMARY and the JSON reconciliation.ok (backward-compat)."""
    healthy = {"ok": True, "cash_match": True, "cash_diff": 0.0}
    csv_path = write_accounting_report_csv(
        positions_result=_report_positions_result(),
        output_dir=tmp_path / "out",
        run_id="stored_ok_csv",
        as_of=pd.Timestamp("2025-01-15", tz="UTC"),
        start_cash=10000.0,
        reconciliation_result=healthy,
        reconciliation_ok=True,
        broker_meta=_STORED_BROKER_META,
    )
    json_path = write_accounting_report_json(
        positions_result=_report_positions_result(),
        output_dir=tmp_path / "out",
        run_id="stored_ok_json",
        as_of=pd.Timestamp("2025-01-15", tz="UTC"),
        start_cash=10000.0,
        reconciliation_result=healthy,
        reconciliation_ok=True,
        broker_meta=_STORED_BROKER_META,
    )
    assert _csv_summary_reconciliation_ok(csv_path) == True  # noqa: E712
    assert _json_reconciliation_ok(json_path) is True


def test_report_stored_snapshot_drift_records_false(tmp_path: Path) -> None:
    """A real stored_snapshot drift (graded False) records False in BOTH the CSV
    SUMMARY and the JSON reconciliation.ok."""
    drift = {"ok": False, "cash_match": False, "cash_diff": 250.0}
    csv_path = write_accounting_report_csv(
        positions_result=_report_positions_result(),
        output_dir=tmp_path / "out",
        run_id="stored_drift_csv",
        as_of=pd.Timestamp("2025-01-15", tz="UTC"),
        start_cash=10000.0,
        reconciliation_result=drift,
        reconciliation_ok=False,
        broker_meta=_STORED_BROKER_META,
    )
    json_path = write_accounting_report_json(
        positions_result=_report_positions_result(),
        output_dir=tmp_path / "out",
        run_id="stored_drift_json",
        as_of=pd.Timestamp("2025-01-15", tz="UTC"),
        start_cash=10000.0,
        reconciliation_result=drift,
        reconciliation_ok=False,
        broker_meta=_STORED_BROKER_META,
    )
    assert _csv_summary_reconciliation_ok(csv_path) == False  # noqa: E712
    assert _json_reconciliation_ok(json_path) is False


def test_report_legacy_call_without_graded_ok_uses_result_ok(tmp_path: Path) -> None:
    """Backward-compat: a legacy caller that does NOT pass reconciliation_ok still
    derives the SUMMARY value from reconciliation_result["ok"] (the sentinel
    default preserves the old behaviour — distinct from explicit graded None)."""
    healthy = {"ok": True, "cash_match": True, "cash_diff": 0.0}
    csv_path = write_accounting_report_csv(
        positions_result=_report_positions_result(),
        output_dir=tmp_path / "out",
        run_id="legacy_csv",
        as_of=pd.Timestamp("2025-01-15", tz="UTC"),
        start_cash=10000.0,
        reconciliation_result=healthy,  # no reconciliation_ok kwarg
        broker_meta=_STORED_BROKER_META,
    )
    json_path = write_accounting_report_json(
        positions_result=_report_positions_result(),
        output_dir=tmp_path / "out",
        run_id="legacy_json",
        as_of=pd.Timestamp("2025-01-15", tz="UTC"),
        start_cash=10000.0,
        reconciliation_result=healthy,
        broker_meta=_STORED_BROKER_META,
    )
    assert _csv_summary_reconciliation_ok(csv_path) == True  # noqa: E712
    assert _json_reconciliation_ok(json_path) is True


def test_end_to_end_paper_view_report_artifact_is_unverified(tmp_path: Path) -> None:
    """End-to-end: build_ledger_from_trades on a paper_view run writes an
    accounting report whose SUMMARY reconciliation_ok is empty/None (graded
    unverified) — the artifact matches the function-level grading, closing the
    B-acct-3 gap at the EOD evidence artifact, not just in the return dict."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "run_paper_artifact"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    result = build_ledger_from_trades(
        orders_df=_minimal_trades(),
        trades_df=_minimal_trades(),
        run_id=run_id,
        output_dir=output_dir,
        as_of_date=as_of,
        start_cash=10000.0,
        broker_snapshot_policy="ignore",  # always paper_view
        is_backtest=False,
    )
    # Function-level grading (unchanged B-acct-3 behaviour).
    assert result["reconciliation_ok"] is None
    assert result["broker_meta"]["broker_view_source"] == "paper_view"

    # Artifact-level: the written CSV SUMMARY must mirror that, not True.
    csv_files = list(output_dir.rglob("accounting_2025-01-15.csv"))
    assert csv_files, "accounting report CSV was not written"
    cell = _csv_summary_reconciliation_ok(csv_files[0])
    assert pd.isna(cell) or cell == "", (
        f"end-to-end paper_view report must not record a True pass; got {cell!r}"
    )

    json_files = list(output_dir.rglob("accounting_2025-01-15.json"))
    assert json_files, "accounting report JSON was not written"
    assert _json_reconciliation_ok(json_files[0]) is None


def test_report_json_graded_ok_without_result_emits_block(tmp_path: Path) -> None:
    """F-senior-5 / F-auditor-4: a caller that grades reconciliation_ok=None
    (unverified) but passes reconciliation_result=None must STILL emit the
    reconciliation block with ok=None — the graded status must not be silently
    omitted just because no result dict was supplied. The pre-fix writer gated
    the whole block on a truthy reconciliation_result and dropped it here."""
    json_path = write_accounting_report_json(
        positions_result=_report_positions_result(),
        output_dir=tmp_path / "out",
        run_id="graded_no_result",
        as_of=pd.Timestamp("2025-01-15", tz="UTC"),
        start_cash=10000.0,
        reconciliation_result=None,  # no result dict at all
        reconciliation_ok=None,  # but GRADED unverified — must surface
    )
    # _json_reconciliation_ok returns "MISSING" when the block is absent.
    assert _json_reconciliation_ok(json_path) is None, (
        "graded reconciliation_ok=None must be emitted as reconciliation.ok=None, "
        "not omitted, even when reconciliation_result is None"
    )
