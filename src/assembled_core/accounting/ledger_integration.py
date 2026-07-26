"""Ledger integration helper for pipeline (Sprint 13 L5).

This module provides functions to integrate ledger/accounting into the execution pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
from src.assembled_core.accounting.accounting_report import (
    write_accounting_report_csv,
    write_accounting_report_json,
)
from src.assembled_core.accounting.broker_snapshot import normalize_broker_snapshot
from src.assembled_core.accounting.broker_snapshot_store import (
    load_broker_snapshot_json,
    load_broker_snapshot_parquet,
)
from src.assembled_core.accounting.evidence_index import write_evidence_index_json
from src.assembled_core.accounting.evidence_pack import build_evidence_pack
from src.assembled_core.accounting.ledger import events_from_orders, events_from_trades
from src.assembled_core.accounting.ledger_store import (
    ledger_base_path,
    store_daily_snapshot_parquet,
    store_ledger_events_parquet,
)
from src.assembled_core.accounting.position_engine import build_positions_from_ledger
from src.assembled_core.accounting.reconciliation import (
    ReconcileSLO,
    evaluate_reconcile_slo,
    reconcile_ledger_vs_broker,
)
from src.assembled_core.accounting.reconciliation_report import (
    write_reconcile_report_csv,
    write_reconcile_report_json,
)
from src.assembled_core.errors import ReconciliationError

logger = logging.getLogger(__name__)


def build_ledger_from_trades(
    orders_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    run_id: str,
    output_dir: Path,
    as_of_date: pd.Timestamp | str | None = None,
    prices_df: pd.DataFrame | None = None,
    start_cash: float = 10000.0,
    *,
    broker_snapshot_policy: str = "prefer",
    write_paper_broker_snapshot: bool = False,
    broker_snapshot_run_id: str | None = None,
    write_evidence_pack: bool = False,
    is_backtest: bool = True,
    reconcile_slo: ReconcileSLO | None = None,
) -> dict[str, Any]:
    """Build ledger from orders and trades, store events, compute positions, and reconcile.

    Args:
        orders_df: DataFrame with orders (columns: timestamp, symbol, side, qty, price)
        trades_df: DataFrame with trades/fills (columns: timestamp, symbol, side, qty, price,
            fill_qty, fill_price, status, total_cost_cash, ...)
        run_id: Run identifier
        output_dir: Base output directory
        as_of_date: Reconciliation date (default: last timestamp in trades_df)
        prices_df: Optional prices DataFrame for unrealized PnL calculation
        start_cash: Starting cash balance
        broker_snapshot_policy: Policy for broker snapshot usage:
            - "ignore": Never use snapshot, always use paper view
            - "prefer": Use snapshot if available, otherwise paper view (default)
            - "require": Snapshot must exist, raise ValueError if missing
        write_paper_broker_snapshot: If True, write paper broker view as snapshot after computation
        broker_snapshot_run_id: Optional run_id for broker snapshot (default: run_id)
        write_evidence_pack: If True, create evidence pack (ZIP + manifest) after evidence index write
        is_backtest: If True (default — SAFE), reconciliation SLO failures are
            classified, surfaced and ERROR-logged but NEVER escalated (no raise,
            no state change), per E-035 backtest-safety. Only live/paper callers
            should pass ``is_backtest=False`` to enable fail-closed escalation.
            Defaulting to True means an unknown mode never escalates.
        reconcile_slo: Optional SLO thresholds; defaults to ``ReconcileSLO()``.

    Returns:
        Dictionary with:
        - ledger_pack_path: Path to ledger directory
        - positions_df: DataFrame with positions
        - cash_balance: Final cash balance
        - reconciliation_result: Reconciliation result dict (if broker snapshot available)
        - reconcile_report_path: Path to reconciliation report (if reconciliation performed)
        - reconciliation_ok: bool | None. True iff reconciled against a real
          independent broker snapshot AND it passed; False on a genuine drift.
          ``None`` means UNVERIFIED — no independent broker snapshot existed
          (paper-vs-paper), so reconciliation was NOT actually performed; this is
          deliberately NOT reported as a healthy pass (B-acct-3).
        - reconciliation_severity: SLO severity in {"ok","warn","fail"},
          ``"unverified"`` for the paper-view (not-reconciled) case, or None
        - reconciliation_violations: list of SLO violation dicts (may be empty)
        - reconciliation_blocked: bool — True iff a live/paper SLO "fail" was
          detected (book must not keep trading); also signalled by raising
          ReconciliationError after the report is written.
        - broker_snapshot_path: Path to broker snapshot directory (if written or used)

    Raises:
        ReconciliationError: If a reconciliation SLO "fail" is detected AND
            ``is_backtest`` is False (live/paper). Raised AFTER the reconciliation
            report is written so the failure is recorded in the EOD pack. Never
            raised in backtest mode.
    """
    # Normalize as_of_date
    if as_of_date is None:
        if not trades_df.empty and "timestamp" in trades_df.columns:
            as_of_date = pd.to_datetime(trades_df["timestamp"].max(), utc=True)
        else:
            as_of_date = pd.Timestamp.now("UTC")
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date, utc=True)
    # Normalize via a Timestamp-annotated local for the type checker. cast()
    # instead of assert/isinstance so degenerate inputs (e.g. NaT from an
    # all-NaT trades timestamp column) keep their pre-existing pass-through
    # behaviour — see the defensive isinstance guards at call-sites below.
    as_of_ts: pd.Timestamp = cast("pd.Timestamp", as_of_date)
    if as_of_ts.tz is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    as_of_date = as_of_ts

    # Generate ledger events
    logger.info(f"Generating ledger events for run_id={run_id}")

    # Generate ORDER_SUBMIT events from orders
    order_events = events_from_orders(orders_df, run_id=run_id, source="paper")
    logger.info(f"Generated {len(order_events)} ORDER_SUBMIT events")

    # Generate FILL/REJECT events from trades
    trade_events = events_from_trades(trades_df, run_id=run_id, source="paper")
    logger.info(f"Generated {len(trade_events)} FILL/REJECT events")

    # Combine events. order_events and trade_events have overlapping but not
    # identical optional columns (e.g. order_type vs commission_cash). Suppress
    # the pandas FutureWarning about all-NA column dtype inference — the current
    # behavior (using the non-NA frame's dtype) is exactly what we want here.
    import warnings as _w

    _frames = [f for f in [order_events, trade_events] if not f.empty]
    if not _frames:
        all_events = order_events.iloc[0:0].copy()
    elif len(_frames) == 1:
        all_events = _frames[0].copy()
    else:
        with _w.catch_warnings():
            _w.simplefilter("ignore", FutureWarning)
            all_events = pd.concat(_frames, ignore_index=True)

    # Store ledger events
    ledger_base = ledger_base_path(output_dir, run_id)
    store_ledger_events_parquet(all_events, output_dir, run_id, mode="replace")
    logger.info(f"Stored ledger events to {ledger_base}")

    # Build positions from ledger
    logger.info("Building positions from ledger events")
    positions_result = build_positions_from_ledger(
        all_events,
        prices_df=prices_df,
        mark_ts=as_of_date,
        start_cash=start_cash,
        missing_price_policy="zero",
    )

    positions_df = positions_result["positions_df"]
    cash_balance = positions_result["cash_balance"]

    logger.info(
        f"Positions built: {len(positions_df)} positions, cash={cash_balance:.2f}"
    )

    # Store daily snapshot (only if positions exist)
    if not positions_df.empty:
        snapshot_df = positions_df.copy()
        snapshot_df["as_of_date"] = as_of_date
        store_daily_snapshot_parquet(snapshot_df, output_dir, run_id, as_of_date)
        logger.info(f"Stored daily snapshot for {as_of_ts.date()}")
    else:
        logger.info(f"No positions to snapshot for {as_of_ts.date()}")

    # Determine broker snapshot run_id (for loading/writing)
    snapshot_run_id = (
        broker_snapshot_run_id if broker_snapshot_run_id is not None else run_id
    )

    # Try to reconcile with broker snapshot
    # Decision logic based on broker_snapshot_policy
    reconciliation_result = None
    reconcile_report_path = None
    # bool | None per the documented return contract: None = "unverified"
    # (B-acct-3 paper_view grading below), True/False = real reconcile outcome.
    reconciliation_ok: bool | None = True
    reconciliation_severity = None
    reconciliation_violations: list[dict] = []
    reconciliation_blocked = False
    broker_snapshot_path = None
    # Deferred so the report is written BEFORE we escalate (fail-closed).
    pending_reconcile_error: ReconciliationError | None = None

    # Decision logic (outside try/except so ValueError for "require" propagates)
    logger.info(
        f"Attempting reconciliation with broker snapshot (policy: {broker_snapshot_policy})"
    )

    # Initialize broker_meta to track which source was used
    broker_meta = {
        "broker_view_source": None,
        "broker_snapshot_run_id": snapshot_run_id,
        "broker_snapshot_date": (
            as_of_date.isoformat()
            if isinstance(as_of_date, pd.Timestamp)
            else str(as_of_date)
        ),
        "broker_snapshot_path": None,
    }

    if broker_snapshot_policy == "ignore":
        # Always use paper view
        logger.info("Broker snapshot policy is 'ignore', using paper broker view")
        broker_positions_df = positions_df[["symbol", "qty"]].copy()
        broker_cash = cash_balance
        broker_meta["broker_view_source"] = "paper_view"
    else:
        # Try to load broker snapshot from store
        broker_snapshot_json = load_broker_snapshot_json(
            output_dir, snapshot_run_id, as_of_date
        )
        broker_snapshot_parquet = load_broker_snapshot_parquet(
            output_dir, snapshot_run_id, as_of_date
        )

        if broker_snapshot_json is not None:
            # Use stored broker snapshot
            logger.info("Using stored broker snapshot for reconciliation")
            broker_cash = broker_snapshot_json["cash"]

            # Load positions from Parquet if available, otherwise from JSON
            if broker_snapshot_parquet is not None:
                broker_positions_df = broker_snapshot_parquet[["symbol", "qty"]].copy()
            else:
                # Convert JSON positions list to DataFrame
                positions_list = broker_snapshot_json["positions"]
                if positions_list:
                    broker_positions_df = pd.DataFrame(positions_list)[
                        ["symbol", "qty"]
                    ].copy()
                else:
                    broker_positions_df = pd.DataFrame(columns=["symbol", "qty"])

            # Normalize broker snapshot (trimming, sorting, filter tiny residuals)
            normalized = normalize_broker_snapshot(
                cash=broker_cash,
                positions_df=broker_positions_df,
                qty_tol=1e-8,
            )
            broker_positions_df = normalized["positions_df"]
            broker_cash = normalized["cash"]

            # Set broker_snapshot_path for manifest and broker_meta
            from src.assembled_core.accounting.broker_snapshot_store import (
                broker_snapshot_base_path,
            )

            snapshot_base = broker_snapshot_base_path(output_dir, snapshot_run_id)
            broker_snapshot_path = snapshot_base
            broker_meta["broker_view_source"] = "stored_snapshot"
            broker_meta["broker_snapshot_path"] = str(
                snapshot_base.relative_to(output_dir)
            )
        else:
            # Snapshot not found
            if broker_snapshot_policy == "require":
                from src.assembled_core.accounting.broker_snapshot_store import (
                    broker_snapshot_base_path,
                )

                expected_path = broker_snapshot_base_path(output_dir, snapshot_run_id)
                raise ValueError(
                    f"Broker snapshot required but not found: "
                    f"run_id={snapshot_run_id}, as_of_date={as_of_ts.date()}. "
                    f"Expected path: {expected_path}"
                )
            # Fallback: Use positions_df as broker snapshot (paper broker view)
            logger.warning(
                "[Reconciliation] No stored broker snapshot found — falling back to "
                "paper broker view. Reconciliation will compare paper vs. paper, "
                "which CANNOT detect real position mismatches."
            )
            broker_positions_df = positions_df[["symbol", "qty"]].copy()
            broker_cash = cash_balance
            broker_meta["broker_view_source"] = "paper_view"

    try:
        reconciliation_result = reconcile_ledger_vs_broker(
            ledger_positions_df=positions_df[["symbol", "qty"]].copy(),
            ledger_cash=cash_balance,
            broker_positions_df=broker_positions_df,
            broker_cash=broker_cash,
            cash_tol=1e-6,
            qty_tol=1e-8,
            fail_fast=False,
        )

        reconciliation_ok = reconciliation_result["ok"]

        if reconciliation_ok:
            logger.info("Reconciliation PASSED: ledger matches broker snapshot")
        else:
            logger.warning(f"Reconciliation FAILED: {reconciliation_result['message']}")

        # Write reconciliation report
        csv_path = write_reconcile_report_csv(
            reconciliation_result,
            output_dir,
            run_id,
            as_of_date,
            ledger_cash=cash_balance,
            broker_cash=broker_cash,
            broker_meta=broker_meta,
        )
        _ = write_reconcile_report_json(
            reconciliation_result,
            output_dir,
            run_id,
            as_of_date,
            ledger_cash=cash_balance,
            broker_cash=broker_cash,
            broker_meta=broker_meta,
        )

        # Use CSV path as primary report path
        reconcile_report_path = csv_path.relative_to(output_dir)
        logger.info(f"Reconciliation report written: {reconcile_report_path}")

    except Exception as e:
        # Broad guard for the report-WRITING path only (reconcile compute + CSV/
        # JSON report). The SLO classification + fail-closed escalation lives in
        # its OWN try/except below so a classifier crash can never be swallowed
        # here into a silent fail-open. If the report path itself fails in
        # live/paper we still escalate fail-closed (we could not even produce a
        # clean reconcile), rather than continuing silently.
        logger.warning(f"Reconciliation failed: {e}", exc_info=True)
        reconciliation_ok = False
        if not is_backtest:
            reconciliation_blocked = True
            pending_reconcile_error = ReconciliationError(
                "Ledger-vs-broker reconciliation report path failed in live/paper "
                f"mode — failing closed: {e}"
            )

    # ── B-acct-3: paper-vs-paper reconcile is UNVERIFIED, never a healthy pass ──
    # When no independent broker snapshot exists (policy "ignore", or "prefer"
    # falling back to the paper view), broker_positions_df/broker_cash are literally
    # a copy of the ledger's own positions/cash. reconcile_ledger_vs_broker then
    # compares a frame to itself → ok=True trivially, and evaluate_reconcile_slo
    # would grade it as severity "ok" — masking the fact that NO real reconciliation
    # happened. Running the B-acct-1 fail-closed SLO escalation on this is meaningless
    # (there is no genuine drift signal to act on) and reporting a healthy pass would
    # hide a real broker drift that we simply never looked at.
    #
    # So: surface an explicit "unverified" state (reconciliation_ok=None,
    # severity="unverified"), emit a loud WARNING, and SKIP the SLO classification +
    # escalation entirely. We do NOT raise — there is no drift signal — but the
    # not-reconciled state is made visible in the return dict + reports, not a silent
    # green. The real-snapshot path (broker_view_source == "stored_snapshot") is left
    # untouched below and keeps the B-acct-1 classify + fail-closed-raise behaviour.
    is_paper_view = broker_meta.get("broker_view_source") == "paper_view"
    if (
        is_paper_view
        and reconciliation_result is not None
        and pending_reconcile_error is None
    ):
        reconciliation_ok = None
        reconciliation_severity = "unverified"
        reconciliation_violations = []
        logger.warning(
            "[Reconciliation] UNVERIFIED: no independent broker snapshot for "
            "run_id=%s as_of=%s (broker_view_source=paper_view). Book was NOT "
            "reconciled against a real broker view — comparing the ledger to itself "
            "cannot detect a genuine position/cash drift. Reporting reconciliation "
            "as unverified (not a pass); SLO fail-closed escalation is skipped "
            "because there is no real drift signal to act on.",
            run_id,
            as_of_date.date() if isinstance(as_of_date, pd.Timestamp) else as_of_date,
        )

    # ── B-acct-1: route the result through the SLO classifier (fail-closed) ──
    # The plain reconcile_ledger_vs_broker(fail_fast=False) only logs a WARNING on
    # drift; the book would keep trading next cycle. We grade the drift against
    # ReconcileSLO thresholds and, in LIVE/PAPER mode only, escalate a hard "fail"
    # so a drifting book cannot keep trading.
    #
    # F2 fix: this verdict is computed in its OWN try/except, NOT inside the broad
    # report-writing try above. A crash in the classifier (alert import error, inf
    # interaction, frame edge) must itself FAIL CLOSED in live/paper — never be
    # swallowed into reconciliation_ok=False with no escalation (the exact
    # fail-open gap this batch closes). Only attempt classification when a reconcile
    # result actually exists (the report path above may have failed before it was
    # produced). B-acct-3: skip entirely for paper_view (handled above as unverified).
    if (
        not is_paper_view
        and reconciliation_result is not None
        and pending_reconcile_error is None
    ):
        try:
            # max_qty_diff: largest per-symbol share-count divergence.
            # position_diffs_df only lists symbols present on BOTH sides; a symbol
            # missing entirely from one side is a full-position mismatch the
            # per-symbol diff cannot see, so we treat any missing symbol as an
            # effectively unbounded qty diff (inf) to guarantee it cannot slip
            # under the SLO threshold.
            pos_diffs = reconciliation_result.get("position_diffs_df")
            if pos_diffs is not None and not pos_diffs.empty:
                max_qty_diff = float(pos_diffs["diff_qty"].abs().max())
            else:
                max_qty_diff = 0.0
            if reconciliation_result.get(
                "missing_in_ledger"
            ) or reconciliation_result.get("missing_in_broker"):
                max_qty_diff = float("inf")

            slo = reconcile_slo if reconcile_slo is not None else ReconcileSLO()
            # E-035: suppress alert + live ops-audit side effects in backtest so a
            # historical-drift replay never emits real production alerts nor writes
            # the live reconciliation_audit.jsonl. Live/paper keeps both.
            slo_verdict = evaluate_reconcile_slo(
                cash_diff=float(reconciliation_result.get("cash_diff", 0.0)),
                broker_cash=float(broker_cash),
                max_qty_diff=max_qty_diff,
                fill_rate=None,
                slippage_p99_bps=None,
                slo=slo,
                suppress_side_effects=is_backtest,
            )
            reconciliation_severity = slo_verdict["severity"]
            reconciliation_violations = slo_verdict["violations"]

            if reconciliation_severity == "fail":
                # A real drift: log at ERROR (not warning) and force
                # reconciliation_ok False so downstream/reports surface the breach.
                reconciliation_ok = False
                logger.error(
                    "[Reconciliation] SLO FAIL (mode=%s): %s | violations=%s",
                    "backtest" if is_backtest else "live/paper",
                    reconciliation_result.get("message", ""),
                    reconciliation_violations,
                )
                # E-035 / backtest-safety: escalation (fail-closed raise) fires
                # ONLY in live/paper. In backtest we classify + surface + ERROR-log
                # but NEVER raise and NEVER change live state. is_backtest defaults
                # to True, so an unknown/unspecified mode is treated as backtest and
                # does NOT escalate.
                if not is_backtest:
                    reconciliation_blocked = True
                    pending_reconcile_error = ReconciliationError(
                        "Ledger-vs-broker reconciliation SLO FAIL in live/paper "
                        "mode — book must not keep trading. "
                        f"violations={reconciliation_violations} | "
                        f"{reconciliation_result.get('message', '')}"
                    )
            elif reconciliation_severity == "warn":
                logger.warning(
                    "[Reconciliation] SLO WARN: %s | violations=%s",
                    reconciliation_result.get("message", ""),
                    reconciliation_violations,
                )
        except Exception as e:
            # F2: failure-to-classify must itself fail closed in live/paper.
            reconciliation_ok = False
            if is_backtest:
                # E-035: backtest never escalates — ERROR-log and continue.
                logger.error(
                    "[Reconciliation] SLO classifier crashed in backtest — "
                    "continuing without escalation: %s",
                    e,
                    exc_info=True,
                )
            else:
                logger.error(
                    "[Reconciliation] SLO classifier crashed in live/paper — "
                    "failing closed: %s",
                    e,
                    exc_info=True,
                )
                reconciliation_blocked = True
                pending_reconcile_error = ReconciliationError(
                    f"reconcile SLO classifier failed - failing closed: {e}"
                )

    # Write paper broker snapshot if requested
    if write_paper_broker_snapshot:
        try:
            logger.info("Writing paper broker snapshot")
            from src.assembled_core.accounting.broker_snapshot_store import (
                store_broker_snapshot_json,
                store_broker_snapshot_parquet,
            )

            # Use paper view (positions_df + cash_balance)
            paper_positions_df = positions_df[["symbol", "qty"]].copy()
            paper_cash = cash_balance

            # Store JSON snapshot
            _ = store_broker_snapshot_json(
                cash=paper_cash,
                positions_df=paper_positions_df,
                output_dir=output_dir,
                run_id=snapshot_run_id,
                as_of_date=as_of_date,
            )

            # Store Parquet snapshot (optional, only if positions exist)
            if not paper_positions_df.empty:
                store_broker_snapshot_parquet(
                    positions_df=paper_positions_df,
                    output_dir=output_dir,
                    run_id=snapshot_run_id,
                    as_of_date=as_of_date,
                )

            # Set broker_snapshot_path for manifest
            from src.assembled_core.accounting.broker_snapshot_store import (
                broker_snapshot_base_path,
            )

            broker_snapshot_path = broker_snapshot_base_path(
                output_dir, snapshot_run_id
            )
            logger.info(f"Paper broker snapshot written: {broker_snapshot_path}")
        except Exception as e:
            logger.warning(f"Failed to write paper broker snapshot: {e}", exc_info=True)

    # Write accounting report (after positions and reconciliation are computed)
    accounting_report_path = None
    evidence_index_path = None
    evidence_pack_path = None
    evidence_pack_manifest_path = None
    try:
        logger.info("Writing accounting report")

        # Extract costs breakdown from trades_df if available
        costs_breakdown = None
        if not trades_df.empty and "total_cost_cash" in trades_df.columns:
            costs_breakdown = {
                "commission_cash": (
                    float(trades_df["commission_cash"].sum())
                    if "commission_cash" in trades_df.columns
                    else 0.0
                ),
                "spread_cash": (
                    float(trades_df["spread_cash"].sum())
                    if "spread_cash" in trades_df.columns
                    else 0.0
                ),
                "slippage_cash": (
                    float(trades_df["slippage_cash"].sum())
                    if "slippage_cash" in trades_df.columns
                    else 0.0
                ),
                "total_cost_cash": float(trades_df["total_cost_cash"].sum()),
            }

        # Write accounting report (CSV and JSON)
        csv_path = write_accounting_report_csv(
            positions_result=positions_result,
            output_dir=output_dir,
            run_id=run_id,
            as_of=as_of_date,
            start_cash=start_cash,
            reconciliation_result=reconciliation_result,
            # B-acct-3: feed the GRADED reconciliation_ok (None/"unverified" on a
            # paper_view run, True/False on a real stored_snapshot reconcile) so
            # the report never records the trivial paper-vs-paper self-comparison
            # True. Mirrors the value passed to write_evidence_index_json below.
            reconciliation_ok=reconciliation_ok,
            ledger_pack_path=ledger_base.relative_to(output_dir).as_posix(),
            reconcile_report_path=(
                reconcile_report_path.as_posix() if reconcile_report_path else None
            ),
            costs_breakdown=costs_breakdown,
            broker_meta=broker_meta,
        )
        _ = write_accounting_report_json(
            positions_result=positions_result,
            output_dir=output_dir,
            run_id=run_id,
            as_of=as_of_date,
            start_cash=start_cash,
            reconciliation_result=reconciliation_result,
            reconciliation_ok=reconciliation_ok,
            ledger_pack_path=ledger_base.relative_to(output_dir).as_posix(),
            reconcile_report_path=(
                reconcile_report_path.as_posix() if reconcile_report_path else None
            ),
            costs_breakdown=costs_breakdown,
            broker_meta=broker_meta,
        )

        # Use CSV path as primary report path
        accounting_report_path = csv_path.relative_to(output_dir)
        logger.info(f"Accounting report written: {accounting_report_path}")

        # Write evidence index (links all accounting-related artifacts)
        # Evidence pack expects file paths, not directories (ledger_events.parquet, snapshot_*.json)
        date_str = as_of_ts.strftime("%Y-%m-%d")
        ledger_events_path = ledger_base / "ledger_events.parquet"
        broker_snapshot_file = (
            (broker_snapshot_path / f"snapshot_{date_str}.json")
            if broker_snapshot_path is not None
            else None
        )
        try:
            evidence_paths = {
                "broker_snapshot_path": broker_snapshot_file,
                "ledger_pack_path": ledger_events_path,
                "reconcile_report_path": (
                    output_dir / reconcile_report_path
                    if reconcile_report_path
                    else None
                ),
                "accounting_report_path": (
                    output_dir / accounting_report_path
                    if accounting_report_path
                    else None
                ),
                "manifest_path": None,
            }
            evidence_json_path = write_evidence_index_json(
                output_dir=output_dir,
                run_id=run_id,
                as_of_date=as_of_date,
                paths=evidence_paths,
                broker_meta=broker_meta,
                reconciliation_ok=reconciliation_ok,
            )
            evidence_index_path = evidence_json_path.relative_to(output_dir)
            logger.info(f"Evidence index written: {evidence_index_path}")

            # Build evidence pack if requested
            if write_evidence_pack:
                try:
                    pack_result = build_evidence_pack(
                        output_dir=output_dir,
                        run_id=run_id,
                        as_of_date=as_of_date,
                        include_optional=True,
                    )
                    evidence_pack_path = pack_result["pack_path"]
                    evidence_pack_manifest_path = pack_result["pack_manifest_path"]
                    logger.info(
                        f"Evidence pack created: {evidence_pack_path} "
                        f"({pack_result['n_files']} files)"
                    )
                except Exception as e:  # best-effort, should not fail the run
                    logger.warning(f"Failed to build evidence pack: {e}", exc_info=True)
        except Exception as e:  # best-effort, should not fail the run
            logger.warning(f"Failed to write evidence index: {e}", exc_info=True)

    except Exception as e:
        logger.warning(f"Accounting report generation failed: {e}", exc_info=True)

    def _posix_path(p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            return p.as_posix() if isinstance(p, Path) else str(p).replace("\\", "/")
        except Exception:
            return str(p).replace("\\", "/")

    # ── B-acct-1 fail-closed escalation ──────────────────────────────────────
    # All reports (reconcile CSV/JSON, accounting CSV/JSON, evidence index/pack)
    # are now written, so the failure is fully recorded in the EOD pack. Only NOW
    # do we raise — and only in live/paper mode (pending_reconcile_error is None
    # in backtest mode by construction). This guarantees the recorded failure is
    # never lost to a raise, while still giving the live caller a hard signal it
    # cannot ignore.
    if pending_reconcile_error is not None:
        logger.error(
            "[Reconciliation] Escalating fail-closed (live/paper): %s",
            pending_reconcile_error,
        )
        raise pending_reconcile_error

    return {
        "ledger_pack_path": ledger_base.relative_to(output_dir).as_posix(),
        "positions_df": positions_df,
        "cash_balance": cash_balance,
        "reconciliation_result": reconciliation_result,
        "reconcile_report_path": (
            _posix_path(reconcile_report_path) if reconcile_report_path else None
        ),
        "reconciliation_ok": reconciliation_ok,
        "reconciliation_severity": reconciliation_severity,
        "reconciliation_violations": reconciliation_violations,
        "reconciliation_blocked": reconciliation_blocked,
        "accounting_report_path": (
            _posix_path(accounting_report_path) if accounting_report_path else None
        ),
        "broker_snapshot_path": (
            broker_snapshot_path.relative_to(output_dir).as_posix()
            if broker_snapshot_path
            else None
        ),
        "broker_meta": broker_meta,
        "evidence_index_path": (
            _posix_path(evidence_index_path) if evidence_index_path else None
        ),
        "evidence_pack_path": evidence_pack_path,
        "evidence_pack_manifest_path": evidence_pack_manifest_path,
    }
