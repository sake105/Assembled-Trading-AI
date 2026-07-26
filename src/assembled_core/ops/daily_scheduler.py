"""
daily_scheduler.py — Autonomous daily operations orchestrator.

Ties together ingest, post-trade analysis, reconcile, and health-check
workers into a coherent autonomous daily cycle. Uses stdlib only — no
external scheduling libraries.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    """Result of a single worker execution."""

    worker_name: str
    status: str  # "ok" | "skip" | "error"
    duration_s: float
    error_msg: Optional[str] = None


def _ingest_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Download EOD price data for the universe and save to output_dir."""
    t0 = time.monotonic()
    if dry_run:
        logger.info("[SKIP] ingest: dry_run=True")
        return WorkerResult(
            worker_name="ingest_worker", status="skip", duration_s=time.monotonic() - t0
        )
    try:
        import yfinance as yf
        from src.assembled_core.data.universe_etf import (
            get_all_symbols,
            load_etf_universe,
        )

        universe = load_etf_universe()
        symbols = get_all_symbols(universe)
        if not symbols:
            logger.warning("[SKIP] ingest: no symbols in universe")
            return WorkerResult(
                worker_name="ingest_worker",
                status="skip",
                duration_s=time.monotonic() - t0,
            )

        # Download last 2 trading days to catch the most recent close
        raw = yf.download(symbols, period="2d", progress=False, auto_adjust=True)
        if raw.empty:
            logger.warning("[WARN] ingest: yfinance returned empty data")
            return WorkerResult(
                worker_name="ingest_worker",
                status="skip",
                duration_s=time.monotonic() - t0,
            )

        out_path = Path(output_dir) / f"prices_{date_str}.parquet"
        if isinstance(raw.columns, __import__("pandas").MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw
        closes.to_parquet(str(out_path))
        logger.info("[OK] ingest: saved %d symbols to %s", len(symbols), out_path)
        return WorkerResult(
            worker_name="ingest_worker", status="ok", duration_s=time.monotonic() - t0
        )
    except ImportError as exc:
        logger.info("[SKIP] ingest: dependency not available (%s)", exc)
        return WorkerResult(
            worker_name="ingest_worker", status="skip", duration_s=time.monotonic() - t0
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[ERROR] ingest: %s", msg)
        return WorkerResult(
            worker_name="ingest_worker",
            status="error",
            duration_s=time.monotonic() - t0,
            error_msg=msg,
        )


def _news_fetch_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Run news worker + RSS→sentiment bridge + sentiment fusion."""
    t0 = time.monotonic()
    import subprocess
    import sys as _sys

    steps = [
        # 1. Refresh RSS events
        [_sys.executable, "scripts/run_news_worker.py", "--once"],
        # 2. Bridge RSS events → sentiment parquet
        [_sys.executable, "scripts/convert_rss_events_to_sentiment.py"],
        # 3. Fuse all sources
        [_sys.executable, "scripts/fuse_news_sentiment.py", "--update-primary"],
    ]
    if dry_run:
        return WorkerResult(
            worker_name="news_fetch_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )

    errors = []
    for cmd in steps:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode != 0:
                errors.append(f"{cmd[1]}: rc={res.returncode}")
        except Exception as exc:
            errors.append(f"{cmd[1]}: {exc}")

    if errors:
        return WorkerResult(
            worker_name="news_fetch_worker",
            status="error",
            duration_s=time.monotonic() - t0,
            error_msg="; ".join(errors),
        )
    return WorkerResult(
        worker_name="news_fetch_worker", status="ok", duration_s=time.monotonic() - t0
    )


def _post_trade_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Run post-trade analysis and write report to output_dir."""
    t0 = time.monotonic()
    if dry_run:
        logger.info("[SKIP] post_trade: dry_run=True")
        return WorkerResult(
            worker_name="post_trade_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    try:
        from src.assembled_core.qa import post_trade_analyzer as pta

        # Locate the most recent fills file in output_dir
        fills_path = Path(output_dir) / "fills.parquet"
        prices_path = Path(output_dir) / f"prices_{date_str}.parquet"
        if not fills_path.exists():
            logger.info("[SKIP] post_trade: no fills file at %s", fills_path)
            return WorkerResult(
                worker_name="post_trade_worker",
                status="skip",
                duration_s=time.monotonic() - t0,
            )

        import pandas as pd

        fills_df = pd.read_parquet(str(fills_path))
        prices_df = pd.read_parquet(str(prices_path)) if prices_path.exists() else None

        report = pta.run_post_trade_analysis(fills_df, prices_df)  # type: ignore[attr-defined]
        report_path = Path(output_dir) / f"post_trade_{date_str}.json"
        import json

        report_path.write_text(
            json.dumps(report, default=str, indent=2), encoding="utf-8"
        )
        logger.info("[OK] post_trade: report written to %s", report_path)
        return WorkerResult(
            worker_name="post_trade_worker",
            status="ok",
            duration_s=time.monotonic() - t0,
        )
    except ImportError as exc:
        logger.info("[SKIP] post_trade: not available (%s)", exc)
        return WorkerResult(
            worker_name="post_trade_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[ERROR] post_trade: %s", msg)
        return WorkerResult(
            worker_name="post_trade_worker",
            status="error",
            duration_s=time.monotonic() - t0,
            error_msg=msg,
        )


def _feedback_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Run FeedbackLoopController after post-trade data has been written.

    Loads the learning store and the most recent factor panel (if available),
    then calls run_feedback_check(). Logs the result with [FEEDBACK] prefix.
    Returns a WorkerResult matching the existing scheduler contract.
    """
    t0 = time.monotonic()
    if dry_run:
        logger.info("[FEEDBACK] dry_run=True — skipping feedback check")
        return WorkerResult(
            worker_name="feedback_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    try:
        import pandas as pd
        from src.assembled_core.ml.feedback_loop import (  # type: ignore
            FeedbackLoopConfig,
            FeedbackLoopController,
        )
        from src.assembled_core.qa.learning_store import (
            DEFAULT_LEARNING_STORE_PATH,
            load_learning_records_as_dataframe,
        )

        out_path = Path(output_dir)

        # Locate learning store — fall back to module default if not in output_dir
        local_store = out_path / "post_trade_learning.jsonl"
        learning_store_path = (
            local_store if local_store.exists() else Path(DEFAULT_LEARNING_STORE_PATH)
        )

        # Locate current model — use a best-effort path; FeedbackLoopController
        # handles a missing model gracefully via its internal guards.
        model_candidates = sorted(out_path.glob("model_*.pkl"), reverse=True)
        current_model_path = (
            model_candidates[0] if model_candidates else out_path / "model.pkl"
        )

        # Load recent factor panel — use most recent factor_scores parquet if present
        panel_files = sorted(out_path.glob("factor_scores_*.parquet"), reverse=True)
        if panel_files:
            panel_df = pd.read_parquet(str(panel_files[0]))
        else:
            panel_df = load_learning_records_as_dataframe(learning_store_path)

        controller = FeedbackLoopController(
            config=FeedbackLoopConfig(),
            state_dir=out_path / "feedback_state",
        )
        result = controller.run_feedback_check(
            learning_store_path=learning_store_path,
            current_model_path=current_model_path,
            panel_df=panel_df,
        )

        logger.info(
            "[FEEDBACK] check complete — signals=%d retrain=%s deployed=%s report=%s",
            result.active_signal_count,
            result.retrain_triggered,
            result.new_model_deployed,
            result.report_path,
        )
        return WorkerResult(
            worker_name="feedback_worker", status="ok", duration_s=time.monotonic() - t0
        )
    except ImportError as exc:
        logger.info("[FEEDBACK] not available (%s)", exc)
        return WorkerResult(
            worker_name="feedback_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[FEEDBACK] error: %s", msg)
        return WorkerResult(
            worker_name="feedback_worker",
            status="error",
            duration_s=time.monotonic() - t0,
            error_msg=msg,
        )


def _reconcile_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Reconcile paper ledger against fills and write reconciliation report."""
    t0 = time.monotonic()
    if dry_run:
        logger.info("[SKIP] reconcile: dry_run=True")
        return WorkerResult(
            worker_name="reconcile_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    try:
        from src.assembled_core.data.ledger_store import LedgerStore

        db_path = Path(output_dir) / "paper_ledger.db"
        ledger = LedgerStore(db_path=db_path)

        positions = ledger.get_positions()
        cash = ledger.get_cash()
        equity_curve = ledger.load_equity_curve()

        import json

        report = {
            "date": date_str,
            "cash": cash,
            "n_positions": len(positions),
            "positions": (
                positions.to_dict(orient="records") if not positions.empty else []
            ),
            "equity_curve_rows": len(equity_curve),
            "last_equity": (
                float(equity_curve["equity"].iloc[-1])
                if not equity_curve.empty
                else cash
            ),
        }
        report_path = Path(output_dir) / f"reconcile_{date_str}.json"
        report_path.write_text(
            json.dumps(report, default=str, indent=2), encoding="utf-8"
        )
        logger.info(
            "[OK] reconcile: %d positions, cash=%.2f, written to %s",
            len(positions),
            cash,
            report_path,
        )
        return WorkerResult(
            worker_name="reconcile_worker",
            status="ok",
            duration_s=time.monotonic() - t0,
        )
    except ImportError as exc:
        logger.info("[SKIP] reconcile: not available (%s)", exc)
        return WorkerResult(
            worker_name="reconcile_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[ERROR] reconcile: %s", msg)
        return WorkerResult(
            worker_name="reconcile_worker",
            status="error",
            duration_s=time.monotonic() - t0,
            error_msg=msg,
        )


def _health_check_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Health check — verifies output directory, data freshness, and module availability."""
    t0 = time.monotonic()
    path = Path(output_dir)
    issues: list[str] = []

    # 1. Output directory writability
    try:
        if not path.exists():
            raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
        test_file = path / ".health_check_probe"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
    except Exception as exc:
        issues.append(f"output_dir_not_writable: {exc}")

    # 2. Data freshness: check if today's price file exists
    prices_path = path / f"prices_{date_str}.parquet"
    if not prices_path.exists():
        issues.append(f"price_data_missing: {prices_path.name} not found")
    else:
        # Check file age
        mtime = prices_path.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        if age_hours > 26:
            issues.append(
                f"price_data_stale: {prices_path.name} is {age_hours:.1f}h old"
            )

    # 3. Module availability check
    optional_modules = {
        "sklearn": "ml models (ridge, random_forest)",
        "yfinance": "price data ingestion",
        "pandas": "data processing",
        "numpy": "numerical computation",
    }
    for mod, purpose in optional_modules.items():
        try:
            __import__(mod)
        except ImportError:
            if mod in ("pandas", "numpy"):
                issues.append(f"critical_module_missing: {mod} ({purpose})")
            else:
                logger.debug(
                    "[HEALTH] optional module unavailable: %s (%s)", mod, purpose
                )

    if issues:
        msg = "; ".join(issues)
        logger.warning("[WARN] health_check: %s", msg)
        # Return "ok" for non-critical issues (data freshness), "error" only for critical
        has_critical = any("critical" in i or "not_writable" in i for i in issues)
        status = "error" if has_critical else "ok"
        return WorkerResult(
            worker_name="health_check_worker",
            status=status,
            duration_s=time.monotonic() - t0,
            error_msg=msg if has_critical else None,
        )

    logger.info("[OK] health_check: all checks passed")
    return WorkerResult(
        worker_name="health_check_worker",
        status="ok",
        duration_s=time.monotonic() - t0,
    )


def _retrain_scheduler_worker(
    date_str: str, output_dir: str, dry_run: bool
) -> WorkerResult:
    """Run RetrainingScheduler after feedback_worker to evaluate 5 retrain signals.

    Reads state from output_dir (equity curve, IC series, regime series) and
    produces a RetrainingRecommendation. auto_deploy is always False.
    Log prefix: [RETRAIN-SCHED]
    """
    t0 = time.monotonic()
    if dry_run:
        logger.info("[RETRAIN-SCHED] dry_run=True — skipping retrain scheduler")
        return WorkerResult(
            worker_name="retrain_scheduler_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    try:
        import json

        from src.assembled_core.ml.retraining_scheduler import (
            RetrainingScheduler,
        )

        out_path = Path(output_dir)
        scheduler = RetrainingScheduler()

        import pandas as _sched_pd

        # --- Optional: load equity curve ---
        equity_since_retrain = None
        try:
            eq_candidates = sorted(
                out_path.glob("equity_curve_*.parquet"), reverse=True
            )
            if not eq_candidates:
                eq_candidates = sorted(
                    out_path.glob("equity_curve.parquet"), reverse=True
                )
            if eq_candidates:
                eq_df = _sched_pd.read_parquet(str(eq_candidates[0]))
                if "equity" in eq_df.columns:
                    equity_since_retrain = eq_df["equity"]
        except Exception as _e:
            logger.debug("[daily_scheduler] equity_curve load skipped: %s", _e)

        # --- Optional: load IC series ---
        ic_series = None
        try:
            ic_candidates = sorted(out_path.glob("ic_series_*.parquet"), reverse=True)
            if ic_candidates:
                ic_df = _sched_pd.read_parquet(str(ic_candidates[0]))
                if "ic" in ic_df.columns:
                    ic_series = ic_df["ic"]
        except Exception as _e:
            logger.debug("[daily_scheduler] ic_series load skipped: %s", _e)

        # --- Optional: load regime series ---
        regime_series = None
        try:
            reg_candidates = sorted(out_path.glob("regime_*.parquet"), reverse=True)
            if reg_candidates:
                reg_df = _sched_pd.read_parquet(str(reg_candidates[0]))
                if "regime" in reg_df.columns:
                    regime_series = reg_df["regime"]
        except Exception as _e:
            logger.debug("[daily_scheduler] regime_series load skipped: %s", _e)

        # --- Optional: load last retrain date ---
        model_last_trained_date = None
        try:
            state_file = out_path / "feedback_state" / "feedback_state.json"
            if state_file.exists():
                state_data = json.loads(
                    state_file.read_text(encoding="ascii", errors="replace")
                )
                last_retrain_str = state_data.get("last_retrain_date")
                if last_retrain_str:
                    from datetime import date as _date

                    model_last_trained_date = _date.fromisoformat(
                        str(last_retrain_str)[:10]
                    )
        except Exception as _e:
            logger.debug("[daily_scheduler] last_retrain_date load skipped: %s", _e)

        rec = scheduler.evaluate(
            model_last_trained_date=model_last_trained_date,
            ic_series=ic_series,
            equity_since_retrain=equity_since_retrain,
            regime_series=regime_series,
        )

        # Persist recommendation JSON
        rec_path = out_path / f"retrain_recommendation_{date_str}.json"
        rec_dict = {
            "checked_at": rec.checked_at,
            "signals_fired": rec.signals_fired,
            "decision": rec.decision,
            "auto_deploy": rec.auto_deploy,
            "notes": rec.notes,
            "signal_details": [
                {
                    "name": d.name,
                    "fired": d.fired,
                    "reason": d.reason,
                    "value": d.value,
                }
                for d in rec.signal_details
            ],
        }
        rec_path.write_text(json.dumps(rec_dict, indent=2), encoding="utf-8")

        logger.info(
            "[RETRAIN-SCHED] decision=%s signals=%d auto_deploy=%s report=%s",
            rec.decision,
            rec.signals_fired,
            rec.auto_deploy,
            rec_path,
        )
        return WorkerResult(
            worker_name="retrain_scheduler_worker",
            status="ok",
            duration_s=time.monotonic() - t0,
        )
    except ImportError as exc:
        logger.info("[RETRAIN-SCHED] not available (%s)", exc)
        return WorkerResult(
            worker_name="retrain_scheduler_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[RETRAIN-SCHED] error: %s", msg)
        return WorkerResult(
            worker_name="retrain_scheduler_worker",
            status="error",
            duration_s=time.monotonic() - t0,
            error_msg=msg,
        )


def _factor_curation_worker(
    date_str: str, output_dir: str, dry_run: bool
) -> WorkerResult:
    """Quarterly factor curation: compute a heuristic IC t-stat per factor, flag decayed ones.

    Only runs when the date is in the first week of a quarter (Jan/Apr/Jul/Oct).
    Quality gate is a raw IC t-stat (IC_mean / IC_std * sqrt(n)) reported as
    ``ic_tstat`` — NOT a Deflated Sharpe Ratio (no deflation / multiple-testing
    correction is applied). Factors with ic_tstat < 0.5 are flagged for removal
    from active bundles (advisory report only). (A35)
    """
    t0 = time.monotonic()

    # Quarter check: only run in first 7 days of Jan/Apr/Jul/Oct
    from datetime import date as _date, datetime as _dt, timezone as _tz

    try:
        d = _date.fromisoformat(date_str)
    except (ValueError, TypeError):
        # date.today() sweep: UTC for stable quarter detection (avoids late-CET
        # rollover producing wrong quarter at month boundaries).
        d = _dt.now(tz=_tz.utc).date()
    if d.month not in (1, 4, 7, 10) or d.day > 7:
        logger.debug("[SKIP] factor_curation: not a quarterly curation window")
        return WorkerResult(
            worker_name="factor_curation_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )

    if dry_run:
        logger.info("[SKIP] factor_curation: dry_run=True")
        return WorkerResult(
            worker_name="factor_curation_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )

    try:
        import json

        import pandas as pd
        from src.assembled_core.qa.factor_analysis import (
            compute_factor_half_life,
            compute_ic_decay_curve,
        )

        out_path = Path(output_dir)
        # Find most recent factor scores file
        score_files = sorted(out_path.glob("factor_scores_*.parquet"), reverse=True)
        if not score_files:
            logger.info(
                "[SKIP] factor_curation: no factor score files found in %s", output_dir
            )
            return WorkerResult(
                worker_name="factor_curation_worker",
                status="skip",
                duration_s=time.monotonic() - t0,
            )

        panel_df = pd.read_parquet(str(score_files[0]))
        if panel_df.empty:
            logger.info("[SKIP] factor_curation: empty factor scores file")
            return WorkerResult(
                worker_name="factor_curation_worker",
                status="skip",
                duration_s=time.monotonic() - t0,
            )

        # Identify factor columns (exclude metadata columns)
        meta_cols = {
            "symbol",
            "timestamp",
            "date",
            "returns",
            "forward_returns",
            "target",
            "close",
            "open",
            "high",
            "low",
            "volume",
        }
        factor_cols = [
            c
            for c in panel_df.columns
            if c not in meta_cols and panel_df[c].dtype in ("float64", "float32")
        ]

        curation_report: dict[str, Any] = {
            "date": date_str,
            "factors": {},
            "flagged_for_removal": [],
        }
        # NOTE: this is a heuristic IC t-stat (IC_mean / IC_std * sqrt(n)), NOT a
        # Deflated Sharpe Ratio — there is NO deflation / multiple-testing
        # correction applied here. Field name kept honest as `ic_tstat` (A35).
        ic_tstat_threshold = 0.5

        for factor_col in factor_cols:
            try:
                ic_curve = compute_ic_decay_curve(
                    panel_df, factor_col, max_horizon_days=60
                )
                half_life = compute_factor_half_life(ic_curve)
                # Heuristic IC t-stat from IC stats: IC_mean / IC_std * sqrt(n).
                # NOT a Deflated Sharpe Ratio — no multiple-testing correction.
                if "ic" in ic_curve.columns:
                    ic_series = ic_curve["ic"].dropna()
                    if len(ic_series) >= 10:
                        ic_mean = ic_series.mean()
                        ic_std = ic_series.std()
                        n = len(ic_series)
                        ic_tstat = (
                            (ic_mean / ic_std * (n**0.5)) if ic_std > 1e-9 else 0.0
                        )
                    else:
                        ic_tstat = 0.0
                else:
                    ic_tstat = 0.0

                curation_report["factors"][factor_col] = {
                    "ic_tstat": round(float(ic_tstat), 4),
                    "half_life_days": round(float(half_life), 1) if half_life else None,
                    "status": (
                        "active" if ic_tstat >= ic_tstat_threshold else "flagged"
                    ),
                }
                if ic_tstat < ic_tstat_threshold:
                    curation_report["flagged_for_removal"].append(factor_col)
            except Exception as exc:
                logger.debug(
                    "factor_curation: error processing %s: %s", factor_col, exc
                )
                curation_report["factors"][factor_col] = {
                    "ic_tstat": None,
                    "half_life_days": None,
                    "status": "error",
                    "error": str(exc),
                }

        report_path = out_path / f"factor_curation_{date_str}.json"
        report_path.write_text(
            json.dumps(curation_report, default=str, indent=2), encoding="utf-8"
        )
        n_flagged = len(curation_report["flagged_for_removal"])
        logger.info(
            "[OK] factor_curation: %d factors analyzed, %d flagged (IC t-stat < %.1f), report at %s",
            len(factor_cols),
            n_flagged,
            ic_tstat_threshold,
            report_path,
        )
        return WorkerResult(
            worker_name="factor_curation_worker",
            status="ok",
            duration_s=time.monotonic() - t0,
        )
    except ImportError as exc:
        logger.info("[SKIP] factor_curation: not available (%s)", exc)
        return WorkerResult(
            worker_name="factor_curation_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[ERROR] factor_curation: %s", msg)
        return WorkerResult(
            worker_name="factor_curation_worker",
            status="error",
            duration_s=time.monotonic() - t0,
            error_msg=msg,
        )


def _alert_health_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Check system-level alert conditions and dispatch via AlertManager (Phase 11).

    CRITICAL: kill-switch engaged, drawdown > 20%, reconciliation failure.
    WARNING: IC degradation, model stale, feature drift.
    INFO: regime change, new equity highs/lows.
    """
    t0 = time.monotonic()
    if dry_run:
        logger.info("[ALERT] dry_run=True -- skipping alert health check")
        return WorkerResult(
            worker_name="alert_health_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    try:
        from src.assembled_core.ops.alert_manager import AlertManager

        mgr = AlertManager(output_dir=str(Path(output_dir) / "alerts"))
        out_path = Path(output_dir)

        # CRITICAL: Kill-switch state
        try:
            from src.assembled_core.execution.kill_switch import (
                is_kill_switch_engaged,
            )

            if is_kill_switch_engaged():
                mgr.alert(
                    "CRITICAL",
                    "kill_switch",
                    "Kill switch is currently engaged",
                    details={"date": date_str},
                )
        except Exception as _ke:
            logger.debug("[ALERT] kill_switch check skipped: %s", _ke)

        # CRITICAL: Reconciliation failure (error files present)
        try:
            error_files = list(out_path.glob("*.error"))
            if error_files:
                mgr.alert(
                    "CRITICAL",
                    "reconciliation",
                    f"Error files found: {[f.name for f in error_files[:3]]}",
                    details={"date": date_str, "n_errors": len(error_files)},
                )
        except Exception as _re:
            logger.debug("[ALERT] reconciliation error check skipped: %s", _re)

        # CRITICAL/WARNING: Reconciliation status escalation (A12).
        # The *.error glob above never matches — nothing in src/ writes *.error.
        # The authoritative reconcile status lives in reconcile_latest.json
        # (ops/reconcile.py sets status="FAIL"/"OK"), with a per-date fallback.
        # Escalate FAIL/WARN through the real delivering AlertManager so a failed
        # cash/equity/positions invariant is actually surfaced, not just recorded.
        try:
            import json

            from src.assembled_core.ops.alerting import AlertManager as _DeliverMgr

            rec_report = None
            for _name in (
                "reconcile_latest.json",
                f"reconcile_{date_str}.json",
            ):
                _p = out_path / _name
                if _p.exists():
                    try:
                        rec_report = json.loads(_p.read_text(encoding="utf-8"))
                        break
                    except Exception as _rpe:
                        logger.warning(
                            "[ALERT] reconcile report %s unreadable: %s", _name, _rpe
                        )

            rec_status = str((rec_report or {}).get("status", "")).upper()
            if rec_status in ("FAIL", "WARN"):
                rule = (
                    "reconciliation_fail"
                    if rec_status == "FAIL"
                    else "reconciliation_warn"
                )
                cash = (rec_report or {}).get("cash") or {}
                notes = (rec_report or {}).get("notes") or []
                ctx = {
                    "cash_diff_bps": cash.get("delta"),
                    "max_qty_diff": None,
                    "violation_count": len(notes),
                    "first_violation": notes[0] if notes else None,
                    "date": date_str,
                }
                _DeliverMgr().fire(rule, ctx)
                logger.warning(
                    "[ALERT] reconcile status=%s — fired %s (notes=%d)",
                    rec_status,
                    rule,
                    len(notes),
                )
        except Exception as _rse2:
            logger.warning("[ALERT] reconcile status escalation failed: %s", _rse2)

        # WARNING: Signal health / IC degradation
        try:
            import json

            diag_path = out_path / "diagnostics" / f"signal_health_{date_str}.json"
            if diag_path.exists():
                diag_data = json.loads(diag_path.read_text(encoding="utf-8"))
                n_alerts = diag_data.get("n_alerts", 0)
                if n_alerts > 0:
                    mgr.alert(
                        "WARNING",
                        "signal_diagnostics",
                        f"IC degradation: {n_alerts} factor alerts",
                        details={
                            "alerts": diag_data.get("alerts", [])[:5],
                            "date": date_str,
                        },
                    )
        except Exception as _sd:
            logger.debug("[ALERT] signal_health check skipped: %s", _sd)

        # WARNING: Model stale (no model updated in >30 days)
        try:
            model_files = sorted(out_path.glob("model_*.pkl"), reverse=True)
            if not model_files:
                model_files = list(out_path.glob("model.pkl"))
            if model_files:
                model_age_days = (time.time() - model_files[0].stat().st_mtime) / 86400
                if model_age_days > 30:
                    mgr.alert(
                        "WARNING",
                        "model_staleness",
                        f"Model file is {model_age_days:.0f} days old (>30)",
                        details={
                            "model": model_files[0].name,
                            "age_days": round(model_age_days, 1),
                        },
                    )
        except Exception as _ms:
            logger.debug("[ALERT] model_stale check skipped: %s", _ms)

        # INFO: Risk state not WATCH (regime change)
        try:
            import json

            kpis_path = out_path / "run_kpis.json"
            if kpis_path.exists():
                kpis = json.loads(kpis_path.read_text(encoding="utf-8"))
                risk_state = kpis.get("risk_state")
                if risk_state and risk_state not in ("WATCH", None):
                    mgr.alert(
                        "INFO",
                        "risk_state_machine",
                        f"Risk state: {risk_state}",
                        details={"state": risk_state, "date": date_str},
                    )
        except Exception as _rse:
            logger.debug("[ALERT] regime_change check skipped: %s", _rse)

        alert_file = mgr.flush_to_json()
        logger.info("[ALERT] health check done: date=%s file=%s", date_str, alert_file)
        return WorkerResult(
            worker_name="alert_health_worker",
            status="ok",
            duration_s=time.monotonic() - t0,
        )
    except ImportError as exc:
        logger.info("[ALERT] not available (%s)", exc)
        return WorkerResult(
            worker_name="alert_health_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[ALERT] alert_health_worker error: %s", msg)
        return WorkerResult(
            worker_name="alert_health_worker",
            status="error",
            duration_s=time.monotonic() - t0,
            error_msg=msg,
        )


def _kpi_export_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Export Prometheus-format KPI metrics to output/metrics/ (Phase 11 / Plan C15)."""
    t0 = time.monotonic()
    if dry_run:
        logger.info("[KPI] dry_run=True -- skipping KPI export")
        return WorkerResult(
            worker_name="kpi_export_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    try:
        import json

        from src.assembled_core.ops.metrics_exporter import export_metrics

        out_path = Path(output_dir)
        metrics: dict = {}

        # Load run_kpis.json if present
        kpis_path = out_path / "run_kpis.json"
        if kpis_path.exists():
            try:
                kpis = json.loads(kpis_path.read_text(encoding="utf-8"))
                mults = kpis.get("multipliers") or {}
                if "georisk" in mults:
                    metrics["assembled_georisk_multiplier"] = float(mults["georisk"])
                if "profit_lock" in mults:
                    metrics["assembled_profit_lock_multiplier"] = float(
                        mults["profit_lock"]
                    )
                if "final_exposure_multiplier" in mults:
                    metrics["assembled_exposure_multiplier"] = float(
                        mults["final_exposure_multiplier"]
                    )
                targets_summary = kpis.get("targets_summary") or {}
                if "n_targets" in targets_summary:
                    metrics["assembled_targets_count"] = float(
                        targets_summary["n_targets"]
                    )
            except Exception as _kje:
                logger.debug("[KPI] run_kpis.json parse error: %s", _kje)

        # Load reconcile report if present
        reconcile_path = out_path / f"reconcile_{date_str}.json"
        if reconcile_path.exists():
            try:
                rec = json.loads(reconcile_path.read_text(encoding="utf-8"))
                if rec.get("n_positions") is not None:
                    metrics["assembled_n_positions"] = float(rec["n_positions"])
                if rec.get("last_equity") is not None:
                    metrics["assembled_equity"] = float(rec["last_equity"])
                if rec.get("cash") is not None:
                    metrics["assembled_cash"] = float(rec["cash"])
            except Exception as _rje:
                logger.debug("[KPI] reconcile parse error: %s", _rje)

        # Signal diagnostics alert count
        diag_path = out_path / "diagnostics" / f"signal_health_{date_str}.json"
        if diag_path.exists():
            try:
                diag = json.loads(diag_path.read_text(encoding="utf-8"))
                metrics["assembled_signal_health_alerts"] = float(
                    diag.get("n_alerts", 0)
                )
                metrics["assembled_signal_health_factors"] = float(
                    diag.get("n_factors", 0)
                )
            except Exception as _dge:
                logger.debug("[KPI] signal_health parse error: %s", _dge)

        if not metrics:
            logger.info("[KPI] no metrics available for %s -- skip", date_str)
            return WorkerResult(
                worker_name="kpi_export_worker",
                status="skip",
                duration_s=time.monotonic() - t0,
            )

        metrics_dir = out_path / "metrics"
        export_result = export_metrics(
            metrics,
            labels={"date": date_str},
            path=metrics_dir / "assembled.prom",
        )
        logger.info(
            "[KPI] exported %d metrics to %s",
            export_result.get("metrics_count", 0),
            export_result.get("file"),
        )
        return WorkerResult(
            worker_name="kpi_export_worker",
            status="ok",
            duration_s=time.monotonic() - t0,
        )
    except ImportError as exc:
        logger.info("[KPI] not available (%s)", exc)
        return WorkerResult(
            worker_name="kpi_export_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[KPI] kpi_export_worker error: %s", msg)
        return WorkerResult(
            worker_name="kpi_export_worker",
            status="error",
            duration_s=time.monotonic() - t0,
            error_msg=msg,
        )


# Default worker registry (callables that accept date_str, output_dir, dry_run)
_DEFAULT_WORKERS: List[Callable] = [
    _news_fetch_worker,  # refresh RSS events + sentiment fusion (runs first)
    _ingest_worker,
    _post_trade_worker,
    _feedback_worker,
    _retrain_scheduler_worker,  # runs after feedback to evaluate 5 retrain signals
    _reconcile_worker,
    _health_check_worker,
    _factor_curation_worker,
    _alert_health_worker,  # Phase 11: alert conditions after health check
    _kpi_export_worker,  # Phase 11: Prometheus metrics export
]


class DailyScheduler:
    """Orchestrates a sequence of daily operational workers."""

    def __init__(self, workers: Optional[List[Callable]] = None) -> None:
        self.workers: List[Callable] = (
            workers if workers is not None else list(_DEFAULT_WORKERS)
        )

    def run_daily_cycle(
        self,
        date_str: str,
        output_dir: str,
        dry_run: bool = False,
    ) -> List[WorkerResult]:
        """Run all workers in sequence and return their results."""
        results: List[WorkerResult] = []
        logger.info("[START] daily_cycle date=%s dry_run=%s", date_str, dry_run)
        for worker_fn in self.workers:
            name = getattr(worker_fn, "__name__", repr(worker_fn))
            try:
                result = worker_fn(date_str, output_dir, dry_run)
                results.append(result)
                logger.info(
                    "[%s] %s duration=%.3fs",
                    result.status.upper(),
                    name,
                    result.duration_s,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                logger.error("[ERROR] %s caught unhandled exception: %s", name, msg)
                results.append(
                    WorkerResult(
                        worker_name=name,
                        status="error",
                        duration_s=0.0,
                        error_msg=msg,
                    )
                )
        logger.info("[OK] daily_cycle complete workers=%d", len(results))
        return results


# Module-level convenience function backed by a default scheduler instance
def run_daily_cycle(
    date_str: str,
    output_dir: str,
    dry_run: bool = False,
) -> List[WorkerResult]:
    """Run the default daily worker cycle and return results."""
    scheduler = DailyScheduler()
    return scheduler.run_daily_cycle(date_str, output_dir, dry_run)


def build_cycle_summary(results: List[WorkerResult]) -> dict:
    """Build a summary dict from a list of WorkerResult objects."""
    ok = sum(1 for r in results if r.status == "ok")
    skip = sum(1 for r in results if r.status == "skip")
    error = sum(1 for r in results if r.status == "error")
    return {
        "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "total": len(results),
        "ok": ok,
        "skip": skip,
        "error": error,
        "workers": [
            {
                "worker_name": r.worker_name,
                "status": r.status,
                "duration_s": r.duration_s,
                "error_msg": r.error_msg,
            }
            for r in results
        ],
    }


def schedule_loop(
    interval_hours: float,
    output_dir: str,
    dry_run: bool = False,
    max_iterations: Optional[int] = None,
) -> None:
    """Run the daily cycle repeatedly at interval_hours cadence.

    Args:
        interval_hours: Hours to sleep between cycles.
        output_dir: Output directory passed to each cycle.
        dry_run: Passed through to each cycle.
        max_iterations: Stop after this many iterations (None = run forever).
                        Useful for testing.
    """
    scheduler = DailyScheduler()
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        results = scheduler.run_daily_cycle(date_str, output_dir, dry_run)
        summary = build_cycle_summary(results)
        logger.info(
            "[OK] schedule_loop iteration=%d summary=%s",
            iteration + 1,
            summary,
        )
        iteration += 1
        if max_iterations is not None and iteration >= max_iterations:
            break
        time.sleep(interval_hours * 3600)
