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
from typing import Callable, List, Optional

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
        return WorkerResult(worker_name="ingest_worker", status="skip",
                            duration_s=time.monotonic() - t0)
    try:
        from src.assembled_core.data.universe_etf import load_etf_universe, get_all_symbols  # type: ignore
        import yfinance as yf  # type: ignore

        universe = load_etf_universe()
        symbols = get_all_symbols(universe)
        if not symbols:
            logger.warning("[SKIP] ingest: no symbols in universe")
            return WorkerResult(worker_name="ingest_worker", status="skip",
                                duration_s=time.monotonic() - t0)

        # Download last 2 trading days to catch the most recent close
        raw = yf.download(symbols, period="2d", progress=False, auto_adjust=True)
        if raw.empty:
            logger.warning("[WARN] ingest: yfinance returned empty data")
            return WorkerResult(worker_name="ingest_worker", status="skip",
                                duration_s=time.monotonic() - t0)

        out_path = Path(output_dir) / f"prices_{date_str}.parquet"
        if isinstance(raw.columns, __import__("pandas").MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw
        closes.to_parquet(str(out_path))
        logger.info("[OK] ingest: saved %d symbols to %s", len(symbols), out_path)
        return WorkerResult(worker_name="ingest_worker", status="ok",
                            duration_s=time.monotonic() - t0)
    except ImportError as exc:
        logger.info("[SKIP] ingest: dependency not available (%s)", exc)
        return WorkerResult(worker_name="ingest_worker", status="skip",
                            duration_s=time.monotonic() - t0)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[ERROR] ingest: %s", msg)
        return WorkerResult(worker_name="ingest_worker", status="error",
                            duration_s=time.monotonic() - t0, error_msg=msg)


def _post_trade_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Run post-trade analysis and write report to output_dir."""
    t0 = time.monotonic()
    if dry_run:
        logger.info("[SKIP] post_trade: dry_run=True")
        return WorkerResult(worker_name="post_trade_worker", status="skip",
                            duration_s=time.monotonic() - t0)
    try:
        from src.assembled_core.qa import post_trade_analyzer as pta  # type: ignore

        # Locate the most recent fills file in output_dir
        fills_path = Path(output_dir) / "fills.parquet"
        prices_path = Path(output_dir) / f"prices_{date_str}.parquet"
        if not fills_path.exists():
            logger.info("[SKIP] post_trade: no fills file at %s", fills_path)
            return WorkerResult(worker_name="post_trade_worker", status="skip",
                                duration_s=time.monotonic() - t0)

        import pandas as pd
        fills_df = pd.read_parquet(str(fills_path))
        prices_df = pd.read_parquet(str(prices_path)) if prices_path.exists() else None

        report = pta.run_post_trade_analysis(fills_df, prices_df)  # type: ignore[attr-defined]
        report_path = Path(output_dir) / f"post_trade_{date_str}.json"
        import json
        report_path.write_text(json.dumps(report, default=str, indent=2))
        logger.info("[OK] post_trade: report written to %s", report_path)
        return WorkerResult(worker_name="post_trade_worker", status="ok",
                            duration_s=time.monotonic() - t0)
    except ImportError as exc:
        logger.info("[SKIP] post_trade: not available (%s)", exc)
        return WorkerResult(worker_name="post_trade_worker", status="skip",
                            duration_s=time.monotonic() - t0)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[ERROR] post_trade: %s", msg)
        return WorkerResult(worker_name="post_trade_worker", status="error",
                            duration_s=time.monotonic() - t0, error_msg=msg)


def _reconcile_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Reconcile paper ledger against fills and write reconciliation report."""
    t0 = time.monotonic()
    if dry_run:
        logger.info("[SKIP] reconcile: dry_run=True")
        return WorkerResult(worker_name="reconcile_worker", status="skip",
                            duration_s=time.monotonic() - t0)
    try:
        from src.assembled_core.data.ledger_store import LedgerStore  # type: ignore

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
            "positions": positions.to_dict(orient="records") if not positions.empty else [],
            "equity_curve_rows": len(equity_curve),
            "last_equity": float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else cash,
        }
        report_path = Path(output_dir) / f"reconcile_{date_str}.json"
        report_path.write_text(json.dumps(report, default=str, indent=2))
        logger.info("[OK] reconcile: %d positions, cash=%.2f, written to %s",
                    len(positions), cash, report_path)
        return WorkerResult(worker_name="reconcile_worker", status="ok",
                            duration_s=time.monotonic() - t0)
    except ImportError as exc:
        logger.info("[SKIP] reconcile: not available (%s)", exc)
        return WorkerResult(worker_name="reconcile_worker", status="skip",
                            duration_s=time.monotonic() - t0)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[ERROR] reconcile: %s", msg)
        return WorkerResult(worker_name="reconcile_worker", status="error",
                            duration_s=time.monotonic() - t0, error_msg=msg)


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
        test_file.write_text("ok")
        test_file.unlink()
    except Exception as exc:
        issues.append(f"output_dir_not_writable: {exc}")

    # 2. Data freshness: check if today's price file exists
    prices_path = path / f"prices_{date_str}.parquet"
    if not prices_path.exists():
        issues.append(f"price_data_missing: {prices_path.name} not found")
    else:
        # Check file age
        import os
        mtime = os.path.getmtime(str(prices_path))
        age_hours = (time.time() - mtime) / 3600
        if age_hours > 26:
            issues.append(f"price_data_stale: {prices_path.name} is {age_hours:.1f}h old")

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
                logger.debug("[HEALTH] optional module unavailable: %s (%s)", mod, purpose)

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


def _factor_curation_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Quarterly factor curation: compute DSR for all active factors, flag decayed ones.

    Only runs when the date is in the first week of a quarter (Jan/Apr/Jul/Oct).
    Uses Deflated Sharpe Ratio (DSR) as the quality gate — factors with DSR < 0.5
    are flagged for removal from active bundles.
    """
    t0 = time.monotonic()

    # Quarter check: only run in first 7 days of Jan/Apr/Jul/Oct
    from datetime import date as _date
    try:
        d = _date.fromisoformat(date_str)
    except (ValueError, TypeError):
        d = _date.today()
    if d.month not in (1, 4, 7, 10) or d.day > 7:
        logger.debug("[SKIP] factor_curation: not a quarterly curation window")
        return WorkerResult(worker_name="factor_curation_worker", status="skip",
                            duration_s=time.monotonic() - t0)

    if dry_run:
        logger.info("[SKIP] factor_curation: dry_run=True")
        return WorkerResult(worker_name="factor_curation_worker", status="skip",
                            duration_s=time.monotonic() - t0)

    try:
        from src.assembled_core.qa.factor_analysis import (  # type: ignore
            compute_ic_decay_curve,
            compute_factor_half_life,
        )
        import pandas as pd
        import json

        out_path = Path(output_dir)
        # Find most recent factor scores file
        score_files = sorted(out_path.glob("factor_scores_*.parquet"), reverse=True)
        if not score_files:
            logger.info("[SKIP] factor_curation: no factor score files found in %s", output_dir)
            return WorkerResult(worker_name="factor_curation_worker", status="skip",
                                duration_s=time.monotonic() - t0)

        panel_df = pd.read_parquet(str(score_files[0]))
        if panel_df.empty:
            logger.info("[SKIP] factor_curation: empty factor scores file")
            return WorkerResult(worker_name="factor_curation_worker", status="skip",
                                duration_s=time.monotonic() - t0)

        # Identify factor columns (exclude metadata columns)
        meta_cols = {"symbol", "timestamp", "date", "returns", "forward_returns",
                     "target", "close", "open", "high", "low", "volume"}
        factor_cols = [c for c in panel_df.columns if c not in meta_cols and panel_df[c].dtype in ("float64", "float32")]

        curation_report = {"date": date_str, "factors": {}, "flagged_for_removal": []}
        dsr_threshold = 0.5

        for factor_col in factor_cols:
            try:
                ic_curve = compute_ic_decay_curve(panel_df, factor_col, max_horizon_days=60)
                half_life = compute_factor_half_life(ic_curve)
                # Approximate DSR from IC stats (simplified: IC_mean / IC_std * sqrt(n))
                if "ic" in ic_curve.columns:
                    ic_series = ic_curve["ic"].dropna()
                    if len(ic_series) >= 10:
                        ic_mean = ic_series.mean()
                        ic_std = ic_series.std()
                        n = len(ic_series)
                        dsr = (ic_mean / ic_std * (n ** 0.5)) if ic_std > 1e-9 else 0.0
                    else:
                        dsr = 0.0
                else:
                    dsr = 0.0

                curation_report["factors"][factor_col] = {
                    "dsr": round(float(dsr), 4),
                    "half_life_days": round(float(half_life), 1) if half_life else None,
                    "status": "active" if dsr >= dsr_threshold else "flagged",
                }
                if dsr < dsr_threshold:
                    curation_report["flagged_for_removal"].append(factor_col)
            except Exception as exc:
                logger.debug("factor_curation: error processing %s: %s", factor_col, exc)
                curation_report["factors"][factor_col] = {
                    "dsr": None,
                    "half_life_days": None,
                    "status": "error",
                    "error": str(exc),
                }

        report_path = out_path / f"factor_curation_{date_str}.json"
        report_path.write_text(json.dumps(curation_report, default=str, indent=2))
        n_flagged = len(curation_report["flagged_for_removal"])
        logger.info("[OK] factor_curation: %d factors analyzed, %d flagged (DSR < %.1f), report at %s",
                    len(factor_cols), n_flagged, dsr_threshold, report_path)
        return WorkerResult(worker_name="factor_curation_worker", status="ok",
                            duration_s=time.monotonic() - t0)
    except ImportError as exc:
        logger.info("[SKIP] factor_curation: not available (%s)", exc)
        return WorkerResult(worker_name="factor_curation_worker", status="skip",
                            duration_s=time.monotonic() - t0)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[ERROR] factor_curation: %s", msg)
        return WorkerResult(worker_name="factor_curation_worker", status="error",
                            duration_s=time.monotonic() - t0, error_msg=msg)


# Default worker registry (callables that accept date_str, output_dir, dry_run)
_DEFAULT_WORKERS: List[Callable] = [
    _ingest_worker,
    _post_trade_worker,
    _reconcile_worker,
    _health_check_worker,
    _factor_curation_worker,
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
