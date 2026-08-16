# src/assembled_core/api/routers/monitoring.py
"""Monitoring endpoints for QA, Risk, and Drift status.

This module provides simplified monitoring endpoints that aggregate key status
information for dashboards and operational monitoring.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from src.assembled_core.api.models import (
    DriftStatusSummary,
    FeatureDriftItem,
    QAStatusSummary,
    RiskStatusSummary,
)
from src.assembled_core.config import (
    OUTPUT_DIR,
    SUPPORTED_FREQS,
    FreqStr,
    get_base_dir,
)
from src.assembled_core.logging_utils import get_logger
from src.assembled_core.pipeline.io import load_orders
from src.assembled_core.qa.metrics import compute_all_metrics
from src.assembled_core.qa.qa_gates import evaluate_all_gates

router = APIRouter()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Path-traversal safe-roots for caller-supplied filesystem paths
# ---------------------------------------------------------------------------
# Several monitoring endpoints accept a caller-supplied db_path / output_dir and
# open/glob/read it. These are unauthenticated GETs, so an unconstrained path is
# an arbitrary-file-read vector. We confine reads to a small set of legitimate
# roots that cover every documented default:
#   - OUTPUT_DIR            -> repo/output         (config canonical output)
#   - get_base_dir()/src/output -> legacy "src/output" default of these handlers
#   - get_base_dir()/data       -> default "data/paper_ledger.db" location
#   - tempfile.gettempdir()     -> test/CI scratch dirs
# A path is accepted iff it equals a root or lives under one (post-resolve()).
_MON_SAFE_ROOTS: tuple[Path, ...] = (
    OUTPUT_DIR.resolve(),
    (get_base_dir() / "src" / "output").resolve(),
    (get_base_dir() / "data").resolve(),
    Path(tempfile.gettempdir()).resolve(),
)


def _is_safe_monitoring_path(resolved: Path) -> bool:
    """Return True iff *resolved* is one of the monitoring safe roots or below it.

    Args:
        resolved: an already-``.resolve()``-d path.

    Returns:
        True if the path is inside an allowed root, else False.
    """
    return any(resolved == root or root in resolved.parents for root in _MON_SAFE_ROOTS)


@router.get("/monitoring/qa_status", response_model=QAStatusSummary)
def get_qa_status_summary(
    freq: str = Query(default="1d", description="Trading frequency"),
) -> QAStatusSummary:
    """Get simplified QA status summary for monitoring.

    Returns a quick overview of QA gate results and key performance metrics.
    Uses the most recent QA evaluation available (from run_manifest or computed on-the-fly).

    Args:
        freq: Trading frequency ("1d" or "5min"), default "1d"

    Returns:
        QAStatusSummary with overall result, gate counts, and key metrics

    Raises:
        HTTPException: 400 if freq is not supported, 404 if no data found, 500 for errors
    """
    # Validate frequency
    if freq not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}",
        )

    try:
        # Try to load from run manifest first (has most recent evaluation)
        manifest_path = OUTPUT_DIR / f"run_manifest_{freq}.json"
        last_updated = None

        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                # Get gate results if available
                gate_counts = {"ok": 0, "warning": 0, "block": 0}
                overall_result = "UNKNOWN"
                key_metrics = {}

                if "qa_gate_result" in manifest and manifest["qa_gate_result"]:
                    gate_dict = manifest["qa_gate_result"]
                    overall_result = str(
                        gate_dict.get("overall_result", "UNKNOWN")
                    ).upper()
                    gate_counts = {
                        "ok": gate_dict.get("passed_gates", 0),
                        "warning": gate_dict.get("warning_gates", 0),
                        "block": gate_dict.get("blocked_gates", 0),
                        # E-066: SKIPPED is its own state. Omitting it here made
                        # an unchecked gate indistinguishable from a passed one.
                        "skipped": gate_dict.get("skipped_gates", 0),
                    }

                # Get metrics if available
                if "qa_metrics" in manifest and manifest["qa_metrics"]:
                    metrics_dict = manifest["qa_metrics"]
                    key_metrics = {
                        "sharpe_ratio": metrics_dict.get("sharpe_ratio"),
                        "max_drawdown_pct": metrics_dict.get("max_drawdown_pct"),
                        "total_return": metrics_dict.get("total_return"),
                        "cagr": metrics_dict.get("cagr"),
                    }

                # Get timestamp from manifest if available
                if "timestamp" in manifest:
                    try:
                        last_updated = datetime.fromisoformat(
                            str(manifest["timestamp"]).replace("Z", "+00:00")
                        )
                    except Exception as exc:
                        logger.warning(
                            "[Monitoring] failed to parse QA manifest timestamp: %s",
                            exc,
                        )

                return QAStatusSummary(
                    overall_result=overall_result,
                    gate_counts=gate_counts,
                    key_metrics=(
                        key_metrics
                        if key_metrics
                        else {
                            "sharpe_ratio": None,
                            "max_drawdown_pct": None,
                            "total_return": None,
                            "cagr": None,
                        }
                    ),
                    last_updated=last_updated,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load from manifest: {e}, computing on-the-fly"
                )

        # Fallback: Compute on-the-fly from equity/trades
        logger.info(f"Computing QA status from equity/trades for freq={freq}")

        # Load equity
        portfolio_equity_file = OUTPUT_DIR / f"portfolio_equity_{freq}.csv"
        backtest_equity_file = OUTPUT_DIR / f"equity_curve_{freq}.csv"

        equity_df = None
        if portfolio_equity_file.exists():
            equity_df = pd.read_csv(
                portfolio_equity_file, dtype={"timestamp": "string"}
            )
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            last_updated = equity_df["timestamp"].max().to_pydatetime()
        elif backtest_equity_file.exists():
            equity_df = pd.read_csv(backtest_equity_file, dtype={"timestamp": "string"})
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            last_updated = equity_df["timestamp"].max().to_pydatetime()

        if equity_df is None or equity_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No equity file found for freq={freq}. Cannot compute QA status.",
            )

        # Load trades (optional)
        trades_df = None
        try:
            trades_df = load_orders(freq, output_dir=OUTPUT_DIR, strict=False)
            if trades_df.empty:
                trades_df = None
        except Exception as exc:
            logger.warning("[Monitoring] failed to load trades for QA status: %s", exc)

        # Compute metrics
        start_capital = (
            equity_df["equity"].iloc[0] if "equity" in equity_df.columns else 10000.0
        )
        metrics = compute_all_metrics(
            equity=equity_df,
            trades=trades_df,
            start_capital=start_capital,
            freq=freq,
            risk_free_rate=0.0,
        )

        # Evaluate gates
        gate_summary = evaluate_all_gates(metrics)

        return QAStatusSummary(
            overall_result=gate_summary.overall_result.value.upper(),
            gate_counts={
                "ok": gate_summary.passed_gates,
                "warning": gate_summary.warning_gates,
                "block": gate_summary.blocked_gates,
                # E-066: "not checked" is never a pass.
                "skipped": gate_summary.skipped_gates,
            },
            key_metrics={
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "total_return": metrics.total_return,
                "cagr": metrics.cagr,
            },
            last_updated=last_updated,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing QA status: {e}", exc_info=True)
        # A27: generic caller-facing detail; real exception logged above.
        raise HTTPException(
            status_code=500, detail="internal error computing QA status"
        )


@router.get("/monitoring/risk_status", response_model=RiskStatusSummary)
def get_risk_status_summary(
    freq: str = Query(default="1d", description="Trading frequency"),
) -> RiskStatusSummary:
    """Get simplified risk status summary for monitoring.

    Returns key risk metrics from the last portfolio evaluation.
    Uses portfolio equity curve if available, falls back to backtest equity.

    Args:
        freq: Trading frequency ("1d" or "5min"), default "1d"

    Returns:
        RiskStatusSummary with sharpe_ratio, max_drawdown_pct, volatility, var_95, current_drawdown

    Raises:
        HTTPException: 400 if freq is not supported, 404 if no data found, 500 for errors
    """
    # Validate frequency
    if freq not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}",
        )

    try:
        # Try to load from portfolio report or risk metrics
        portfolio_equity_file = OUTPUT_DIR / f"portfolio_equity_{freq}.csv"
        backtest_equity_file = OUTPUT_DIR / f"equity_curve_{freq}.csv"

        equity_df = None
        last_updated = None

        if portfolio_equity_file.exists():
            equity_df = pd.read_csv(
                portfolio_equity_file, dtype={"timestamp": "string"}
            )
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            last_updated = equity_df["timestamp"].max().to_pydatetime()
        elif backtest_equity_file.exists():
            equity_df = pd.read_csv(backtest_equity_file, dtype={"timestamp": "string"})
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            last_updated = equity_df["timestamp"].max().to_pydatetime()

        if equity_df is None or equity_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No equity file found for freq={freq}. Cannot compute risk status.",
            )

        # Compute risk metrics using qa.risk_metrics module
        from src.assembled_core.qa.risk_metrics import compute_portfolio_risk_metrics

        equity_series = equity_df.set_index("timestamp")["equity"].sort_index()
        # freq is validated against SUPPORTED_FREQS above; FreqStr is the
        # central Literal kept in sync with it (config/__init__.py).
        risk_metrics = compute_portfolio_risk_metrics(
            equity_series, freq=cast(FreqStr, freq)
        )

        # Compute current drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = equity_series - rolling_max
        current_drawdown = float(drawdown.iloc[-1])

        return RiskStatusSummary(
            sharpe_ratio=risk_metrics.get("ann_sharpe"),
            max_drawdown_pct=risk_metrics.get("max_drawdown_pct"),
            volatility=risk_metrics.get("ann_vol"),
            var_95=risk_metrics.get("var_95"),
            current_drawdown=current_drawdown,
            last_updated=last_updated,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing risk status: {e}", exc_info=True)
        # A27: generic caller-facing detail; real exception logged above.
        raise HTTPException(
            status_code=500, detail="internal error computing risk status"
        )


@router.get("/monitoring/drift_status", response_model=DriftStatusSummary)
def get_drift_status_summary(
    freq: str = Query(default="1d", description="Trading frequency"),
    top_n: int = Query(
        default=10, ge=1, le=50, description="Number of top features to return"
    ),
) -> DriftStatusSummary:
    """Get drift status summary for monitoring.

    Returns the status of the last feature drift analysis, showing which features
    have drifted and their severity. Reads persisted drift results written by
    `qa.drift_detection.save_drift_results()` to `output/drift_analysis_{freq}.parquet`.

    Args:
        freq: Trading frequency ("1d" or "5min"), default "1d"
        top_n: Number of top features with drift to return (default: 10, max: 50)

    Returns:
        DriftStatusSummary with overall severity, top features with drift, and total features checked

    Raises:
        HTTPException: 400 if `freq` is unsupported; 503 if no drift analysis has
            been persisted yet for this frequency; 500 for unexpected errors.
    """
    # Validate frequency
    if freq not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}",
        )

    try:
        # Reads persisted drift results written by qa.drift_detection.save_drift_results()
        drift_results_file = OUTPUT_DIR / f"drift_analysis_{freq}.parquet"

        if drift_results_file.exists():
            # Load drift results
            drift_df = pd.read_parquet(drift_results_file)

            if (
                not drift_df.empty
                and "feature" in drift_df.columns
                and "psi" in drift_df.columns
            ):
                # Sort by PSI descending and take top N
                drift_df_sorted = drift_df.sort_values("psi", ascending=False).head(
                    top_n
                )

                features_with_drift = [
                    FeatureDriftItem(
                        feature=row.feature,
                        psi=float(row.psi),
                        drift_flag=str(getattr(row, "drift_flag", "NONE")),
                    )
                    for row in drift_df_sorted.itertuples(index=False)
                ]

                # Determine overall severity (worst case)
                if "drift_flag" in drift_df.columns:
                    if (drift_df["drift_flag"] == "SEVERE").any():
                        overall_severity = "SEVERE"
                    elif (drift_df["drift_flag"] == "MODERATE").any():
                        overall_severity = "MODERATE"
                    else:
                        overall_severity = "NONE"
                else:
                    # Fallback: use PSI thresholds
                    max_psi = drift_df["psi"].max()
                    if max_psi >= 0.3:
                        overall_severity = "SEVERE"
                    elif max_psi >= 0.2:
                        overall_severity = "MODERATE"
                    else:
                        overall_severity = "NONE"

                # Get last updated from file modification time
                last_updated = datetime.fromtimestamp(
                    drift_results_file.stat().st_mtime
                )

                return DriftStatusSummary(
                    overall_severity=overall_severity,
                    features_with_drift=features_with_drift,
                    total_features_checked=len(drift_df),
                    last_updated=last_updated,
                )

        # No drift analysis present — fail loud rather than masquerade as "NONE".
        # Operators must know there is no signal, not believe everything is clean.
        # (Audit C3-023 / C4-033 — replace dummy with 503.)
        logger.warning(
            "[Monitoring] No drift analysis file for freq=%s (expected at %s)",
            freq,
            drift_results_file,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Drift analysis not available for freq={freq}. "
                f"Expected file: {drift_results_file}. "
                "Run scripts/run_drift_check.py to generate."
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing drift status: {e}", exc_info=True)
        # A27: generic caller-facing detail; real exception logged above.
        raise HTTPException(
            status_code=500, detail="internal error computing drift status"
        )


# ---------------------------------------------------------------------------
# Live Trading / Portfolio Dashboard Endpoints
# ---------------------------------------------------------------------------


@router.get("/monitoring/portfolio")
def get_portfolio_status(
    db_path: str = Query(
        default="data/paper_ledger.db", description="Path to SQLite ledger"
    ),
) -> dict:
    """Return current portfolio state: positions, P&L, cash, equity.

    Reads from the SQLite paper ledger (LedgerStore). Falls back to empty
    state if no ledger file found.
    """
    try:
        from pathlib import Path as _Path

        from src.assembled_core.data.ledger_store import LedgerStore

        # EHRLICHKEIT (Audit-Plan 5.2, 2026-08-16): eine data/paper_ledger.db
        # existiert NIRGENDS im Betrieb — kein Prozess schreibt sie; der reale
        # Pilot-Ledger lebt im accounting/-Pfad (CSV/JSONL unter output/) und
        # ist an diesen Endpoint NICHT angebunden. no_ledger ist hier also der
        # strukturelle Dauerzustand, kein leeres Portfolio.
        no_ledger = {
            "status": "no_ledger",
            "producer_exists": False,
            "message": (
                "no SQLite ledger is written by any live process (audit "
                "2026-08-16); the real pilot ledger lives in the accounting/ "
                "path and is not wired to this endpoint"
            ),
            "cash": 0.0,
            "positions": [],
            "equity": 0.0,
            "n_positions": 0,
        }

        if not _is_safe_monitoring_path(_Path(db_path).resolve()):
            logger.warning(
                "[monitoring/portfolio] rejected out-of-bounds db_path: %s", db_path
            )
            return no_ledger

        if not _Path(db_path).exists():
            return no_ledger

        ledger = LedgerStore(db_path=db_path)
        positions = ledger.get_positions()
        cash = ledger.get_cash()
        equity_curve = ledger.load_equity_curve()
        last_equity = (
            float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else cash
        )

        return {
            "status": "ok",
            "cash": round(cash, 2),
            "equity": round(last_equity, 2),
            "n_positions": len(positions),
            "positions": (
                json.loads(positions.to_json(orient="records"))
                if not positions.empty
                else []
            ),
            "last_updated": (
                equity_curve["as_of"].iloc[-1].isoformat()
                if not equity_curve.empty
                else None
            ),
        }
    except Exception as exc:
        logger.error("Error fetching portfolio status: %s", exc)
        # A27: generic caller-facing detail; real exception logged above.
        raise HTTPException(
            status_code=500, detail="internal error fetching portfolio status"
        )


@router.get("/monitoring/regime")
def get_regime_status(
    output_dir: str = Query(
        default="src/output", description="Output directory for regime state"
    ),
) -> dict:
    """Return current market regime state.

    Reads the most recent regime state from the risk module. Falls back to
    a default 'unknown' state if no regime data is available.
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        # Look for most recent regime state file
        out_path = _Path(output_dir)
        if not _is_safe_monitoring_path(out_path.resolve()):
            logger.warning(
                "[monitoring/regime] rejected out-of-bounds output_dir: %s",
                output_dir,
            )
            return {
                "status": "stale",
                "regime": "unknown",
                "message": "no regime_state_*.json files in output_dir",
                "last_run": None,
            }
        if not out_path.exists():
            # The output dir missing is not the same failure as "no regime
            # data yet" — collapsing both to status=unavailable makes
            # dashboards pin regime=unknown as steady state instead of
            # alerting on a broken pipeline path.
            raise HTTPException(
                status_code=503,
                detail=f"output_dir {out_path} does not exist — regime pipeline has not run",
            )
        regime_files = sorted(out_path.glob("regime_state_*.json"), reverse=True)
        if regime_files:
            data = _json.loads(regime_files[0].read_text(encoding="utf-8"))
            return {
                "status": "ok",
                "regime": data.get("regime", "unknown"),
                "regime_score": data.get("regime_score", 0.0),
                "source_file": regime_files[0].name,
            }

        # EHRLICHKEIT (Audit-Plan 5.2, 2026-08-16): kein Modul im Repo
        # schreibt regime_state_*.json — dieser Zustand ist kein Cold-Start,
        # sondern strukturell: der Endpoint hat KEINEN Producer. Solange kein
        # Writer verdrahtet ist, kann hier nie etwas anderes stehen.
        return {
            "status": "stale",
            "regime": "unknown",
            "producer_exists": False,
            "message": (
                "no regime_state_*.json files — NO producer writes this "
                "artifact anywhere in the repo (audit 2026-08-16); endpoint "
                "cannot deliver data until a writer is wired"
            ),
            "last_run": None,
        }
    except HTTPException:
        # The intentional 503 (output_dir missing) must survive the broad
        # except below — otherwise it is silently reconverted to a generic 500.
        raise
    except Exception as exc:
        logger.error("Error fetching regime status: %s", exc)
        # A27: generic caller-facing detail; real exception logged above.
        raise HTTPException(
            status_code=500, detail="internal error fetching regime status"
        )


@router.get("/monitoring/alerts")
def get_active_alerts(
    db_path: str = Query(
        default="data/paper_ledger.db", description="Path to SQLite ledger"
    ),
    output_dir: str = Query(default="src/output", description="Output directory"),
) -> dict:
    """Return active system alerts: zombies, correlation guard, kill-switch status.

    Aggregates alerts from multiple risk subsystems.
    """
    empty_alerts = {"status": "ok", "n_alerts": 0, "alerts": []}

    # Guard both caller-supplied paths before any filesystem access. Reject to
    # the benign empty-alerts response (identical to a cold-start), so an
    # out-of-bounds path cannot be used to probe arbitrary files or leak
    # existence. SEIT 2026-08-16 (F-senior-8): die Zombie-/Corr-Bloecke lesen
    # NICHT mehr aus output_dir, sondern aus default_shadow_root() — die
    # Guards bleiben als defence-in-depth fuer db_path und kuenftige Leser;
    # ob output_dir hier Parameter bleibt, ist eine offene Signaturfrage.
    # BETRIEBSANNAHME (F-senior-9, E-146-Klasse): default_shadow_root() ist
    # ohne ATI_SHADOW_ROOT CWD-relativ — konsistent mit dem Producer, aber
    # nur solange API-Prozess und Pipeline dieselbe CWD (Repo-Root) haben.
    if not _is_safe_monitoring_path(Path(db_path).resolve()):
        logger.warning(
            "[monitoring/alerts] rejected out-of-bounds db_path: %s", db_path
        )
        return empty_alerts
    if not _is_safe_monitoring_path(Path(output_dir).resolve()):
        logger.warning(
            "[monitoring/alerts] rejected out-of-bounds output_dir: %s", output_dir
        )
        return empty_alerts

    alerts = []

    # Zombie positions.
    # FIX 2026-08-16 (Audit-Plan 5.1): this block previously globbed
    # {output_dir}/zombie_report_*.json — a filename NO producer ever wrote.
    # The zombie killer records via ops.shadow_recorder ->
    # output/shadow/zombie_killer_<date>.json with an envelope schema
    # ({module, snapshot_date, payload: {would_apply: {zombie_symbols}}}),
    # so this alert could never fire (consumer without producer).
    # Freshness guard: shadow snapshots from BACKTESTS carry historical
    # snapshot_dates (files back to 2021 exist) — only recent snapshots may
    # alert, or test/backtest residue produces phantom alerts.
    try:
        import json as _json
        from datetime import datetime as _dt
        from datetime import timedelta as _td
        from datetime import timezone as _tz

        from src.assembled_core.ops.shadow_mode import default_shadow_root

        _fresh_cutoff = (_dt.now(tz=_tz.utc) - _td(days=7)).date().isoformat()
        zombie_files = sorted(
            default_shadow_root().glob("zombie_killer_*.json"), reverse=True
        )
        if zombie_files:
            envelope = _json.loads(zombie_files[0].read_text(encoding="utf-8"))
            snap_date = str(envelope.get("snapshot_date", ""))
            zombie_syms = (
                (envelope.get("payload") or {}).get("would_apply") or {}
            ).get("zombie_symbols", [])
            if zombie_syms and snap_date >= _fresh_cutoff:
                alerts.append(
                    {
                        "type": "zombie_positions",
                        "severity": "HIGH",
                        "message": f"{len(zombie_syms)} zombie position(s) detected: {zombie_syms}",
                        "source": zombie_files[0].name,
                    }
                )
    except Exception as exc:
        logger.warning("[Monitoring] failed to load zombie alerts: %s", exc)

    # Kill-switch state
    # NOTE: the kill switch is a function-based API in execution.kill_switch
    # (get_kill_switch_state), NOT a risk.kill_switch.KillSwitch class. The previous
    # import raised ImportError on every call and was swallowed below, so an engaged
    # kill switch never surfaced an alert on this dashboard (Diagnostik A8, BLOCKER).
    try:
        from src.assembled_core.execution.kill_switch import get_kill_switch_state

        ks_state = get_kill_switch_state()
        if ks_state.get("engaged"):
            reason = ks_state.get("persistent", {}).get("reason") or "unknown"
            alerts.append(
                {
                    "type": "kill_switch",
                    "severity": "CRITICAL",
                    "message": f"Kill switch is ACTIVE — trading halted (reason: {reason})",
                }
            )
    except Exception as exc:
        logger.warning("[Monitoring] failed to check kill-switch state: %s", exc)

    # Correlation guard.
    # FIX 2026-08-16 (Audit-Plan 5.1): same consumer-without-producer defect —
    # the guard records via shadow_recorder into output/shadow/ with the
    # envelope schema, not {output_dir}/correlation_guard_*.json. Same
    # 7-day freshness guard against backtest residue.
    try:
        import json as _json
        from datetime import datetime as _dt
        from datetime import timedelta as _td
        from datetime import timezone as _tz

        from src.assembled_core.ops.shadow_mode import default_shadow_root

        _fresh_cutoff = (_dt.now(tz=_tz.utc) - _td(days=7)).date().isoformat()
        corr_files = sorted(
            default_shadow_root().glob("correlation_guard_*.json"), reverse=True
        )
        if corr_files:
            envelope = _json.loads(corr_files[0].read_text(encoding="utf-8"))
            snap_date = str(envelope.get("snapshot_date", ""))
            payload = envelope.get("payload") or {}
            would_apply = payload.get("would_apply") or {}
            triggered = bool(
                would_apply.get("guard_triggered") or payload.get("guard_triggered")
            )
            if triggered and snap_date >= _fresh_cutoff:
                alerts.append(
                    {
                        "type": "correlation_guard",
                        "severity": "MEDIUM",
                        "message": "Correlation guard active — position weights scaled down",
                        "source": corr_files[0].name,
                    }
                )
    except Exception as exc:
        logger.warning("[Monitoring] failed to check correlation guard: %s", exc)

    return {
        "status": "ok",
        "n_alerts": len(alerts),
        "alerts": alerts,
    }


@router.get("/monitoring/signals")
def get_signal_scores(
    output_dir: str = Query(
        default="output/signals",
        description="Output directory for signal scores",
    ),
    top_n: int = Query(
        default=20, description="Number of top/bottom symbols to return"
    ),
) -> dict:
    """Return most recent composite signal scores per symbol.

    PRODUCER (seit Audit-Plan 5.3, 2026-08-16):
    ``scripts/generate_attribution_report.py`` schreibt
    ``output/signals/signal_scores_<ts>.json`` — vorher suchte dieser Endpoint
    in ``src/output`` nach Dateien, die nie jemand schrieb.
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        out_path = _Path(output_dir)
        if not _is_safe_monitoring_path(out_path.resolve()):
            logger.warning(
                "[monitoring/signals] rejected out-of-bounds output_dir: %s",
                output_dir,
            )
            return {"status": "unavailable", "message": "No signal score files found"}
        score_files = sorted(out_path.glob("signal_scores_*.json"), reverse=True)
        if not score_files:
            # Try parquet
            score_files_pq = sorted(
                out_path.glob("signal_scores_*.parquet"), reverse=True
            )
            if score_files_pq:
                df = pd.read_parquet(str(score_files_pq[0]))
                scores = (
                    {
                        k: float(v)
                        for k, v in df.set_index("symbol")["score"].to_dict().items()
                    }
                    if "score" in df.columns
                    else {}
                )
                return {
                    "status": "ok",
                    "source": score_files_pq[0].name,
                    "top_long": sorted(scores.items(), key=lambda x: -x[1])[:top_n],
                    "top_short": sorted(scores.items(), key=lambda x: x[1])[:top_n],
                    "n_symbols": len(scores),
                }
            # F-senior-1 (Stage 2, 2026-08-16): seit Audit-Plan 5.3 EXISTIERT
            # der Producer (scripts/generate_attribution_report.py) — ein
            # leeres Verzeichnis heisst jetzt "noch keine Daten" (Erstlauf,
            # Retention, falsches CWD), NICHT "strukturell tot". Die zwei
            # Zustaende duerfen nie wieder ein Feld teilen (E-162).
            return {
                "status": "no_data_yet",
                "producer_exists": True,
                "producer": "scripts/generate_attribution_report.py",
                "message": (
                    "no signal_scores_* files yet — producer exists but has "
                    "not written into this directory"
                ),
            }

        data = _json.loads(score_files[0].read_text(encoding="utf-8"))
        scores = data.get("scores", {})
        return {
            "status": "ok",
            "source": score_files[0].name,
            "top_long": sorted(scores.items(), key=lambda x: -x[1])[:top_n],
            "top_short": sorted(scores.items(), key=lambda x: x[1])[:top_n],
            "n_symbols": len(scores),
        }
    except Exception as exc:
        logger.error("Error fetching signal scores: %s", exc)
        # A27: generic caller-facing detail; real exception logged above.
        raise HTTPException(
            status_code=500, detail="internal error fetching signal scores"
        )


@router.get("/monitoring/data-quality")
def get_data_quality(
    price_path: str = Query(
        default="output/aggregates/daily.parquet",
        description="Path to the operational EOD price panel",
    ),
) -> dict:
    """Return data freshness and quality status of the operational price panel.

    FIX (Audit-Plan 5.2, 2026-08-16): dieser Endpoint suchte vorher
    ``prices_*.parquet`` in ``src/output`` — ein Muster, das kein Producer je
    geschrieben hat (Antwort war immer "unavailable"), und zaehlte n_symbols
    als ``len(df.columns)`` obwohl das Panel LONG-Format hat. Jetzt liest er
    den ECHTEN operativen Cache und zaehlt Symbole korrekt.
    """
    try:
        from datetime import datetime as _dt
        from pathlib import Path as _Path

        p = _Path(price_path)
        if not _is_safe_monitoring_path(p.resolve()):
            logger.warning(
                "[monitoring/data-quality] rejected out-of-bounds price_path: %s",
                price_path,
            )
            return {"status": "unavailable", "message": "price_path out of bounds"}
        if not p.exists():
            raise HTTPException(
                status_code=503,
                detail=f"price panel {p} does not exist — ingest has not run",
            )

        mtime = p.stat().st_mtime
        file_age_hours = (time.time() - mtime) / 3600

        # F-senior-7: fehlende Spalten sollen eine diagnostizierbare 503
        # liefern, kein generisches 500 — read_parquet(columns=...) wirft
        # sonst VOR jedem Fallback.
        try:
            df = pd.read_parquet(str(p), columns=["timestamp", "symbol"])
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"price panel {p.name} lacks timestamp/symbol columns: {exc}",
            )

        n_symbols = int(df["symbol"].nunique())
        last_bar_ts = pd.to_datetime(df["timestamp"].max(), utc=True)

        # F-senior-2 / E-163: Frische aus dem LETZTEN BAR, nicht aus dem
        # Datei-mtime — ein Backfill-Rewrite ist ein Dateiereignis, kein
        # Datenereignis (gemessen: mtime 15.08. bei last bar 05.08.).
        bar_age_days = (pd.Timestamp.now(tz="UTC") - last_bar_ts).days
        freshness = "fresh" if bar_age_days <= 4 else "stale"

        return {
            "status": "ok",
            "latest_file": p.name,
            "freshness": freshness,
            "bar_age_days": int(bar_age_days),
            "file_age_hours": round(file_age_hours, 1),
            "n_symbols": n_symbols,
            "last_bar": str(last_bar_ts),
            "last_modified": _dt.fromtimestamp(mtime).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error fetching data quality: %s", exc)
        # A27: generic caller-facing detail; real exception logged above.
        raise HTTPException(
            status_code=500, detail="internal error fetching data quality"
        )
