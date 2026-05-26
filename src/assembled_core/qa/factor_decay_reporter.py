"""Factor Decay Monitoring — daily IC-decay reporting.

This module provides a thin monitoring wrapper around
``qa.factor_analysis.compute_ic_decay_curve`` that runs as part of the EOD
pipeline.  It is intentionally *non-blocking*: any exception during computation
is logged and swallowed, so the caller always returns cleanly.

Output:
    ``output/qa/factor_decay_log.jsonl`` — one JSON line appended per run.

Usage (EOD pipeline)::

    from src.assembled_core.qa.factor_decay_reporter import run_factor_decay_monitoring

    run_factor_decay_monitoring(prices, factor_col="close")  # minimal smoke call

    # Full call with explicit factor column list
    run_factor_decay_monitoring(
        panel_df=prices_with_factors,
        factor_cols=["returns_12m", "rv_20"],
        run_date="2026-05-26",
    )
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path("output") / "qa" / "factor_decay_log.jsonl"


def run_factor_decay_monitoring(
    panel_df: pd.DataFrame | None,
    factor_cols: Sequence[str] | None = None,
    factor_col: str | None = None,
    price_col: str = "close",
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    run_date: str | None = None,
    log_path: Path | str | None = None,
    max_horizon_days: int = 60,
) -> dict:
    """Compute IC-decay curve for each factor and append a summary to JSONL log.

    This function is a *monitoring* call — it never raises.  If data is
    unavailable or computation fails, it logs ``[SKIP]`` or ``[ERROR]`` and
    returns a dict describing what happened.

    Args:
        panel_df: Panel DataFrame with symbol, timestamp, close price and one or
            more factor columns.  If ``None`` or empty, the call is a no-op.
        factor_cols: List of factor column names to analyse.  If omitted, falls
            back to *factor_col* (single-column shorthand).
        factor_col: Single factor column shorthand.  Ignored when *factor_cols*
            is given.
        price_col: Name of the close-price column (default: ``"close"``).
        symbol_col: Symbol column name (default: ``"symbol"``).
        timestamp_col: Timestamp column name (default: ``"timestamp"``).
        run_date: ISO date string for the log entry (default: today UTC).
        log_path: Path to the JSONL log file.  Defaults to
            ``output/qa/factor_decay_log.jsonl``.
        max_horizon_days: Maximum prediction horizon to evaluate (default: 60).

    Returns:
        Dict with keys ``status``, ``run_date``, ``factors_computed``,
        ``message``, and optionally ``results`` (list of per-factor dicts).
    """
    run_date = run_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    log_path = Path(log_path) if log_path else _DEFAULT_LOG_PATH

    # --- Guard: no data ---
    if panel_df is None or (hasattr(panel_df, "empty") and panel_df.empty):
        msg = "[SKIP] factor decay — panel_df is None or empty"
        logger.info("[FACTOR-DECAY] %s (run_date=%s)", msg, run_date)
        return {
            "status": "skip",
            "run_date": run_date,
            "message": msg,
            "factors_computed": 0,
        }

    # Resolve factor list
    _cols: list[str] = []
    if factor_cols:
        _cols = list(factor_cols)
    elif factor_col:
        _cols = [factor_col]
    else:
        # Auto-detect: any column that isn't a standard OHLCV/meta column
        _ohlcv = {
            price_col,
            symbol_col,
            timestamp_col,
            "open",
            "high",
            "low",
            "volume",
            "date",
            "adj_close",
        }
        _cols = [c for c in panel_df.columns if c not in _ohlcv]

    # Filter to columns that actually exist
    _available = [c for c in _cols if c in panel_df.columns]
    _missing = [c for c in _cols if c not in panel_df.columns]
    if _missing:
        logger.warning(
            "[FACTOR-DECAY] Requested factor cols not in panel: %s — skipping them",
            _missing,
        )

    # Guard: price column or symbol column missing
    for required in (price_col, symbol_col, timestamp_col):
        if required not in panel_df.columns:
            msg = f"[SKIP] factor decay — required column '{required}' not in panel_df"
            logger.info("[FACTOR-DECAY] %s (run_date=%s)", msg, run_date)
            return {
                "status": "skip",
                "run_date": run_date,
                "message": msg,
                "factors_computed": 0,
            }

    if not _available:
        msg = "[SKIP] factor decay — no valid factor columns found in panel_df"
        logger.info("[FACTOR-DECAY] %s (run_date=%s)", msg, run_date)
        return {
            "status": "skip",
            "run_date": run_date,
            "message": msg,
            "factors_computed": 0,
        }

    # --- Compute ---
    try:
        from src.assembled_core.qa.factor_analysis import (
            compute_ic_decay_curve,
            estimate_alpha_decay_halflife,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "[FACTOR-DECAY] import failed with %s: %s", type(exc).__name__, exc
        )
        msg = f"[SKIP] factor decay — could not import factor_analysis: {exc}"
        logger.warning("[FACTOR-DECAY] %s", msg)
        return {
            "status": "skip",
            "run_date": run_date,
            "message": msg,
            "factors_computed": 0,
        }

    per_factor_results: list[dict] = []
    for fc in _available:
        try:
            decay_df = compute_ic_decay_curve(
                panel_df=panel_df,
                factor_col=fc,
                price_col=price_col,
                symbol_col=symbol_col,
                timestamp_col=timestamp_col,
                max_horizon_days=max_horizon_days,
            )
            if decay_df.empty:
                logger.info(
                    "[FACTOR-DECAY] factor=%s — decay curve empty (insufficient data), run_date=%s",
                    fc,
                    run_date,
                )
                per_factor_results.append(
                    {"factor": fc, "status": "empty", "half_life_days": None}
                )
                continue

            half_life_info = estimate_alpha_decay_halflife(decay_df)
            h1_row = decay_df[decay_df["horizon_days"] == 1]
            ic_1d = float(h1_row["ic_mean"].iloc[0]) if not h1_row.empty else None

            result_entry = {
                "factor": fc,
                "status": "ok",
                "ic_at_1d": ic_1d,
                "half_life_days": half_life_info.get("half_life_days"),
                "ic_0": half_life_info.get("ic_0"),
                "r_squared": half_life_info.get("r_squared"),
                "decay_curve_rows": len(decay_df),
            }
            per_factor_results.append(result_entry)

            logger.info(
                "[FACTOR-DECAY] factor=%s ic_1d=%.4f half_life=%.1fd r2=%.3f run_date=%s",
                fc,
                ic_1d or 0.0,
                half_life_info.get("half_life_days") or float("nan"),
                half_life_info.get("r_squared") or 0.0,
                run_date,
            )

        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.warning(
                "[FACTOR-DECAY] factor=%s raised exception (non-blocking): %s",
                fc,
                tb,
            )
            per_factor_results.append(
                {"factor": fc, "status": "error", "traceback": tb}
            )

    # --- Write JSONL ---
    log_record = {
        "run_date": run_date,
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
        "factors_computed": len(
            [r for r in per_factor_results if r.get("status") == "ok"]
        ),
        "results": per_factor_results,
        "status": "ok",
    }
    _write_jsonl_row(log_path, log_record)

    return {
        "status": "ok",
        "run_date": run_date,
        "factors_computed": log_record["factors_computed"],
        "results": per_factor_results,
        "message": f"computed {log_record['factors_computed']}/{len(_available)} factors",
    }


def _write_jsonl_row(path: Path, record: dict) -> None:
    """Append *record* as a JSON line to *path*, creating parent dirs if needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:  # noqa: BLE001
        logger.warning(
            "[FACTOR-DECAY] Failed to write log to %s: %s",
            path,
            traceback.format_exc(),
        )
