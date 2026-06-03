# src/assembled_core/api/routers/performance.py
"""Performance endpoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from src.assembled_core.api.models import EquityCurveResponse, EquityPoint, Frequency
from src.assembled_core.api.routers.health import _is_safe_output_dir
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.logging_utils import get_logger
from src.assembled_core.pipeline.backtest import compute_metrics

router = APIRouter()
logger = get_logger(__name__)


@router.get("/performance/{freq}/backtest-curve", response_model=EquityCurveResponse)
def get_backtest_curve(freq: Frequency) -> EquityCurveResponse:
    """Get backtest equity curve for a given frequency.

    Args:
        freq: Trading frequency ("1d" or "5min")

    Returns:
        EquityCurveResponse with equity curve points

    Raises:
        HTTPException: 404 if equity curve file not found, 500 if data is malformed
    """
    curve_file = OUTPUT_DIR / f"equity_curve_{freq.value}.csv"

    if not curve_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Equity curve file not found: {curve_file}. Run backtest first.",
        )

    try:
        try:
            df = pd.read_csv(curve_file, dtype={"timestamp": "string"})
        except (IOError, OSError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read equity curve file: {exc}",
            ) from exc

        if "timestamp" not in df.columns or "equity" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Malformed equity curve: missing required columns. Found: {list(df.columns)}",
            )

        # Convert timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["equity"] = pd.to_numeric(df["equity"], errors="coerce")

        # Drop NaNs
        df = df.dropna(subset=["timestamp", "equity"])

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Equity curve file is empty or contains only invalid data",
            )

        # Convert to EquityPoint models
        points = [
            EquityPoint(timestamp=row.timestamp, equity=float(row.equity))
            for row in df.itertuples(index=False)
        ]

        start_equity = float(df["equity"].iloc[0])
        end_equity = float(df["equity"].iloc[-1])

        return EquityCurveResponse(
            frequency=freq,
            points=points,
            count=len(points),
            start_equity=start_equity,
            end_equity=end_equity,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading equity curve: {e}")


@router.get("/performance/{freq}/metrics")
def get_performance_metrics(freq: Frequency) -> dict:
    """Get performance metrics for a given frequency.

    Args:
        freq: Trading frequency ("1d" or "5min")

    Returns:
        Dictionary with performance metrics (final_pf, sharpe, rows, first, last)

    Raises:
        HTTPException: 404 if equity curve file not found, 500 if data is malformed
    """
    curve_file = OUTPUT_DIR / f"equity_curve_{freq.value}.csv"

    if not curve_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Equity curve file not found: {curve_file}. Run backtest first.",
        )

    try:
        try:
            df = pd.read_csv(curve_file, dtype={"timestamp": "string"})
        except (IOError, OSError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read equity curve file: {exc}",
            ) from exc

        if "timestamp" not in df.columns or "equity" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Malformed equity curve: missing required columns. Found: {list(df.columns)}",
            )

        # Convert timestamps and equity
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["equity"] = pd.to_numeric(df["equity"], errors="coerce")

        # Drop NaNs
        df = df.dropna(subset=["timestamp", "equity"])

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Equity curve file is empty or contains only invalid data",
            )

        # Compute metrics using pipeline function
        metrics = compute_metrics(df)

        # Convert timestamps to ISO format strings for JSON serialization
        return {
            "freq": freq.value,
            "final_pf": metrics["final_pf"],
            "sharpe": metrics["sharpe"],
            "rows": metrics["rows"],
            "first": (
                metrics["first"].isoformat()
                if hasattr(metrics["first"], "isoformat")
                else str(metrics["first"])
            ),
            "last": (
                metrics["last"].isoformat()
                if hasattr(metrics["last"], "isoformat")
                else str(metrics["last"])
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing metrics: {e}")


_DEFAULT_LEDGER = str(OUTPUT_DIR / "runs/_paper_ledger/ledger_state.json")


@router.get("/performance/{freq}/live-curve", response_model=EquityCurveResponse)
def get_live_curve(
    freq: Frequency,
    ledger_path: str = Query(
        default=_DEFAULT_LEDGER,
        description="Path to the paper ledger JSON state file.",
    ),
) -> EquityCurveResponse:
    """Equity curve from the running paper pilot.

    Same response schema as ``/performance/{freq}/backtest-curve`` so the
    frontend can render both with the same component.

    When no pilot data exists (ledger absent or equity_curve empty) returns
    an empty but valid EquityCurveResponse — never 404 or 500.

    Args:
        freq: Trading frequency ("1d" or "5min").
        ledger_path: Override the default paper-ledger JSON path.

    Returns:
        EquityCurveResponse with pilot equity curve points.
    """
    jpath = Path(ledger_path)
    if not jpath.is_absolute():
        jpath = OUTPUT_DIR.parent / ledger_path

    _empty = EquityCurveResponse(
        frequency=freq.value, points=[], count=0, start_equity=0.0, end_equity=0.0
    )

    # Reject unauthenticated path-traversal to files outside the output dir (Diagnostik A6).
    if not _is_safe_output_dir(jpath.resolve()):
        logger.warning(
            "[live-curve] rejected out-of-bounds ledger_path: %s", ledger_path
        )
        return _empty

    if not jpath.exists():
        return _empty

    try:
        from src.assembled_core.ops.paper_ledger import load_ledger_state

        # Sentinel start_capital so a silent loader fallback is detectable (mirror /ledger, A7).
        state = load_ledger_state(jpath, start_capital=-1.0)
    except Exception as exc:
        # Fail closed to an empty curve (documented contract: never 404/500) instead of
        # leaking the exception text via HTTP 500 (Diagnostik A7/A27/E-025).
        logger.warning(
            "[live-curve] failed to load pilot ledger from %s: %s", jpath, exc
        )
        return _empty

    # Detect the silent loader fallback (corrupt/unreadable) → empty curve, not a fake one.
    if state.get("updated_utc") is None and (state.get("cash") or 0) < 0:
        logger.warning(
            "[live-curve] possible corrupt ledger at %s — returning empty curve", jpath
        )
        return _empty

    equity_curve: list[dict] = state.get("equity_curve") or []
    if not equity_curve:
        return _empty

    points: list[EquityPoint] = []
    for entry in equity_curve:
        try:
            ts = pd.to_datetime(entry.get("utc"), utc=True)
            eq = float(entry.get("equity", 0))
            if pd.notna(ts):
                points.append(EquityPoint(timestamp=ts, equity=eq))
        except Exception:
            continue

    if not points:
        return _empty

    return EquityCurveResponse(
        frequency=freq.value,
        points=points,
        count=len(points),
        start_equity=points[0].equity,
        end_equity=points[-1].equity,
    )
