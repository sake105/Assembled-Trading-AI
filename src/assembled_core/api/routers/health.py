# src/assembled_core/api/routers/health.py
"""Rich health-check endpoint (GO_LIVE F2).

Mirrors the checks performed by ops/daily_scheduler._health_check_worker and
execution/broker_adapter.BrokerAdapter.health_check().  The existing thin
/health (uptime-only) in app.py is replaced by this router.

HTTP 200  — all critical checks pass (output_dir writability).
HTTP 503  — at least one critical check fails.
Non-critical checks (data freshness, broker, kill-switch) are reported in
'checks' but never elevate to 503 on their own.
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.assembled_core.config import OUTPUT_DIR

router = APIRouter()

_SAFE_ROOTS: tuple[Path, ...] = (
    OUTPUT_DIR.resolve(),
    Path(tempfile.gettempdir()).resolve(),
)


def _is_safe_output_dir(resolved: Path) -> bool:
    return any(resolved == root or root in resolved.parents for root in _SAFE_ROOTS)


@router.get("/health", tags=["ops"])
def get_health(
    output_dir: str = Query(
        default="output",
        description="Output directory to probe for writability and data freshness",
    ),
    check_broker: bool = Query(
        default=False,
        description="Set to true to include a live broker connectivity check (makes an outbound API call).",
    ),
) -> JSONResponse:
    """Machine-readable health check.

    Returns:
        JSON with ``status``, ``timestamp_utc``, and ``checks`` dict.
        HTTP 200 when healthy, 503 when a critical check fails.
    """
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    checks: dict[str, dict] = {}
    critical_failed = False

    # ── 1. Output directory writability (critical) ───────────────────
    try:
        _path = Path(output_dir)
        _resolved = _path.resolve()
        if not _is_safe_output_dir(_resolved):
            checks["output_dir"] = {
                "ok": False,
                "detail": "output_dir is outside safe roots (OUTPUT_DIR or system temp)",
            }
            critical_failed = True
        else:
            _path.mkdir(parents=True, exist_ok=True)
            _probe = _path / ".health_check_probe"
            try:
                _probe.write_text("ok", encoding="utf-8")
                checks["output_dir"] = {"ok": True, "detail": None}
            except Exception as exc:
                checks["output_dir"] = {"ok": False, "detail": f"not writable: {exc}"}
                critical_failed = True
            finally:
                _probe.unlink(missing_ok=True)
    except Exception as exc:
        checks["output_dir"] = {"ok": False, "detail": f"path error: {exc}"}
        critical_failed = True

    # ── 2. Data freshness (non-critical; skipped when output_dir is unsafe) ──
    _safe_path = None if critical_failed else Path(output_dir).resolve()
    if _safe_path is not None:
        try:
            _price_files = sorted(_safe_path.glob("prices_*.parquet"), reverse=True)
            if not _price_files:
                checks["data_freshness"] = {
                    "ok": False,
                    "detail": "no price files found",
                }
            else:
                _age_h = (time.time() - _price_files[0].stat().st_mtime) / 3600
                checks["data_freshness"] = {
                    "ok": _age_h < 26,
                    "detail": f"latest: {_price_files[0].name}, age: {_age_h:.1f}h",
                }
        except Exception as exc:
            checks["data_freshness"] = {"ok": False, "detail": str(exc)}
    else:
        checks["data_freshness"] = {
            "ok": False,
            "detail": "skipped — output_dir unsafe",
        }

    # ── 3. Broker connectivity (opt-in — makes outbound API call) ────
    if check_broker:
        try:
            from src.assembled_core.execution.broker_adapter import AlpacaAdapter

            _broker_result = AlpacaAdapter().health_check()
            checks["broker"] = {
                "ok": bool(_broker_result.get("ok", False)),
                "detail": _broker_result.get("message"),
            }
        except Exception as exc:
            checks["broker"] = {"ok": False, "detail": f"check skipped: {exc}"}
    else:
        checks["broker"] = {
            "ok": None,
            "detail": "skipped — pass ?check_broker=true to enable",
        }

    # ── 4. Kill-switch readable (non-critical) ───────────────────────
    try:
        from src.assembled_core.execution.kill_switch import get_kill_switch_state

        _ks = get_kill_switch_state()
        checks["kill_switch"] = {
            "ok": True,
            "detail": f"state={_ks.get('state', 'unknown')}",
        }
    except Exception as exc:
        checks["kill_switch"] = {"ok": False, "detail": str(exc)}

    payload = {
        "status": "unhealthy" if critical_failed else "healthy",
        "timestamp_utc": timestamp_utc,
        "checks": checks,
    }
    return JSONResponse(content=payload, status_code=503 if critical_failed else 200)
