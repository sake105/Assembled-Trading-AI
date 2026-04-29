# src/assembled_core/api/app.py
"""FastAPI application factory."""

from __future__ import annotations

import time

from fastapi import FastAPI
from src.assembled_core.api.middleware import add_middleware
from src.assembled_core.api.routers import (
    monitoring,
    oms,
    orders,
    paper_trading,
    performance,
    portfolio,
    qa,
    risk,
    signals,
)

_APP_START_TIME: float = time.time()


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI instance with all routers included
    """
    app = FastAPI(
        title="Assembled Trading AI API",
        description="Trading pipeline API with read and command endpoints",
        version="2.0.0",
    )

    # ── Middleware ───────────────────────────────────────────────────
    add_middleware(app)

    # ── Health / Readiness / Liveness probes ────────────────────────
    @app.get("/health", tags=["ops"])
    def health():
        """Basic health check — returns 200 if the process is alive."""
        return {"status": "ok", "uptime_s": round(time.time() - _APP_START_TIME, 1)}

    @app.get("/ready", tags=["ops"])
    def ready():
        """Readiness probe — checks that critical dependencies are available."""
        checks: dict[str, bool] = {}
        # Check kill switch state is readable
        try:
            from src.assembled_core.execution.kill_switch import get_kill_switch_state
            get_kill_switch_state()
            checks["kill_switch"] = True
        except Exception:
            checks["kill_switch"] = False

        all_ok = all(checks.values())
        return {"ready": all_ok, "checks": checks}

    @app.get("/live", tags=["ops"])
    def live():
        """Liveness probe — returns 200 if event loop is responsive."""
        return {"alive": True}

    # ── Kill Switch command endpoints (POST) ────────────────────────
    @app.post("/api/v1/kill-switch/activate", tags=["risk-commands"])
    def activate_kill_switch_endpoint(
        throttle_pct: float = 0.0,
        reason: str = "",
        actor: str = "api",
    ):
        """Activate the kill switch with optional fractional throttle."""
        from src.assembled_core.execution.kill_switch import (
            activate_kill_switch,
            get_kill_switch_state,
        )
        activate_kill_switch(throttle_pct=throttle_pct, reason=reason, actor=actor)
        return {"action": "activated", "state": get_kill_switch_state()}

    @app.post("/api/v1/kill-switch/deactivate", tags=["risk-commands"])
    def deactivate_kill_switch_endpoint(
        reason: str = "",
        actor: str = "api",
    ):
        """Deactivate the kill switch."""
        from src.assembled_core.execution.kill_switch import (
            deactivate_kill_switch,
            get_kill_switch_state,
        )
        deactivate_kill_switch(reason=reason, actor=actor)
        return {"action": "deactivated", "state": get_kill_switch_state()}

    @app.get("/api/v1/kill-switch/state", tags=["risk-commands"])
    def get_kill_switch_state_endpoint():
        """Get current kill switch state."""
        from src.assembled_core.execution.kill_switch import get_kill_switch_state
        return get_kill_switch_state()

    # ── Existing routers ────────────────────────────────────────────
    app.include_router(orders.router, prefix="/api/v1", tags=["orders"])
    app.include_router(performance.router, prefix="/api/v1", tags=["performance"])
    app.include_router(risk.router, prefix="/api/v1", tags=["risk"])
    app.include_router(signals.router, prefix="/api/v1", tags=["signals"])
    app.include_router(portfolio.router, prefix="/api/v1", tags=["portfolio"])
    app.include_router(qa.router, prefix="/api/v1", tags=["qa"])
    app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])
    app.include_router(
        paper_trading.router, prefix="/api/v1/paper", tags=["paper-trading"]
    )
    app.include_router(oms.router, prefix="/api/v1/oms", tags=["oms"])

    return app
