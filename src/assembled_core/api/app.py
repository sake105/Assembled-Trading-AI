# src/assembled_core/api/app.py
"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI

from src.assembled_core.api.routers import orders, performance, portfolio, qa, risk, signals


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI instance with all routers included
    """
    app = FastAPI(
        title="Assembled Trading AI API",
        description="Read-only API for trading pipeline outputs",
        version="1.0.0",
    )
    
    # Include routers under /api/v1 prefix
    app.include_router(orders.router, prefix="/api/v1", tags=["orders"])
    app.include_router(performance.router, prefix="/api/v1", tags=["performance"])
    app.include_router(risk.router, prefix="/api/v1", tags=["risk"])
    app.include_router(signals.router, prefix="/api/v1", tags=["signals"])
    app.include_router(portfolio.router, prefix="/api/v1", tags=["portfolio"])
    app.include_router(qa.router, prefix="/api/v1", tags=["qa"])
    
    return app

