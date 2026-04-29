"""FastAPI backend modules.

This package provides:
- FastAPI application factory (``create_app``)
- Pydantic request/response models
- API routers (signals, orders, portfolio, performance, risk, qa)
"""

from __future__ import annotations

from src.assembled_core.api import models  # noqa: F401 - registers the models module
from src.assembled_core.api.app import create_app

__all__ = [
    "create_app",
    "models",
]
