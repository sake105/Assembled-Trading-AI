# scripts/run_api.py
"""CLI entry point for FastAPI server."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn

from src.assembled_core.api.app import create_app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
