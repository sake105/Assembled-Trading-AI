"""Assembled Trading AI - Core Backend Package.

This package provides the core trading pipeline functionality:
- Data I/O and resampling
- Signal generation (EMA-based strategies)
- Order generation and execution simulation
- Backtesting and portfolio simulation with cost models
- QA/Health checks
- FastAPI backend for read-only API access

Main modules:
- pipeline: Core trading pipeline (signals, orders, backtest, portfolio)
- api: FastAPI endpoints for accessing pipeline outputs
- qa: Quality assurance and health checks
- config: Central configuration (paths, frequencies)
"""

from __future__ import annotations

def _get_version() -> str:
    """Read version from installed package metadata (pyproject.toml project.version). ASCII-only fallback."""
    try:
        from importlib.metadata import version
        v = version("assembled-trading-core")
        if v and isinstance(v, str):
            return v.encode("ascii", errors="ignore").decode("ascii") or "0.0.0+unknown"
        return "0.0.0+unknown"
    except Exception:
        return "0.0.0+unknown"


__version__ = _get_version()
