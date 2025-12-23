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

__version__ = "0.0.1"
