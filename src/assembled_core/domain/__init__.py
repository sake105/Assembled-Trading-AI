"""Domain layer — pure business logic, no I/O, no framework imports.

Audit C-001..C-007 hexagonal-architecture skeleton. Five Bounded
Contexts live under this package:

- ``trading``    — Strategy, Signal, Order (FSM), Position
- ``risk``       — KillSwitch, RiskLimits, DriftReport
- ``accounting`` — Ledger, Fill, PnL, Reconciliation
- ``research``   — Backtest, CVRun, Experiment
- ``operations`` — HealthCheck, AuditEvent, Heartbeat

Allowed dependencies inside the domain layer:
  - stdlib
  - numpy, pandas, scipy (the numerical core)
  - other ``assembled_core.domain.*`` modules (siblings)

Forbidden:
  - httpx, fastapi, sqlalchemy, yfinance, alpaca-py, polygon, ...
  - any module under ``assembled_core.adapters.*``
  - any module under ``assembled_core.application.*``

Anything in those forbidden categories belongs in ports + adapters.
This invariant is checked by ``tests/test_hexagonal_layering.py``.
"""
