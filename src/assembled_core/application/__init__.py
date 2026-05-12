"""Application layer — Use-Cases / Command-Handlers / Query-Handlers.

Audit C-001 / C-005 / C-006: the application layer orchestrates ports
and domain objects to fulfil a single use-case (e.g. "submit an order",
"record a kill-switch trip"). It does NOT contain business rules —
those belong in the domain layer. It does NOT contain I/O — that
belongs in adapters.

Use-Cases live under ``assembled_core.application.use_cases``.
"""
