# src/assembled_core/ports/audit_logger.py
"""AuditLogger port — append-only event logging for the domain.

A use-case calls ``audit.append({...})`` and the adapter persists the
record per the project audit-log retention policy (10y WORM via
Backblaze B2 Object Lock — see docs/AUDIT_LOG_RETENTION.md +
docs/GOBD_WORM_POLICY.md). The domain does not care where it is
persisted.

The hash-chain semantics (audit C4-016) belong in the *adapter*, not
the port: a domain caller never needs to know that the record carries
a ``prev_hash`` / ``hash`` pair — that is a durability concern.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class AuditLogger(Protocol):
    """Append a single event to the audit log."""

    def append(self, event: Mapping[str, Any]) -> None: ...

    def verify_chain(self) -> tuple[bool, int]:
        """Return ``(ok, n_records)`` — for tamper-evidence checks.

        Adapters that don't support chain verification return
        ``(True, n_records)`` so callers can treat ``ok=False`` as a
        definite breach.
        """
        ...
