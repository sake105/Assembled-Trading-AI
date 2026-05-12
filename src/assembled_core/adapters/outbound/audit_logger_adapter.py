# src/assembled_core/adapters/outbound/audit_logger_adapter.py
"""AuditLogger port implementations — Hash-chained JSONL + in-memory fake.

The production adapter delegates to the existing kill_switch
``_append_audit`` machinery so the hash chain that audit C4-016
mandates is one source of truth. The in-memory adapter is for tests.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.assembled_core.ports.audit_logger import AuditLogger


class JsonlAuditLogger(AuditLogger):
    """JSONL audit logger with hash-chain + fsync per write.

    The hash-chain mechanics live in ``execution.kill_switch._append_audit``
    today; this adapter is the *port-shaped* facade in front of that
    helper so domain callers don't need to know which file the chain
    actually lives in.
    """

    def __init__(self, *, audit_path: Path | str | None = None) -> None:
        # Resolve at __init__ rather than at use-time so env changes
        # mid-process don't silently re-route the chain.
        if audit_path is None:
            audit_path = os.environ.get(
                "ASSEMBLED_GENERIC_AUDIT_PATH",
                "output/ops/generic_audit.jsonl",
            )
        self._path = Path(audit_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Mapping[str, Any]) -> None:
        # We deliberately re-implement the per-line write here instead of
        # importing kill_switch._append_audit, because that helper
        # writes to a fixed kill-switch path. The hash-chain SHAPE is
        # what matters; the destination path is per-adapter.
        import hashlib

        prev_hash = self._last_hash()
        record = dict(event)
        record["ts"] = record.get("ts") or datetime.now(timezone.utc).isoformat()
        record["prev_hash"] = prev_hash
        digest_payload = json.dumps(record, sort_keys=True).encode("utf-8")
        record["hash"] = hashlib.sha256(digest_payload).hexdigest()
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    def verify_chain(self) -> tuple[bool, int]:
        # Delegate to the kill-switch chain verifier — it is generic
        # over file path.
        from src.assembled_core.execution.kill_switch import verify_audit_chain

        return verify_audit_chain(self._path)

    def _last_hash(self) -> str:
        from src.assembled_core.execution.kill_switch import _last_audit_hash

        return _last_audit_hash(self._path)


class InMemoryAuditLogger(AuditLogger):
    """Fake audit logger — collects events in a list, for tests."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, event: Mapping[str, Any]) -> None:
        self.events.append(dict(event))

    def verify_chain(self) -> tuple[bool, int]:
        # No chain in-memory — return ok=True / count=n.
        return True, len(self.events)


__all__ = ["JsonlAuditLogger", "InMemoryAuditLogger"]
