"""Hash-chained Tamper-Evident Audit Log.

Implements an append-only log where each entry contains the SHA-256 hash
of the previous entry, making manipulation detectable (blockchain principle
without blockchain overhead).

Events logged: Orders, Fills, Risk-Breaches, Model-Retraining, Config-Changes.

MiFID II / SEC audit trail requirements:
- Immutable record of all trading decisions
- Timestamped to millisecond precision
- Verifiable chain of integrity
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of auditable events."""
    ORDER_CREATED = "order_created"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FILLED = "order_filled"
    RISK_BREACH = "risk_breach"
    KILL_SWITCH = "kill_switch"
    MODEL_RETRAIN = "model_retrain"
    CONFIG_CHANGE = "config_change"
    POSITION_CHANGE = "position_change"
    RECONCILIATION = "reconciliation"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class AuditEntry:
    """Single audit log entry with hash chain."""
    sequence: int
    timestamp: str  # ISO-8601 UTC
    event_type: str
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AuditEntry":
        return cls(**d)


def _compute_hash(sequence: int, timestamp: str, event_type: str,
                  payload: dict, prev_hash: str) -> str:
    """Compute SHA-256 hash of entry fields."""
    content = json.dumps({
        "seq": sequence,
        "ts": timestamp,
        "type": event_type,
        "payload": payload,
        "prev": prev_hash,
    }, sort_keys=True, default=str)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only hash-chained audit log.

    Args:
        log_path: Path to JSON lines log file. If None, in-memory only.
        auto_flush: Write each entry immediately to disk.
    """

    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        log_path: str | Path | None = None,
        auto_flush: bool = True,
    ) -> None:
        self._entries: list[AuditEntry] = []
        self._log_path = Path(log_path) if log_path else None
        self._auto_flush = auto_flush
        self._last_hash = self.GENESIS_HASH
        self._sequence = 0

        # Load existing entries if file exists
        if self._log_path and self._log_path.exists():
            self._load_from_file()

    def append(
        self,
        event_type: AuditEventType | str,
        payload: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Append a new entry to the audit log.

        Args:
            event_type: Type of audit event.
            payload: Event-specific data.

        Returns:
            The created AuditEntry.
        """
        self._sequence += 1
        ts = datetime.now(timezone.utc).isoformat()
        event_str = event_type.value if isinstance(event_type, AuditEventType) else str(event_type)
        payload = payload or {}

        entry_hash = _compute_hash(
            self._sequence, ts, event_str, payload, self._last_hash
        )

        entry = AuditEntry(
            sequence=self._sequence,
            timestamp=ts,
            event_type=event_str,
            payload=payload,
            prev_hash=self._last_hash,
            entry_hash=entry_hash,
        )

        self._entries.append(entry)
        self._last_hash = entry_hash

        if self._auto_flush and self._log_path:
            self._flush_entry(entry)

        logger.debug("[AuditLog] #%d %s", self._sequence, event_str)
        return entry

    def verify_chain(self) -> tuple[bool, int]:
        """Verify the hash chain integrity.

        Returns:
            Tuple of (is_valid, first_broken_sequence).
            If valid, first_broken_sequence = -1.
        """
        if not self._entries:
            return True, -1

        prev_hash = self.GENESIS_HASH
        for entry in self._entries:
            expected = _compute_hash(
                entry.sequence, entry.timestamp, entry.event_type,
                entry.payload, prev_hash,
            )
            if entry.entry_hash != expected:
                logger.error(
                    "[AuditLog] Chain broken at #%d: expected %s, got %s",
                    entry.sequence, expected[:16], entry.entry_hash[:16],
                )
                return False, entry.sequence
            if entry.prev_hash != prev_hash:
                return False, entry.sequence
            prev_hash = entry.entry_hash

        return True, -1

    def get_entries(
        self,
        event_type: AuditEventType | str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit log entries.

        Args:
            event_type: Filter by event type.
            since: ISO-8601 timestamp to filter entries after.
            limit: Maximum entries to return.

        Returns:
            List of matching AuditEntry objects.
        """
        result = self._entries
        if event_type:
            et = event_type.value if isinstance(event_type, AuditEventType) else str(event_type)
            result = [e for e in result if e.event_type == et]
        if since:
            result = [e for e in result if e.timestamp >= since]
        return result[-limit:]

    @property
    def length(self) -> int:
        return len(self._entries)

    @property
    def last_hash(self) -> str:
        return self._last_hash

    def _flush_entry(self, entry: AuditEntry) -> None:
        """Append entry to log file as JSON line."""
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    def _load_from_file(self) -> None:
        """Load entries from existing log file."""
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    entry = AuditEntry.from_dict(data)
                    self._entries.append(entry)
                    self._last_hash = entry.entry_hash
                    self._sequence = entry.sequence
            logger.info("[AuditLog] Loaded %d entries from %s", len(self._entries), self._log_path)
        except Exception as e:
            logger.warning("[AuditLog] Failed to load log: %s", e)


__all__ = [
    "AuditEventType",
    "AuditEntry",
    "AuditLog",
]
