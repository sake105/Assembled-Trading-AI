"""Event-Replay engine for deterministic session replay.

From 42_EVENT_REPLAY_SYSTEM.md.

Usage:
    store = EventStore("data/events/events.db")
    replayer = Replayer(store)
    replayer.register_handler("market_tick_received", on_tick)
    outputs = replayer.replay_session("session-uuid-here")
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.assembled_core.events.store import EventStore

logger = logging.getLogger(__name__)

Handler = Callable[[dict], Any]


@dataclass
class ReplayResult:
    session_id: str
    n_events_replayed: int
    outputs: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    ended_at: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


class Replayer:
    """Replays stored event sessions through registered handlers.

    Handlers are registered per event_type.  A '*' wildcard handler
    receives every event.
    """

    def __init__(self, store: EventStore) -> None:
        self._store = store
        self._handlers: dict[str, list[Handler]] = {}

    def register_handler(self, event_type: str, handler: Handler) -> None:
        """Register a handler for an event_type (or '*' for all events)."""
        self._handlers.setdefault(event_type, []).append(handler)

    def replay_session(
        self, session_id: str, stop_on_error: bool = False
    ) -> ReplayResult:
        """Replay all events for *session_id* in sequence order.

        Args:
            session_id: Session to replay.
            stop_on_error: If True, abort replay on first handler exception.

        Returns:
            ReplayResult with all outputs collected from handlers.
        """
        events = self._store.load_session(session_id)
        result = ReplayResult(session_id=session_id, n_events_replayed=len(events))

        for row in events:
            try:
                payload = json.loads(row.get("payload_json", "{}"))
            except json.JSONDecodeError as _exc:
                logger.warning(
                    "[replayer] malformed payload_json in session %s seq %s: %s",
                    session_id,
                    row.get("sequence"),
                    _exc,
                )
                payload = {}
            event_dict = {
                "session_id": row["session_id"],
                "sequence": row["sequence"],
                "event_type": row["event_type"],
                "source": row["source"],
                "occurred_at": row["occurred_at"],
                "payload": payload,
            }

            for etype in (row["event_type"], "*"):
                for handler in self._handlers.get(etype, []):
                    try:
                        out = handler(event_dict)
                        if out is not None:
                            result.outputs.append(out)
                    except Exception as exc:
                        msg = f"Handler error at seq={row['sequence']}: {exc}"
                        logger.warning(msg)
                        result.errors.append(msg)
                        if stop_on_error:
                            result.ended_at = datetime.now(tz=timezone.utc)
                            return result

        result.ended_at = datetime.now(tz=timezone.utc)
        logger.info(
            "Replay session=%s: %d events, %d outputs, %d errors, %.2fs",
            session_id,
            result.n_events_replayed,
            len(result.outputs),
            len(result.errors),
            result.duration_seconds or 0,
        )
        return result

    def compare_replays(
        self,
        session_id: str,
        reference_outputs: list[Any],
        key_fn: Callable[[Any], Any] = lambda x: x,
    ) -> dict[str, Any]:
        """Replay session and compare outputs to a reference run.

        Args:
            session_id: Session to replay.
            reference_outputs: Outputs from a previous replay run.
            key_fn: Function to extract a comparable key from each output.

        Returns:
            Dict with 'match', 'n_new', 'n_ref', 'divergences'.
        """
        result = self.replay_session(session_id)
        new_keys = [key_fn(o) for o in result.outputs]
        ref_keys = [key_fn(o) for o in reference_outputs]

        divergences = []
        for i, (nk, rk) in enumerate(zip(new_keys, ref_keys)):
            if nk != rk:
                divergences.append({"index": i, "new": nk, "ref": rk})

        return {
            "match": len(divergences) == 0 and len(new_keys) == len(ref_keys),
            "n_new": len(new_keys),
            "n_ref": len(ref_keys),
            "divergences": divergences,
        }


__all__ = ["Replayer", "ReplayResult"]
