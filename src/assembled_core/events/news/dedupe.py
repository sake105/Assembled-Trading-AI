from __future__ import annotations

from typing import Dict, Iterable, List

from .models import NewsEvent


def dedupe_events(events: Iterable[NewsEvent]) -> List[NewsEvent]:
    """Dedupe events by canonical_url (preferred) then fingerprint.

    When duplicates are found:
      - prefer event with earlier published_utc (older news first)
      - if published_utc is equal, prefer event with longer summary
    """
    by_key: Dict[str, NewsEvent] = {}

    for ev in events:
        key = ev.canonical_url or ev.fingerprint
        if not key:
            key = ev.event_id
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = ev
            continue

        # Compare published_utc lexicographically (ISO strings are orderable)
        if ev.published_utc < existing.published_utc:
            better = ev
        elif ev.published_utc > existing.published_utc:
            better = existing
        else:
            # Same published time: prefer longer summary (if any)
            existing_len = len(existing.summary or "")
            new_len = len(ev.summary or "")
            better = ev if new_len > existing_len else existing

        by_key[key] = better

    return list(by_key.values())

