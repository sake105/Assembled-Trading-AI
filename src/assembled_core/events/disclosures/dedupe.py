"""Dedupe disclosure events by fingerprint."""

from __future__ import annotations

from typing import List

from .models import DisclosureEvent


def dedupe_events(events: List[DisclosureEvent]) -> List[DisclosureEvent]:
    """Dedupe by fingerprint; keep first occurrence."""
    seen: set[str] = set()
    out: List[DisclosureEvent] = []
    for ev in events:
        fp = ev.fingerprint or ""
        if fp and fp in seen:
            continue
        if fp:
            seen.add(fp)
        out.append(ev)
    return out
