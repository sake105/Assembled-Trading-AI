"""Dedupe disclosure events by fingerprint."""

from __future__ import annotations

import logging
from typing import List

from .models import DisclosureEvent

logger = logging.getLogger(__name__)


def dedupe_events(events: List[DisclosureEvent]) -> List[DisclosureEvent]:
    """Dedupe by fingerprint; keep first occurrence.

    Events with an empty fingerprint are still kept (current behavior
    preserved) but a WARNING is logged per empty-fp event — a normalizer
    bug that leaves fingerprint="" would otherwise silently let every
    downstream copy of the same broken event flow through, amplifying
    parse errors into fake trigger volume.
    """
    seen: set[str] = set()
    out: List[DisclosureEvent] = []
    n_empty_fp = 0
    for ev in events:
        fp = ev.fingerprint or ""
        if fp and fp in seen:
            continue
        if fp:
            seen.add(fp)
        else:
            n_empty_fp += 1
        out.append(ev)
    if n_empty_fp > 0:
        logger.warning(
            "[Dedupe] %d disclosure event(s) had empty fingerprint and "
            "bypassed dedup — normalizer may be producing malformed events",
            n_empty_fp,
        )
    return out
