from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from .models import NewsHealth, NewsStatus


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_health(
    sources: Iterable[str],
    items_raw: int,
    items_after_dedupe: int,
    failures: List[Dict[str, Any]],
    *,
    fetched_utc: str | None = None,
    min_sources_ok: int = 1,
) -> NewsHealth:
    """Compute NewsHealth from basic counts and failures."""
    sources_list = list(sources)
    sources_total = len(sources_list)
    sources_failed = len(failures)
    sources_ok = max(sources_total - sources_failed, 0)

    if fetched_utc is None:
        fetched_utc = _now_utc_iso()

    status: NewsStatus
    notes: List[str] = []

    if sources_ok == 0:
        status = "ERROR"
        notes.append("No sources succeeded (sources_ok == 0).")
    elif sources_ok < max(min_sources_ok, 1) or sources_failed > 0:
        status = "DEGRADED"
        if sources_ok < max(min_sources_ok, 1):
            notes.append("Sources below min_sources_ok.")
        if sources_failed > 0:
            notes.append("One or more sources failed.")
    else:
        status = "OK"

    if items_after_dedupe <= 0:
        notes.append("no_new_items")

    return NewsHealth(
        status=status,
        fetched_utc=fetched_utc,
        sources_total=sources_total,
        sources_ok=sources_ok,
        sources_failed=sources_failed,
        items_raw=items_raw,
        items_after_dedupe=items_after_dedupe,
        failures=list(failures),
        notes=notes,
    )

