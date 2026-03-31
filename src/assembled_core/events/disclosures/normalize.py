"""Normalize raw disclosure items into DisclosureEvent."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, Optional

from .models import DisclosureEvent


def now_utc_iso() -> str:
    """Current UTC timestamp as ISO string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _canonical_key(raw: Dict[str, Any], source_id: str, published_utc: str) -> str:
    """Build a canonical string for fingerprinting."""
    # House PTR
    if source_id == "house_ptr":
        base = "house_ptr|" + str(raw.get("doc_id") or raw.get("link") or "")
        pdf_meta = (
            raw.get("pdf_meta") if isinstance(raw.get("pdf_meta"), dict) else None
        )
        if pdf_meta and pdf_meta.get("sha256"):
            base = base + "|" + str(pdf_meta["sha256"])
        return base
    # Form 4 (edgar): fingerprint by edgar_form4|accession|link
    if raw.get("link") and (
        raw.get("company") is not None or raw.get("cik") is not None
    ):
        key = "edgar_form4|" + str(raw.get("accession") or raw.get("link") or "")
        return key
    parts = [
        source_id,
        str(raw.get("source_id") or ""),
        str(raw.get("published_utc") or published_utc),
        str(raw.get("person_or_entity") or ""),
        str(raw.get("ticker") or ""),
        str(raw.get("action_type") or ""),
    ]
    return "|".join(parts)


def normalize_raw_item(
    raw: Dict[str, Any],
    source_id: str,
    source_name: str,
    source_domain: str,
    fetched_utc: str,
) -> Optional[DisclosureEvent]:
    """Convert a raw item to DisclosureEvent. Returns None if invalid."""
    if not raw or not isinstance(raw, dict):
        return None
    published_utc = str(raw.get("published_utc") or raw.get("published") or fetched_utc)
    event_id = str(raw.get("event_id") or uuid.uuid4().hex[:16])

    # House PTR
    if source_id == "house_ptr":
        person_or_entity = str(raw.get("person") or raw.get("title") or "").strip()
        ticker = None
        action_type = "HOUSE_PTR_FILED"
        notional = None
    # Form 4 (edgar): raw has link + company/cik -> action_type FORM4_FILED
    elif raw.get("link") and (
        raw.get("company") is not None or raw.get("cik") is not None
    ):
        person_or_entity = str(raw.get("company") or "").strip()
        ticker = None
        action_type = "FORM4_FILED"
        notional = None
    else:
        person_or_entity = str(
            raw.get("person_or_entity") or raw.get("person") or ""
        ).strip()
        ticker = raw.get("ticker")
        if ticker is not None:
            ticker = str(ticker).strip() or None
        action_type = str(raw.get("action_type") or "").strip()
        notional = raw.get("notional")
        if notional is not None:
            try:
                notional = float(notional)
            except (TypeError, ValueError):
                notional = None

    canonical = _canonical_key(raw, source_id, published_utc)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return DisclosureEvent(
        event_id=event_id,
        source_id=source_id,
        source_name=source_name,
        source_domain=source_domain,
        published_utc=published_utc,
        fetched_utc=fetched_utc,
        person_or_entity=person_or_entity,
        ticker=ticker,
        action_type=action_type,
        notional=notional,
        raw=dict(raw),
        fingerprint=fingerprint,
    )
