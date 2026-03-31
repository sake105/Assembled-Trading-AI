from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import urlparse, urlunparse, parse_qsl

from dateutil import parser as date_parser

from .entities import _canonical_country_code, extract_countries, extract_entities
from .fingerprint import simhash64
from .models import NewsEvent


def sanitize_text(text: str, strip_html: bool, max_chars: int) -> str:
    """Strip simple HTML, normalize whitespace and cap length."""
    if text is None:
        return ""
    s = str(text)
    if strip_html:
        import re

        s = re.sub(r"<[^>]+>", " ", s)
    s = " ".join(s.split())  # collapse whitespace
    if max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars]
    return s


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}


def canonicalize_url(url: str) -> str:
    """Normalize URL by stripping tracking params and fragments; lowercase scheme/host."""
    if not url:
        return ""
    parsed = urlparse(url)
    # Drop fragment
    fragment = ""
    # Filter query params
    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS
    ]
    query = "&".join(f"{k}={v}" for k, v in query_pairs) if query_pairs else ""
    netloc = parsed.netloc.lower()
    scheme = (parsed.scheme or "https").lower()

    # Remove default ports
    if ":" in netloc:
        host, port = netloc.split(":", 1)
        if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
            netloc = host

    # Normalize trailing slash: ensure "/path" and "/path/" normalize gleich
    path = parsed.path or ""
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    cleaned = parsed._replace(
        scheme=scheme,
        netloc=netloc,
        path=path,
        query=query,
        fragment=fragment,
    )
    return urlunparse(cleaned)


def _parse_published(published: str | None, fetched_utc: str) -> str:
    if not published:
        return fetched_utc
    try:
        dt = date_parser.parse(published)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return fetched_utc


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_raw_item(
    raw: Dict[str, Any],
    *,
    source_id: str,
    source_name: str,
    source_domain: str,
    fetched_utc: str,
) -> NewsEvent | None:
    """Normalize a raw RSS/GDELT item into NewsEvent. May return None for bad items."""
    title = str(raw.get("title") or "").strip()
    url = str(raw.get("link") or "").strip()
    canonical_url = canonicalize_url(url)

    published_raw = raw.get("published")
    published_utc = _parse_published(
        str(published_raw) if published_raw is not None else None, fetched_utc
    )

    summary = raw.get("summary")
    summary_str = str(summary) if summary is not None else None

    # Drop extremely short titles; try fallback from summary
    def _is_too_short(text: str) -> bool:
        words = text.split()
        return len(text) < 12 or len(words) < 2

    if _is_too_short(title):
        if summary_str and not _is_too_short(summary_str):
            # Fallback: derive title from summary
            fallback = " ".join(summary_str.split())
            title = fallback[:80]
        else:
            # Both title and summary too weak -> drop event
            return None

    # Final whitespace normalization and hard caps
    title = sanitize_text(title, strip_html=False, max_chars=300)
    if summary_str is not None:
        summary_str = sanitize_text(summary_str, strip_html=False, max_chars=800)

    # Entity & country extraction (use GDELT-provided fields when available)
    countries: list[str] = []
    raw_inner = raw.get("raw", raw)
    if isinstance(raw_inner, dict):
        country_codes: set[str] = set()
        # Locations list, if present (GDELT-style)
        locations = raw_inner.get("locations")
        if isinstance(locations, list):
            for loc in locations:
                if not isinstance(loc, dict):
                    continue
                for key in (
                    "countryCode",
                    "country",
                    "country_name",
                    "adm1countryCode",
                ):
                    val = loc.get(key)
                    if isinstance(val, str):
                        code = _canonical_country_code(val)
                        if code:
                            country_codes.add(code)

        # Direct source country fields
        for key in (
            "sourceCountryCode",
            "sourcecountrycode",
            "sourceCountry",
            "sourcecountry",
        ):
            val = raw_inner.get(key)
            if isinstance(val, str):
                code = _canonical_country_code(val)
                if code:
                    country_codes.add(code)

        if country_codes:
            countries = sorted(country_codes)

    combined_text = f"{title} {summary_str or ''}".strip()
    if not countries:
        countries = extract_countries(combined_text)
    entities = extract_entities(combined_text)

    # 64-bit content fingerprint (SimHash) from title+summary
    fp64_int = simhash64(combined_text)
    fingerprint64 = f"{fp64_int:016x}"

    if canonical_url:
        fingerprint = _sha256_hex(canonical_url)
    else:
        key = f"{title}|{source_domain}|{published_utc}"
        fingerprint = _sha256_hex(key)

    event_id = f"news_{fingerprint[:12]}"

    return NewsEvent(
        event_id=event_id,
        source_id=source_id,
        title=title,
        url=url,
        canonical_url=canonical_url or url,
        source_name=source_name,
        source_domain=source_domain,
        published_utc=published_utc,
        fetched_utc=fetched_utc,
        summary=summary_str,
        language=None,
        raw=raw,
        fingerprint=fingerprint,
        fingerprint64=fingerprint64,
        entities=entities,
        countries=countries,
    )


def now_utc_iso() -> str:
    """Helper to get current UTC timestamp in ISO8601."""
    return datetime.now(timezone.utc).isoformat()
