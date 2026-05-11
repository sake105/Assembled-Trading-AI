"""EDGAR Filing-Stream — Poll-basiert für Live-Pipeline.

Strategie
---------
1. Polling SEC-RSS-Feed alle N-Sekunden (~60s ist erlaubt).
2. State persistiert in ``last_seen_accession`` — gibt nur NEUE Filings zurück.
3. Klassifizierung via Form-Type + Items (für 8-K).

Endpoints (alle frei)
---------------------
- https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=40&output=atom
- https://efts.sec.gov/LATEST/search-index?q=&dateRange=custom&startdt=...&forms=8-K

User-Agent required (SEC).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from erweiterung._base import (
    rate_limited,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)


@dataclass
class EdgarFiling:
    accession: str
    cik: str
    company: str
    form_type: str
    filed_at: pd.Timestamp
    primary_doc_url: str | None = None
    items: list[str] = field(default_factory=list)


_USER_AGENT = (
    "AssembledTradingAI-Erweiterung/0.1 (research; contact: hans.oertel2@gmail.com)"
)


@rate_limited(min_interval_s=0.15)
@retry_with_backoff(max_attempts=3, base_delay=2.0)
def fetch_edgar_atom(form_type: str = "8-K", count: int = 40) -> list[EdgarFiling]:
    """Fetch current filings via SEC Atom-Feed.

    Returns:
        Liste neuer EdgarFiling-Records.
    """
    import requests

    url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcurrent&type={form_type}&dateb=&owner=include&count={count}&output=atom"
    )
    r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=20)
    r.raise_for_status()
    return _parse_atom_feed(r.text, form_type)


_ENTRY_RE = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
_ACCESSION_RE = re.compile(r"accession-number=([0-9\-]+)")
_TITLE_RE = re.compile(r"<title>([^<]+)</title>")
_LINK_RE = re.compile(r'<link[^>]*href="([^"]+)"')
_UPDATED_RE = re.compile(r"<updated>([^<]+)</updated>")
_COMPANY_RE = re.compile(r"\([^)]*(?:Filer:)?[^)]*\)")


def _parse_atom_feed(xml: str, form_type: str) -> list[EdgarFiling]:
    out: list[EdgarFiling] = []
    for entry_match in _ENTRY_RE.finditer(xml):
        block = entry_match.group(1)
        title = (
            _TITLE_RE.search(block).group(1).strip() if _TITLE_RE.search(block) else ""
        )
        link = _LINK_RE.search(block).group(1) if _LINK_RE.search(block) else None
        updated = (
            _UPDATED_RE.search(block).group(1) if _UPDATED_RE.search(block) else None
        )
        if updated:
            try:
                filed_at = pd.Timestamp(updated)
            except (TypeError, ValueError):
                filed_at = pd.Timestamp.utcnow()
        else:
            filed_at = pd.Timestamp.utcnow()
        # accession from link
        accession_match = re.search(r"(\d{10}-\d{2}-\d{6})", link or "")
        accession = accession_match.group(1) if accession_match else "unknown"
        # CIK
        cik_match = re.search(r"CIK=(\d+)", link or "")
        cik = cik_match.group(1) if cik_match else ""
        # Company from title (typically: "Form-X - COMPANY NAME (CIK ...)")
        company_match = re.search(r"-\s+([A-Z][\w\s,.\-&']+?)\s+\(", title)
        company = company_match.group(1).strip() if company_match else ""
        out.append(
            EdgarFiling(
                accession=accession,
                cik=cik,
                company=company,
                form_type=form_type,
                filed_at=filed_at,
                primary_doc_url=link,
            )
        )
    return out


@dataclass
class EdgarStreamState:
    """Persistent stream-state."""

    last_seen_accession: set[str] = field(default_factory=set)
    state_file: Path | None = None

    def load(self) -> None:
        if self.state_file and self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.last_seen_accession = set(data.get("last_seen_accession", []))
            except (json.JSONDecodeError, OSError):
                logger.warning("[edgar-stream] state-file unreadable, starting fresh")

    def save(self) -> None:
        if self.state_file is None:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.state_file.write_text(
                json.dumps(
                    {"last_seen_accession": sorted(self.last_seen_accession)}, indent=2
                )
            )
        except OSError as e:
            logger.warning("[edgar-stream] cannot save state: %s", e)


def poll_new_filings(
    form_type: str = "8-K",
    state: EdgarStreamState | None = None,
    count: int = 40,
) -> tuple[list[EdgarFiling], EdgarStreamState]:
    """Poll EDGAR, return only NEW filings since last poll.

    Args:
        form_type: SEC form code (e.g. '8-K', '13D', '4').
        state: persistent stream-state. If None, fresh (all current = new).
        count: max filings per poll.

    Returns:
        (new_filings, updated_state).
    """
    state = state or EdgarStreamState()
    state.load()
    filings = fetch_edgar_atom(form_type=form_type, count=count)
    new_filings: list[EdgarFiling] = []
    for f in filings:
        if f.accession in state.last_seen_accession:
            continue
        new_filings.append(f)
        state.last_seen_accession.add(f.accession)
    # Trim memory: keep only last 5000 accessions
    if len(state.last_seen_accession) > 5000:
        state.last_seen_accession = set(list(state.last_seen_accession)[-5000:])
    state.save()
    return new_filings, state


def filings_to_dataframe(filings: list[EdgarFiling]) -> pd.DataFrame:
    """Convert EdgarFiling list to DataFrame."""
    if not filings:
        return pd.DataFrame()
    rows = []
    for f in filings:
        rows.append(
            {
                "accession": f.accession,
                "cik": f.cik,
                "company": f.company,
                "form_type": f.form_type,
                "filed_at": f.filed_at,
                "primary_doc_url": f.primary_doc_url,
                "items": ",".join(f.items),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "EdgarFiling",
    "EdgarStreamState",
    "fetch_edgar_atom",
    "poll_new_filings",
    "filings_to_dataframe",
]
