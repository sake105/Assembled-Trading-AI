"""SEC EDGAR data source — insider trades (Form 4) and company filings.

Fetches Form 4 insider transaction data from the SEC EDGAR Atom feed.
No API key required. SEC requires a descriptive User-Agent header.

API docs: https://www.sec.gov/developer

Usage::

    from assembled_core.data.sources.edgar_source import fetch_insider_trades

    df = fetch_insider_trades(ticker="AAPL", max_results=40)
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
import pandas as pd

logger = logging.getLogger(__name__)

_EMPTY = pd.DataFrame(
    columns=["filed_at", "ticker", "company", "cik", "insider_name", "form_type", "url"]
)
_ATOM_NS = "http://www.w3.org/2005/Atom"
_USER_AGENT = "Assembled-Trading-AI/1.0 research@assembled-trading-ai.local"
# SEC EDGAR full-text search for Form 4 filings by company ticker
_EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms=4"
# EDGAR company search Atom feed
_EDGAR_ATOM_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=4&dateb=&owner=include&count={count}&search_text=&output=atom"


def fetch_insider_trades(
    ticker: str,
    *,
    max_results: int = 40,
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """Fetch recent Form 4 insider trade filings for a ticker from SEC EDGAR.

    Args:
        ticker:      Stock ticker symbol, e.g. "AAPL".
        max_results: Maximum number of filings to return (default: 40).
        from_date:   Optional start date filter "YYYY-MM-DD".
        to_date:     Optional end date filter "YYYY-MM-DD".

    Returns:
        DataFrame with columns: filed_at (UTC), ticker, company, cik,
        insider_name, form_type, url.
        Empty DataFrame on error or no filings found.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.error("[ERROR] requests not installed.")
        return _EMPTY.copy()

    url = _EDGAR_ATOM_URL.format(ticker=ticker.upper(), count=max_results)
    headers = {"User-Agent": _USER_AGENT}

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        xml_text = resp.text
    except Exception as exc:
        logger.error("[ERROR] edgar: request failed for %s — %s", ticker, exc)
        return _EMPTY.copy()

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("[ERROR] edgar: XML parse failed for %s — %s", ticker, exc)
        return _EMPTY.copy()

    ns = {"atom": _ATOM_NS}
    entries = root.findall("atom:entry", ns)

    if not entries:
        logger.warning("[WARN] edgar: no Form 4 entries found for %s", ticker)
        return _EMPTY.copy()

    from_ts = pd.Timestamp(from_date, tz="UTC") if from_date else None
    to_ts = pd.Timestamp(to_date, tz="UTC") if to_date else None

    rows = []
    for entry in entries:
        title_el = entry.find("atom:title", ns)
        updated_el = entry.find("atom:updated", ns)
        link_el = entry.find("atom:link", ns)
        id_el = entry.find("atom:id", ns)

        title = (title_el.text or "").strip() if title_el is not None else ""
        updated_raw = (updated_el.text or "").strip() if updated_el is not None else ""
        filing_url = (link_el.get("href") or "") if link_el is not None else ""
        entry_id = (id_el.text or "") if id_el is not None else ""

        # Parse timestamp
        try:
            filed_at = pd.Timestamp(updated_raw, tz="UTC")
        except Exception:
            filed_at = pd.NaT

        # Date filter
        if from_ts and pd.notna(filed_at) and filed_at < from_ts:
            continue
        if to_ts and pd.notna(filed_at) and filed_at > to_ts:
            continue

        # Extract CIK from entry id
        cik = ""
        m = re.search(r"CIK=(\d+)", entry_id, re.IGNORECASE)
        if m:
            cik = m.group(1)

        # Best-effort company name from title
        company = title
        m2 = re.search(r"4\s*[-–]\s*(.+)", title, re.IGNORECASE)
        if m2:
            company = m2.group(1).strip()

        rows.append(
            {
                "filed_at": filed_at,
                "ticker": ticker.upper(),
                "company": company,
                "cik": cik,
                "insider_name": "",  # not in Atom feed; would need to fetch individual filing
                "form_type": "4",
                "url": filing_url,
            }
        )

    if not rows:
        return _EMPTY.copy()

    result = pd.DataFrame(rows)
    result = result.sort_values("filed_at", ascending=False).reset_index(drop=True)
    logger.info("[OK] edgar: %d Form 4 filings for %s", len(result), ticker)
    return result
