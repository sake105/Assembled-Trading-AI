"""SEC EDGAR Form 4 ingester — free replacement for paid insider-trade feeds.

Enumerates Form 4 / 4/A filings from the EDGAR daily-index, fetches the raw
ownership XML, classifies the SEC ``transactionCode`` (fixing the legacy
"100% unknown type" defect where every row was stamped ``transaction_type='unknown'``),
and emits a PIT-correct insider-trades frame to ``output/insider_form4.parquet``.

PIT contract
------------
- ``available_at`` = the SGML header ``ACCEPTANCE-DATETIME`` (the instant the
  filing became public), parsed ``America/New_York`` -> UTC. This is the only
  defensible availability timestamp.
- ``filing_date`` = ``FILED AS OF DATE`` (date-only); what
  :func:`altdata_loader.load_insider_filings` and
  :func:`earnings_insider_wrapper.compute_earnings_insider_factors` gate on
  (``filing_date <= as_of``).
- ``transaction_date`` / ``event_date`` = the economic transaction date; it is
  kept immutable and is NEVER used for availability (that would be look-ahead).
  Availability is a SEPARATE derived column — avoiding the E-038 boundary-shift
  anti-pattern.

Schema
------
The emitted frame is a SUPERSET serving both downstream consumer shapes:
- ``earnings_insider_wrapper``: ``symbol, filing_date, transaction_type, value_usd``
  (``transaction_type in {P, S}``; ``value_usd`` is GROSS positive — the wrapper
  applies the +/- sign from ``transaction_type``).
- ``insider_features.add_insider_features``: ``timestamp, symbol, net_shares,
  trades_count, role`` (+ ``event_date``/``disclosure_date`` for PIT windows).

Compliance (SEC fair-access policy)
-----------------------------------
- A declared ``User-Agent`` (real org + contact email) is REQUIRED; an
  undeclared UA returns HTTP 403 ("Undeclared Automated Tool").
- <=8 req/s (0.12s spacing) + exponential backoff on 403/429/503. The 10/s hard
  cap is budgeted GLOBALLY across ``www.sec.gov`` / ``efts.sec.gov`` /
  ``data.sec.gov`` — this module only hits ``www.sec.gov``.
"""

from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "classify_transaction_code",
    "acceptance_datetime_to_utc",
    "parse_form4_index",
    "parse_form4_submission",
    "form4_rows_to_dataframe",
    "enumerate_form4_filings",
    "fetch_submission",
    "ingest_form4",
    "resolve_user_agent",
    "FORM4_COLUMNS",
]

EDGAR_ARCHIVES = "https://www.sec.gov/Archives/"
_DAILY_INDEX_FMT = (
    "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{qtr}/form.{ymd}.idx"
)
_EASTERN = "America/New_York"
MIN_REQUEST_SPACING_S = 0.12  # ~8 req/s, under the 10/s SEC hard cap

# Authoritative SEC Form 345 transaction-code legend (for WARNING context only).
# Open-market P (purchase) and S (sale) are the only codes we treat as directional
# buy/sell; everything else (grants, exercises, gifts, tax withholding, ...) is
# classified 'unknown' — surfaced, never silently coerced.
_KNOWN_CODE_LEGEND = {
    "P": "Open-market purchase",
    "S": "Open-market sale",
    "A": "Grant/award/other acquisition",
    "D": "Disposition to the issuer",
    "F": "Payment of exercise price / tax by share withholding",
    "I": "Discretionary transaction",
    "M": "Exercise/conversion of derivative",
    "C": "Conversion of derivative",
    "E": "Expiration (short)",
    "H": "Expiration (long)",
    "O": "Exercise of out-of-the-money derivative",
    "X": "Exercise of in-the-money derivative",
    "G": "Gift",
    "L": "Small-acquisition exemption",
    "W": "Acquisition/disposition by will or inheritance",
    "Z": "Deposit/withdrawal from voting trust",
    "J": "Other (described in footnote)",
    "K": "Equity swap or similar",
    "U": "Disposition pursuant to a tender of shares",
    "V": "Transaction voluntarily reported earlier than required",
}

FORM4_COLUMNS = [
    "accession",
    "symbol",
    "issuer_cik",
    "issuer_name",
    "reporting_owner_cik",
    "reporting_owner_name",
    "role",
    "security_title",
    "is_derivative",
    "transaction_date",
    "event_date",
    "filing_date",
    "available_at",
    "timestamp",
    "disclosure_date",
    "transaction_code",
    "transaction_type",
    "acquired_disposed",
    "shares",
    "price",
    "value_usd",
    "net_shares",
    "trades_count",
]


# ---------------------------------------------------------------------------
# Pure parsing / classification (unit-tested offline)
# ---------------------------------------------------------------------------


def classify_transaction_code(code: str | None) -> str:
    """Classify a raw SEC transactionCode into ``'P'``, ``'S'`` or ``'unknown'``.

    Only open-market purchases (``P``) and sales (``S``) are treated as
    directional. Every other code (grants, exercises, gifts, tax withholding,
    ...) maps to ``'unknown'`` — never silently coerced to a buy/sell. Callers
    surface a per-code WARNING + an end-of-run unknown-share summary.
    """
    if code is None:
        return "unknown"
    c = str(code).strip().upper()
    if c in ("P", "S"):
        return c
    return "unknown"


def acceptance_datetime_to_utc(raw: str) -> pd.Timestamp:
    """Parse an EDGAR ``ACCEPTANCE-DATETIME`` (``YYYYMMDDHHMMSS``, US/Eastern) to UTC.

    The acceptance timestamp is wall-clock US/Eastern (the SEC's timezone) and
    is DST-aware: October => EDT (UTC-4), January => EST (UTC-5).
    """
    s = str(raw).strip()
    if len(s) < 14 or not s[:14].isdigit():
        raise ValueError(f"unparseable ACCEPTANCE-DATETIME: {raw!r}")
    naive = pd.Timestamp(
        year=int(s[0:4]),
        month=int(s[4:6]),
        day=int(s[6:8]),
        hour=int(s[8:10]),
        minute=int(s[10:12]),
        second=int(s[12:14]),
    )
    return naive.tz_localize(
        _EASTERN, ambiguous=True, nonexistent="shift_forward"
    ).tz_convert("UTC")


def parse_form4_index(idx_text: str) -> list[dict[str, str]]:
    """Parse an EDGAR daily ``form.<date>.idx`` into Form 4 / 4/A entries.

    The ``.idx`` is fixed-width but each data row ends with a space-free
    ``edgar/data/<cik>/<accession>.txt`` path, so we split on whitespace and
    read the trailing ``[cik, date_filed, filename]`` columns; the company name
    is whatever lies between the form-type and the CIK. Header lines are skipped
    because their tokens never match a Form-4 shape.
    """
    entries: list[dict[str, str]] = []
    for line in idx_text.splitlines():
        tokens = line.split()
        if len(tokens) < 5:
            continue
        form_type = tokens[0]
        filename = tokens[-1]
        if form_type not in ("4", "4/A") or not filename.startswith("edgar/"):
            continue
        entries.append(
            {
                "form_type": form_type,
                "company": " ".join(tokens[1:-3]),
                "cik": tokens[-3],
                "date_filed": tokens[-2],
                "filename": filename,
                "url": EDGAR_ARCHIVES + filename,
            }
        )
    return entries


def _find_text(el: ET.Element | None, path: str) -> str:
    if el is None:
        return ""
    node = el.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return ""


def _to_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _strip_namespaces(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _relationship_role(rel: ET.Element | None) -> str:
    """Collapse the reportingOwnerRelationship flags into a single role label."""
    if rel is None:
        return "Unknown"

    def _is_true(tag: str) -> bool:
        return _find_text(rel, tag).strip().lower() in ("true", "1")

    if _is_true("isOfficer"):
        return "Officer"
    if _is_true("isDirector"):
        return "Director"
    if _is_true("isTenPercentOwner"):
        return "10%Owner"
    if _is_true("isOther"):
        return "Other"
    return "Unknown"


def _extract_ownership_xml(text: str) -> str | None:
    m = re.search(r"<ownershipDocument>.*?</ownershipDocument>", text, re.DOTALL)
    return m.group(0) if m else None


def parse_form4_submission(
    text: str, *, accession: str | None = None
) -> list[dict[str, Any]]:
    """Parse a full Form 4 submission ``.txt`` into one row per TRANSACTION.

    Holdings (``nonDerivativeHolding`` / ``derivativeHolding``) are skipped —
    they carry no transactionCode and are not trades. Both the non-derivative
    and derivative transaction tables are parsed. ``available_at`` comes from the
    SGML ``ACCEPTANCE-DATETIME`` header; ``filing_date`` from ``FILED AS OF DATE``.
    """
    acc_m = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", text)
    filed_m = re.search(r"FILED AS OF DATE:\s*(\d{8})", text)
    available_at = acceptance_datetime_to_utc(acc_m.group(1)) if acc_m else pd.NaT
    filing_date = (
        pd.Timestamp(filed_m.group(1)) if filed_m else pd.NaT
    )  # YYYYMMDD -> midnight naive
    disclosure_date = filing_date  # the day the filing became public

    xml_str = _extract_ownership_xml(text)
    if xml_str is None:
        logger.warning(
            "[WARN] form4_parse: no <ownershipDocument> found accession=%s", accession
        )
        return []
    try:
        root = _strip_namespaces(ET.fromstring(xml_str))
    except ET.ParseError as exc:
        logger.warning(
            "[WARN] form4_parse: XML parse error accession=%s: %s", accession, exc
        )
        return []

    issuer = root.find("issuer")
    symbol = _find_text(issuer, "issuerTradingSymbol").upper()
    issuer_cik = _find_text(issuer, "issuerCik")
    issuer_name = _find_text(issuer, "issuerName")

    owners = root.findall("reportingOwner")
    primary = owners[0] if owners else None
    owner_cik = (
        _find_text(primary, "reportingOwnerId/rptOwnerCik")
        if primary is not None
        else ""
    )
    owner_name = (
        _find_text(primary, "reportingOwnerId/rptOwnerName")
        if primary is not None
        else ""
    )
    role = _relationship_role(
        primary.find("reportingOwnerRelationship") if primary is not None else None
    )

    rows: list[dict[str, Any]] = []
    txn_nodes = [(el, False) for el in root.findall(".//nonDerivativeTransaction")] + [
        (el, True) for el in root.findall(".//derivativeTransaction")
    ]
    for txn, is_deriv in txn_nodes:
        code_raw = _find_text(txn, "transactionCoding/transactionCode")
        txn_date_raw = _find_text(txn, "transactionDate/value")
        shares = _to_float(
            _find_text(txn, "transactionAmounts/transactionShares/value")
        )
        # A priceless open-market trade (price not reported) yields value_usd=0.0
        # -> contributes zero signed dollar flow downstream (conservative, never
        # inflated); net_shares is still signed by acquired/disposed.
        price = _to_float(
            _find_text(txn, "transactionAmounts/transactionPricePerShare/value")
        )
        acq_disp = _find_text(
            txn, "transactionAmounts/transactionAcquiredDisposedCode/value"
        ).upper()
        sign = 1.0 if acq_disp == "A" else (-1.0 if acq_disp == "D" else 0.0)
        txn_date = pd.Timestamp(txn_date_raw) if txn_date_raw else pd.NaT

        rows.append(
            {
                "accession": accession,
                "symbol": symbol,
                "issuer_cik": issuer_cik,
                "issuer_name": issuer_name,
                "reporting_owner_cik": owner_cik,
                "reporting_owner_name": owner_name,
                "role": role,
                "security_title": _find_text(txn, "securityTitle/value"),
                "is_derivative": is_deriv,
                "transaction_date": txn_date,
                "event_date": txn_date,
                "filing_date": filing_date,
                "available_at": available_at,
                "timestamp": available_at,
                "disclosure_date": disclosure_date,
                "transaction_code": code_raw,
                "transaction_type": classify_transaction_code(code_raw),
                "acquired_disposed": acq_disp,
                "shares": shares,
                "price": price,
                "value_usd": abs(shares * price),
                "net_shares": shares * sign,
                "trades_count": 1,
            }
        )
    return rows


def form4_rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble parsed rows into a typed, PIT-safe DataFrame (stable schema)."""
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in FORM4_COLUMNS})

    df = pd.DataFrame(rows)
    for c in FORM4_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[FORM4_COLUMNS]

    for c in ("transaction_date", "event_date", "filing_date", "disclosure_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("available_at", "timestamp"):
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    for c in ("shares", "price", "value_usd", "net_shares"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["trades_count"] = (
        pd.to_numeric(df["trades_count"], errors="coerce").fillna(1).astype("int64")
    )
    df["is_derivative"] = df["is_derivative"].astype(bool)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Network layer (exercised live in Phase-4 verification, not in unit tests)
# ---------------------------------------------------------------------------


def resolve_user_agent(user_agent: str | None = None) -> str:
    """Resolve the SEC-compliant User-Agent (real org + contact email).

    Resolution order: explicit arg -> ``SEC_USER_AGENT`` env -> compose from
    ``SEC_CONTACT_EMAIL`` env -> project settings. Raises if none configured —
    an undeclared/fake UA gets the project blocked (HTTP 403) by the SEC.
    """
    if user_agent and user_agent.strip():
        return user_agent.strip()
    env_ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if env_ua:
        return env_ua
    email = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    if email:
        return f"Assembled-Trading-AI {email}"
    try:
        from src.assembled_core.config.settings import get_settings  # noqa: PLC0415

        ua = getattr(get_settings(), "sec_user_agent", "") or ""
        if ua.strip():
            return ua.strip()
    except Exception:  # settings optional here
        pass
    raise ValueError(
        "SEC User-Agent not configured. Set SEC_USER_AGENT (e.g. "
        "'Assembled-Trading-AI you@example.com') or SEC_CONTACT_EMAIL. "
        "SEC fair-access policy requires a declared UA with a real contact."
    )


class _RateLimiter:
    def __init__(self, min_spacing: float = MIN_REQUEST_SPACING_S) -> None:
        self.min_spacing = min_spacing
        self._last = 0.0

    def wait(self) -> None:
        delta = time.monotonic() - self._last
        if delta < self.min_spacing:
            time.sleep(self.min_spacing - delta)
        self._last = time.monotonic()


def _http_get(
    url: str,
    user_agent: str,
    *,
    limiter: _RateLimiter,
    max_retries: int = 5,
    timeout: float = 30.0,
) -> bytes:
    """GET with declared UA, global pacing, and exponential backoff on 403/429/503."""
    import urllib.error
    import urllib.request

    headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        limiter.wait()
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return bytes(resp.read())
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (403, 429, 503) and attempt < max_retries - 1:
                backoff = 0.5 * (2**attempt)
                logger.warning(
                    "[WARN] edgar_get http=%s attempt=%d backoff=%.1fs url=%s",
                    exc.code,
                    attempt + 1,
                    backoff,
                    url,
                )
                time.sleep(backoff)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(0.5 * (2**attempt))
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"edgar_get failed without exception: {url}")


def enumerate_form4_filings(
    date: pd.Timestamp | str,
    user_agent: str | None = None,
    *,
    limiter: _RateLimiter | None = None,
) -> list[dict[str, str]]:
    """Enumerate all Form 4 / 4/A filings accepted on ``date`` via the daily-index."""
    ua = resolve_user_agent(user_agent)
    limiter = limiter or _RateLimiter()
    d = pd.Timestamp(date)
    url = _DAILY_INDEX_FMT.format(
        year=d.year, qtr=(d.month - 1) // 3 + 1, ymd=d.strftime("%Y%m%d")
    )
    try:
        raw = _http_get(url, ua, limiter=limiter)
    except Exception as exc:  # weekend/holiday -> 404; report, do not crash
        logger.warning("[WARN] form4_enumerate no index for %s: %s", d.date(), exc)
        return []
    return parse_form4_index(raw.decode("latin-1"))


def fetch_submission(
    filename_or_url: str,
    user_agent: str | None = None,
    *,
    limiter: _RateLimiter | None = None,
) -> str:
    """Fetch a full submission ``.txt`` (header + embedded ownership XML)."""
    ua = resolve_user_agent(user_agent)
    limiter = limiter or _RateLimiter()
    url = (
        filename_or_url
        if filename_or_url.startswith("http")
        else EDGAR_ARCHIVES + filename_or_url
    )
    return _http_get(url, ua, limiter=limiter).decode("latin-1")


def ingest_form4(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str | None = None,
    *,
    symbols: list[str] | None = None,
    user_agent: str | None = None,
    out_path: Path | str | None = None,
    max_filings: int | None = None,
) -> pd.DataFrame:
    """Ingest Form 4 filings over ``[start_date, end_date]`` into a typed frame.

    Writes ``out_path`` (default ``output/insider_form4.parquet``) and logs a
    completeness summary (P/S/unknown counts, % unknown, requests issued). Does
    NOT overwrite the legacy ``insider_trading.parquet``.
    """
    ua = resolve_user_agent(user_agent)
    limiter = _RateLimiter()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize() if end_date is not None else start
    sym_set = {s.upper() for s in symbols} if symbols else None

    all_rows: list[dict[str, Any]] = []
    n_filings = 0
    unknown_codes: dict[str, int] = {}
    started = time.monotonic()

    for d in pd.date_range(start, end, freq="D"):
        entries = enumerate_form4_filings(d, ua, limiter=limiter)
        for entry in entries:
            if max_filings is not None and n_filings >= max_filings:
                break
            try:
                text = fetch_submission(entry["url"], ua, limiter=limiter)
            except Exception as exc:
                logger.warning(
                    "[WARN] form4_fetch_skip cik=%s file=%s reason=%s",
                    entry.get("cik"),
                    entry.get("filename"),
                    exc,
                )
                continue
            n_filings += 1
            rows = parse_form4_submission(text, accession=Path(entry["filename"]).stem)
            for r in rows:
                if sym_set is not None and r["symbol"] not in sym_set:
                    continue
                if r["transaction_type"] == "unknown" and r["transaction_code"]:
                    unknown_codes[r["transaction_code"]] = (
                        unknown_codes.get(r["transaction_code"], 0) + 1
                    )
                all_rows.append(r)

    df = form4_rows_to_dataframe(all_rows)

    total = len(df)
    n_ps = int((df["transaction_type"].isin(["P", "S"])).sum()) if total else 0
    n_unknown = total - n_ps
    pct_unknown = (100.0 * n_unknown / total) if total else 0.0
    elapsed = max(time.monotonic() - started, 1e-9)
    logger.info(
        "[OK] form4_ingest filings=%d rows=%d P/S=%d unknown=%d pct_unknown=%.1f%% "
        "req_rate=%.2f/s window=%s..%s",
        n_filings,
        total,
        n_ps,
        n_unknown,
        pct_unknown,
        n_filings / elapsed,
        start.date(),
        end.date(),
    )
    if unknown_codes:
        logger.warning(
            "[WARN] form4_unknown_codes (not P/S, classified unknown): %s",
            ", ".join(
                f"{c}={n} ({_KNOWN_CODE_LEGEND.get(c, 'unrecognized')})"
                for c, n in sorted(unknown_codes.items(), key=lambda kv: -kv[1])
            ),
        )

    out = Path(out_path) if out_path else Path("output") / "insider_form4.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
        logger.info("[OK] form4_ingest wrote %d rows -> %s", total, out)
    except Exception as exc:
        logger.error("[ERROR] form4_ingest parquet write failed -> %s: %s", out, exc)
    return df
