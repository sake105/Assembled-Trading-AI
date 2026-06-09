"""Free Congress (STOCK Act) trades ingester — replaces the paid QuiverQuant feed.

Congress trades are STOCK-Act Periodic Transaction Reports (PTRs), a SEPARATE
universe from SEC Form 4 insider trades — never merge them. This module fetches
free, structured, ticker-level disclosures and emits a PIT-correct frame to
``output/congress_trades.parquet`` consumed by
:func:`congress_features.add_congress_features` (which the pipeline wires via
``feature_cfg.include_congress`` + ``congress_data_path``).

Sources (volunteer GitHub mirrors of the official filings; verified live
2026-06-09 — no API key, no ToS click-gate):
  * ``kadoa``        kadoa-org/congress-trading-monitor — BOTH chambers, ISO
                     dates, numeric amount range, ``filing_date`` = disclosure.
  * ``house_watcher`` TattooedHead/house-stock-watcher-data — House only,
                     MM/DD/YYYY dates, pre-computed ``amount_mid``.
The durable bedrock (never goes stale) is the official US House Clerk
``<YEAR>FD.zip`` + per-PTR PDFs and the Senate eFD — both require PDF parsing and
are documented as the fallback path, NOT built here. The mirrors have no SLA;
the predecessor (house/senate-stock-watcher S3) is now dead (HTTP 403).

PIT contract
------------
``disclosure_date`` (the filing/notification date) is the availability key —
``available_at``/``timestamp`` are set from it, NEVER from ``transaction_date``
(STOCK Act allows up to 45 days lag → using the trade date injects look-ahead).
The economic ``transaction_date``/``event_date`` is kept immutable; when a feed
omits the disclosure date the +``CONGRESS_DAYS`` (45) STOCK Act fallback fills a
SEPARATE ``disclosure_date`` column and never mutates the trade date (E-038).

Amounts are coarse statutory RANGES (e.g. "$1,001 - $15,000"); there is no exact
figure anywhere in the disclosure system. We map a range to its arithmetic
midpoint and flag it as low-resolution.

LEGAL / COMPLIANCE caveat (must be honoured by callers)
-------------------------------------------------------
Financial-disclosure records carry a statutory restriction (5 U.S.C. app.
§13107): unlawful to use for a COMMERCIAL purpose. A private research backtest is
the defensible/intended use; a LIVE money-making trading system is legally
uncertain and must be escalated before production use. Data is 100% public,
statutorily-delayed — no MNPI concern.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from src.assembled_core.data.source_latencies import CONGRESS_DAYS

logger = logging.getLogger(__name__)

__all__ = [
    "amount_midpoint",
    "parse_amount_range",
    "parse_kadoa_records",
    "parse_house_watcher_records",
    "normalize_congress",
    "dedupe_congress",
    "load_congress_sample",
    "fetch_congress_trades",
    "ingest_congress",
    "CONGRESS_COLUMNS",
    "KADOA_URL",
    "HOUSE_WATCHER_URL",
]

KADOA_URL = (
    "https://raw.githubusercontent.com/kadoa-org/congress-trading-monitor/"
    "main/public/data/trades.json"
)
KADOA_CDN = (
    "https://cdn.jsdelivr.net/gh/kadoa-org/congress-trading-monitor@main/"
    "public/data/trades.json"
)
HOUSE_WATCHER_URL = (
    "https://raw.githubusercontent.com/TattooedHead/house-stock-watcher-data/"
    "main/data/all_transactions.json"
)
_DEFAULT_UA = "Mozilla/5.0 (Assembled-Trading-AI congress-ingest)"
_NULL_TICKERS = {"", "--", "N/A", "NONE", "NULL"}

CONGRESS_COLUMNS = [
    "symbol",
    "transaction_date",
    "event_date",
    "disclosure_date",
    "timestamp",
    "available_at",
    "amount",
    "amount_low",
    "amount_high",
    "transaction_type",
    "type",
    "member",
    "chamber",
    "party",
    "source",
]


def _normalize_side(raw: Any) -> str | None:
    """Normalize a source transaction-type label to ``'buy'`` / ``'sell'`` / None.

    Emitted as the ``type`` column so sign-aware consumers
    (e.g. ``congress_features.compute_congress_net_buy_score``, which keys on a
    ``type`` column) get a correct sign instead of fail-open defaulting a Sale to
    a buy. ``'Exchange'`` and unrecognized labels -> None.
    """
    s = str(raw).lower() if raw is not None else ""
    if "purchas" in s or s == "buy":
        return "buy"
    if "sale" in s or "sell" in s:
        return "sell"
    return None


# ---------------------------------------------------------------------------
# Pure parsing / mapping (unit-tested offline)
# ---------------------------------------------------------------------------


def amount_midpoint(low: float | int | None, high: float | int | None) -> float:
    """Arithmetic midpoint of a statutory amount range.

    Both bounds missing -> NaN. One bound missing -> the present bound.
    """
    if low is None and high is None:
        return float("nan")
    if low is None:
        return float(high)  # type: ignore[arg-type]
    if high is None:
        return float(low)
    return (float(low) + float(high)) / 2.0


def parse_amount_range(label: str | None) -> tuple[float | None, float | None]:
    """Parse ``"$1,001 - $15,000"`` -> ``(1001.0, 15000.0)`` (fallback for string amounts)."""
    if not label:
        return (None, None)
    nums = [n.replace(",", "") for n in re.findall(r"[\d,]+", str(label))]
    vals = [float(n) for n in nums if n.isdigit()]
    if len(vals) >= 2:
        return (vals[0], vals[1])
    if len(vals) == 1:
        return (vals[0], vals[0])
    return (None, None)


def _ticker(raw: Any) -> str:
    t = (str(raw) if raw is not None else "").strip().upper()
    return "" if t in _NULL_TICKERS else t


def _ts(value: Any, fmt: str | None = None) -> pd.Timestamp:
    if value is None or value == "":
        return pd.NaT
    try:
        if fmt:
            return pd.to_datetime(value, format=fmt, errors="coerce")
        return pd.to_datetime(value, errors="coerce")
    except (ValueError, TypeError):
        return pd.NaT


def _with_disclosure_fallback(
    event_date: pd.Timestamp, disclosure: pd.Timestamp
) -> pd.Timestamp:
    """Disclosure date, or trade-date + STOCK-Act lag when the feed omits it.

    The economic ``event_date`` is NEVER mutated — only a missing disclosure date
    is derived (E-038 boundary discipline).
    """
    if pd.notna(disclosure):
        return disclosure
    if pd.notna(event_date):
        return event_date + pd.Timedelta(days=CONGRESS_DAYS)
    return pd.NaT


def parse_kadoa_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse kadoa-org/congress-trading-monitor records (both chambers, ISO dates)."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        symbol = _ticker(rec.get("ticker"))
        if not symbol:
            continue
        event_date = _ts(rec.get("transaction_date"))
        disclosure = _with_disclosure_fallback(event_date, _ts(rec.get("filing_date")))
        low, high = rec.get("amount_range_low"), rec.get("amount_range_high")
        rows.append(
            {
                "symbol": symbol,
                "transaction_date": event_date,
                "event_date": event_date,
                "disclosure_date": disclosure,
                "amount": amount_midpoint(low, high),
                "amount_low": low,
                "amount_high": high,
                "transaction_type": rec.get("transaction_type"),
                "member": rec.get("filer_name"),
                "chamber": rec.get("chamber"),
                "party": rec.get("party"),
                "source": "kadoa",
            }
        )
    return rows


def parse_house_watcher_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse TattooedHead/house-stock-watcher records (House, MM/DD/YYYY, amount_mid)."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        symbol = _ticker(rec.get("ticker"))
        if not symbol:
            continue
        event_date = _ts(rec.get("transaction_date"), fmt="%m/%d/%Y")
        disclosure = _with_disclosure_fallback(
            event_date, _ts(rec.get("disclosure_date"), fmt="%m/%d/%Y")
        )
        amount = rec.get("amount_mid")
        if amount is None:
            low, high = parse_amount_range(rec.get("amount"))
            amount = amount_midpoint(low, high)
        else:
            amount = float(amount)
            low, high = parse_amount_range(rec.get("amount"))
        rows.append(
            {
                "symbol": symbol,
                "transaction_date": event_date,
                "event_date": event_date,
                "disclosure_date": disclosure,
                "amount": amount,
                "amount_low": low,
                "amount_high": high,
                "transaction_type": rec.get("type"),
                "member": rec.get("representative"),
                "chamber": "House",
                "party": rec.get("party"),
                "source": "house_watcher",
            }
        )
    return rows


def normalize_congress(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble parsed rows into a typed, PIT-safe DataFrame (stable schema)."""
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in CONGRESS_COLUMNS})

    df = pd.DataFrame(rows)
    for c in CONGRESS_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    for c in ("transaction_date", "event_date", "disclosure_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    # PIT anchors derived from the disclosure date (the public-availability day).
    df["timestamp"] = pd.to_datetime(df["disclosure_date"], errors="coerce", utc=True)
    df["available_at"] = df["timestamp"]
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype("float64")
    df["amount_low"] = pd.to_numeric(df["amount_low"], errors="coerce")
    df["amount_high"] = pd.to_numeric(df["amount_high"], errors="coerce")
    df["chamber"] = df["chamber"].map(lambda x: str(x).title() if pd.notna(x) else x)
    df["type"] = df["transaction_type"].map(_normalize_side)

    # Data-quality guard: a PTR discloses a PAST trade, so
    # transaction_date > disclosure_date is logically impossible (source
    # data-entry error). Drop such rows LOUDLY rather than letting a bad/future
    # trade date pollute the feature windows.
    both = df["transaction_date"].notna() & df["disclosure_date"].notna()
    impossible = both & (df["transaction_date"] > df["disclosure_date"])
    n_bad = int(impossible.sum())
    if n_bad:
        logger.warning(
            "[WARN] congress_normalize dropped %d/%d rows with "
            "transaction_date > disclosure_date (source data error)",
            n_bad,
            len(df),
        )
        df = df[~impossible]

    return df[CONGRESS_COLUMNS].reset_index(drop=True)


def dedupe_congress(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cross-source duplicates of the same economic trade.

    Keys on the NORMALIZED side (``type``), NOT the raw ``transaction_type``:
    kadoa emits "Sale (Partial)"/"Sale (Full)" while house_watcher emits "Sale"
    for the SAME House PTR, so keying on the raw label would let an identical
    sell survive in both mirrors and double-count the (unsigned) amount feature
    in ``add_congress_features``. None/unknown sides fall back to the raw label
    so distinct Exchange/unknown trades are not over-collapsed.
    """
    if df.empty:
        return df
    side_key = df["type"].fillna(df["transaction_type"])
    return (
        df.assign(_side_key=side_key)
        .drop_duplicates(
            subset=["symbol", "event_date", "disclosure_date", "_side_key"],
            keep="first",
        )
        .drop(columns="_side_key")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Pipeline entry point (imported by trading_cycle_shared)
# ---------------------------------------------------------------------------

_SAMPLE_RECORDS = [
    {
        "ticker": "AAPL",
        "transaction_date": "2024-01-10",
        "filing_date": "2024-02-05",
        "transaction_type": "Purchase",
        "amount_range_low": 1001,
        "amount_range_high": 15000,
        "filer_name": "Sample Member",
        "chamber": "House",
        "party": "I",
    }
]


def load_congress_sample(
    path: Path | str | None = None, *, allow_sample: bool = False
) -> pd.DataFrame:
    """Load congress trades for the pipeline (mirrors insider_ingest's fail-loud gate).

    With ``path`` -> read the saved ingest parquet/csv. Without ``path`` ->
    ``allow_sample=True`` is REQUIRED to opt into built-in dummy data; otherwise a
    ``ValueError`` is raised (no silent phantom-data fallback).
    """
    if path is not None:
        path = Path(path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(
                f"Unsupported congress file format: {path.suffix}. Use .parquet or .csv"
            )
        for c in ("timestamp", "available_at"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        for c in ("transaction_date", "event_date", "disclosure_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    if not allow_sample:
        raise ValueError(
            "load_congress_sample() received no path and no explicit "
            "allow_sample=True. Production callers must provide a real congress "
            "data file (run ingest_congress); tests/dev callers must pass "
            "allow_sample=True to opt into dummy sample data."
        )
    return normalize_congress(parse_kadoa_records(_SAMPLE_RECORDS))


# ---------------------------------------------------------------------------
# Network layer (exercised live in Phase-4 verification, not in unit tests)
# ---------------------------------------------------------------------------


def _http_get_json(url: str, user_agent: str, timeout: float = 60.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_congress_trades(
    source: str = "kadoa",
    *,
    url: str | None = None,
    user_agent: str | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Fetch raw records for ``source`` (``kadoa`` | ``house_watcher``).

    kadoa falls back from raw.githubusercontent to the jsDelivr CDN mirror on
    failure (same bytes, different infra). Returns ``[]`` on total failure and
    logs a DEGRADED warning — never silently returns empty without a signal.
    """
    ua = user_agent or _DEFAULT_UA
    if source == "kadoa":
        urls = [url or KADOA_URL, KADOA_CDN]
    elif source == "house_watcher":
        urls = [url or HOUSE_WATCHER_URL]
    else:
        raise ValueError(f"unknown congress source: {source!r}")

    last_exc: Exception | None = None
    for u in urls:
        try:
            data = _http_get_json(u, ua, timeout=timeout)
            if isinstance(data, list):
                logger.info(
                    "[OK] congress_fetch source=%s rows=%d url=%s", source, len(data), u
                )
                return data
            logger.warning(
                "[WARN] congress_fetch source=%s non-list payload url=%s", source, u
            )
        except Exception as exc:  # noqa: BLE001 — best-effort multi-mirror fetch
            last_exc = exc
            logger.warning(
                "[WARN] congress_fetch source=%s url=%s failed: %s", source, u, exc
            )
    logger.error(
        "[ERROR] congress_fetch source=%s DEGRADED — all mirrors failed: %s",
        source,
        last_exc,
    )
    return []


def ingest_congress(
    out_path: Path | str | None = None,
    *,
    sources: tuple[str, ...] = ("kadoa", "house_watcher"),
    user_agent: str | None = None,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Fetch + normalize + (optionally) dedupe congress trades into a typed frame.

    Writes ``out_path`` (default ``output/congress_trades.parquet``) and logs a
    completeness summary. Cross-source dedupe keeps the first occurrence per
    ``(symbol, event_date, disclosure_date, transaction_type)`` so the House rows
    shared by kadoa + house_watcher are not double-counted.
    """
    rows: list[dict[str, Any]] = []
    for source in sources:
        records = fetch_congress_trades(source, user_agent=user_agent)
        if source == "kadoa":
            rows.extend(parse_kadoa_records(records))
        elif source == "house_watcher":
            rows.extend(parse_house_watcher_records(records))

    df = normalize_congress(rows)

    n_before = len(df)
    if dedupe and not df.empty:
        df = dedupe_congress(df)
    n_dupes = n_before - len(df)

    total = len(df)
    n_disc = int(df["disclosure_date"].notna().sum()) if total else 0
    by_chamber = df["chamber"].fillna("?").value_counts().to_dict() if total else {}
    logger.info(
        "[OK] congress_ingest rows=%d dupes_dropped=%d with_disclosure=%d chambers=%s",
        total,
        n_dupes,
        n_disc,
        by_chamber,
    )

    out = Path(out_path) if out_path else Path("output") / "congress_trades.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
        logger.info("[OK] congress_ingest wrote %d rows -> %s", total, out)
    except Exception as exc:
        logger.error("[ERROR] congress_ingest parquet write failed -> %s: %s", out, exc)
    return df
