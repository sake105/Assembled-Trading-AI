"""SEC EDGAR XBRL Company-Facts fundamentals ingester — free, PIT-correct
replacement for paid fundamentals / earnings-estimate feeds.

Pulls *as-reported* financial-statement facts (EPS, net income, revenue, share
counts, ...) per issuer from the SEC Company Facts API
(``data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json``) and emits a tall/long,
restatement-versioned, PIT-correct frame to ``output/fundamentals_xbrl.parquet``.

Scope (Phase 3, evidence-gated rollout)
---------------------------------------
This module ONLY ingests + loads fundamentals and proves coverage. It does NOT
re-activate any factor weights — that is a separate, evidence-gated step.

It reanimates the PEAD / SUE path (``features/pead_sue.py`` consumes an actual
EPS time-series), which free XBRL *can* serve. It does NOT (and cannot) revive a
true analyst-consensus ``earnings_surprise_z``: SEC Company Facts contains
*actuals only* — there is no consensus estimate in EDGAR. That factor therefore
stays dead unless redefined as a model-based surprise.

PIT contract
------------
- ``available_at`` = the EDGAR ACCEPTANCE instant of the SUBMISSION that first
  reported (or re-stated) the datapoint — NEVER the fiscal ``period_end``. The
  Company Facts JSON gives only a date-granular ``filed``; the microsecond
  acceptance instant is resolved from the issuer's submissions feed
  (``acceptanceDateTime``). NOTE on zones (Phase-4 cross-checked against the SGML
  ``ACCEPTANCE-DATETIME`` header for a real filing): the submissions-feed
  ``acceptanceDateTime`` is ISO-8601 and ALREADY UTC (trailing ``Z``) — it is
  parsed honoring its declared zone. Only the bare 14-digit SGML header (no zone)
  is US/Eastern wall-clock, converted via
  :func:`edgar_form4_ingest.acceptance_datetime_to_utc`. (An earlier draft
  mis-treated the feed value as Eastern, yielding an ``available_at`` 4h too
  late — conservative but inaccurate; fixed.)
- ``filed_date`` / ``disclosure_date`` = ``filed`` (FILED-AS-OF date, date-only);
  the date-granular gating fallback when ``available_at`` could not be resolved
  (then ``available_at`` is left ``NaT`` — we NEVER stamp ``now()``, which would
  fabricate availability; the PIT firewall drops the row instead). Fallback uses
  :data:`source_latencies.EDGAR_DAYS` so a same-day filing only becomes visible
  at the next-bar boundary.
- ``period_start`` / ``period_end`` = the economic fiscal boundaries; kept
  immutable and NEVER used for availability gating (avoids the E-038
  boundary-shift / E-041 tail-read anti-patterns).

Restatement / amendment discipline
-----------------------------------
APPEND-ONLY, versioned by accession. A 10-K/A (or a later 10-Q) re-stating a
prior period emits a NEW datapoint with the SAME ``(tag, period_end)`` but a
DIFFERENT ``accession`` and a LATER acceptance — we persist BOTH as separate
rows, each with its own ``available_at``. PIT consumption (:func:`select_pit_rows`)
then selects, per ``(symbol, namespace, tag, period_end)``, the row with the
MAXIMAL availability that is still ``<= as_of`` — so a restatement filed later
can never retroactively change an earlier-as-of value (E-038 discipline reused).

Compliance (SEC fair-access policy)
-----------------------------------
Reuses the proven Form 4 networking layer wholesale (NO second HTTP / rate-limit
truth — Rule 50): a declared ``User-Agent`` (real org + contact email) is
REQUIRED, <=8 req/s global pacing + exponential backoff on 403/429/503. The 10/s
hard cap is budgeted GLOBALLY across ``www.sec.gov`` / ``data.sec.gov``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Deliberate package-internal reuse of the Form 4 networking + acceptance-parse
# layer so there is exactly ONE SEC HTTP / rate-limit / ET->UTC truth (Rule 50).
from src.assembled_core.data.edgar_form4_ingest import (
    _http_get,
    _RateLimiter,
    acceptance_datetime_to_utc,
    fetch_cik_map,
    resolve_user_agent,
)
from src.assembled_core.data.source_latencies import EDGAR_DAYS

logger = logging.getLogger(__name__)

__all__ = [
    "FUNDAMENTAL_TAGS",
    "XBRL_COLUMNS",
    "parse_acceptance_datetime",
    "build_accession_acceptance_map",
    "submission_page_names",
    "parse_company_facts",
    "company_facts_rows_to_dataframe",
    "is_amendment",
    "attach_available_at",
    "select_pit_rows",
    "coalesce_field",
    "fetch_company_facts",
    "fetch_submissions",
    "fetch_acceptance_map",
    "ingest_fundamentals_xbrl",
]

_COMPANY_FACTS_FMT = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_SUBMISSIONS_FMT = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# field -> ordered (namespace, tag) candidates, most-preferred first. The ingester
# extracts the UNION of these; the per-field choice is made at consumption time by
# :func:`coalesce_field` (ordered-coalesce, recording which tag won).
FUNDAMENTAL_TAGS: dict[str, list[tuple[str, str]]] = {
    "eps_diluted": [
        ("us-gaap", "EarningsPerShareDiluted"),
        ("us-gaap", "EarningsPerShareBasicAndDiluted"),
        ("us-gaap", "IncomeLossFromContinuingOperationsPerDilutedShare"),
    ],
    "eps_basic": [
        ("us-gaap", "EarningsPerShareBasic"),
        ("us-gaap", "EarningsPerShareBasicAndDiluted"),
    ],
    "net_income": [
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "ProfitLoss"),
        ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic"),
    ],
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
    ],
    # INSTANT cover-page share count (dei namespace) — used for market-cap
    # normalisation; distinct from the duration weighted-average below.
    "shares_outstanding": [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ],
    # DURATION weighted-average shares — the denominator when deriving EPS from
    # NetIncomeLoss for filers that omit the EPS tag.
    "weighted_diluted_shares": [
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"),
    ],
    "total_assets": [("us-gaap", "Assets")],
    "equity": [
        ("us-gaap", "StockholdersEquity"),
        (
            "us-gaap",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    ],
    # REALISED repurchases (cash-flow line) — economically DIFFERENT from the 8-K
    # buyback-announcement signal; provided additively, NOT a drop-in for
    # buyback_drift_score.
    "buybacks": [
        ("us-gaap", "PaymentsForRepurchaseOfCommonStock"),
        ("us-gaap", "PaymentsForRepurchaseOfEquity"),
    ],
}

XBRL_COLUMNS = [
    "symbol",
    "cik",
    "namespace",
    "tag",
    "unit",
    "val",
    "period_start",
    "period_end",
    "fy",
    "fp",
    "frame",
    "form",
    "is_amendment",
    "accession",
    "filed_date",
    "disclosure_date",
    "available_at",
    "timestamp",
]

_ISO_DT_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})")


def _wanted_tags() -> set[tuple[str, str]]:
    return {nt for cands in FUNDAMENTAL_TAGS.values() for nt in cands}


# ---------------------------------------------------------------------------
# Pure parsing / classification / PIT logic (unit-tested offline)
# ---------------------------------------------------------------------------


def parse_acceptance_datetime(raw: str) -> pd.Timestamp:
    """Parse a SEC acceptance datetime to UTC, by FORM (zones differ — see below).

    Two recognised forms with DIFFERENT zone semantics (Phase-4 cross-checked):
    - 14-digit SGML header ``YYYYMMDDHHMMSS`` (no zone): US/Eastern wall-clock,
      DST-aware, via :func:`acceptance_datetime_to_utc` (the proven Form 4 path).
    - ISO-8601 submissions-feed value (e.g. ``2026-05-01T10:01:00.000Z``): carries
      its own zone — the SEC feed is UTC (trailing ``Z``). Parsed honoring that
      zone; a naive ISO (no offset) is assumed UTC. It is NOT re-interpreted as
      Eastern (doing so produced an ``available_at`` 4h too late).

    Raises:
        ValueError: if the input matches neither recognised form.
    """
    s = str(raw).strip()
    if len(s) >= 14 and s[:14].isdigit():
        return acceptance_datetime_to_utc(s[:14])
    if _ISO_DT_RE.match(s):
        ts = pd.Timestamp(s)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    raise ValueError(f"unparseable acceptance datetime: {raw!r}")


def build_accession_acceptance_map(
    submissions: dict[str, Any],
) -> dict[str, pd.Timestamp]:
    """Map ``accessionNumber -> acceptance instant (UTC)`` from a submissions block.

    Accepts BOTH shapes: the main submissions JSON (arrays under
    ``filings.recent``) AND a paginated page file (``filings.files[].name``), which
    carries the same parallel arrays at the TOP LEVEL. Unparseable / missing
    acceptance datetimes are skipped with a WARNING (never silently defaulted).
    """
    filings = submissions.get("filings")
    if isinstance(filings, dict) and isinstance(filings.get("recent"), dict):
        block = filings["recent"]
    else:
        block = submissions  # flat paginated page: arrays at top level
    accs = block.get("accessionNumber", []) or []
    adts = block.get("acceptanceDateTime", []) or []
    out: dict[str, pd.Timestamp] = {}
    for i, acc in enumerate(accs):
        if not acc or i >= len(adts) or not adts[i]:
            continue
        try:
            out[str(acc)] = parse_acceptance_datetime(adts[i])
        except ValueError:
            logger.warning(
                "[WARN] xbrl_acceptance_unparseable acc=%s raw=%r", acc, adts[i]
            )
    return out


def submission_page_names(submissions: dict[str, Any]) -> list[str]:
    """Names of the older paginated submission pages (``filings.files[].name``)."""
    files = (submissions.get("filings", {}) or {}).get("files", []) or []
    return [str(f["name"]) for f in files if isinstance(f, dict) and f.get("name")]


def parse_company_facts(
    payload: dict[str, Any],
    *,
    symbol: str,
    cik: str | int | None = None,
    wanted: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Walk a Company Facts JSON into one row per (namespace, tag, unit, datapoint).

    Only ``(namespace, tag)`` pairs in ``wanted`` (default: the union of
    :data:`FUNDAMENTAL_TAGS`) are emitted — every other tag is skipped. Reads
    BOTH the ``us-gaap`` and ``dei`` namespaces (the dei namespace carries the
    cover-page share count). ``available_at`` / ``is_amendment`` are NOT set here
    (resolved later via :func:`attach_available_at`).
    """
    want = wanted if wanted is not None else _wanted_tags()
    sym = str(symbol).strip().upper()
    cik_val = cik if cik is not None else payload.get("cik")
    facts = payload.get("facts", {}) or {}
    rows: list[dict[str, Any]] = []
    for namespace, tagmap in facts.items():
        for tag, body in (tagmap or {}).items():
            if (namespace, tag) not in want:
                continue
            units = (body or {}).get("units", {}) or {}
            for unit, datapoints in units.items():
                for dp in datapoints or []:
                    rows.append(
                        {
                            "symbol": sym,
                            "cik": cik_val,
                            "namespace": namespace,
                            "tag": tag,
                            "unit": unit,
                            "val": dp.get("val"),
                            "period_start": dp.get("start"),
                            "period_end": dp.get("end"),
                            "fy": dp.get("fy"),
                            "fp": dp.get("fp"),
                            "frame": dp.get("frame"),
                            "form": dp.get("form"),
                            "accession": dp.get("accn"),
                            "filed": dp.get("filed"),
                        }
                    )
    return rows


def _coerce_xbrl_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("period_start", "period_end", "filed_date", "disclosure_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")  # naive, economic/date-only
    for c in ("available_at", "timestamp"):
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    df["val"] = pd.to_numeric(df["val"], errors="coerce").astype("float64")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    # Build a clean numpy-bool column without an object-dtype fillna (which emits
    # a pandas downcasting FutureWarning): NA/None/NaN -> False.
    df["is_amendment"] = [False if pd.isna(v) else bool(v) for v in df["is_amendment"]]
    for c in (
        "symbol",
        "cik",
        "namespace",
        "tag",
        "unit",
        "fp",
        "frame",
        "form",
        "accession",
    ):
        df[c] = df[c].astype("object")
    return df.reset_index(drop=True)


def company_facts_rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble parsed rows into a typed, PIT-safe DataFrame (stable schema)."""
    if not rows:
        df = pd.DataFrame({c: pd.Series(dtype="object") for c in XBRL_COLUMNS})
        return _coerce_xbrl_dtypes(df)

    df = pd.DataFrame(rows)
    # disclosure_date / filed_date both come from the FILED-AS-OF date (the day
    # the submission became public). available_at (finer) is attached later.
    if "filed" in df.columns:
        df["filed_date"] = df["filed"]
        df["disclosure_date"] = df["filed"]
    for c in XBRL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[XBRL_COLUMNS]
    return _coerce_xbrl_dtypes(df)


def is_amendment(form: str | None) -> bool:
    """True if ``form`` is an amendment (ends with ``/A``, e.g. ``10-K/A``)."""
    return bool(form) and str(form).strip().upper().endswith("/A")


def attach_available_at(
    df: pd.DataFrame, accn_acceptance: dict[str, pd.Timestamp]
) -> pd.DataFrame:
    """Join ``accession -> acceptance instant`` onto ``available_at`` + set ``is_amendment``.

    Accessions with no resolved acceptance get ``available_at = NaT`` (we NEVER
    stamp ``now()``); :func:`select_pit_rows` then falls back to
    ``filed_date + EDGAR_DAYS`` for the date-granular gate.
    """
    out = df.copy()
    if "accession" in out.columns:
        out["available_at"] = out["accession"].map(
            lambda a: accn_acceptance.get(str(a), pd.NaT)
        )
        out["available_at"] = pd.to_datetime(
            out["available_at"], errors="coerce", utc=True
        )
    if "form" in out.columns:
        out["is_amendment"] = out["form"].map(is_amendment).astype(bool)
    return out


def _effective_availability(df: pd.DataFrame, latency_days: int) -> pd.Series:
    """Per-row availability instant (UTC): ``available_at`` else ``filed_date + latency``.

    Computes a LOCAL series — never mutates the source timestamp columns (E-038).
    The fallback is conservative: a same-day filing becomes visible only at the
    next-bar boundary.
    """
    avail = df["available_at"]
    filed = pd.to_datetime(df["filed_date"], errors="coerce")
    # Defensive: filed_date is coerced naive by _coerce_xbrl_dtypes today, but
    # PIT-critical code must not crash on an already-tz-aware value (future caller
    # / parquet round-trip) — localize when naive, convert when aware.
    filed = (
        filed.dt.tz_convert("UTC")
        if filed.dt.tz is not None
        else filed.dt.tz_localize("UTC")
    )
    fallback = filed + pd.Timedelta(days=latency_days)
    return avail.where(avail.notna(), fallback)


def select_pit_rows(
    df: pd.DataFrame,
    as_of: pd.Timestamp | str,
    *,
    symbols: list[str] | None = None,
    latency_days: int = EDGAR_DAYS,
) -> pd.DataFrame:
    """PIT-select: visible rows as-of, then the latest restatement per period.

    1. Keep rows whose effective availability (:func:`_effective_availability`)
       is ``<= as_of``.
    2. Per ``(symbol, namespace, tag, period_end)``, keep the row with the
       MAXIMAL availability (the latest filing accepted on-or-before ``as_of``) —
       so a later restatement never retroactively rewrites an earlier value.
    """
    if df.empty:
        return df.copy()
    as_of_ts = pd.Timestamp(as_of)
    as_of_ts = (
        as_of_ts.tz_localize("UTC")
        if as_of_ts.tzinfo is None
        else as_of_ts.tz_convert("UTC")
    )
    work = df.copy()
    work["_eff"] = _effective_availability(work, latency_days)
    mask = work["_eff"].notna() & (work["_eff"] <= as_of_ts)
    if symbols:
        wanted_syms = {str(s).strip().upper() for s in symbols}
        mask &= work["symbol"].isin(wanted_syms)
    work = work.loc[mask]
    if work.empty:
        return work.drop(columns=["_eff"]).reset_index(drop=True)
    # period_start MUST be part of the key: a single 10-K legitimately emits a Q4
    # quarterly fact (start = Oct-1) AND an FY annual fact (start = Jan-1) with the
    # SAME period_end — keying without period_start collapses them and one
    # silently overwrites the other (corrupts the per-quarter series the SUE/PEAD
    # consumer reads). Tie-break is deterministic + PIT-meaningful: max
    # availability, then amendment-wins, then accession — so a parquet round-trip
    # or cross-symbol concat (no canonical row order), and the common date-only
    # filed_date+latency tie, never make the as-reported value order-dependent.
    work = (
        work.sort_values(["_eff", "is_amendment", "accession"])
        .groupby(
            ["symbol", "namespace", "tag", "period_end", "period_start"],
            dropna=False,
            as_index=False,
        )
        .tail(1)
    )
    return work.drop(columns=["_eff"]).sort_index().reset_index(drop=True)


def coalesce_field(
    df: pd.DataFrame,
    field: str,
    *,
    tags: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Ordered-coalesce a tall frame into one value per ``(symbol, period_end)``.

    For each ``(symbol, period_end, period_start)`` group, picks the first
    ``(namespace, tag)`` in priority order (default: ``FUNDAMENTAL_TAGS[field]``)
    that carries a non-null value, and records the winning ``source_tag`` /
    ``source_namespace``. Returns columns ``[symbol, period_end, period_start,
    <field>, source_tag, source_namespace, fp, fy]``.

    ``period_start`` is part of the group key (and emitted) so a Q4 quarterly
    fact and an FY annual fact sharing the same ``period_end`` stay DISTINCT rows
    rather than one silently overwriting the other — the consumer then selects
    the duration it needs (e.g. the 3-month QTD value for a quarterly SUE series).
    """
    candidates = tags if tags is not None else FUNDAMENTAL_TAGS.get(field, [])
    cols = [
        "symbol",
        "period_end",
        "period_start",
        field,
        "source_tag",
        "source_namespace",
        "fp",
        "fy",
    ]
    if df.empty or not candidates:
        return pd.DataFrame(columns=cols)
    priority = {nt: i for i, nt in enumerate(candidates)}

    nt_pairs = list(zip(df["namespace"], df["tag"]))
    keep_mask = [p in priority for p in nt_pairs]
    sub = df.loc[keep_mask].copy()
    sub = sub[sub["val"].notna()]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    sub["_prio"] = [priority[p] for p in zip(sub["namespace"], sub["tag"])]
    sub = sub.sort_values("_prio")

    out_rows: list[dict[str, Any]] = []
    for (sym, pend, pstart), g in sub.groupby(
        ["symbol", "period_end", "period_start"], dropna=False
    ):
        best = g.iloc[0]
        out_rows.append(
            {
                "symbol": sym,
                "period_end": pend,
                "period_start": pstart,
                field: best["val"],
                "source_tag": best["tag"],
                "source_namespace": best["namespace"],
                "fp": best["fp"],
                "fy": best["fy"],
            }
        )
    return pd.DataFrame(out_rows, columns=cols)


# ---------------------------------------------------------------------------
# Network layer (exercised live in Phase-4 verification, not in unit tests)
# ---------------------------------------------------------------------------


def fetch_company_facts(
    cik: str | int,
    user_agent: str | None = None,
    *,
    limiter: _RateLimiter | None = None,
) -> dict[str, Any]:
    """Fetch + parse the Company Facts JSON for one issuer CIK."""
    ua = resolve_user_agent(user_agent)
    limiter = limiter or _RateLimiter()
    raw = _http_get(_COMPANY_FACTS_FMT.format(cik=int(cik)), ua, limiter=limiter)
    payload: dict[str, Any] = json.loads(raw)
    return payload


def fetch_submissions(
    cik: str | int,
    user_agent: str | None = None,
    *,
    limiter: _RateLimiter | None = None,
) -> dict[str, Any]:
    """Fetch + parse the submissions feed (for ``accession -> acceptance``)."""
    ua = resolve_user_agent(user_agent)
    limiter = limiter or _RateLimiter()
    raw = _http_get(_SUBMISSIONS_FMT.format(cik=int(cik)), ua, limiter=limiter)
    payload: dict[str, Any] = json.loads(raw)
    return payload


def fetch_acceptance_map(
    cik: str | int,
    user_agent: str | None = None,
    *,
    limiter: _RateLimiter | None = None,
) -> dict[str, pd.Timestamp]:
    """Full ``accession -> acceptance instant`` map for an issuer (all pages).

    Fetches the main submissions feed AND every older paginated page
    (``filings.files``) so acceptance instants are resolved beyond the ~1000
    most-recent filings — lifting ``available_at`` coverage for deep history. A
    failure of the MAIN feed degrades to ``{}`` (the caller then falls back to
    ``filed_date + EDGAR_DAYS``); a single page failure is logged and skipped.
    """
    ua = resolve_user_agent(user_agent)
    limiter = limiter or _RateLimiter()
    try:
        main = fetch_submissions(cik, ua, limiter=limiter)
    except Exception as exc:
        logger.warning(
            "[WARN] xbrl_submissions_fetch_failed cik=%s: %s — available_at=NaT",
            cik,
            exc,
        )
        return {}
    acc_map = build_accession_acceptance_map(main)
    for name in submission_page_names(main):
        try:
            raw = _http_get(
                f"https://data.sec.gov/submissions/{name}", ua, limiter=limiter
            )
            acc_map.update(build_accession_acceptance_map(json.loads(raw)))
        except Exception as exc:
            logger.warning("[WARN] xbrl_submissions_page_failed name=%s: %s", name, exc)
    return acc_map


def ingest_fundamentals_xbrl(
    symbols: list[str],
    *,
    user_agent: str | None = None,
    out_path: Path | str | None = None,
    wanted: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Ingest XBRL Company Facts for ``symbols`` into a tall, PIT-correct frame.

    Per symbol: resolve CIK (``company_tickers.json``), fetch Company Facts +
    the submissions feed (for the acceptance instant), parse the wanted tags, and
    attach ``available_at``. Writes ``out_path`` (default
    ``output/fundamentals_xbrl.parquet``) and logs a coverage summary. Does NOT
    overwrite the legacy yfinance ``output/fundamentals.parquet``.
    """
    ua = resolve_user_agent(user_agent)
    limiter = _RateLimiter()
    cik_map = fetch_cik_map(ua, limiter=limiter)

    frames: list[pd.DataFrame] = []
    n_no_cik = 0
    n_facts_fail = 0
    started = time.monotonic()

    for sym in symbols:
        s = sym.strip().upper()
        cik = cik_map.get(s)
        if not cik:
            n_no_cik += 1
            logger.warning("[WARN] xbrl_no_cik symbol=%s", sym)
            continue
        try:
            facts = fetch_company_facts(cik, ua, limiter=limiter)
        except Exception as exc:
            n_facts_fail += 1
            logger.warning(
                "[WARN] xbrl_facts_fetch_failed symbol=%s cik=%s: %s", s, cik, exc
            )
            continue
        df_sym = company_facts_rows_to_dataframe(
            parse_company_facts(facts, symbol=s, cik=str(int(cik)), wanted=wanted)
        )
        # All pages resolved; main-feed failure degrades to {} -> available_at NaT
        # -> filed_date+EDGAR_DAYS fallback (never now()).
        accn_map = fetch_acceptance_map(cik, ua, limiter=limiter)
        frames.append(attach_available_at(df_sym, accn_map))

    df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else company_facts_rows_to_dataframe([])
    )
    # Ops-only ingest stamp (never used for PIT gating).
    df["timestamp"] = pd.Timestamp.now(tz="UTC")

    total = len(df)
    n_avail = int(df["available_at"].notna().sum()) if total else 0
    pct_avail = (100.0 * n_avail / total) if total else 0.0
    elapsed = max(time.monotonic() - started, 1e-9)
    logger.info(
        "[OK] xbrl_ingest symbols=%d no_cik=%d facts_fail=%d rows=%d "
        "available_at_resolved=%.1f%% elapsed=%.1fs",
        len(symbols),
        n_no_cik,
        n_facts_fail,
        total,
        pct_avail,
        elapsed,
    )

    out = Path(out_path) if out_path else Path("output") / "fundamentals_xbrl.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
        logger.info("[OK] xbrl_ingest wrote %d rows -> %s", total, out)
    except Exception as exc:
        logger.error("[ERROR] xbrl_ingest parquet write failed -> %s: %s", out, exc)
    return df
