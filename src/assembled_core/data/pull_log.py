"""Per-request ingest protocol (anti-pattern E-112).

WHY THIS EXISTS
---------------
E-112 ("Ein ``ls`` beantwortet keine Frage ueber einen Anbieter") records a real
incident: coverage of a data vendor was inferred from which files happened to
exist on disk. The conclusion — an "enrichment factor" of 3.06 — was wrong,
because 8 of 8 symbols in the supposed failure group were never requested at
all. The vendor had never been asked.

The rule that came out of it (docs/CLAUDE_CODING_ERRORS.md, E-112):

    Every ingest writes a request protocol - symbol, window, HTTP status,
    bar count - ALSO AND ESPECIALLY for empty results.

That last clause is the whole point. A pull that returns nothing must leave a
trace, otherwise "no file" is indistinguishable from "never asked", and any
later coverage statement is a guess wearing a number.

This module is the shared primitive for that protocol. It is deliberately tiny
and never raises: an ingest run must not fail because its bookkeeping failed.

VOCABULARY
----------
``ok`` / ``empty`` / ``error`` are deckungsgleich with
:mod:`assembled_core.data.feed_status`, so the per-request protocol and the
per-frame stamp can be joined on those three. ``skipped`` is ADDITIONAL and has
no counterpart there — feed_status deliberately keeps three mutually exclusive
states, so a join on ``skipped`` finds nothing.

  ``ok``     - request succeeded and returned at least one row
  ``empty``  - request succeeded but returned zero rows (a legitimate empty
               window OR a symbol the vendor does not cover; the protocol
               records the fact, it does not interpret it)
  ``error``  - request failed (HTTP error, timeout, parse failure)
  ``skipped`` - no request was made at all (unmapped key, filtered out). Kept
               out of ``empty_keys`` so a key that was never asked about can
               never be mistaken for one the vendor does not cover.

Note the deliberate asymmetry with plain logging: ``empty`` is a first-class
recorded outcome here, not an absence.

USAGE
-----
    from src.assembled_core.data.pull_log import PullLog

    plog = PullLog(source="eodhd", run_id="20260815T1830Z")
    for sym in symbols:
        try:
            bars = fetch(sym, start, end)
            plog.record(sym, window=(start, end), http_status=200, n_rows=len(bars))
        except HTTPError as exc:
            plog.record(sym, window=(start, end), http_status=exc.code, error=str(exc))
    plog.write()   # -> output/ops/pull_log_eodhd_20260815T1830Z.json

A control group matters (anti-pattern E-113): a protocol that only ever
requested delisted tickers cannot distinguish "vendor lacks delisted coverage"
from "credentials are dead". Record the control symbols too.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: v2 (2026-08-15) added: log_lines, duplicate_keys, skipped/skipped_keys,
#: ok_rows_unknown/ok_rows_unknown_keys. A v1 artefact lacks them; a consumer
#: must branch on this value rather than assume the keys exist.
SCHEMA_VERSION = "pull_log.v2"

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_ERROR = "error"
#: The request was deliberately NOT made (no mapping, filtered out, ...).
#: Its own state on purpose: forcing such a key into `empty` would put something
#: that was never asked about into `empty_keys`, and that list is documented and
#: tested as "only requests we actually made" — the exact confusion E-112 is
#: about. Forcing it into `error` is equally wrong: a consumer reads that as a
#: vendor or transport failure.
STATUS_SKIPPED = "skipped"

_VALID_STATUSES = frozenset({STATUS_OK, STATUS_EMPTY, STATUS_ERROR, STATUS_SKIPPED})

#: Repo root: src/assembled_core/data/pull_log.py -> up 4.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default output directory, following the ops-artifact convention used by
#: scripts/ops/refresh_daily_cache_from_eodhd.py and friends.
#:
#: Anchored at the repo root, NOT CWD-relative. A bare ``Path("output")/"ops"``
#: silently scattered protocols into whatever directory a puller happened to be
#: started from — verified: a run from /tmp wrote /tmp/output/ops/. A protocol
#: that lands somewhere nobody looks is the same as no protocol, which is the
#: whole failure this module exists to prevent (and E-146 in its own right).
DEFAULT_LOG_DIR = _REPO_ROOT / "output" / "ops"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _coerce_window(window: Any) -> tuple[str | None, str | None]:
    """Best-effort normalisation of a (start, end) pair to ISO strings."""
    if window is None:
        return (None, None)
    try:
        start, end = window
    except (TypeError, ValueError):
        return (str(window), None)
    return (
        None if start is None else str(start),
        None if end is None else str(end),
    )


def _default_run_id() -> str:
    """UTC timestamp used when the caller does not supply a run id.

    Without one, ``default_path`` produces a single slot that every run
    overwrites — and the question E-112 was written for is retrospective
    ("were those eight symbols ever requested?"). A file that only ever holds
    the newest run cannot answer it.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class PullLog:
    """In-memory collector for per-request ingest outcomes.

    Every method is non-raising by contract. Bookkeeping must never be the
    reason an ingest run dies.
    """

    source: str
    run_id: str | None = field(default_factory=_default_run_id)
    entries: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now_iso)

    # ------------------------------------------------------------------ record
    def record(
        self,
        key: str,
        *,
        window: Any = None,
        http_status: int | None = None,
        n_rows: int | None = 0,
        error: str | None = None,
        status: str | None = None,
        **extra: Any,
    ) -> None:
        """Record the outcome of one request.

        ``key`` is whatever identifies the request to the vendor - usually a
        symbol, but it can be a CIK, an accession number, or a URL.

        ``status`` is normally derived: an ``error`` string means ``error``,
        zero rows means ``empty``, anything else means ``ok``. Pass it
        explicitly only to override that derivation.

        ``n_rows=None`` means "row count UNKNOWN at this layer" and is NOT the
        same as zero. A transport that cannot count rows (a text body, a dict
        envelope) records None; the caller that parses the payload records the
        real count. Recording None yields ``ok`` with a null count, so an
        unknown never masquerades as either a success with data or an empty
        result.

        Record ONE entry per key per run. Logging the same key from both the
        transport and the parse layer makes that key appear twice with two
        different statuses, and :meth:`summary` then reports counts that no
        longer answer "how many symbols did we ask about".
        """
        # None means "row count unknown at this layer" — NOT "zero rows".
        # A transport that cannot count rows (a text body, a dict envelope)
        # must not be able to turn a successful request into an "empty" one,
        # and it must not be able to claim rows it never counted either.
        rows: int | None
        if n_rows is None:
            rows = None
        else:
            try:
                # NaN und Inf sind "unbekannt", nicht "null". Beide wuerden ueber
                # den except-Zweig zu rows=0 und damit zu status "empty" - und
                # empty_keys ist genau die Liste, aus der spaeter Coverage-
                # Aussagen gebaut werden. Ein unbekannter Wert darf dort nicht
                # als "Anbieter liefert nichts" erscheinen (E-112).
                if isinstance(n_rows, float) and not math.isfinite(n_rows):
                    rows = None
                else:
                    rows = int(n_rows)
            except (TypeError, ValueError, OverflowError):
                # OverflowError inherits from ArithmeticError, NOT ValueError:
                # float("inf") arriving via **extra would otherwise escape the
                # never-raise contract that io_utils explicitly relies on.
                rows = 0

        if status is None:
            if error is not None:
                status = STATUS_ERROR
            elif rows is None:
                # Request succeeded, row count unknown. Recorded as ok, but the
                # null n_rows keeps it distinguishable from a counted success,
                # so a coverage claim built on this cannot silently assume data.
                status = STATUS_OK
            elif rows <= 0:
                status = STATUS_EMPTY
            else:
                status = STATUS_OK
        elif status not in _VALID_STATUSES:
            logger.debug(
                "[pull_log] unknown status %r for key %r - recording as %r",
                status,
                key,
                STATUS_ERROR,
            )
            status = STATUS_ERROR

        window_start, window_end = _coerce_window(window)

        entry: dict[str, Any] = {
            "key": str(key),
            "status": status,
            "window_start": window_start,
            "window_end": window_end,
            "http_status": http_status,
            "n_rows": rows,
            "error": None if error is None else str(error)[:500],
            "recorded_at": _utc_now_iso(),
        }
        if extra:
            entry.update(extra)

        self.entries.append(entry)

        if status == STATUS_ERROR:
            logger.warning(
                "[pull_log] %s: request FAILED for %s (http=%s, reason=%s)",
                self.source,
                key,
                http_status,
                error,
            )

    # --------------------------------------------------------------- aggregate
    def summary(self) -> dict[str, Any]:
        """Aggregate counts over all recorded requests.

        Aggregated per KEY, not per log line. Counting lines would make
        ``requested`` a row count instead of a request count, and any coverage
        ratio built on it would be invented — the same failure shape as the
        retracted 3.06 "enrichment factor" that produced E-112, one level up.
        If a key was somehow recorded more than once, the LAST entry wins and
        ``duplicate_keys`` names it, so the double-logging is visible rather
        than silently folded into the totals.
        """
        final: dict[str, dict[str, Any]] = {}
        duplicates: set[str] = set()
        for entry in self.entries:
            key = entry["key"]
            if key in final:
                duplicates.add(key)
            final[key] = entry

        requested = len(final)
        n_ok = sum(1 for e in final.values() if e["status"] == STATUS_OK)
        n_empty = sum(1 for e in final.values() if e["status"] == STATUS_EMPTY)
        n_error = sum(1 for e in final.values() if e["status"] == STATUS_ERROR)
        n_skipped = sum(1 for e in final.values() if e["status"] == STATUS_SKIPPED)
        # "ok, but we never counted the rows" is a third kind of success and
        # must stay separable. Coverage ratios are built from requested/ok; if
        # unknown-count successes fold indistinguishably into `ok`, the ratio
        # asserts data that was never observed — the shape of the retracted
        # 3.06 enrichment factor (E-112, E-148).
        n_ok_rows_unknown = sum(
            1
            for e in final.values()
            if e["status"] == STATUS_OK and e.get("n_rows") is None
        )
        returned_rows = sum(int(e.get("n_rows") or 0) for e in final.values())
        return {
            "requested": requested,
            "log_lines": len(self.entries),
            "duplicate_keys": sorted(duplicates),
            "ok": n_ok,
            "empty": n_empty,
            "error": n_error,
            "skipped": n_skipped,
            "ok_rows_unknown": n_ok_rows_unknown,
            "returned_rows": returned_rows,
            # The keys that actually answer a coverage question. Anything that
            # was never requested does NOT appear here - by construction this
            # protocol can only speak about requests it made (E-112).
            # skipped keys are deliberately absent here: this list answers
            # "which requests came back empty", not "which keys have no data".
            "empty_keys": sorted(
                k for k, e in final.items() if e["status"] == STATUS_EMPTY
            ),
            "skipped_keys": sorted(
                k for k, e in final.items() if e["status"] == STATUS_SKIPPED
            ),
            "ok_rows_unknown_keys": sorted(
                k
                for k, e in final.items()
                if e["status"] == STATUS_OK and e.get("n_rows") is None
            ),
            "error_keys": sorted(
                k for k, e in final.items() if e["status"] == STATUS_ERROR
            ),
        }

    # ------------------------------------------------------------------- write
    def default_path(self, log_dir: Path | None = None) -> Path:
        directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
        suffix = f"_{self.run_id}" if self.run_id else ""
        return directory / f"pull_log_{self.source}{suffix}.json"

    def write(
        self, path: Path | None = None, *, log_dir: Path | None = None
    ) -> Path | None:
        """Write the protocol atomically. Returns the path, or None on failure.

        Never raises: a failed write is logged and swallowed, because losing the
        protocol must not lose the data the run just fetched.
        """
        # default_path() und summary() gehoeren MIT in den Schutz: der
        # Docstring verspricht "never raises", und summary() kann ueber ein
        # per **extra hereingereichtes, nicht-numerisches n_rows werfen.
        # Narrow on purpose: these are the realistic failure modes (a bad path
        # type, a non-numeric n_rows arriving via **extra, an unusable log_dir).
        # A blanket `except Exception` here would also swallow genuine defects
        # in summary() — and this repo runs a broad-except ratchet precisely to
        # stop that habit from spreading.
        try:
            target = Path(path) if path is not None else self.default_path(log_dir)
            summary_data: dict[str, Any] = self.summary()
        except (
            TypeError,
            ValueError,
            OverflowError,
            AttributeError,
            KeyError,
            OSError,
        ) as exc:
            logger.warning("[pull_log] could not assemble protocol: %s", exc)
            return None
        # Bind the summary to its own name before building the payload: reading
        # it back out of the dict widens it to the union of all value types and
        # is no longer indexable for a type checker.
        summary = summary_data
        payload: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "source": self.source,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": _utc_now_iso(),
            "summary": summary,
            "entries": self.entries,
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            tmp.replace(target)
        except Exception as exc:  # pragma: no cover - must never break a run
            logger.warning("[pull_log] could not write protocol to %s: %s", target, exc)
            return None

        if summary["duplicate_keys"]:
            # E-148 Punkt 3 verlangt, Mehrfacheintraege SICHTBAR zu machen.
            # Nur im JSON zu stehen heisst: sichtbar fuer den, der die Datei
            # oeffnet - also fuer niemanden im Regelbetrieb.
            logger.warning(
                "[pull_log] %s: %d key(s) logged more than once (%s). The last "
                "entry wins per key, so a later 'ok' can mask an earlier "
                "'empty'. Exactly one layer should record each key.",
                self.source,
                len(summary["duplicate_keys"]),
                ", ".join(summary["duplicate_keys"][:5]),
            )

        logger.info(
            "[pull_log] %s: %d requested (%d ok, %d empty, %d error, %d skipped), "
            "%d rows -> %s",
            self.source,
            summary["requested"],
            summary["ok"],
            summary["empty"],
            summary["error"],
            summary["skipped"],
            summary["returned_rows"],
            target,
        )
        return target


__all__ = [
    "SCHEMA_VERSION",
    "STATUS_OK",
    "STATUS_EMPTY",
    "STATUS_ERROR",
    "STATUS_SKIPPED",
    "DEFAULT_LOG_DIR",
    "PullLog",
]
