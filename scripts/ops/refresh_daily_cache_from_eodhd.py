"""Refresh output/aggregates/daily.parquet from EODHD (paid feed, 100k calls/day).

Bridge for the recurring yfinance-429 outage: the pilot's 195-symbol batch fetch is
rate-limited by Yahoo (since ~2026-07-08), leaving the cache stale and the cycle
BLOCKING (fail-closed, correct). This script fills the SAME cache with EODHD daily
bars so the pilot's cache-fresh path is satisfied.

Semantics (verified 2026-06-01, memory session note): the cache `close` column is
TOTAL-RETURN adjusted (matches yf auto_adjust Adj Close ~0.0bps median). EODHD
delivers raw OHLC + adjusted_close. Mapping used here:
  factor      = adjusted_close / close_raw          (per row)
  open/high/low -> raw * factor                     (mimics yf auto_adjust OHLC)
  close       = adjusted_close
  adj_close   = adjusted_close
  volume      = raw volume

Idempotent: appends only rows with timestamp > per-symbol cache max (F-RX-2 idiom),
dedupes (symbol, timestamp) keep=last, atomic tmp+replace (F-RX-4), status JSON
sidecar (F-RX-5) shared with the panel-refresh consumer path.

Usage:
    python scripts/ops/refresh_daily_cache_from_eodhd.py --dry-run
    python scripts/ops/refresh_daily_cache_from_eodhd.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

logger = logging.getLogger(__name__)

CACHE_PATH = ROOT / "output" / "aggregates" / "daily.parquet"
STATUS_PATH = ROOT / "output" / "ops" / "refresh_cache_status.json"


def _write_status(
    *,
    rc: int,
    cache_latest,
    new_latest,
    rows_appended: int,
    n_fail: int = 0,
    error: str | None = None,
) -> None:
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "source": "eodhd",
            "rc": int(rc),
            "ok": rc >= 0,
            "cache_latest": str(cache_latest) if cache_latest is not None else None,
            "panel_latest": str(new_latest) if new_latest is not None else None,
            "rows_appended": int(rows_appended),
            "n_fetch_fail": int(n_fail),
            "error": error,
        }
        tmp = STATUS_PATH.with_name(STATUS_PATH.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(STATUS_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[refresh-eodhd] failed to write status JSON: %s", exc)


def _fetch_symbol(tok: str, sym: str, frm: str) -> list[dict]:
    url = f"https://eodhd.com/api/eod/{sym}.US?api_token={tok}&fmt=json&from={frm}"
    req = urllib.request.Request(url, headers={"User-Agent": "research ops"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())


def refresh(*, dry_run: bool) -> int:
    import pandas as pd

    # Protokoll ZUERST: "kein Token" und "kein Cache" sind Betriebszustaende,
    # die genauso belegt gehoeren wie ein Anbieterausfall - und sie waren die
    # letzten beiden der sieben return-Pfade ohne Evidenz.
    plog = None
    try:
        from src.assembled_core.data.pull_log import PullLog

        plog = PullLog(source="eodhd_eod")
    except Exception as _plexc:  # pragma: no cover - must not block the refresh
        logger.warning("[refresh-eodhd] pull_log unavailable: %s", _plexc)

    def _abort(reason: str) -> None:
        if plog is not None:
            plog.record("__run__", status="skipped", n_rows=0, skipped_reason=reason)
            plog.write()

    tok = os.environ.get("EODHD_API_TOKEN")
    if not tok:
        logger.error("[refresh-eodhd] EODHD_API_TOKEN fehlt in .env")
        _abort("no EODHD_API_TOKEN - no request was made")
        _write_status(
            rc=-1, cache_latest=None, new_latest=None, rows_appended=0, error="no token"
        )
        return -1
    if not CACHE_PATH.exists():
        logger.error("[refresh-eodhd] cache not found: %s", CACHE_PATH)
        _abort(f"cache missing at {CACHE_PATH} - no request was made")
        _write_status(
            rc=-1,
            cache_latest=None,
            new_latest=None,
            rows_appended=0,
            error="cache missing",
        )
        return -1

    cache = pd.read_parquet(CACHE_PATH)
    cache["timestamp"] = pd.to_datetime(cache["timestamp"], utc=True)
    cache_latest = cache["timestamp"].max()
    per_sym_max = cache.groupby("symbol")["timestamp"].max()
    today = pd.Timestamp.now(tz="UTC").normalize()
    stale = per_sym_max[per_sym_max < today - pd.Timedelta(days=1)]
    logger.info(
        "[refresh-eodhd] cache latest=%s; %d/%d symbols stale",
        cache_latest,
        len(stale),
        len(per_sym_max),
    )
    # E-112 request protocol for the primary price ingest. The existing status
    # JSON is a RUN aggregate (rc, rows_appended); it cannot say which symbols
    # were asked for or what each one returned. That per-symbol record is what
    # a later coverage question actually needs.
    #
    # Initialised BEFORE the stale.empty return so that even a run which asks
    # for nothing leaves a protocol saying "0 requested". "No protocol file" and
    # "we asked and got nothing" must not look the same afterwards — that
    # ambiguity is the whole of E-112.
    if stale.empty:
        if plog is not None:
            plog.write()
        _write_status(
            rc=0, cache_latest=cache_latest, new_latest=cache_latest, rows_appended=0
        )
        return 0

    frames = []
    n_fail = 0

    for sym, cmax in stale.items():
        frm = (cmax - pd.Timedelta(days=5)).date().isoformat()
        try:
            rows = _fetch_symbol(tok, sym, frm)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            logger.warning("[refresh-eodhd] %s fetch failed: %s", sym, str(exc)[:80])
            if plog is not None:
                plog.record(
                    sym,
                    window=(frm, "today"),
                    http_status=getattr(exc, "code", None),
                    error=f"{type(exc).__name__}: {str(exc)[:200]}",
                )
            continue
        if not rows:
            # E-112, the canonical case: a successful request that returned
            # nothing used to `continue` without a trace. "No new rows for LEH"
            # and "LEH was never requested" then look identical afterwards, and
            # a coverage claim built on that is a guess. This is the ingest the
            # EODHD outage in KNOWN_ISSUES §0.0 is about, so it is exactly the
            # one that must be able to answer "did we ask, and what came back".
            if plog is not None:
                plog.record(sym, window=(frm, "today"), n_rows=0)
            continue
        df = pd.DataFrame(rows)
        need = {"date", "open", "high", "low", "close", "adjusted_close", "volume"}
        if not need.issubset(df.columns):
            n_fail += 1
            if plog is not None:
                plog.record(
                    sym,
                    window=(frm, "today"),
                    n_rows=len(rows),
                    error=f"schema mismatch, missing: {sorted(need - set(df.columns))}",
                )
            continue
        if plog is not None:
            plog.record(sym, window=(frm, "today"), n_rows=len(rows))
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
        # BLOCKER-Fix (Review Stage-2, E-041-Klasse): PIT-Cutoff — EODHD /eod/ liefert
        # während offener US-Session einen FORMING same-day Bar (empirisch verifiziert).
        # Der 21:10-CEST-Task läuft VOR US-Close → ohne Cutoff würde ein partieller
        # Tagesbar zum Signal-/Ausführungspreis (multifactor tail(1)). Idiom gespiegelt
        # von refresh_sector_etf_cache.py:204 (timestamp < today_utc).
        today_utc = pd.Timestamp.now(tz="UTC").normalize()
        n_today = int((df["timestamp"] >= today_utc).sum())
        if n_today:
            logger.info(
                "[refresh-eodhd] %s: %d same-day/forming Bar(s) gedroppt (PIT-Cutoff)",
                sym,
                n_today,
            )
        df = df[(df["timestamp"] > cmax) & (df["timestamp"] < today_utc)]
        # MAJOR-Fix (Review TR): defekte Zeilen DROPPEN statt factor=1.0-maskieren —
        # close_raw<=0 oder adjusted_close=NaN würde sonst einen intern inkonsistenten
        # bzw. NaN-Bar in den LIVE-Cache schreiben. Gedroppte Symbole bleiben stale und
        # werden vom Pilot per F-RX-1 laut gedroppt (fail-loud statt still-falsch).
        df = df[
            (pd.to_numeric(df["close"], errors="coerce") > 0)
            & pd.to_numeric(df["adjusted_close"], errors="coerce").notna()
        ]
        if df.empty:
            continue
        raw_close = df["close"].astype(float)
        factor = df["adjusted_close"].astype(float) / raw_close
        out = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "symbol": sym,
                "open": df["open"].astype(float) * factor,
                "high": df["high"].astype(float) * factor,
                "low": df["low"].astype(float) * factor,
                "close": df["adjusted_close"].astype(float),
                "adj_close": df["adjusted_close"].astype(float),
                "volume": df["volume"].astype(float),
            }
        )
        frames.append(out)
        time.sleep(0.03)

    # Written before every `return` below, so the protocol survives each of
    # them — including "all fetches failed", the run where it matters most.
    # NOTE the precise claim: this covers the RETURN paths. An exception
    # escaping outside the per-symbol try (say a malformed date field) still
    # ends refresh() without a protocol file. Wrapping the whole loop in
    # try/finally would close that too and is the better shape; it is not
    # done here because this is the live cache writer and the change would
    # need its own risk review. Tracked in KNOWN_ISSUES §0.06 (c).
    if plog is not None:
        plog.write()

    if not frames:
        # MAJOR-Fix (Review TR): 100%-Fetch-Failure darf NICHT als ok:true/rc=0
        # durchgehen — sonst ist der Alert-Konsument blind für einen toten Feed.
        if n_fail > 0 and n_fail >= len(stale):
            logger.error(
                "[refresh-eodhd] ALLE %d stale-Symbole failten — Feed tot?", n_fail
            )
            _write_status(
                rc=-1,
                cache_latest=cache_latest,
                new_latest=cache_latest,
                rows_appended=0,
                n_fail=n_fail,
                error="all fetches failed",
            )
            return -1
        logger.info("[refresh-eodhd] keine neuen Zeilen (fails=%d)", n_fail)
        _write_status(
            rc=0,
            cache_latest=cache_latest,
            new_latest=cache_latest,
            rows_appended=0,
            n_fail=n_fail,
        )
        return 0

    new_rows = pd.concat(frames, ignore_index=True)[cache.columns.tolist()]
    logger.info(
        "[refresh-eodhd] %d rows für %d Symbole, ts %s..%s (fetch-fails=%d)",
        len(new_rows),
        new_rows["symbol"].nunique(),
        new_rows["timestamp"].min(),
        new_rows["timestamp"].max(),
        n_fail,
    )
    if dry_run:
        logger.info("[refresh-eodhd] --dry-run, nichts geschrieben")
        _write_status(
            rc=int(len(new_rows)),
            cache_latest=cache_latest,
            new_latest=new_rows["timestamp"].max(),
            rows_appended=int(len(new_rows)),
            n_fail=n_fail,
        )
        return len(new_rows)

    merged = pd.concat([cache, new_rows], ignore_index=True)
    merged = (
        merged.sort_values(["symbol", "timestamp"])
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .reset_index(drop=True)
    )
    tmp = CACHE_PATH.with_name(CACHE_PATH.name + ".tmp")
    try:
        merged.to_parquet(tmp, index=False)
        tmp.replace(CACHE_PATH)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    logger.info(
        "[refresh-eodhd] wrote %s — new latest=%s",
        CACHE_PATH,
        merged["timestamp"].max(),
    )
    _write_status(
        rc=int(len(new_rows)),
        cache_latest=cache_latest,
        new_latest=merged["timestamp"].max(),
        rows_appended=int(len(new_rows)),
        n_fail=n_fail,
    )
    return len(new_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    rc = refresh(dry_run=args.dry_run)
    return 0 if rc >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
