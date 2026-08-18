"""Refresh sector-ETF + SPY closes in output/aggregates/daily.parquet via yfinance.

The live ``multifactor_v2`` sector_rotation_bias factor reads its prices from the
sector store, which loads ``output/aggregates/daily.parquet`` (via
``load_eod_prices(SECTOR_ETFS + ["SPY"])`` → ``get_default_price_path("1d")``).
That factor applies a live-staleness guard (``SECTOR_STORE_STALE_DAYS = 7``):
when the newest sector bar is older than 7 days it neutralises the factor to 0.0.

None of the existing daily steps refresh the sector ETFs:
- ``refresh_daily_cache_from_panel.py`` copies from the master universe panel,
  which does not reliably carry the sector ETFs.
- ``prewarm_price_cache.py`` is driven by ``configs/watchlist.txt``, and the
  sector ETFs are not on the watchlist.

So sector-ETF rows in daily.parquet silently go stale and the factor is
neutralised. This script closes that gap: it fetches fresh OHLCV for the 8
sector ETFs + SPY and idempotently merges the newer rows into daily.parquet,
so the factor computes on fresh data instead of degrading to neutral.

PIT safety: today's bar is EXCLUDED. The pilot runs at 21:30 CET (15:30 ET),
before the US close — a same-day yfinance bar would be partial/forming. The
factor's PIT slice (``timestamp <= as_of``) would otherwise treat that partial
bar as a settled close. We only append bars strictly before today (UTC date).

Schemas:
- daily.parquet:        [timestamp, symbol, open, high, low, close, adj_close, volume]
- fetch_prices_yfinance: [timestamp, symbol, open, high, low, close, volume]
  (no separate adj_close column — appended rows MIRROR close, matching
  refresh_daily_cache_from_panel.py. Until 2026-08-15 they carried a NaN
  "sentinel" instead; that rested on the assumption that close was unadjusted,
  which is false here — this fetcher calls yfinance with auto_adjust=True. See
  the comment at the adj_close branch below for the measurements.)

Idempotent: appends only rows with timestamp strictly greater than each
symbol's current cache max; drops exact (symbol, timestamp) duplicates as a
final safety net. A yfinance outage degrades gracefully (0 rows appended, WARN,
cache unchanged → factor falls back to its existing neutral behaviour).

Usage:
    python scripts/ops/refresh_sector_etf_cache.py
    python scripts/ops/refresh_sector_etf_cache.py --dry-run
    python scripts/ops/refresh_sector_etf_cache.py --lookback-days 120
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.sources.yfinance_source import (  # noqa: E402
    YFinanceRateLimitError,
    fetch_prices_yfinance,
)
from src.assembled_core.signals.sector_rotation import SECTOR_ETFS  # noqa: E402

logger = logging.getLogger(__name__)

CACHE_PATH = ROOT / "output" / "aggregates" / "daily.parquet"
STATUS_PATH = ROOT / "output" / "ops" / "refresh_sector_etf_status.json"

# Target symbols: the 8 sector ETFs the factor ranks, plus SPY as the
# market benchmark for relative-strength. Kept in sync with SECTOR_ETFS.
TARGET_SYMBOLS = list(SECTOR_ETFS) + ["SPY"]

# Default fetch window. 120 calendar days (~83 trading days) comfortably
# bridges the largest observed sector-ETF staleness gap (~31d) while staying
# cheap for 9 symbols. The per-symbol "strictly newer" merge discards any
# overlap, so an over-wide window is harmless.
DEFAULT_LOOKBACK_DAYS = 120

# Informational only: mirrors multifactor_v2.SECTOR_STORE_STALE_DAYS so the
# post-merge log can report whether the factor's live guard will pass. Not
# imported (multifactor_v2 pulls a heavy module chain); kept as a local
# reference for a log line, with no functional effect.
_STALE_DAYS_REF = 7


def _write_status(
    *,
    rc: int,
    cache_latest: object | None,
    fetch_latest: object | None,
    rows_appended: int,
    error: str | None = None,
    dropped_symbols: list[str] | None = None,
    status_path: Path | None = None,
) -> None:
    """Write a status JSON for ops monitoring.

    Mirrors refresh_daily_cache_from_panel._write_status: the .bat only logs a
    WARN to a per-day file (no alert surface), so this sidecar JSON gives
    downstream consumers a single load-then-check path. Best-effort: an
    exception here must not abort the refresh outcome.
    """
    if status_path is None:
        status_path = STATUS_PATH
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "rc": int(rc),
            "ok": rc >= 0,
            "cache_latest": str(cache_latest) if cache_latest is not None else None,
            "fetch_latest": str(fetch_latest) if fetch_latest is not None else None,
            "rows_appended": int(rows_appended),
            "symbols": TARGET_SYMBOLS,
            "dropped_symbols": dropped_symbols or [],
            "error": error,
        }
        tmp = status_path.with_name(status_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(status_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[refresh-sector] failed to write status JSON: %s", exc)


def refresh(
    cache_path: Path,
    *,
    dry_run: bool,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    today: _dt.date | None = None,
) -> int:
    """Append fresh sector-ETF + SPY rows into the daily cache.

    Returns rows appended (>= 0), or -1 on a hard error (missing cache).
    A yfinance outage is NOT a hard error: returns 0 with the cache unchanged.
    """
    import pandas as pd

    if not cache_path.exists():
        logger.error("[refresh-sector] cache not found: %s", cache_path)
        _write_status(
            rc=-1,
            cache_latest=None,
            fetch_latest=None,
            rows_appended=0,
            error=f"cache not found: {cache_path}",
        )
        return -1

    # PIT cutoff: only bars strictly before today (UTC) may enter, so a
    # same-day partial bar can never be treated as a settled close.
    today_utc = (
        today if today is not None else _dt.datetime.now(_dt.timezone.utc).date()
    )
    start_date = (today_utc - _dt.timedelta(days=lookback_days)).isoformat()
    end_date = (
        today_utc.isoformat()
    )  # advisory only — the PIT cutoff below is the real exclusion guard

    logger.info(
        "[refresh-sector] fetching %d symbols %s..%s (today=%s excluded, PIT)",
        len(TARGET_SYMBOLS),
        start_date,
        end_date,
        today_utc,
    )

    try:
        fetched = fetch_prices_yfinance(TARGET_SYMBOLS, start_date, end_date)
    except YFinanceRateLimitError as exc:
        # 2026-08-18 (Telegram-Alert "Sector-ETF-Refresh degradiert"): ein
        # Rate-Limit ist NUR dann ein Problem, wenn die Sektor-Daten dadurch
        # veralten. Seit alle 9 TARGET_SYMBOLS ueber prewarm_price_cache
        # laufen (6 via watchlist, 3 via CACHE_ONLY_SYMBOLS) — inklusive
        # Alpaca-Fallback — sind sie in aller Regel bereits frisch, wenn
        # dieser Refresher an yfinance scheitert. Ein blindes error-Feld
        # erzeugte daraus einen TAEGLICHEN Degradiert-Alarm ohne realen
        # Mangel: Alert-Fatigue (E-189). Deshalb: Frische MESSEN und nur
        # melden, wenn sie wirklich fehlt.
        _cache_ts = pd.to_datetime(
            pd.read_parquet(cache_path, columns=["timestamp"])["timestamp"], utc=True
        )
        cache_latest = _cache_ts.max()
        _sector_age_days = (pd.Timestamp(today_utc, tz="UTC") - cache_latest).days
        _fresh = _sector_age_days <= _STALE_DAYS_REF
        logger.warning(
            "[refresh-sector] yfinance rate-limited (429): %s — cache unchanged. "
            "Sector data age=%dd (guard=%dd) -> %s",
            exc,
            _sector_age_days,
            _STALE_DAYS_REF,
            "still FRESH (prewarm/Alpaca keeps it current), no degradation"
            if _fresh
            else "STALE, sector_rotation_bias will be neutralised",
        )
        _write_status(
            rc=0,
            cache_latest=cache_latest,
            fetch_latest=None,
            rows_appended=0,
            # Nur ein echter Frische-Mangel ist ein Fehler. Sonst: kein
            # error-Feld -> der Watchdog-Konsument schweigt zu Recht.
            error=(
                None
                if _fresh
                else f"yfinance_rate_limited_and_stale (age={_sector_age_days}d)"
            ),
        )
        return 0

    cache = pd.read_parquet(cache_path)
    cache["timestamp"] = pd.to_datetime(cache["timestamp"], utc=True)
    cache_latest = cache["timestamp"].max()

    if fetched is None or fetched.empty:
        logger.warning(
            "[refresh-sector] yfinance returned no rows — cache unchanged, "
            "sector_rotation_bias keeps its existing freshness state."
        )
        _write_status(
            rc=0,
            cache_latest=cache_latest,
            fetch_latest=None,
            rows_appended=0,
            error="yfinance_empty",
        )
        return 0

    fetched = fetched.copy()
    fetched["timestamp"] = pd.to_datetime(fetched["timestamp"], utc=True)

    # PIT guard: drop today's (and any future) bar so a forming partial bar
    # never enters the store.
    cutoff = pd.Timestamp(today_utc, tz="UTC")
    before = len(fetched)
    fetched = fetched[fetched["timestamp"] < cutoff]
    if len(fetched) < before:
        logger.info(
            "[refresh-sector] PIT: dropped %d bar(s) dated >= %s (today)",
            before - len(fetched),
            today_utc,
        )
    if fetched.empty:
        logger.info("[refresh-sector] nothing left after PIT cutoff — no append")
        _write_status(
            rc=0,
            cache_latest=cache_latest,
            fetch_latest=None,
            rows_appended=0,
        )
        return 0

    fetch_latest = fetched["timestamp"].max()
    logger.info(
        "[refresh-sector] cache latest=%s, fetch latest=%s (post-PIT)",
        cache_latest,
        fetch_latest,
    )

    # Per-symbol freshness delta — NUR fuers Reporting (rows_appended):
    # der eigentliche Merge bekommt seit 2026-08-17 den VOLLEN fetched-Frame
    # inkl. Overlap, denn der Overlap traegt die Semantik-Pruefung der
    # guarded_merge (Overlap-Re-Adjustierung bei Corporate Actions zwischen
    # den Adjustierungs-Ankern + fail-closed Naht-Guard, E-165/E-166 —
    # dieser Schreiber war der fuenfte OHNE Guard, F-auditor-8).
    very_old = pd.Timestamp("1900-01-01", tz="UTC")
    cache_per_sym = (
        cache[cache["symbol"].isin(TARGET_SYMBOLS)]
        .groupby("symbol")["timestamp"]
        .max()
        .rename("_cache_max")
        .reset_index()
    )
    fetched_cmax = fetched.merge(cache_per_sym, on="symbol", how="left")
    fetched_cmax["_cache_max"] = fetched_cmax["_cache_max"].fillna(very_old)
    new_rows = fetched_cmax[
        fetched_cmax["timestamp"] > fetched_cmax["_cache_max"]
    ].drop(columns=["_cache_max"])

    if new_rows.empty:
        logger.info(
            "[refresh-sector] no fetched rows strictly newer than per-symbol cache max"
        )
        _write_status(
            rc=0,
            cache_latest=cache_latest,
            fetch_latest=fetch_latest,
            rows_appended=0,
        )
        return 0

    n_syms = new_rows["symbol"].nunique()
    logger.info(
        "[refresh-sector] %d rows to append for %d symbols, ts %s..%s",
        len(new_rows),
        n_syms,
        new_rows["timestamp"].min(),
        new_rows["timestamp"].max(),
    )

    if dry_run:
        logger.info("[refresh-sector] --dry-run set, not writing")
        _write_status(
            rc=int(len(new_rows)),
            cache_latest=cache_latest,
            fetch_latest=fetch_latest,
            rows_appended=int(len(new_rows)),
        )
        return len(new_rows)

    # adj_close-Spiegel uebernimmt guarded_merge unbedingt am Schreibpunkt
    # (E-170); der historische F-RX-3-Kontext bleibt in der Git-Historie.
    from src.assembled_core.data.price_cache_merge import (
        SeamGuardError,
        guarded_merge,
    )

    # Schema angleichen (Feed-Status-Spalten von yfinance verwerfen; adj_close
    # darf fehlen — der Helper stellt die Invariante her).
    keep_cols = [c for c in cache.columns.tolist() if c in fetched.columns]
    fetched_clean = fetched[keep_cols]

    try:
        result = guarded_merge(cache, fetched_clean)
    except SeamGuardError as exc:
        # F-TR-1 (Stage 1, 2026-08-17): der Abbruch muss den Operator
        # ERREICHEN — rc=-2 (negativ -> ok=false im Status, Exit 1 im main,
        # WARN-Zweig im .bat feuert) + error-Feld. Ein positives rc waere mit
        # "N rows appended" des Erfolgspfads ambig gewesen und ok=true haette
        # den Guard-Treffer als Erfolg maskiert (E-142-Klasse).
        logger.error("[refresh-sector] %s", exc)
        _write_status(
            rc=-2,
            cache_latest=cache_latest,
            fetch_latest=fetch_latest,
            rows_appended=0,
            error=f"seam_guard: {exc}",
        )
        return -2
    if result.dropped_symbols:
        logger.warning(
            "[refresh-sector] dropped (overlap ratio not constant): %s",
            result.dropped_symbols,
        )
    merged = result.combined.reset_index(drop=True)
    # F-TR-4: real angehaengte Zeilen zaehlen (Drops der Overlap-Pruefung
    # duerfen nicht als "appended" berichtet werden).
    rows_appended_real = max(0, len(merged) - len(cache))  # F-senior-6

    # Atomic write (Path.replace is atomic on the same filesystem).
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    try:
        merged.to_parquet(tmp, index=False)
        tmp.replace(cache_path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    # Post-merge freshness report against the factor's live guard (info only).
    sector_newest = merged[merged["symbol"].isin(TARGET_SYMBOLS)]["timestamp"].max()
    age_days = (pd.Timestamp(today_utc, tz="UTC") - sector_newest).days
    guard_state = "FRESH" if age_days <= _STALE_DAYS_REF else "STILL-STALE"
    logger.info(
        "[refresh-sector] wrote %s | sector newest=%s (age %dd, guard=%dd → %s)",
        cache_path,
        sector_newest,
        age_days,
        _STALE_DAYS_REF,
        guard_state,
    )
    _write_status(
        rc=int(rows_appended_real),  # F-senior-5: rc == Rueckgabewert
        cache_latest=cache_latest,
        fetch_latest=fetch_latest,
        rows_appended=int(rows_appended_real),
        # F-senior-1: Drop-Forensik zusaetzlich ins Status-JSON (E-140/E-172),
        # nicht nur in die Tageslogdatei. EHRLICHE GRENZE (E-176): das Feld
        # wird geschrieben, aber ein Konsument/Alarm fehlt noch (Follow-up);
        # der Drop-Pfad endet mit Exit 0.
        dropped_symbols=result.dropped_symbols,
        error=(
            f"overlap_ratio_not_constant: {sorted(result.dropped_symbols)}"
            if result.dropped_symbols
            else None
        ),
    )
    return int(rows_appended_real)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without writing",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Fetch window in calendar days (default {DEFAULT_LOOKBACK_DAYS})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    rc = refresh(CACHE_PATH, dry_run=args.dry_run, lookback_days=args.lookback_days)
    return 0 if rc >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
