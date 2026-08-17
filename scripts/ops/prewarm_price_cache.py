"""Pre-warm output/aggregates/daily.parquet with missing watchlist symbols.

Pilot R6-followup: after switching the watchlist to the 195-symbol master
universe, 125 of those symbols are missing from the EOD price cache.
The pilot's load_eod_prices reads daily.parquet first and only falls back
to yfinance when stale — but missing symbols still need a one-time fetch.

This script:
1. Reads watchlist.txt (skipping comment lines)
2. Loads existing cache at output/aggregates/daily.parquet
3. Computes the gap (watchlist - cache)
4. Fetches gap symbols via yfinance (2-year history by default)
5. Merges + sorts + writes back to cache atomically (tmp + replace)

Usage:
    python scripts/ops/prewarm_price_cache.py            # default: ~2y history
    python scripts/ops/prewarm_price_cache.py --years 5  # longer history
    python scripts/ops/prewarm_price_cache.py --dry-run  # show gap only
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.sources.yfinance_source import YFinanceRateLimitError  # noqa: E402

logger = logging.getLogger(__name__)

WATCHLIST_PATH = ROOT / "watchlist.txt"
CACHE_PATH = ROOT / "output" / "aggregates" / "daily.parquet"


def load_watchlist(path: Path = WATCHLIST_PATH) -> list[str]:
    """Read watchlist.txt, skipping comments + blanks."""
    if not path.exists():
        raise FileNotFoundError(f"Watchlist not found: {path}")
    return [
        s.strip()
        for s in path.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.startswith("#")
    ]


def cache_symbols(path: Path = CACHE_PATH) -> set[str]:
    """Return symbols currently in the price cache."""
    if not path.exists():
        return set()
    import pandas as pd

    df = pd.read_parquet(path, columns=["symbol"])
    return set(df["symbol"].unique())


def stale_cache_symbols(
    watchlist: list[str],
    max_age_days: int,
    path: Path = CACHE_PATH,
) -> list[str]:
    """Return watchlist symbols in the cache whose own latest bar is > max_age_days old.

    F-RX-6 §9.12 (d) follow-up: prewarm previously refreshed only MISSING
    symbols (watchlist - cache). Symbols PRESENT in cache but stale per-symbol
    (e.g. KO/PEP/BRK-B/PG @ 2026-05-01 while panel-refreshed peers are at
    2026-05-18) stayed frozen forever — refresh_daily_cache_from_panel.py
    can't fix them because they're not in the master_universe_panel. This
    helper surfaces them so the prewarm path can yfinance-refresh them too.

    Returns symbols sorted by ascending freshness (oldest first), so a
    --max-symbols budget caps the work to the most-urgent ones.
    """
    if not path.exists():
        return []
    import pandas as pd

    df = pd.read_parquet(path, columns=["symbol", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["symbol"].isin(watchlist)]
    per_sym = df.groupby("symbol")["timestamp"].max()
    today = pd.Timestamp.now("UTC").normalize()
    ages = (today - per_sym.dt.normalize()).dt.days
    stale = ages[ages > max_age_days].sort_values(ascending=False)
    return list(stale.index)


def fetch_missing_alpaca(missing: list[str], years: int) -> "pd.DataFrame":
    """Fetch symbols via Alpaca bars API (fallback when yfinance is rate-limited).

    Requires ALPACA_API_KEY and ALPACA_API_SECRET environment variables.
    Returns empty DataFrame if SDK unavailable or credentials missing.
    """
    import os

    import pandas as pd

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_API_SECRET", "")
    if not api_key or not secret_key:
        logger.error(
            "[prewarm] Alpaca fallback unavailable: ALPACA_API_KEY/ALPACA_API_SECRET not set"
        )
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        )

    try:
        from alpaca.data import StockHistoricalDataClient  # type: ignore[import]
        from alpaca.data.requests import StockBarsRequest  # type: ignore[import]
        from alpaca.data.timeframe import TimeFrame  # type: ignore[import]
    except ImportError:
        logger.error(
            "[prewarm] alpaca-py SDK not installed; cannot use Alpaca fallback"
        )
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        )

    # Exclude today: Alpaca may return a partial intraday bar for the current
    # session; yesterday is the exclusive upper bound. (Adjustierung wird oben
    # explizit per Adjustment.ALL gesetzt — nicht mit dieser Datumslogik verwechseln.)
    end_dt = datetime.now(tz=timezone.utc) - timedelta(days=1)
    start_dt = end_dt - timedelta(days=int(years * 366))
    logger.info(
        "[prewarm] Alpaca fallback: fetching %d symbols (%s to %s)",
        len(missing),
        start_dt.date().isoformat(),
        end_dt.date().isoformat(),
    )

    try:
        from alpaca.data.enums import Adjustment  # type: ignore[import]

        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        # BLOCKER-Fix 2026-08-17 (E-165): der API-Default ist RAW. Der Cache
        # ist total-return-adjustiert — RAW-Bars erzeugten am 17.08. eine
        # Naht mit bis +2444 % (BKNG) und Splits als -90-%-Crashs mitten in
        # der Reihe. adjustment=ALL = split+dividend-adjustiert, dieselbe
        # Basis wie der Bestand. NIE den Default einer Preis-API annehmen.
        request = StockBarsRequest(
            symbol_or_symbols=missing,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            adjustment=Adjustment.ALL,
        )
        bars = client.get_stock_bars(request)
        df = bars.df.reset_index()
    except Exception as exc:
        logger.error(
            "[prewarm] Alpaca bars fetch failed (%s): %s", type(exc).__name__, exc
        )
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        )

    if df.empty or "symbol" not in df.columns:
        logger.warning("[prewarm] Alpaca returned empty or schema-less DataFrame")
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        )

    # Lowercase column names (defensive against SDK schema drift)
    df = df.rename(columns=str.lower)
    # Alpaca timestamps are timezone-aware; normalize to UTC date
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    keep = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    # Spiegel-Konvention wie die anderen beiden Cache-Schreiber (F-senior-2,
    # 2026-08-17): ohne adj_close riss dieser DRITTE Schreiber die am 15.08.
    # reparierte 0-NaN-Invariante wieder auf (gemessen 97.859 NaN).
    df["adj_close"] = df["close"]
    got = set(df["symbol"].unique())
    failed = set(missing) - got
    if failed:
        logger.warning(
            "[prewarm] Alpaca: %d/%d symbols had no data: %s",
            len(failed),
            len(missing),
            sorted(failed)[:20],
        )
    logger.info(
        "[prewarm] Alpaca: fetched %d rows for %d/%d symbols",
        len(df),
        len(got),
        len(missing),
    )
    return df


_FAILED_SYMBOLS_PATH = ROOT / "output" / "prewarm_failed_symbols.json"


def write_failed_symbols(
    symbols: list[str],
    reason: str,
    path: Path = _FAILED_SYMBOLS_PATH,
) -> None:
    """Persist symbols that could not be fetched for cross-run monitoring.

    stale_cache_symbols() will re-surface them on the next run, but this file
    lets an operator (or future health-check script) detect persistent per-symbol
    failures without tailing per-day scheduler logs.
    """
    import json

    payload = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "reason": reason,
        "symbols": sorted(symbols),
        "count": len(symbols),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.warning(
        "[prewarm] %d symbols unfetched — logged to %s (reason: %s)",
        len(symbols),
        path,
        reason,
    )


def fetch_missing(missing: list[str], years: int) -> "pd.DataFrame":
    """Fetch the missing symbols via yfinance; raises YFinanceRateLimitError on HTTP 429."""
    from src.assembled_core.data.sources.yfinance_source import fetch_prices_yfinance

    end = datetime.now(tz=timezone.utc).date()
    start = end - timedelta(days=int(years * 366))  # buffer for leap years
    logger.info(
        "[prewarm] fetching %d symbols from yfinance (%s to %s)",
        len(missing),
        start.isoformat(),
        end.isoformat(),
    )
    # YFinanceRateLimitError propagates to caller for Alpaca fallback
    df = fetch_prices_yfinance(missing, start.isoformat(), end.isoformat())
    if df.empty:
        logger.error("[prewarm] yfinance returned EMPTY DataFrame for all symbols")
        return df
    got = set(df["symbol"].unique())
    failed = set(missing) - got
    if failed:
        logger.warning(
            "[prewarm] %d/%d symbols had no data: %s",
            len(failed),
            len(missing),
            sorted(failed)[:20],
        )
    logger.info(
        "[prewarm] fetched %d rows for %d/%d symbols",
        len(df),
        len(got),
        len(missing),
    )
    return df


def merge_and_save(new_df: "pd.DataFrame", cache_path: Path = CACHE_PATH) -> int:
    """Merge new rows into cache. Returns rows-after-merge count."""
    import pandas as pd

    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
    else:
        existing = pd.DataFrame(columns=new_df.columns)

    # OVERLAP-RE-ADJUSTIERUNG (E-165 Teil 2, 2026-08-17): auch adjusted-zu-
    # adjusted ist nicht nahtfrei, wenn zwischen dem Anker des Bestands
    # (letzter EODHD-Bar) und dem Anker der neuen Quelle (heute) eine
    # Corporate Action lag — die neue Reihe ist dann RUECKWIRKEND anders
    # skaliert (gemessen: AAPL, BE am 17.08.). Korrekte Operation: im
    # Ueberlappungsfenster die Ratio neu/alt messen; ist sie KONSTANT und
    # != 1, wird der gesamte BESTAND des Symbols auf den neuen Anker
    # reskaliert (normale rueckwaertige Adjustierung). Nicht-konstante
    # Ratio = inkonsistente Quelle -> Symbol verwerfen, failed-Liste.
    _price_cols = [
        c
        for c in ("open", "high", "low", "close", "adj_close")
        if c in existing.columns
    ]
    _verified: set[str] = set()  # Symbole mit per Overlap BEWIESENER Semantik
    _drop_syms: list[str] = []
    if not existing.empty and not new_df.empty:
        for _sym in new_df["symbol"].unique():
            _old_s = existing[existing["symbol"] == _sym].set_index("timestamp")
            _new_s = new_df[new_df["symbol"] == _sym].set_index("timestamp")
            _common = _old_s.index.intersection(_new_s.index)
            if len(_common) < 5:
                continue  # kein belastbarer Overlap -> Naht-Guard entscheidet
            _ratio = _new_s.loc[_common, "close"].astype(float) / _old_s.loc[
                _common, "close"
            ].astype(float)
            _med = float(_ratio.median())
            _spread = float((_ratio / _med - 1).abs().max())
            if _spread <= 0.02:
                _verified.add(_sym)
            if _spread > 0.02:
                logger.warning(
                    "[prewarm] %s: overlap ratio NOT constant (spread %.1f%%) "
                    "— dropping its new rows, flagging for review",
                    _sym,
                    _spread * 100,
                )
                _drop_syms.append(_sym)
                continue
            if abs(_med - 1.0) > 0.001:
                logger.info(
                    "[prewarm] %s: corporate action between anchors — "
                    "rescaling existing history by %.6f",
                    _sym,
                    _med,
                )
                # F-auditor-7 (2026-08-17): grosse Faktoren sind meist echte
                # Splits, KOENNEN aber auch ein recyceltes Ticker-Symbol oder
                # ein Einheitenwechsel sein — konstant heisst konsistent,
                # nicht zwingend legitim. Nicht blocken, aber LAUT machen.
                # Bewusst NICHT mitskaliert: volume (Split-Historie bleibt
                # stueckseitig alt; ADV ueber die Grenze ist dann unscharf).
                if abs(_med - 1.0) > 0.25:
                    logger.warning(
                        "[prewarm] %s: LARGE rescale factor %.4f over %d rows "
                        "— verify this is a real split, not ticker recycling",
                        _sym,
                        _med,
                        int((existing["symbol"] == _sym).sum()),
                    )
                _mask = existing["symbol"] == _sym
                for _c in _price_cols:
                    existing.loc[_mask, _c] = (
                        existing.loc[_mask, _c].astype(float) * _med
                    )
        if _drop_syms:
            new_df = new_df[~new_df["symbol"].isin(_drop_syms)]
            # F-senior-4: NICHT hier protokollieren — der Naht-Guard unten
            # kann noch abbrechen ("Nothing written" muss wahr bleiben);
            # geschrieben wird direkt vor dem atomic write.

    combined = pd.concat([existing, new_df], ignore_index=True)
    # Dedupe on (symbol, timestamp) — last-write-wins favors the fresh fetch
    combined = combined.drop_duplicates(
        subset=["symbol", "timestamp"], keep="last"
    ).sort_values(["symbol", "timestamp"])

    # INVARIANTE AM SCHREIBPUNKT (E-166, 2026-08-17): adj_close == close ist
    # die Cache-Invariante (backfill_adj_close 15.08.). Sie je FETCH-Pfad zu
    # spiegeln hat zweimal versagt (Alpaca-Pfad am 16.08., yfinance-Pfad am
    # 17.08. — liefert die Spalte gar nicht). Hier, am einzigen Schreibpunkt,
    # kann kein vierter Pfad sie mehr aufreissen.
    # F-senior-2 (Kompakt-Review 2026-08-17): Invariante UNBEDINGT herstellen
    # — der Spalte-fehlt-Fall (frischer Cache + Quelle ohne adj_close) ist die
    # haerteste Verletzung und war vom bedingten Guard ausgeschlossen (E-170).
    if "adj_close" not in combined.columns:
        combined["adj_close"] = combined["close"]
    _na = combined["adj_close"].isna()
    if bool(_na.any()):
        combined.loc[_na, "adj_close"] = combined.loc[_na, "close"]

    # NAHT-GUARD (E-165, fail-closed, 2026-08-17): zwei Preisquellen sind nur
    # mergebar, wenn die Werte-SEMANTIK identisch ist — Symbol+Timestamp als
    # Merge-Key reicht nicht. Der 17.08.-Vorfall (RAW-Bars in TR-Cache:
    # BKNG +2444 %, NFLX-Split als -90 %) waere hier gestoppt worden.
    # PRAEZISIERUNG (gleicher Tag): geprueft wird NUR die QUELLEN-NAHT
    # (letzter Alt-Bar -> erster Neu-Bar) und NUR fuer Symbole OHNE per
    # Overlap bewiesene Semantik — echte historische Extremtage im Bestand
    # (AAPL -51,8 % Sep-2000, BE +59 % Nov-2024, beide real) duerfen einen
    # Merge nicht blocken; der Fehlerfall ist der Ratio-Sprung ZWISCHEN
    # Quellen, nicht ein Extremtag innerhalb einer Quelle.
    _new_syms = set(new_df["symbol"].unique()) if not new_df.empty else set()
    _bad: list[str] = []
    for _sym in _new_syms - _verified:
        _old_ts = (
            existing.loc[existing["symbol"] == _sym, "timestamp"].max()
            if not existing.empty
            else None
        )
        if _old_ts is None or pd.isna(_old_ts):
            continue  # neues Symbol: keine Naht
        _after = new_df[
            (new_df["symbol"] == _sym) & (new_df["timestamp"] > _old_ts)
        ].sort_values("timestamp")
        if _after.empty:
            continue
        _old_close = float(
            existing.loc[
                (existing["symbol"] == _sym) & (existing["timestamp"] == _old_ts),
                "close",
            ].iloc[0]
        )
        _first_new = float(_after["close"].iloc[0])
        if _old_close > 0 and abs(_first_new / _old_close - 1.0) > 0.5:
            _bad.append(_sym)
    if _bad:
        raise RuntimeError(
            f"[prewarm] MERGE ABORTED (seam guard): {len(_bad)} symbol(s) with "
            f"|daily move| > 50% after merge — adjustment-basis mismatch or "
            f"corrupt feed. Nothing written. Symbols: {sorted(_bad)[:15]}"
        )

    if _drop_syms:
        write_failed_symbols(_drop_syms, "overlap_ratio_not_constant")

    # Atomic write
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(cache_path)
    return len(combined)


def main(argv: list[str] | None = None) -> int:
    # .env HIER laden, nicht auf Modulebene: tests/test_prewarm_price_cache.py
    # laedt dieses Script per importlib.exec_module — ein Modulebenen-
    # load_dotenv wuerde echte Credentials in den pytest-Prozess injizieren
    # (F-senior-3/E-168). override=False: Task-gesetzte ENV gewinnt.
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        logger.warning("[WARN] python-dotenv fehlt — Alpaca-Fallback ohne .env-Keys")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="History years to fetch for missing symbols (default 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report the gap, do not fetch or write",
    )
    parser.add_argument(
        "--max-stale-days",
        type=int,
        default=3,
        help=(
            "Refresh cache-present watchlist symbols whose own latest bar is "
            "older than this many calendar days (default 3, aligned with "
            "_drop_per_symbol_stale_rows max_age_days in run_live_paper.py so "
            "there is no silent dead-zone of stale-but-not-prewarmed symbols "
            "— F-RX-FU-4). Set to 0 to skip stale-row refresh entirely."
        ),
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=30,
        help=(
            "Hard budget on the number of symbols yfinance will be asked for "
            "in one invocation (default 30). Caps wall-clock time when "
            "rate-limited so the Task Scheduler ExecutionTimeLimit isn't hit. "
            "Stale symbols are processed oldest-first."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    watchlist = load_watchlist()
    cached = cache_symbols()
    missing = sorted(set(watchlist) - cached)
    stale = (
        stale_cache_symbols(watchlist, max_age_days=args.max_stale_days)
        if args.max_stale_days > 0
        else []
    )

    print(
        f"[prewarm] watchlist={len(watchlist)} cached={len(cached)} "
        f"missing={len(missing)} stale(>{args.max_stale_days}d)={len(stale)}"
    )

    if not missing and not stale:
        print("[prewarm] no gap, no stale rows — cache fully fresh")
        return 0

    # Budget: missing first (truly absent), then stale (refresh-eligible).
    targets = missing + [s for s in stale if s not in missing]
    if args.max_symbols > 0 and len(targets) > args.max_symbols:
        print(
            f"[prewarm] {len(targets)} targets exceeds --max-symbols={args.max_symbols} "
            f"budget; deferring tail to next invocation"
        )
        targets = targets[: args.max_symbols]

    print(f"[prewarm] will fetch (first 20): {targets[:20]}")
    if args.dry_run:
        print("[prewarm] DRY RUN — no fetch performed")
        return 0

    fetch_reason = "yfinance"
    try:
        df = fetch_missing(targets, years=args.years)
    except YFinanceRateLimitError as exc:
        print(
            f"[prewarm] yfinance rate-limited ({exc}) — falling back to Alpaca bars API"
        )
        fetch_reason = "alpaca_fallback"
        df = fetch_missing_alpaca(targets, years=args.years)
    except Exception as exc:  # noqa: BLE001
        print(f"[prewarm] fetch failed: {exc} — aborting merge")
        write_failed_symbols(targets, reason=f"fetch_exception:{type(exc).__name__}")
        return 1

    if df.empty:
        print("[prewarm] no data fetched — aborting merge")
        write_failed_symbols(targets, reason=f"{fetch_reason}_empty")
        return 1

    # Record any symbols that still had no data after the fetch
    fetched_syms = set(df["symbol"].unique()) if "symbol" in df.columns else set()
    still_missing = [s for s in targets if s not in fetched_syms]
    if still_missing:
        write_failed_symbols(still_missing, reason=f"{fetch_reason}_partial")

    total = merge_and_save(df)
    print(f"[prewarm] cache updated: {total:,} total rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
