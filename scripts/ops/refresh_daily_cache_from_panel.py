"""Refresh output/aggregates/daily.parquet from data/sample/master_universe_panel.parquet.

Bridges the data freshness gap that breaks the paper pilot when yfinance is
rate-limited at 21:30. The pilot's `_load_prices` in scripts/run_live_paper.py
checks if daily.parquet is <= 3 days old; if not, it falls through to a
sequential per-symbol yfinance fetch that — with 197 symbols and a 15-minute
Task Scheduler ExecutionTimeLimit — gets hard-terminated by Windows.

The master_universe_panel.parquet is built earlier in the daily cycle and
typically contains fresher OHLCV. This script copies its newer rows into
daily.parquet so the pilot's cache-fresh path stays satisfied.

Schemas:
- daily.parquet: [timestamp, symbol, open, high, low, close, adj_close, volume]
- master_universe_panel.parquet: [timestamp, symbol, open, high, low, close, volume]
  (no adj_close — we default it to close for the appended rows)

Idempotent: appends only rows with timestamp > cache.timestamp.max(); drops
exact (symbol, timestamp) duplicates as a final safety net.

Usage:
    python scripts/ops/refresh_daily_cache_from_panel.py
    python scripts/ops/refresh_daily_cache_from_panel.py --dry-run
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

logger = logging.getLogger(__name__)

CACHE_PATH = ROOT / "output" / "aggregates" / "daily.parquet"
PANEL_PATH = ROOT / "data" / "sample" / "master_universe_panel.parquet"
STATUS_PATH = ROOT / "output" / "ops" / "refresh_cache_status.json"


def _write_status(
    *,
    rc: int,
    cache_latest: object | None,
    panel_latest: object | None,
    rows_appended: int,
    error: str | None = None,
    status_path: Path | None = None,
) -> None:
    """Write a status JSON for ops monitoring (F-RX-5 follow-up §9.12 (c)).

    The .bat catches errorlevel 1 and logs WARN to a per-day file, which has
    no alert surface. This sidecar JSON gives downstream consumers (alerting,
    halt-flag triggers, dashboards) a single load-then-check path. Best-effort
    write: an exception here must not abort the refresh outcome.
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
            "panel_latest": str(panel_latest) if panel_latest is not None else None,
            "rows_appended": int(rows_appended),
            "error": error,
        }
        tmp = status_path.with_name(status_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(status_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[refresh-cache] failed to write status JSON: %s", exc)


def refresh(cache_path: Path, panel_path: Path, *, dry_run: bool) -> int:
    """Append fresher panel rows into the daily cache. Returns rows appended."""
    import pandas as pd

    if not cache_path.exists():
        logger.error("[refresh-cache] cache not found: %s", cache_path)
        _write_status(
            rc=-1,
            cache_latest=None,
            panel_latest=None,
            rows_appended=0,
            error=f"cache not found: {cache_path}",
        )
        return -1
    if not panel_path.exists():
        logger.error("[refresh-cache] panel not found: %s", panel_path)
        _write_status(
            rc=-1,
            cache_latest=None,
            panel_latest=None,
            rows_appended=0,
            error=f"panel not found: {panel_path}",
        )
        return -1

    cache = pd.read_parquet(cache_path)
    panel = pd.read_parquet(panel_path)

    cache["timestamp"] = pd.to_datetime(cache["timestamp"], utc=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)

    # K6 / E-053 (2026-07-21, GESAMTBEWERTUNG): enforce the PIT cutoff in
    # THIS writer instead of trusting the panel builder's fetch semantics.
    # Every ingest path into the live price store must drop same-day
    # (potentially forming) bars itself — mirrors the guard in
    # scripts/ops/refresh_daily_cache_from_eodhd.py and
    # refresh_sector_etf_cache.py. Daily bars carry their trading day at
    # midnight UTC, so `timestamp < today` keeps only completed sessions.
    today_utc = pd.Timestamp.now("UTC").normalize()
    n_before_cutoff = len(panel)
    panel = panel[panel["timestamp"] < today_utc]
    if len(panel) < n_before_cutoff:
        logger.warning(
            "[refresh-cache] PIT cutoff dropped %d same-day/future panel row(s) "
            "(>= %s) — forming bars must never enter daily.parquet (E-053)",
            n_before_cutoff - len(panel),
            today_utc.date(),
        )

    cache_latest = cache["timestamp"].max()
    panel_latest = panel["timestamp"].max()
    logger.info(
        "[refresh-cache] cache latest=%s, panel latest=%s",
        cache_latest,
        panel_latest,
    )

    # F-RX-2 (audit 2026-05-21): per-symbol comparison instead of global max.
    # Even if global panel.max() == global cache.max(), individual symbols
    # may have stale rows in cache that the panel has refreshed. Compare
    # per-symbol latest timestamps so heterogeneous freshness is fixed at
    # the per-symbol level.
    very_old = pd.Timestamp("1900-01-01", tz="UTC")
    cache_per_sym = cache.groupby("symbol")["timestamp"].max().rename("_cache_max")

    # Merge each panel row with the cache's per-symbol max; absent symbols
    # default to very_old so all their rows are treated as new.
    cache_max_df = cache_per_sym.reset_index()
    panel_with_cmax = panel.merge(cache_max_df, on="symbol", how="left")
    panel_with_cmax["_cache_max"] = panel_with_cmax["_cache_max"].fillna(very_old)
    new_rows = panel_with_cmax[
        panel_with_cmax["timestamp"] > panel_with_cmax["_cache_max"]
    ].drop(columns=["_cache_max"])
    if new_rows.empty:
        logger.info(
            "[refresh-cache] no panel rows strictly newer than per-symbol cache max"
        )
        _write_status(
            rc=0,
            cache_latest=cache_latest,
            panel_latest=panel_latest,
            rows_appended=0,
        )
        return 0

    n_syms = new_rows["symbol"].nunique()
    ts_min = new_rows["timestamp"].min()
    ts_max = new_rows["timestamp"].max()
    logger.info(
        "[refresh-cache] %d rows to append for %d symbols, ts %s..%s",
        len(new_rows),
        n_syms,
        ts_min,
        ts_max,
    )

    if dry_run:
        logger.info("[refresh-cache] --dry-run set, not writing")
        _write_status(
            rc=int(len(new_rows)),
            cache_latest=cache_latest,
            panel_latest=panel_latest,
            rows_appended=int(len(new_rows)),
        )
        return len(new_rows)

    # F-RX-3 §9.12 (a): panel has no adj_close column. Live-paper hot-path
    # is unaffected (load_eod_prices:146 strips adj_close before the pilot
    # sees it). BUT direct-parquet consumers (backtests, factor stores) read
    # the column. Setting adj_close = close would silently mis-handle
    # ex-dividend dates. Use NaN as sentinel: consumers that compute returns
    # from adj_close get NaN propagation (loud failure), and any consumer
    # that uses .fillna(close) is making the same fallback choice explicit.
    if "adj_close" not in new_rows.columns:
        import numpy as np

        new_rows["adj_close"] = np.nan
        logger.warning(
            "[refresh-cache] panel lacks adj_close — appended rows have "
            "adj_close=NaN (sentinel). Direct-parquet consumers must guard "
            "or fillna(close) explicitly; live-paper hot-path strips "
            "adj_close at load_eod_prices:146 and is unaffected."
        )

    # Reorder to match cache schema (drop any extra cols panel may carry).
    new_rows = new_rows[cache.columns.tolist()]

    merged = pd.concat([cache, new_rows], ignore_index=True)
    merged = (
        merged.sort_values(["symbol", "timestamp"])
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .reset_index(drop=True)
    )

    logger.info(
        "[refresh-cache] merged total rows=%d (%d symbols), new latest=%s",
        len(merged),
        merged["symbol"].nunique(),
        merged["timestamp"].max(),
    )

    # F-RX-4 §9.12 (b): use Path.replace (atomic on same FS) instead of
    # shutil.move (falls back to copy+remove across filesystems). Matches
    # the idiom used in scripts/ops/prewarm_price_cache.py for consistency.
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
    logger.info("[refresh-cache] wrote %s", cache_path)
    _write_status(
        rc=int(len(new_rows)),
        cache_latest=cache_latest,
        panel_latest=panel_latest,
        rows_appended=int(len(new_rows)),
    )
    return len(new_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without writing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    rc = refresh(CACHE_PATH, PANEL_PATH, dry_run=args.dry_run)
    return 0 if rc >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
