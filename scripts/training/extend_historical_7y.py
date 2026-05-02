"""extend_historical_7y.py — Extend historical price data to 7+ years.

Downloads missing OHLCV history from yfinance for all universe tickers,
merges with existing parquet data, validates coverage, and saves a quality
report.

Usage::

    python scripts/training/extend_historical_7y.py
    python scripts/training/extend_historical_7y.py --target-start 2018-01-01
    python scripts/training/extend_historical_7y.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PREFIX = "[DATA-EXT]"

# ---------------------------------------------------------------------------
# Default universe — 59 major tickers used as hard fallback
# ---------------------------------------------------------------------------

DEFAULT_UNIVERSE: list[str] = [
    # Tech / Mega-cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Financials / Consumer
    "JPM", "V", "JNJ", "UNH", "PG", "HD", "MA", "DIS",
    # Software / Cloud
    "ADBE", "CRM", "NFLX", "COST", "PEP", "CMCSA",
    # Semiconductors
    "INTC", "AMD", "QCOM", "TXN", "AVGO", "MU", "AMAT", "LRCX", "KLAC",
    # Fintech / Payments
    "PYPL", "SQ",
    # High-growth / New economy
    "ABNB", "COIN", "SNOW", "NET", "DDOG", "ZS", "CRWD", "PANW",
    # International e-commerce
    "SHOP", "MELI", "SE", "BABA", "JD", "PDD",
    # EV
    "NIO", "LI", "XPEV",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY", "MPC", "VLO", "PSX", "EOG", "DVN",
]

# ---------------------------------------------------------------------------
# Retry config (mirrors yfinance_source.py pattern)
# ---------------------------------------------------------------------------

_RETRY_MAX = 3
_RETRY_BACKOFF_BASE = 2.0  # seconds; doubles each retry
_RATE_LIMIT_SLEEP = 0.5    # seconds between tickers


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------

def _load_universe(universe_file: Path | None) -> list[str]:
    """Load tickers from file, or fall back to DEFAULT_UNIVERSE.

    Search order:
    1. Explicit ``universe_file`` argument (if provided).
    2. ``data/universe/default_universe.csv`` relative to repo root.
    3. Watchlist files under ``data/universe/``.
    4. Hardcoded DEFAULT_UNIVERSE.
    """
    # -- explicit path --
    if universe_file is not None:
        if universe_file.exists():
            tickers = _read_ticker_file(universe_file)
            logger.info("%s Loaded %d tickers from %s", PREFIX, len(tickers), universe_file)
            return tickers
        logger.warning("%s universe_file not found: %s — searching defaults", PREFIX, universe_file)

    # -- repo-relative defaults --
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "data" / "universe" / "default_universe.csv",
        repo_root / "data" / "universe" / "watchlist.csv",
        repo_root / "data" / "universe" / "universe.csv",
    ]
    for path in candidates:
        if path.exists():
            tickers = _read_ticker_file(path)
            if tickers:
                logger.info("%s Loaded %d tickers from %s", PREFIX, len(tickers), path)
                return tickers

    logger.warning(
        "%s No universe file found — using hardcoded DEFAULT_UNIVERSE (%d tickers)",
        PREFIX,
        len(DEFAULT_UNIVERSE),
    )
    return list(DEFAULT_UNIVERSE)


def _read_ticker_file(path: Path) -> list[str]:
    """Read tickers from CSV/TXT.  Accepts first-column CSVs or plain lists."""
    try:
        df = pd.read_csv(path, header=None)
        # If the first cell looks like a header (non-ticker string) skip it
        raw = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
        tickers = [t for t in raw if t and not t.lower().startswith("ticker") and not t.lower().startswith("symbol")]
        return tickers
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s Failed to read %s — %s", PREFIX, path, exc)
        return []


# ---------------------------------------------------------------------------
# yfinance download with retry
# ---------------------------------------------------------------------------

def _fetch_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """Download adjusted OHLCV for one symbol with retry/backoff.

    Returns None on failure.  Uses ``auto_adjust=True`` so Close reflects
    adjusted prices (handles splits and dividends).
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        logger.error("%s yfinance not installed. Run: pip install yfinance>=0.2.40", PREFIX)
        return None

    last_exc: Exception | None = None
    for attempt in range(1, _RETRY_MAX + 1):
        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,   # adjusted OHLCV — correct for corporate actions
                actions=False,
            )
            if raw is None or raw.empty:
                logger.warning(
                    "%s No data for %s (%s – %s) on attempt %d",
                    PREFIX, symbol, start_date, end_date, attempt,
                )
                return None

            raw = raw.reset_index()
            date_col = "Date" if "Date" in raw.columns else "Datetime"
            raw = raw.rename(columns={date_col: "date"})
            raw["date"] = pd.to_datetime(raw["date"]).dt.normalize().dt.tz_localize(None)
            raw["symbol"] = symbol

            rename_map = {
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            }
            raw = raw.rename(columns=rename_map)

            cols = [c for c in ["date", "open", "high", "low", "close", "volume", "symbol"]
                    if c in raw.columns]
            return raw[cols].copy()

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = _RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "%s Attempt %d/%d failed for %s — %s. Retry in %.1fs.",
                PREFIX, attempt, _RETRY_MAX, symbol, exc, wait,
            )
            if attempt < _RETRY_MAX:
                time.sleep(wait)

    logger.error(
        "%s All %d retries exhausted for %s — %s",
        PREFIX, _RETRY_MAX, symbol, last_exc,
    )
    return None


# ---------------------------------------------------------------------------
# Per-ticker merge + validate
# ---------------------------------------------------------------------------

def _load_existing(parquet_path: Path) -> pd.DataFrame | None:
    """Load existing parquet for a symbol.  Returns None if absent/unreadable.

    Normalises the date column: existing files may use either ``timestamp``
    (UTC-aware DatetimeTZDtype) or ``date``; both are converted to a plain
    tz-naive ``date`` column for internal consistency.
    """
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path)
        # Normalise column name: legacy files store the date as "timestamp"
        if "timestamp" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"timestamp": "date"})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize().dt.tz_localize(None)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s Could not read %s — %s", PREFIX, parquet_path, exc)
        return None


def _merge_frames(existing: pd.DataFrame | None, new: pd.DataFrame | None) -> pd.DataFrame | None:
    """Combine existing and newly-downloaded frames; dedup + sort by date."""
    frames = [f for f in (existing, new) if f is not None and not f.empty]
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize().dt.tz_localize(None)
    merged = merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return merged


def _validate(
    df: pd.DataFrame,
    symbol: str,
    min_rows: int,
    max_gap_days: int,
) -> dict[str, Any]:
    """Run coverage and gap checks.  Returns a validation summary dict."""
    result: dict[str, Any] = {
        "symbol": symbol,
        "rows": len(df),
        "min_rows_ok": len(df) >= min_rows,
        "earliest_date": None,
        "latest_date": None,
        "max_gap_calendar_days": None,
        "gap_ok": True,
        "valid": True,
        "issues": [],
    }

    if df.empty:
        result["valid"] = False
        result["issues"].append("empty dataframe")
        return result

    dates = df["date"].sort_values()
    result["earliest_date"] = str(dates.iloc[0].date())
    result["latest_date"] = str(dates.iloc[-1].date())

    if not result["min_rows_ok"]:
        result["valid"] = False
        result["issues"].append(f"only {len(df)} rows, need {min_rows}")

    # Gap check — find largest calendar gap between consecutive trading days
    if len(dates) > 1:
        gaps = dates.diff().dt.days.dropna()
        max_gap = int(gaps.max())
        result["max_gap_calendar_days"] = max_gap
        if max_gap > max_gap_days:
            result["gap_ok"] = False
            result["valid"] = False
            result["issues"].append(f"max calendar gap {max_gap}d > {max_gap_days}d")
    else:
        result["max_gap_calendar_days"] = 0

    return result


# ---------------------------------------------------------------------------
# Main extension function
# ---------------------------------------------------------------------------

def extend_historical_data(
    universe_file: Path | None = None,
    target_start: str = "2019-01-01",
    output_dir: Path = Path("data/raw/equities_eod/yfinance"),
    validate: bool = True,
    min_rows: int = 1500,
    max_gap_days: int = 5,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Extend historical OHLCV data to cover target_start through today.

    Parameters
    ----------
    universe_file:
        Optional path to a CSV of tickers.  Falls back to defaults.
    target_start:
        Earliest date to ensure coverage from (``YYYY-MM-DD``).
    output_dir:
        Root directory for per-ticker parquet files.
    validate:
        If True, run row-count and gap checks after merge.
    min_rows:
        Minimum trading-day rows required per ticker.
    max_gap_days:
        Maximum allowed calendar-day gap between consecutive rows.
    dry_run:
        If True, download and validate but do not write files.

    Returns
    -------
    dict[symbol -> result_dict]
        Per-ticker outcome dicts with status, validation, and error info.
    """
    tickers = _load_universe(universe_file)
    today_str = date.today().isoformat()
    target_dt = pd.Timestamp(target_start)

    # Resolve output dir relative to repo root when given as relative path
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "%s Starting extension run — %d tickers, target_start=%s, output=%s",
        PREFIX, len(tickers), target_start, output_dir,
    )

    results: dict[str, dict] = {}

    for idx, symbol in enumerate(tickers, 1):
        logger.info("%s [%d/%d] Processing %s", PREFIX, idx, len(tickers), symbol)
        ticker_result: dict[str, Any] = {
            "symbol": symbol,
            "status": "pending",
            "downloaded_rows": 0,
            "existing_rows": 0,
            "merged_rows": 0,
            "download_start": None,
            "download_end": None,
            "validation": None,
            "error": None,
            "skipped": False,
        }

        parquet_path = output_dir / f"{symbol}.parquet"

        try:
            # -- Load existing data --
            existing_df = _load_existing(parquet_path)
            if existing_df is not None:
                ticker_result["existing_rows"] = len(existing_df)
                existing_earliest = existing_df["date"].min()
            else:
                existing_earliest = None

            # -- Determine download range --
            if existing_earliest is not None and existing_earliest <= target_dt:
                # Already have data from at least target_start; top-up to today
                dl_start = existing_df["date"].max() + timedelta(days=1)
                dl_start_str = dl_start.strftime("%Y-%m-%d")
                if dl_start_str >= today_str:
                    logger.info(
                        "%s %s already up-to-date through %s — skipping download",
                        PREFIX, symbol, existing_df["date"].max().date(),
                    )
                    ticker_result["skipped"] = True
                    new_df = None
                else:
                    logger.info(
                        "%s %s has data from %s; downloading %s → %s",
                        PREFIX, symbol, existing_earliest.date(), dl_start_str, today_str,
                    )
                    download_start_str = dl_start_str
                    download_end_str = today_str
                    ticker_result["download_start"] = download_start_str
                    ticker_result["download_end"] = download_end_str
                    new_df = None if dry_run else _fetch_symbol(
                        symbol, download_start_str, download_end_str
                    )
                    if new_df is not None:
                        ticker_result["downloaded_rows"] = len(new_df)
            else:
                # Need full history from target_start
                dl_end = today_str
                ticker_result["download_start"] = target_start
                ticker_result["download_end"] = dl_end
                logger.info(
                    "%s %s needs full history %s → %s",
                    PREFIX, symbol, target_start, dl_end,
                )
                new_df = None if dry_run else _fetch_symbol(symbol, target_start, dl_end)
                if new_df is not None:
                    ticker_result["downloaded_rows"] = len(new_df)

            if dry_run:
                ticker_result["status"] = "dry_run"
                ticker_result["skipped"] = True
                results[symbol] = ticker_result
                time.sleep(_RATE_LIMIT_SLEEP)
                continue

            # -- Merge --
            if not ticker_result["skipped"]:
                merged_df = _merge_frames(existing_df, new_df)
            else:
                merged_df = existing_df

            if merged_df is None or merged_df.empty:
                logger.warning("%s %s — no data after merge", PREFIX, symbol)
                ticker_result["status"] = "no_data"
                results[symbol] = ticker_result
                time.sleep(_RATE_LIMIT_SLEEP)
                continue

            ticker_result["merged_rows"] = len(merged_df)

            # -- Validate --
            if validate:
                val = _validate(merged_df, symbol, min_rows, max_gap_days)
                ticker_result["validation"] = val
                if not val["valid"]:
                    logger.warning(
                        "%s %s validation FAILED: %s", PREFIX, symbol, val["issues"]
                    )

            # -- Save --
            if not ticker_result["skipped"] or (
                existing_df is not None and new_df is not None
            ):
                merged_df.to_parquet(parquet_path, index=False)
                logger.info(
                    "%s %s saved %d rows → %s",
                    PREFIX, symbol, len(merged_df), parquet_path,
                )

            ticker_result["status"] = "ok"

        except Exception as exc:  # noqa: BLE001
            logger.error("%s %s FAILED — %s", PREFIX, symbol, exc, exc_info=True)
            ticker_result["status"] = "error"
            ticker_result["error"] = str(exc)

        results[symbol] = ticker_result
        time.sleep(_RATE_LIMIT_SLEEP)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    ok_count = sum(1 for r in results.values() if r["status"] == "ok")
    skip_count = sum(1 for r in results.values() if r.get("skipped") and r["status"] != "error")
    fail_count = sum(1 for r in results.values() if r["status"] == "error")
    nodata_count = sum(1 for r in results.values() if r["status"] == "no_data")

    logger.info(
        "%s Done — ok=%d  skipped=%d  no_data=%d  errors=%d  (total=%d)",
        PREFIX, ok_count, skip_count, nodata_count, fail_count, len(results),
    )
    if fail_count:
        failed = [s for s, r in results.items() if r["status"] == "error"]
        logger.warning("%s Failed tickers: %s", PREFIX, failed)

    # ---------------------------------------------------------------------------
    # Quality report
    # ---------------------------------------------------------------------------
    report_dir = Path(__file__).resolve().parents[2] / "output" / "data_quality"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "extension_report.json"

    report: dict[str, Any] = {
        "run_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "target_start": target_start,
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "summary": {
            "total": len(results),
            "ok": ok_count,
            "skipped": skip_count,
            "no_data": nodata_count,
            "errors": fail_count,
        },
        "tickers": results,
    }

    if not dry_run:
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("%s Quality report saved → %s", PREFIX, report_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend historical OHLCV data to 7+ years for all universe tickers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-start",
        default="2019-01-01",
        help="Earliest date to ensure coverage from (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/equities_eod/yfinance",
        help="Directory for per-ticker parquet files (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--universe-file",
        default=None,
        help="Path to a CSV of tickers (first column).  Falls back to built-in defaults.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without downloading or writing any files.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1500,
        help="Minimum trading-day rows required per ticker for validation.",
    )
    parser.add_argument(
        "--max-gap-days",
        type=int,
        default=5,
        help="Maximum calendar-day gap allowed between consecutive rows.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip row-count and gap validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    extend_historical_data(
        universe_file=Path(args.universe_file) if args.universe_file else None,
        target_start=args.target_start,
        output_dir=Path(args.output_dir),
        validate=not args.no_validate,
        min_rows=args.min_rows,
        max_gap_days=args.max_gap_days,
        dry_run=args.dry_run,
    )
