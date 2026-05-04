"""Download pre-2020 EOD data and merge into extended watchlist panel.

Downloads 2007-01-01 to 2019-12-31 for all symbols in the existing watchlist panel,
merges with watchlist_22_2020_2026.parquet, and saves as watchlist_2007_2026.parquet.

Usage:
    python scripts/build_pre2020_panel.py
    python scripts/build_pre2020_panel.py --dry-run
    python scripts/build_pre2020_panel.py --start 2008-01-01 --end 2019-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_pre2020_panel")

EXISTING_PANEL = ROOT / "data" / "sample" / "watchlist_22_2020_2026.parquet"
OUTPUT_PANEL = ROOT / "data" / "sample" / "watchlist_2007_2026.parquet"


def _download_ticker(
    symbol: str, start: str, end: str, retries: int = 3
) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start, end=end, auto_adjust=True)
            if hist.empty:
                logger.warning("%s: no data returned", symbol)
                return pd.DataFrame()
            hist = hist.reset_index()
            # Normalize column names
            hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]
            date_col = next(
                (c for c in hist.columns if c in ("date", "datetime")), None
            )
            if date_col is None:
                logger.warning(
                    "%s: no date column found in %s", symbol, hist.columns.tolist()
                )
                return pd.DataFrame()
            hist = hist.rename(columns={date_col: "timestamp"})
            hist["symbol"] = symbol
            hist["timestamp"] = pd.to_datetime(hist["timestamp"]).dt.tz_localize(None)
            keep = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            present = [c for c in keep if c in hist.columns]
            return hist[present].copy()
        except Exception as exc:
            logger.warning("%s attempt %d/%d: %s", symbol, attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(2**attempt)
    return pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build extended pre-2020 price panel")
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument(
        "--dry-run", action="store_true", help="Download but do not write"
    )
    parser.add_argument("--existing", default=str(EXISTING_PANEL))
    parser.add_argument("--out", default=str(OUTPUT_PANEL))
    args = parser.parse_args(argv)

    if not Path(args.existing).exists():
        logger.error("Existing panel not found: %s", args.existing)
        return 1

    existing = pd.read_parquet(args.existing)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"]).dt.tz_localize(None)
    symbols = sorted(existing["symbol"].unique().tolist())
    logger.info(
        "Downloading %d symbols from %s to %s", len(symbols), args.start, args.end
    )

    frames = []
    for i, sym in enumerate(symbols, 1):
        logger.info("[%d/%d] %s", i, len(symbols), sym)
        df = _download_ticker(sym, args.start, args.end)
        if not df.empty:
            frames.append(df)
        time.sleep(0.3)  # gentle rate limit

    if not frames:
        logger.error("No data downloaded")
        return 1

    pre2020 = pd.concat(frames, ignore_index=True)
    logger.info(
        "Downloaded %d rows (%d symbols)", len(pre2020), pre2020["symbol"].nunique()
    )

    # Merge with existing panel
    combined = pd.concat([pre2020, existing], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"]).dt.tz_localize(None)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["timestamp", "symbol"])
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info(
        "Combined: %d rows (%d dropped as duplicates)",
        len(combined),
        before - len(combined),
    )
    logger.info(
        "Date range: %s to %s",
        combined["timestamp"].min().date(),
        combined["timestamp"].max().date(),
    )
    logger.info("Symbols: %d", combined["symbol"].nunique())

    if args.dry_run:
        logger.info("--dry-run: not writing output")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.info("Saved -> %s (%d rows)", out_path, len(combined))
    return 0


if __name__ == "__main__":
    sys.exit(main())
