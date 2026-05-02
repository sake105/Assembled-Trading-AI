"""EDCL Phase C — Pre-compute event_beta FeatureStore view.

For each (TriggerType, asset) pair, compute the median absolute forward return
in the N-day window following historical geo-trigger events. Writes results to
the FeatureStore as the 'event_beta' view so that conviction_engine.py can
load them via ASOF-join (PIT-safe).

Data requirements:
  --price-path:   Parquet panel with columns [date, symbol, close]. Required.
  --events-path:  CSV/Parquet with columns [event_date, trigger_type, source_tier].
                  Defaults to data/intel/geo_events_historical.parquet.
  --output-root:  FeatureStore root. Defaults to data/feature_store/.

Usage:
    python scripts/training/compute_event_betas.py \
        --price-path data/prices/panel_2018_2026.parquet \
        --events-path data/intel/geo_events_historical.parquet \
        --lookback-days 5 \
        --output-root data/feature_store

Output schema (written to <output-root>/event_beta/<date>.parquet):
    ticker | inference_ts | beta_<TRIGGER_TYPE>_<N>d
    -------|--------------|--------------------------
    XLE    | 2023-06-01   | 0.034
    ...

The file is keyed by the last event date seen (inference_ts = max(event_date)).
When new events are added, re-run this script to update the view.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make src importable when run as script
sys.path.insert(0, str(Path(__file__).parents[2]))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute EDCL event_beta FeatureStore view")
    parser.add_argument("--price-path", required=True, help="Panel parquet with date/symbol/close")
    parser.add_argument(
        "--events-path",
        default="data/intel/geo_events_historical.parquet",
        help="Historical geo-trigger events (event_date, trigger_type, ...)",
    )
    parser.add_argument("--lookback-days", type=int, default=5, help="Forward-return window")
    parser.add_argument("--output-root", default="data/feature_store", help="FeatureStore root")
    parser.add_argument("--min-events", type=int, default=3, help="Min events to compute beta")
    args = parser.parse_args()

    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        log.error("pandas / numpy required: %s", e)
        sys.exit(1)

    price_path = Path(args.price_path)
    events_path = Path(args.events_path)
    output_root = Path(args.output_root)

    if not price_path.exists():
        log.error("Price file not found: %s", price_path)
        sys.exit(1)

    if not events_path.exists():
        log.error(
            "Events file not found: %s\n"
            "Create it by running the geo-trigger ingestion pipeline or exporting\n"
            "crisis_state.json history to a tabular format with columns:\n"
            "  event_date (datetime), trigger_type (str), source_tier (int)",
            events_path,
        )
        sys.exit(1)

    # Load data
    log.info("Loading prices from %s", price_path)
    prices = pd.read_parquet(price_path)
    required_cols = {"date", "symbol", "close"}
    if not required_cols.issubset(prices.columns):
        log.error("Price file must have columns: %s — found: %s", required_cols, set(prices.columns))
        sys.exit(1)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["symbol", "date"])

    log.info("Loading events from %s", events_path)
    if events_path.suffix == ".csv":
        events = pd.read_csv(events_path)
    else:
        events = pd.read_parquet(events_path)
    events["event_date"] = pd.to_datetime(events["event_date"])

    trigger_types = events["trigger_type"].unique()
    symbols = prices["symbol"].unique()
    N = args.lookback_days
    min_events = args.min_events

    log.info(
        "Computing %d-day event betas: %d trigger types × %d symbols",
        N, len(trigger_types), len(symbols),
    )

    # Pivot prices to (date × symbol) matrix for efficient forward-return lookup
    price_matrix = prices.pivot(index="date", columns="symbol", values="close")

    # Compute forward returns for each (event_date, symbol)
    rows = []
    for symbol in symbols:
        if symbol not in price_matrix.columns:
            continue
        sym_prices = price_matrix[symbol].dropna()
        dates_idx = sym_prices.index

        for ttype in trigger_types:
            type_events = events[events["trigger_type"] == ttype]["event_date"].sort_values()
            returns: list[float] = []

            for ev_date in type_events:
                # Find price on or after event_date
                start_locs = dates_idx.searchsorted(ev_date, side="left")
                end_locs = start_locs + N
                if end_locs >= len(dates_idx):
                    continue
                p_start = sym_prices.iloc[start_locs]
                p_end = sym_prices.iloc[end_locs]
                if p_start > 0:
                    ret = (p_end - p_start) / p_start
                    returns.append(abs(float(ret)))

            if len(returns) < min_events:
                continue

            median_beta = float(np.median(returns))
            rows.append({
                "ticker": symbol,
                f"beta_{ttype}_{N}d": median_beta,
            })

    if not rows:
        log.warning("No event betas computed — check that events overlap with price history")
        return

    # Aggregate: one row per ticker with all trigger-type beta columns
    result = pd.DataFrame(rows)
    result = result.groupby("ticker").first().reset_index()
    inference_ts = events["event_date"].max()
    result["inference_ts"] = inference_ts

    # Write to FeatureStore
    view_dir = output_root / "event_beta"
    view_dir.mkdir(parents=True, exist_ok=True)
    out_path = view_dir / f"{inference_ts.date()}.parquet"
    result.to_parquet(out_path, index=False)
    log.info(
        "Wrote event_beta view: %s (%d tickers, %d trigger types, as_of=%s)",
        out_path, len(result), len(trigger_types), inference_ts.date(),
    )


if __name__ == "__main__":
    main()
