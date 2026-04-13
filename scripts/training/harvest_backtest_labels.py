"""Harvest trade labels from existing backtest results.

Scans output/accounting_report_backtest_* directories for ledger events,
reconstructs round-trip trades (entry/exit), joins with the factor panel
at the entry date to get feature values, and emits a labelled training
dataset for supervised learning.

Label:  1 if realized_return > 0, else 0

Output: output/training_data/backtest_trade_labels.parquet

Usage
-----
python scripts/training/harvest_backtest_labels.py
python scripts/training/harvest_backtest_labels.py \\
    --output-path output/training_data/backtest_trade_labels.parquet \\
    --panel-path  output/factor_panels/full_panel_7y.parquet \\
    --bt-root     output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# sys.path -- allow running from any cwd without pip install
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)
_TAG = "[HARVEST]"


def _log(msg: str) -> None:
    logger.info("%s %s", _TAG, msg)


def _warn(msg: str) -> None:
    logger.warning("%s %s", _TAG, msg)


# ---------------------------------------------------------------------------
# Trade reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_trades_from_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    """Read a ledger_events.parquet and return a list of round-trip trade dicts.

    Pairs ORDER_SUBMIT events (buy) with subsequent same-symbol ORDER_SUBMIT
    events (sell, indicated by negative qty) to form entry/exit pairs.
    Falls back to pairing by sign-change in qty when no FILL events exist.
    """
    try:
        df = pd.read_parquet(ledger_path)
    except Exception as exc:
        _warn(f"Cannot read {ledger_path}: {exc!r}")
        return []

    if df.empty:
        return []

    required = {"event_ts", "symbol", "qty", "price", "event_type"}
    if not required.issubset(df.columns):
        _warn(f"Missing columns in {ledger_path}: {required - set(df.columns)}")
        return []

    df = df.copy()
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True, errors="coerce")
    df = df.sort_values("event_ts").reset_index(drop=True)

    # Prefer FILL events; fall back to ORDER_SUBMIT if none present
    fills = df[df["event_type"] == "FILL"] if "FILL" in df["event_type"].values else df[df["event_type"] == "ORDER_SUBMIT"]

    if fills.empty:
        return []

    trades: list[dict[str, Any]] = []

    for symbol, grp in fills.groupby("symbol", sort=False):
        grp = grp.sort_values("event_ts").reset_index(drop=True)
        open_positions: list[dict[str, Any]] = []

        for _, row in grp.iterrows():
            qty = float(row["qty"]) if not pd.isna(row["qty"]) else 0.0
            price = float(row["price"]) if not pd.isna(row["price"]) else np.nan
            ts = row["event_ts"]

            if qty > 0:
                # Entry (buy)
                open_positions.append({"entry_ts": ts, "entry_price": price, "qty": qty})
            elif qty < 0 and open_positions:
                # Exit (sell) -- match FIFO
                entry = open_positions.pop(0)
                if np.isnan(entry["entry_price"]) or np.isnan(price):
                    continue
                entry_price = entry["entry_price"]
                exit_price = price
                holding_days = max(1, (ts - entry["entry_ts"]).days)
                realized_return = (exit_price - entry_price) / entry_price if entry_price != 0 else 0.0
                trades.append({
                    "entry_date": pd.Timestamp(entry["entry_ts"]).normalize().tz_localize(None),
                    "exit_date": pd.Timestamp(ts).normalize().tz_localize(None),
                    "symbol": symbol,
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(exit_price, 6),
                    "realized_return": round(realized_return, 8),
                    "holding_days": holding_days,
                })

    return trades


def _load_all_trades(bt_root: Path) -> pd.DataFrame:
    """Scan all ledger_backtest_* dirs under bt_root and harvest trades."""
    ledger_dirs = sorted(bt_root.glob("ledger_backtest_*"))
    _log(f"Found {len(ledger_dirs)} ledger directories under {bt_root}")

    all_trades: list[dict[str, Any]] = []

    for ld in ledger_dirs:
        ledger_file = ld / "ledger_events.parquet"
        if not ledger_file.exists():
            continue
        trades = _reconstruct_trades_from_ledger(ledger_file)
        if trades:
            for t in trades:
                t["source_run"] = ld.name
            all_trades.extend(trades)

    _log(f"Total raw trades harvested: {len(all_trades)}")

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.drop_duplicates(subset=["entry_date", "exit_date", "symbol", "entry_price", "exit_price"])
    _log(f"Unique trades after dedup: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Factor panel join
# ---------------------------------------------------------------------------

def _join_with_panel(trades_df: pd.DataFrame, panel_path: Path) -> pd.DataFrame:
    """Left-join trades with factor panel at entry_date x symbol."""
    if not panel_path.exists():
        _warn(f"Panel not found at {panel_path} -- returning trades without features")
        return trades_df

    _log(f"Loading factor panel from {panel_path}")
    try:
        panel = pd.read_parquet(panel_path)
    except Exception as exc:
        _warn(f"Failed to load panel: {exc!r}")
        return trades_df

    # Normalise date column
    date_col = None
    for c in ("timestamp", "date"):
        if c in panel.columns:
            date_col = c
            break
    if date_col is None:
        _warn("Panel has no 'timestamp' or 'date' column -- skipping join")
        return trades_df

    panel = panel.copy()
    panel[date_col] = pd.to_datetime(panel[date_col]).dt.normalize()
    panel = panel.rename(columns={date_col: "_join_date"})

    if "symbol" not in panel.columns:
        _warn("Panel has no 'symbol' column -- skipping join")
        return trades_df

    # Keep only feature-like numeric columns (exclude OHLCV meta)
    exclude = {
        "_join_date", "symbol", "open", "high", "low", "close", "volume",
        "date", "timestamp",
    }
    feature_cols = [c for c in panel.columns if c not in exclude and pd.api.types.is_numeric_dtype(panel[c])]
    _log(f"Panel feature columns available: {len(feature_cols)}")

    panel_slim = panel[["_join_date", "symbol"] + feature_cols].copy()

    trades_df = trades_df.copy()
    trades_df["_join_date"] = pd.to_datetime(trades_df["entry_date"]).dt.normalize()

    merged = trades_df.merge(panel_slim, on=["_join_date", "symbol"], how="left")
    merged = merged.drop(columns=["_join_date"])

    n_matched = int(merged[feature_cols[0]].notna().sum()) if feature_cols else 0
    _log(f"Panel join: {len(merged)} rows, {n_matched} with at least one feature value")

    return merged


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def _add_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary label: 1 if realized_return > 0, else 0."""
    df = df.copy()
    df["label"] = (df["realized_return"] > 0).astype(int)
    pos = int(df["label"].sum())
    neg = int((df["label"] == 0).sum())
    _log(f"Labels: {pos} positive ({pos / max(len(df), 1) * 100:.1f}%), {neg} negative")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def harvest_labels(
    bt_root: Path,
    panel_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Full harvest pipeline: scan -> reconstruct -> join -> label -> save."""
    _log("=" * 60)
    _log("harvest_backtest_labels.py -- START")
    _log(f"bt_root     : {bt_root}")
    _log(f"panel_path  : {panel_path}")
    _log(f"output_path : {output_path}")
    _log("=" * 60)

    # 1. Load all trades
    trades_df = _load_all_trades(bt_root)
    if trades_df.empty:
        _warn("No trades found -- output will be empty.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=[
            "entry_date", "exit_date", "symbol",
            "entry_price", "exit_price", "realized_return",
            "holding_days", "label", "source_run",
        ])
        empty.to_parquet(output_path, index=False)
        _log(f"Empty parquet written to {output_path}")
        return empty

    # 2. Join with factor panel
    labelled_df = _join_with_panel(trades_df, panel_path)

    # 3. Add label column
    labelled_df = _add_label(labelled_df)

    # 4. Sort and save
    labelled_df = labelled_df.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labelled_df.to_parquet(output_path, index=False)
    _log(f"[OK] Written {len(labelled_df)} labelled trades to {output_path}")
    _log(f"Columns: {labelled_df.columns.tolist()}")

    # Summary stats
    if "realized_return" in labelled_df.columns:
        rr = labelled_df["realized_return"]
        _log(
            f"Return stats: mean={rr.mean():.4f}, "
            f"median={rr.median():.4f}, "
            f"std={rr.std():.4f}, "
            f"min={rr.min():.4f}, "
            f"max={rr.max():.4f}"
        )

    _log("harvest_backtest_labels.py -- DONE")
    return labelled_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Harvest labelled trade data from backtest accounting reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bt-root",
        type=Path,
        default=Path("output"),
        help="Root directory containing ledger_backtest_* subdirs (default: output/).",
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=Path("output/factor_panels/full_panel_7y.parquet"),
        help="Path to factor panel parquet for feature join.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("output/training_data/backtest_trade_labels.parquet"),
        help="Destination parquet path.",
    )
    args = parser.parse_args(argv)

    harvest_labels(
        bt_root=args.bt_root.resolve(),
        panel_path=args.panel_path.resolve(),
        output_path=args.output_path.resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
