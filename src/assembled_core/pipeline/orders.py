# src/assembled_core/pipeline/orders.py
"""Order generation from trading signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.pipeline.io import ensure_cols


def signals_to_orders(signals: pd.DataFrame) -> pd.DataFrame:
    """Convert signals to orders (on signal changes).

    Args:
        signals: DataFrame with columns: timestamp, symbol, sig, price
        sig: -1 (SELL), 0 (neutral), +1 (BUY)

    Returns:
        DataFrame with columns: timestamp, symbol, side, qty, price
        side: "BUY" or "SELL"
        qty: Quantity (default 1.0)
        Sorted by timestamp, then symbol
    """
    orders = (
        signals.groupby("symbol", group_keys=False)
        .apply(_gen_orders_for_symbol, include_groups=False)
        .reset_index(drop=True)
    )

    # Validate and normalize
    for c in ["timestamp", "symbol", "side", "qty", "price"]:
        if c not in orders.columns:
            raise KeyError(f"Orders-Spalte fehlt: {c}")

    orders["timestamp"] = pd.to_datetime(orders["timestamp"], utc=True)
    orders["qty"] = pd.to_numeric(orders["qty"], errors="coerce").astype("float64")
    orders["price"] = pd.to_numeric(orders["price"], errors="coerce").astype("float64")
    orders["side"] = orders["side"].astype("string")
    orders["symbol"] = orders["symbol"].astype("string")
    orders = orders.dropna(subset=["timestamp", "symbol", "side", "qty", "price"])
    orders = orders.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return orders


def _gen_orders_for_symbol(d: pd.DataFrame) -> pd.DataFrame:
    """Generate orders for a single symbol from signals.

    Args:
        d: DataFrame with columns: timestamp, symbol, sig, price

    Returns:
        DataFrame with columns: timestamp, symbol, side, qty, price
    """
    # Wenn include_groups=False, fehlt 'symbol' → aus d.name rekonstruieren
    if "symbol" not in d.columns:
        d = d.assign(symbol=d.name)

    d = d.sort_values("timestamp").reset_index(drop=True)
    d = ensure_cols(d, ["timestamp", "symbol", "sig", "price"])

    sig = d["sig"].fillna(0).astype("int8")
    sig_prev = sig.shift(1).fillna(0).astype("int8")
    delta = sig - sig_prev

    chg = d.loc[delta != 0, ["timestamp", "symbol", "price"]].copy()
    chg["side"] = np.where(sig.loc[chg.index] > sig_prev.loc[chg.index], "BUY", "SELL")
    chg["qty"] = 1.0
    return chg[["timestamp", "symbol", "side", "qty", "price"]].reset_index(drop=True)


def write_orders(
    orders: pd.DataFrame, freq: str, output_dir: Path | str | None = None
) -> Path:
    """Write orders to CSV file.

    Args:
        orders: DataFrame with columns: timestamp, symbol, side, qty, price
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)

    Returns:
        Path to written CSV file

    Side effects:
        Creates output directory if it doesn't exist
        Writes CSV file: output_dir/orders_{freq}.csv
    """
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"orders_{freq}.csv"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        orders.to_csv(path, index=False)
    except (IOError, OSError) as exc:
        raise RuntimeError(f"Failed to write orders CSV to {path}") from exc
    return path
