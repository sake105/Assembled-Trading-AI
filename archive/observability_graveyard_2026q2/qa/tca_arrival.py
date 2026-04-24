# src/assembled_core/qa/tca_arrival.py
"""TCA: Implementation Shortfall vs Arrival Price (Sprint 2 / C11).

Pure sidecar module. Does not modify ``qa/tca.py``. Computes per-fill
implementation shortfall in basis points against an arrival-price
benchmark (mid-price at the decision timestamp).

Sign convention
---------------
    is_bps = (fill_price - arrival_price) / arrival_price * 10000 * sign(side)

with ``sign(BUY) = +1`` and ``sign(SELL) = -1``.

Interpretation
--------------
* Positive ``is_bps`` = **unfavorable** slippage
  (BUY filled above arrival, or SELL filled below arrival).
* Negative ``is_bps`` = **favorable** slippage
  (BUY filled below arrival, or SELL filled above arrival).

This module is additive. It does not write ``arrival_price`` anywhere
upstream; persisting the decision-time arrival price into the order
record is a separate future item (see plan C11 follow-up). This module
only *consumes* the column when it is provided.

Layering: ``qa`` layer (analysis/reporting). No side effects, no I/O.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_REQUIRED_FILL_COLS = ("timestamp", "symbol", "side", "qty", "fill_price")
_REQUIRED_ARRIVAL_COLS = ("timestamp", "symbol", "arrival_price")

_OUTPUT_COLS = (
    "timestamp",
    "symbol",
    "side",
    "qty",
    "fill_price",
    "arrival_price",
    "is_bps",
)


def _empty_output() -> pd.DataFrame:
    """Return an empty output frame with the canonical column schema."""
    return pd.DataFrame({col: pd.Series(dtype="object") for col in _OUTPUT_COLS})


def _validate_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _sign_from_side(side: Any) -> float:
    """Map side string to +1 (BUY) or -1 (SELL). Case-insensitive.

    Unknown sides produce NaN so they surface as unmatched rather than
    silently producing a wrong sign.
    """
    if not isinstance(side, str):
        return float("nan")
    s = side.strip().upper()
    if s == "BUY":
        return 1.0
    if s == "SELL":
        return -1.0
    return float("nan")


def compute_implementation_shortfall(
    fills: pd.DataFrame,
    arrival_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Compute implementation shortfall per fill in basis points.

    Parameters
    ----------
    fills : pd.DataFrame
        Required columns: ``timestamp, symbol, side, qty, fill_price``.
        ``side`` is ``"BUY"`` or ``"SELL"`` (case-insensitive).
    arrival_prices : pd.DataFrame
        Required columns: ``timestamp, symbol, arrival_price``.
        ``arrival_price`` is the mid-price at the decision timestamp.

    Returns
    -------
    pd.DataFrame
        ``fills`` merged with ``arrival_price`` and ``is_bps``. Rows
        without a matching arrival price (or with zero / NaN arrival
        price, or unparseable side) receive ``is_bps = NaN``. The merge
        is an exact left-join on ``(timestamp, symbol)`` — no dropping.
    """
    if fills is None or len(fills) == 0:
        return _empty_output()

    _validate_columns(fills, _REQUIRED_FILL_COLS, "fills")

    if arrival_prices is None or len(arrival_prices) == 0:
        # No arrivals → every fill is unmatched, is_bps = NaN.
        out = fills.loc[:, list(_REQUIRED_FILL_COLS)].copy()
        out["arrival_price"] = np.nan
        out["is_bps"] = np.nan
        return out.loc[:, list(_OUTPUT_COLS)]

    _validate_columns(arrival_prices, _REQUIRED_ARRIVAL_COLS, "arrival_prices")

    # De-duplicate arrivals on (timestamp, symbol) to make the left-join
    # strictly 1:1 and deterministic. Keep the first record per key.
    arrivals = (
        arrival_prices.loc[:, list(_REQUIRED_ARRIVAL_COLS)]
        .drop_duplicates(subset=["timestamp", "symbol"], keep="first")
    )

    merged = fills.loc[:, list(_REQUIRED_FILL_COLS)].merge(
        arrivals,
        how="left",
        on=["timestamp", "symbol"],
        validate="many_to_one",
    )

    arrival = pd.to_numeric(merged["arrival_price"], errors="coerce")
    fill_px = pd.to_numeric(merged["fill_price"], errors="coerce")

    # Guard: zero or non-finite arrival → NaN (no divide-by-zero).
    safe_arrival = arrival.where(
        arrival.notna() & np.isfinite(arrival) & (arrival != 0.0)
    )

    sign = merged["side"].map(_sign_from_side).astype(float)

    is_bps = (fill_px - safe_arrival) / safe_arrival * 10000.0 * sign

    out = merged.copy()
    out["is_bps"] = is_bps
    return out.loc[:, list(_OUTPUT_COLS)]


def summarize_implementation_shortfall(
    shortfall_df: pd.DataFrame,
) -> dict:
    """Aggregate implementation-shortfall metrics.

    Parameters
    ----------
    shortfall_df : pd.DataFrame
        Output of :func:`compute_implementation_shortfall`.

    Returns
    -------
    dict
        ``{
            "n_fills":   int,
            "n_buy":     int,
            "n_sell":    int,
            "matched":   int,
            "unmatched": list[tuple],   # (timestamp, symbol, side) of NaN rows
            "mean_bps":   float | nan,
            "median_bps": float | nan,
            "p95_bps":    float | nan,
            "per_symbol": dict[str, dict[str, float]],
        }``

    Mean/median/p95 are computed only over matched rows. For an empty
    frame or no matched rows these are ``NaN`` (not an error).
    """
    n_fills = 0 if shortfall_df is None else int(len(shortfall_df))

    if shortfall_df is None or n_fills == 0:
        return {
            "n_fills": 0,
            "n_buy": 0,
            "n_sell": 0,
            "matched": 0,
            "unmatched": [],
            "mean_bps": float("nan"),
            "median_bps": float("nan"),
            "p95_bps": float("nan"),
            "per_symbol": {},
        }

    sides_upper = shortfall_df["side"].astype(str).str.upper()
    n_buy = int((sides_upper == "BUY").sum())
    n_sell = int((sides_upper == "SELL").sum())

    is_bps = pd.to_numeric(shortfall_df["is_bps"], errors="coerce")
    matched_mask = is_bps.notna()
    matched_count = int(matched_mask.sum())

    unmatched_rows = shortfall_df.loc[~matched_mask, ["timestamp", "symbol", "side"]]
    unmatched_list = [
        (r.timestamp, r.symbol, r.side) for r in unmatched_rows.itertuples(index=False)
    ]

    if matched_count == 0:
        mean_bps = float("nan")
        median_bps = float("nan")
        p95_bps = float("nan")
    else:
        matched_vals = is_bps[matched_mask]
        mean_bps = float(matched_vals.mean())
        median_bps = float(matched_vals.median())
        p95_bps = float(np.percentile(matched_vals.to_numpy(), 95))

    per_symbol: dict[str, dict[str, float]] = {}
    if matched_count > 0:
        for sym, grp in shortfall_df.loc[matched_mask].groupby("symbol", sort=True):
            vals = pd.to_numeric(grp["is_bps"], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            per_symbol[str(sym)] = {
                "n_fills": int(len(vals)),
                "mean_bps": float(vals.mean()),
                "median_bps": float(vals.median()),
            }

    return {
        "n_fills": n_fills,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "matched": matched_count,
        "unmatched": unmatched_list,
        "mean_bps": mean_bps,
        "median_bps": median_bps,
        "p95_bps": p95_bps,
        "per_symbol": per_symbol,
    }


__all__ = [
    "compute_implementation_shortfall",
    "summarize_implementation_shortfall",
]
