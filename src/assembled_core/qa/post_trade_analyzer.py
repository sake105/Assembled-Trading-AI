"""Post-Trade Analyzer — M11: Post-Trade Learning Loop.

Analyzes completed trades against actual price outcomes to compute signal quality metrics
and feed learning records into the learning store.

PIT-safe: only uses price data available after trade close.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Default forward-return horizon in calendar days
DEFAULT_HORIZON_DAYS = 5


def compute_forward_returns(
    prices_df: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Compute N-day forward returns for each symbol from a price DataFrame.

    Args:
        prices_df: DataFrame with columns: timestamp (UTC), symbol, close
        horizon_days: Number of calendar days to look forward.

    Returns:
        DataFrame with columns: timestamp, symbol, close, forward_return
        forward_return = close_t+horizon / close_t - 1 (NaN if future price missing)
    """
    required = {"timestamp", "symbol", "close"}
    if not required.issubset(prices_df.columns):
        missing = required - set(prices_df.columns)
        raise ValueError(f"prices_df missing columns: {missing}")

    df = prices_df[["timestamp", "symbol", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    rows = []
    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp").copy()
        grp["future_ts"] = grp["timestamp"] + pd.Timedelta(days=horizon_days)
        # Map future_ts to closest available close
        ts_idx = grp.set_index("timestamp")["close"]
        for _, row in grp.iterrows():
            future_ts = row["future_ts"]
            # Find closest price at or after future_ts
            candidates = ts_idx[ts_idx.index >= future_ts]
            if candidates.empty:
                fwd = float("nan")
            else:
                fwd = candidates.iloc[0] / row["close"] - 1.0
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "symbol": symbol,
                    "close": row["close"],
                    "forward_return": fwd,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "close", "forward_return"])
    return pd.DataFrame(rows)


def compute_signal_hit_rate(
    trades_df: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    *,
    tolerance_days: int = 2,
) -> pd.DataFrame:
    """Match trades to forward returns and compute hit rate per symbol.

    A trade is a "hit" if:
    - BUY and forward_return > 0
    - SELL and forward_return < 0

    Args:
        trades_df: DataFrame with columns: symbol, side (BUY/SELL), event_ts, qty, price
        forward_returns_df: Output from compute_forward_returns().
        tolerance_days: Max days difference for timestamp matching.

    Returns:
        DataFrame with columns: symbol, total_trades, hits, hit_rate, avg_forward_return
    """
    if trades_df.empty or forward_returns_df.empty:
        return pd.DataFrame(
            columns=["symbol", "total_trades", "hits", "hit_rate", "avg_forward_return"]
        )

    trades = trades_df.copy()
    trades["event_ts"] = pd.to_datetime(trades["event_ts"], utc=True)

    fwd = forward_returns_df.copy()
    fwd["timestamp"] = pd.to_datetime(fwd["timestamp"], utc=True)

    results = []
    for symbol in trades["symbol"].unique():
        sym_trades = trades[trades["symbol"] == symbol]
        sym_fwd = fwd[fwd["symbol"] == symbol].set_index("timestamp")["forward_return"]

        hits = 0
        total = 0
        fwd_returns = []

        for _, trade in sym_trades.iterrows():
            ts = trade["event_ts"]
            side = str(trade.get("side", "")).upper()
            if not side:
                # Try event_type: FILL implies BUY from sign of qty
                qty = float(trade.get("qty", 0))
                side = "BUY" if qty > 0 else "SELL"

            # Find closest forward return within tolerance
            tol = pd.Timedelta(days=tolerance_days)
            candidates = sym_fwd[
                (sym_fwd.index >= ts - tol) & (sym_fwd.index <= ts + tol)
            ]
            if candidates.empty:
                continue

            # Use the closest timestamp (not just first in sorted order)
            time_diffs = pd.Series(
                [(abs((idx - ts).total_seconds())) for idx in candidates.index],
                index=candidates.index,
            )
            closest_idx = int(time_diffs.argmin())
            fwd_ret = candidates.iloc[closest_idx]
            if pd.isna(fwd_ret):
                continue

            fwd_returns.append(fwd_ret)
            total += 1
            if (side == "BUY" and fwd_ret > 0) or (side == "SELL" and fwd_ret < 0):
                hits += 1

        if total > 0:
            results.append(
                {
                    "symbol": symbol,
                    "total_trades": total,
                    "hits": hits,
                    "hit_rate": hits / total,
                    "avg_forward_return": sum(fwd_returns) / len(fwd_returns),
                }
            )

    if not results:
        return pd.DataFrame(
            columns=["symbol", "total_trades", "hits", "hit_rate", "avg_forward_return"]
        )
    return pd.DataFrame(results).sort_values("symbol").reset_index(drop=True)


def build_learning_record(
    run_id: str,
    analysis_date: str,
    hit_rate_df: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured learning record for storage.

    Args:
        run_id: Unique run identifier.
        analysis_date: ISO date string (YYYY-MM-DD).
        hit_rate_df: Output from compute_signal_hit_rate().
        horizon_days: Forward horizon used.
        extra: Optional extra fields.

    Returns:
        Dict suitable for JSON serialization.
    """
    overall_hits = int(hit_rate_df["hits"].sum()) if not hit_rate_df.empty else 0
    overall_total = (
        int(hit_rate_df["total_trades"].sum()) if not hit_rate_df.empty else 0
    )
    overall_hit_rate = overall_hits / overall_total if overall_total > 0 else 0.0

    record: dict[str, Any] = {
        "run_id": run_id,
        "analysis_date": analysis_date,
        "horizon_days": horizon_days,
        "overall_hit_rate": round(overall_hit_rate, 4),
        "overall_total_trades": overall_total,
        "per_symbol": [],
    }

    if not hit_rate_df.empty:
        record["per_symbol"] = [
            {
                "symbol": row["symbol"],
                "total_trades": int(row["total_trades"]),
                "hits": int(row["hits"]),
                "hit_rate": round(float(row["hit_rate"]), 4),
                "avg_forward_return": round(float(row["avg_forward_return"]), 4),
            }
            for _, row in hit_rate_df.iterrows()
        ]

    if extra:
        record.update(extra)

    return record
