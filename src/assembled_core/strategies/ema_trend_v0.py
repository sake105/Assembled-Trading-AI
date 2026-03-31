"""BENCH-0: Minimal EOD benchmark strategy — EMA20/EMA60 long-only."""

from __future__ import annotations

import pandas as pd


def compute_signals(
    prices_df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 60,
) -> pd.DataFrame:
    """Generate LONG signals when EMA fast > EMA slow (per symbol, last bar).

    Args:
        prices_df: DataFrame with columns timestamp, symbol, close (and optionally other cols).
        ema_fast: Fast EMA span.
        ema_slow: Slow EMA span.

    Returns:
        DataFrame with columns: timestamp, symbol, direction, score.
        Only rows with direction="LONG" (score=1.0). Empty if no signals.
    """
    if (
        prices_df.empty
        or "close" not in prices_df.columns
        or "symbol" not in prices_df.columns
    ):
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    if "timestamp" not in prices_df.columns:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

    out = []
    for sym, grp in prices_df.groupby("symbol", group_keys=False):
        g = grp.sort_values("timestamp").reset_index(drop=True)
        if len(g) < ema_slow:
            continue
        close = pd.to_numeric(g["close"], errors="coerce").ffill()
        ema_f = close.ewm(span=ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=ema_slow, adjust=False).mean()
        last_idx = len(g) - 1
        if ema_f.iloc[last_idx] > ema_s.iloc[last_idx]:
            ts = g["timestamp"].iloc[last_idx]
            out.append(
                {"timestamp": ts, "symbol": sym, "direction": "LONG", "score": 1.0}
            )

    if not out:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    return pd.DataFrame(out)


def compute_target_positions(
    signals: pd.DataFrame,
    total_capital: float,
    equal_weight: bool = True,
    prices_latest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute target positions from signals. Equal weight across signaled symbols.

    Args:
        signals: DataFrame with columns symbol, (optional timestamp, direction, score).
        total_capital: Total capital to allocate.
        equal_weight: If True, each symbol gets 1/n weight.
        prices_latest: Optional DataFrame with columns symbol, close for target_qty.
            If None, target_qty is set to 0.

    Returns:
        DataFrame with columns: symbol, target_weight, target_qty.
    """
    empty = pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    if signals is None or signals.empty:
        return empty
    if "symbol" not in signals.columns:
        return empty

    syms = signals["symbol"].drop_duplicates().tolist()
    if not syms:
        return empty

    n = len(syms)
    weight = 1.0 / n if equal_weight and n else 0.0
    rows = []
    for sym in syms:
        row = {"symbol": sym, "target_weight": weight, "target_qty": 0.0}
        if (
            prices_latest is not None
            and not prices_latest.empty
            and "close" in prices_latest.columns
            and "symbol" in prices_latest.columns
        ):
            sub = prices_latest[prices_latest["symbol"] == sym]
            if not sub.empty:
                close = float(
                    pd.to_numeric(sub["close"].iloc[-1], errors="coerce") or 0
                )
                if close > 0:
                    row["target_qty"] = (total_capital * weight) / close
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["compute_signals", "compute_target_positions"]
