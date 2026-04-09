"""BENCH-0: EOD benchmark strategy — EMA20/EMA60 with score-based sizing and exit signals."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_signals(
    prices_df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 60,
) -> pd.DataFrame:
    """Generate LONG signals when EMA fast > EMA slow (per symbol, last bar).

    Score is the normalized EMA spread: (ema_fast - ema_slow) / ema_slow.
    Larger spread = stronger trend = higher score.

    Args:
        prices_df: DataFrame with columns timestamp, symbol, close.
        ema_fast: Fast EMA span.
        ema_slow: Slow EMA span.

    Returns:
        DataFrame with columns: timestamp, symbol, direction, score.
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
        fast_val = ema_f.iloc[last_idx]
        slow_val = ema_s.iloc[last_idx]
        if fast_val > slow_val and slow_val > 0:
            ts = g["timestamp"].iloc[last_idx]
            # Score = normalized EMA spread (typically 0.01 to 0.15)
            spread = (fast_val - slow_val) / slow_val
            out.append(
                {"timestamp": ts, "symbol": sym, "direction": "LONG", "score": spread}
            )

    if not out:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    return pd.DataFrame(out)


def compute_target_positions(
    signals: pd.DataFrame,
    total_capital: float,
    equal_weight: bool = True,
    prices_latest: pd.DataFrame | None = None,
    max_positions: int = 0,
    min_position_weight: float = 0.0,
    target_invested_pct: float = 1.0,
) -> pd.DataFrame:
    """Compute target positions from signals.

    Supports both equal-weight and score-proportional sizing.

    Args:
        signals: DataFrame with columns symbol, score, (optional direction).
        total_capital: Total capital to allocate.
        equal_weight: If True, each symbol gets 1/n weight. If False, weight by score.
        prices_latest: Not used (kept for API compatibility).
        max_positions: If > 0, limit to top-N symbols by score.
        min_position_weight: Minimum weight per position (e.g. 0.03 = 3%).
        target_invested_pct: Fraction of capital to invest (e.g. 0.80 = 80%).

    Returns:
        DataFrame with columns: symbol, target_weight, target_qty.
        target_qty is in NOTIONAL (dollar amount), NOT shares.
    """
    empty = pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    if signals is None or signals.empty:
        return empty
    if "symbol" not in signals.columns:
        return empty

    # Sort by score descending to pick best signals
    if "score" in signals.columns:
        sig = signals.sort_values("score", ascending=False).copy()
    else:
        sig = signals.copy()
        sig["score"] = 1.0

    # Limit to top-N positions
    if max_positions > 0 and len(sig) > max_positions:
        sig = sig.head(max_positions)

    syms = sig["symbol"].drop_duplicates().tolist()
    scores = sig.set_index("symbol")["score"].to_dict()
    if not syms:
        return empty

    n = len(syms)
    available_capital = total_capital * min(target_invested_pct, 1.0)

    if equal_weight:
        weight = 1.0 / n
        weights = {sym: weight for sym in syms}
    else:
        # Score-proportional weights
        total_score = sum(scores.get(s, 0.0) for s in syms)
        if total_score <= 0:
            weight = 1.0 / n
            weights = {sym: weight for sym in syms}
        else:
            weights = {
                sym: scores.get(sym, 0.0) / total_score for sym in syms
            }

    # Apply min_position_weight: drop positions below minimum
    if min_position_weight > 0:
        filtered_syms = [s for s in syms if weights[s] >= min_position_weight]
        if not filtered_syms:
            # All below minimum: take top-N that would meet minimum
            max_possible = int(1.0 / min_position_weight)
            filtered_syms = syms[:max_possible] if max_possible > 0 else syms[:1]
        syms = filtered_syms
        # Rebalance weights for remaining symbols
        n = len(syms)
        if equal_weight:
            weights = {sym: 1.0 / n for sym in syms}
        else:
            total_score = sum(scores.get(s, 0.0) for s in syms)
            if total_score > 0:
                weights = {
                    sym: scores.get(sym, 0.0) / total_score for sym in syms
                }
            else:
                weights = {sym: 1.0 / n for sym in syms}

    # Scale weights by target_invested_pct
    rows = []
    for sym in syms:
        w = weights[sym] * min(target_invested_pct, 1.0)
        # target_qty is NOTIONAL (dollar amount)
        rows.append({
            "symbol": sym,
            "target_weight": w,
            "target_qty": available_capital * weights[sym],
        })

    return pd.DataFrame(rows)


def check_exit_signals(
    current_positions: dict,
    prices_latest: pd.DataFrame,
    strategy_cfg: dict | None = None,
) -> pd.DataFrame:
    """Check exit conditions for current positions.

    Exit signals are generated for positions that hit stop-loss, trailing-stop,
    or take-profit thresholds.

    Args:
        current_positions: dict of {symbol: {qty, avg_price, hwm (optional)}}.
        prices_latest: DataFrame with columns symbol, close.
        strategy_cfg: Strategy config with stop_loss_pct, trailing_stop_pct, take_profit_pct.

    Returns:
        DataFrame with columns: symbol, direction, exit_reason, exit_qty_pct.
        direction is always "FLAT" (sell signal).
        exit_qty_pct: 1.0 for full exit, 0.5 for partial (take-profit).
    """
    cfg = strategy_cfg or {}
    stop_loss_pct = float(cfg.get("stop_loss_pct", 0.0))
    trailing_stop_pct = float(cfg.get("trailing_stop_pct", 0.0))
    take_profit_pct = float(cfg.get("take_profit_pct", 0.0))

    if not current_positions or prices_latest is None or prices_latest.empty:
        return pd.DataFrame(
            columns=["symbol", "direction", "exit_reason", "exit_qty_pct"]
        )

    price_map = {}
    if "symbol" in prices_latest.columns and "close" in prices_latest.columns:
        price_map = dict(
            zip(prices_latest["symbol"].values, prices_latest["close"].values)
        )

    exits = []
    for sym, pos in current_positions.items():
        qty = float(pos.get("qty", 0))
        if qty <= 0:
            continue
        avg_price = float(pos.get("avg_price", 0))
        hwm = float(pos.get("hwm", avg_price))
        if avg_price <= 0:
            continue

        current_price = float(price_map.get(sym, 0))
        if current_price <= 0:
            continue

        # Update HWM
        if current_price > hwm:
            hwm = current_price

        # Check stop-loss (hard stop below entry)
        if stop_loss_pct > 0:
            stop_price = avg_price * (1 - stop_loss_pct)
            if current_price <= stop_price:
                exits.append({
                    "symbol": sym,
                    "direction": "FLAT",
                    "exit_reason": f"stop_loss ({current_price:.2f} <= {stop_price:.2f})",
                    "exit_qty_pct": 1.0,
                })
                continue

        # Check trailing stop (below HWM)
        if trailing_stop_pct > 0 and hwm > avg_price:
            trail_price = hwm * (1 - trailing_stop_pct)
            if current_price <= trail_price:
                exits.append({
                    "symbol": sym,
                    "direction": "FLAT",
                    "exit_reason": f"trailing_stop ({current_price:.2f} <= {trail_price:.2f}, hwm={hwm:.2f})",
                    "exit_qty_pct": 1.0,
                })
                continue

        # Check take-profit (partial sell at +N%)
        if take_profit_pct > 0:
            tp_price = avg_price * (1 + take_profit_pct)
            if current_price >= tp_price:
                exits.append({
                    "symbol": sym,
                    "direction": "FLAT",
                    "exit_reason": f"take_profit ({current_price:.2f} >= {tp_price:.2f})",
                    "exit_qty_pct": 0.5,  # Sell 50%, let rest run with trailing stop
                })
                continue

    if not exits:
        return pd.DataFrame(
            columns=["symbol", "direction", "exit_reason", "exit_qty_pct"]
        )
    return pd.DataFrame(exits)


__all__ = ["compute_signals", "compute_target_positions", "check_exit_signals"]
