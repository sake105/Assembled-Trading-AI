"""trend_baseline strategy module — MA-crossover trend follower with exits.

Originally a script-level helper in scripts/run_backtest_strategy.py (used as
the +43.02% CAGR / 1.44 Sharpe OOS 2025-01..05 baseline against which mfv2
underperformed). Promoted to a first-class paper strategy module on
2026-05-21 as part of §9.6 (b) Phase-2 — switching the live pilot's primary
strategy from mfv2 to trend_baseline.

Strategy contract (matches ema_trend_v0 + multifactor_v2 interface):
  - compute_signals(prices, ...) -> DataFrame(timestamp, symbol, direction, score)
  - compute_target_positions(signals, capital, ...) -> DataFrame(symbol, target_weight, target_qty)
  - check_exit_signals(current_positions, prices_latest, strategy_cfg)
        -> DataFrame(symbol, direction, exit_reason, exit_qty_pct)

Exit logic (Phase-2 pre-condition (i) closure 2026-05-21):
  - Stop-loss: hard exit when current_price <= avg_price * (1 - stop_loss_pct)
  - Trailing-stop: exit when price drops `trailing_stop_pct` below the
    high-water-mark per position (HWM tracked in ledger.positions[sym].hwm)
  - Take-profit: partial 50% exit at avg_price * (1 + take_profit_pct);
    remaining stays under trailing-stop protection
  - LONG → FLAT MA-crossover already handled by compute_signals output

Without check_exit_signals, trend_baseline relied solely on MA-flip to close
positions. For a $91k pilot already at -7.5% drawdown that is unacceptable
risk-bearing — Phase-2 promotion is gated on this exit-discipline.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.assembled_core.signals.rules_trend import (
    generate_trend_signals_from_prices,
)

logger = logging.getLogger(__name__)


def compute_signals(
    prices: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 60,
    volume_threshold: float | None = None,
    min_volume_multiplier: float = 1.0,
) -> pd.DataFrame:
    """MA-crossover trend signal — LATEST-BAR slice per symbol.

    Internally calls rules_trend.generate_trend_signals_from_prices to get the
    state-based LONG/FLAT signal for every (date, symbol) row, then reduces to
    one row per symbol (the latest bar) matching the contract used by
    ema_trend_v0.compute_signals and multifactor_v2.compute_signals. This is
    what the paper trading_cycle expects — the FULL signal history would cause
    sizing to split capital across thousands of historical signal rows.

    Args:
        prices: panel with timestamp, symbol, close, optional volume.
        ma_fast / ma_slow: window sizes for the fast/slow moving averages.
        volume_threshold / min_volume_multiplier: optional volume gate.

    Returns:
        DataFrame with one row per symbol where direction is LONG at the
        latest bar. Schema: timestamp, symbol, direction (always LONG in the
        output — FLAT rows are dropped), score (the MA spread or 1.0 if
        unavailable). Matches ema_trend_v0.compute_signals contract.
    """
    full = generate_trend_signals_from_prices(
        prices,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        volume_threshold=volume_threshold,
        min_volume_multiplier=min_volume_multiplier,
    )
    if full is None or full.empty or "direction" not in full.columns:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

    # Latest bar per symbol; keep only LONGs (FLAT means no position).
    latest = (
        full.sort_values("timestamp")
        .groupby("symbol", group_keys=False)
        .tail(1)
        .reset_index(drop=True)
    )
    longs = latest[latest["direction"] == "LONG"].copy()
    if longs.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    return longs[["timestamp", "symbol", "direction", "score"]]


def compute_target_positions(
    signals: pd.DataFrame,
    capital: float,
    *,
    max_positions: int = 0,
    target_invested_pct: float = 1.0,
) -> pd.DataFrame:
    """Equal-weight sizing on LONG signals.

    Args:
        signals: DataFrame with at least symbol + direction; score used to
            rank when max_positions cap binds.
        capital: total capital base used for weight→qty derivation downstream.
        max_positions: optional cap; when > 0 and len(LONG) > cap, keep
            top-N by score.
        target_invested_pct: fraction of capital to deploy total across LONGs.

    Returns:
        DataFrame with columns symbol, target_weight, target_qty (qty=0 — the
        downstream sizing pipeline turns target_weight into qty using
        current prices).
    """
    if signals is None or signals.empty or "direction" not in signals.columns:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])

    longs = signals[signals["direction"] == "LONG"].copy()
    if longs.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])

    if max_positions > 0 and len(longs) > max_positions:
        if "score" in longs.columns:
            longs = longs.sort_values("score", ascending=False).head(max_positions)
        else:
            longs = longs.head(max_positions)

    n = len(longs)
    weight = float(target_invested_pct) / float(n) if n > 0 else 0.0
    return pd.DataFrame(
        {
            "symbol": longs["symbol"].values,
            "target_weight": [weight] * n,
            "target_qty": [0.0] * n,
        }
    )


def check_exit_signals(
    current_positions: dict,
    prices_latest: pd.DataFrame | None,
    strategy_cfg: dict | None = None,
) -> pd.DataFrame:
    """Exit-discipline for trend_baseline positions (Phase-2 pre-cond (i)).

    Mirrors the ema_trend_v0 contract: stop-loss, trailing-stop on HWM,
    partial take-profit. Returns FLAT signals with exit_reason + exit_qty_pct.

    Args:
        current_positions: dict ``{symbol: {qty, avg_price, hwm (optional)}}``
            from the paper ledger state.
        prices_latest: DataFrame with at least symbol + close (one row per
            symbol — typically ``groupby('symbol').last()`` of the daily
            panel filtered to as_of).
        strategy_cfg: dict with keys ``stop_loss_pct``, ``trailing_stop_pct``,
            ``take_profit_pct``. A value of ``0.0`` disables that gate.

    Returns:
        DataFrame with columns ``symbol, direction, exit_reason, exit_qty_pct``.
        ``direction`` is always ``"FLAT"``. ``exit_qty_pct`` is ``1.0`` for a
        full exit, ``0.5`` for a partial take-profit.

    Sequence per symbol: stop_loss > trailing_stop > take_profit. First
    triggered gate wins (continue to next symbol). HWM is updated based on
    the bar's close and persisted in the returned signals only indirectly —
    the paper ledger reads HWM from its own state on the next bar.
    """
    cfg = strategy_cfg or {}
    stop_loss_pct = float(cfg.get("stop_loss_pct", 0.0))
    trailing_stop_pct = float(cfg.get("trailing_stop_pct", 0.0))
    take_profit_pct = float(cfg.get("take_profit_pct", 0.0))

    empty_cols = ["symbol", "direction", "exit_reason", "exit_qty_pct"]
    if not current_positions or prices_latest is None or prices_latest.empty:
        return pd.DataFrame(columns=empty_cols)

    if "symbol" not in prices_latest.columns or "close" not in prices_latest.columns:
        return pd.DataFrame(columns=empty_cols)

    price_map = dict(zip(prices_latest["symbol"].values, prices_latest["close"].values))

    exits: list[dict] = []
    for sym, pos in current_positions.items():
        qty = float(pos.get("qty", 0.0))
        if qty <= 0:
            continue
        avg_price = float(pos.get("avg_price", 0.0))
        if avg_price <= 0:
            continue
        current_price = float(price_map.get(sym, 0.0))
        if current_price <= 0:
            continue

        # HWM update: per-symbol high-water-mark since entry. Initialized
        # to avg_price if the ledger hasn't seen this position before.
        hwm = float(pos.get("hwm", avg_price))
        if current_price > hwm:
            hwm = current_price

        # Stop-loss
        if stop_loss_pct > 0.0:
            stop_price = avg_price * (1.0 - stop_loss_pct)
            if current_price <= stop_price:
                exits.append(
                    {
                        "symbol": sym,
                        "direction": "FLAT",
                        "exit_reason": (
                            f"stop_loss ({current_price:.2f} <= " f"{stop_price:.2f})"
                        ),
                        "exit_qty_pct": 1.0,
                    }
                )
                continue

        # Trailing-stop
        if trailing_stop_pct > 0.0 and hwm > avg_price:
            trail_price = hwm * (1.0 - trailing_stop_pct)
            if current_price <= trail_price:
                exits.append(
                    {
                        "symbol": sym,
                        "direction": "FLAT",
                        "exit_reason": (
                            f"trailing_stop ({current_price:.2f} <= "
                            f"{trail_price:.2f}, hwm={hwm:.2f})"
                        ),
                        "exit_qty_pct": 1.0,
                    }
                )
                continue

        # Take-profit (partial)
        if take_profit_pct > 0.0:
            tp_price = avg_price * (1.0 + take_profit_pct)
            if current_price >= tp_price:
                exits.append(
                    {
                        "symbol": sym,
                        "direction": "FLAT",
                        "exit_reason": (
                            f"take_profit ({current_price:.2f} >= " f"{tp_price:.2f})"
                        ),
                        "exit_qty_pct": 0.5,
                    }
                )
                continue

    if not exits:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(exits)


__all__ = ["compute_signals", "compute_target_positions", "check_exit_signals"]
