"""Turnover budget gate: cap realized turnover per run (daily/weekly).

Scales target deltas when estimated turnover exceeds policy cap.
No new signals; cost-aware gate only.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def _latest_prices_per_symbol(prices: pd.DataFrame) -> pd.Series:
    """Return series symbol -> close (latest) from prices with timestamp, symbol, close."""
    if prices is None or prices.empty or "close" not in prices.columns or "symbol" not in prices.columns:
        return pd.Series(dtype=float)
    if "timestamp" in prices.columns:
        idx = prices.groupby("symbol", group_keys=False)["timestamp"].idxmax()
        out = prices.loc[idx, ["symbol", "close"]].set_index("symbol")["close"]
    else:
        out = prices.groupby("symbol", group_keys=False)["close"].last()
    return out


def estimate_turnover(
    current_positions: pd.DataFrame | None,
    target_positions: pd.DataFrame,
    prices: pd.DataFrame | None,
    portfolio_value: float = 1.0,
) -> float:
    """Estimate turnover as sum(|delta_weight|) / 2.

    current_positions: columns symbol, qty (optional).
    target_positions: columns symbol, target_weight (optional target_qty).
    prices: columns symbol, close (and optionally timestamp for latest).
    portfolio_value: used to convert current notional to weights (default 1.0).

    Returns float. If prices missing/empty, returns float('inf') so gate can scale to zero.
    """
    if target_positions is None or target_positions.empty:
        return 0.0
    if "symbol" not in target_positions.columns:
        return 0.0

    price_series = _latest_prices_per_symbol(prices) if prices is not None else pd.Series(dtype=float)
    if price_series.empty and (current_positions is None or current_positions.empty):
        return 0.0
    if price_series.empty:
        return float("inf")

    symbols = sorted(target_positions["symbol"].unique().tolist())
    if not symbols:
        return 0.0

    # Current weight per symbol: (qty * price) / portfolio_value
    if portfolio_value <= 0:
        portfolio_value = 1.0
    current_weight = pd.Series(index=symbols, data=0.0, dtype=float)
    if current_positions is not None and not current_positions.empty and "qty" in current_positions.columns:
        for _, row in current_positions.iterrows():
            sym = row.get("symbol")
            if sym in current_weight.index:
                qty = float(row.get("qty", 0) or 0)
                pr = float(price_series.get(sym, 0) or 0)
                current_weight[sym] = (qty * pr) / portfolio_value

    # Target weight per symbol
    target_weight = pd.Series(index=symbols, data=0.0, dtype=float)
    if "target_weight" in target_positions.columns:
        for _, row in target_positions.iterrows():
            sym = row.get("symbol")
            if sym in target_weight.index:
                target_weight[sym] = float(row.get("target_weight", 0) or 0)
    else:
        # No target_weight: use 0 (full exit) or derive from target_qty if needed
        pass

    delta = target_weight - current_weight
    turnover = float((delta.abs().sum()) / 2.0)
    return turnover


def apply_turnover_gate(
    target_positions: pd.DataFrame,
    current_positions: pd.DataFrame | None,
    cap: float,
    estimated_turnover: float,
    behavior: str = "scale",
    prices: pd.DataFrame | None = None,
    portfolio_value: float = 1.0,
) -> Tuple[pd.DataFrame, float]:
    """Apply turnover cap: scale target deltas if turnover exceeds cap.

    Returns (new_target_positions, scale_factor). scale_factor 1.0 when no scaling.
    """
    if target_positions is None or target_positions.empty:
        return target_positions, 1.0
    if cap <= 0:
        return target_positions, 1.0
    if estimated_turnover <= cap:
        return target_positions.copy(), 1.0

    price_series = _latest_prices_per_symbol(prices) if prices is not None else pd.Series(dtype=float)
    pv = portfolio_value if portfolio_value > 0 else 1.0

    scale_factor = cap / estimated_turnover
    if behavior == "block":
        # Block: set targets to current (no trades)
        out = target_positions.copy()
        symbols_out = out["symbol"].tolist()
        cw = {}
        cq = {}
        if current_positions is not None and not current_positions.empty:
            for _, row in current_positions.iterrows():
                sym = row.get("symbol")
                cq[sym] = float(row.get("qty", 0) or 0)
                pr = float(price_series.get(sym, 0) or 0) if not price_series.empty else 0.0
                cw[sym] = (cq[sym] * pr) / pv
        for sym in symbols_out:
            cw.setdefault(sym, 0.0)
            cq.setdefault(sym, 0.0)
        if "target_weight" in out.columns:
            for i, sym in enumerate(out["symbol"]):
                out.iloc[i, out.columns.get_loc("target_weight")] = cw[sym]
        if "target_qty" in out.columns:
            for i, sym in enumerate(out["symbol"]):
                out.iloc[i, out.columns.get_loc("target_qty")] = cq[sym]
        return out, 0.0

    # scale: new_target = current + scale_factor * (target - current)

    out = target_positions.copy()
    symbols = out["symbol"].tolist()
    current_w = {}
    current_q = {}
    if current_positions is not None and not current_positions.empty:
        for _, row in current_positions.iterrows():
            sym = row.get("symbol")
            qty = float(row.get("qty", 0) or 0)
            current_q[sym] = qty
            pr = float(price_series.get(sym, 0) or 0) if not price_series.empty else 0.0
            current_w[sym] = (qty * pr) / pv
    for sym in symbols:
        if sym not in current_w:
            current_w[sym] = 0.0
        if sym not in current_q:
            current_q[sym] = 0.0

    if "target_weight" in out.columns:
        for i, sym in enumerate(out["symbol"]):
            tw = float(out.iloc[i]["target_weight"])
            cw = current_w.get(sym, 0.0)
            out.iloc[i, out.columns.get_loc("target_weight")] = cw + scale_factor * (tw - cw)
    if "target_qty" in out.columns and not price_series.empty:
        for i, sym in enumerate(out["symbol"]):
            tq = float(out.iloc[i]["target_qty"])
            cq = current_q.get(sym, 0.0)
            out.iloc[i, out.columns.get_loc("target_qty")] = cq + scale_factor * (tq - cq)

    return out, scale_factor


__all__ = ["estimate_turnover", "apply_turnover_gate"]
