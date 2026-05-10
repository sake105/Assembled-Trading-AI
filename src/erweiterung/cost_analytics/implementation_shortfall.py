"""Implementation Shortfall (Perold 1988).

Definition
----------
IS = (price_decision - price_execution) × shares + commission

Decomposition (Almgren-Chriss-style):
- **Delay Cost** = (price_decision - price_arrival) × shares
  Cost of delay between decision and order placement.
- **Trading Cost** = (price_arrival - price_execution_avg) × shares
  Cost of market impact + spread during execution.
- **Opportunity Cost** = (price_execution_avg - price_close) × unfilled_shares
  Cost of unfilled portion.

Anwendung
---------
- Trade-Cost-Analyse (TCA) für Execution-Quality-Bewertung
- Algorithm-Selection (TWAP vs VWAP vs Almgren-Chriss)
- Counterparty-Performance-Tracking

Reference
---------
- Perold, A. (1988). The Implementation Shortfall: Paper versus Reality.
  *J. Portfolio Management* 14.
- Kissell, R. (2014). *The Science of Algorithmic Trading*.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TradeContext:
    side: int  # +1 buy, -1 sell
    intended_shares: float
    decision_price: float
    arrival_price: float
    avg_execution_price: float
    close_price: float
    filled_shares: float
    commission: float = 0.0


def implementation_shortfall(ctx: TradeContext) -> dict:
    """Compute Implementation Shortfall + decomposition.

    All costs returned in **absolute USD** (positive = bad for buyer).

    Returns:
        dict mit total, delay, trading, opportunity, commission, bps_total.
    """
    side = ctx.side
    intended = ctx.intended_shares
    filled = ctx.filled_shares
    unfilled = intended - filled

    # Delay cost: from decision to arrival (both prices observed BEFORE execution)
    delay = side * (ctx.arrival_price - ctx.decision_price) * intended

    # Trading cost: market-impact during execution (arrival → exec)
    trading = side * (ctx.avg_execution_price - ctx.arrival_price) * filled

    # Opportunity cost: unfilled shares (exec → close)
    opportunity = side * (ctx.close_price - ctx.avg_execution_price) * unfilled

    total = delay + trading + opportunity + ctx.commission

    notional = abs(intended) * ctx.decision_price
    bps_total = total / notional * 10000 if notional > 0 else float("nan")

    return {
        "total_cost_usd": float(total),
        "delay_cost_usd": float(delay),
        "trading_cost_usd": float(trading),
        "opportunity_cost_usd": float(opportunity),
        "commission_usd": float(ctx.commission),
        "total_bps": float(bps_total),
        "notional_usd": float(notional),
        "fill_rate": float(filled / intended) if intended != 0 else float("nan"),
    }


def slippage_decomposition(
    target_price: float,
    fill_price: float,
    bid_ask_mid: float,
    side: int,
) -> dict:
    """Slippage in 3 Komponenten zerlegen.

    Half-Spread + Adverse-Selection + Timing-Cost.

    Args:
        target_price: was wir wollten zu zahlen.
        fill_price: tatsächlich gezahlt.
        bid_ask_mid: Mid-Price zum Zeitpunkt der Order.
        side: +1 buy / -1 sell.

    Returns:
        dict mit Half-Spread + Adverse-Selection + Timing.
    """
    half_spread_cost = side * (fill_price - bid_ask_mid)
    adverse_selection = side * (bid_ask_mid - target_price)
    total = side * (fill_price - target_price)
    return {
        "total_slippage_usd": float(total),
        "half_spread_usd": float(half_spread_cost),
        "adverse_selection_usd": float(adverse_selection),
        "total_bps": (
            float(total / target_price * 10000) if target_price > 0 else float("nan")
        ),
    }


def aggregate_trade_costs(
    trades_df: pd.DataFrame,
) -> dict:
    """Aggregierte Cost-Metriken über mehrere Trades.

    Args:
        trades_df: DataFrame mit Spalten ``cost_bps`` und ``notional``.

    Returns:
        dict mit weighted avg bps, total cost, distribution stats.
    """
    df = trades_df.dropna(subset=["cost_bps", "notional"])
    if df.empty:
        return {"error": "empty trades"}
    total_notional = float(df["notional"].sum())
    if total_notional == 0:
        return {"error": "zero notional"}
    weighted_bps = float((df["cost_bps"] * df["notional"]).sum() / total_notional)
    return {
        "n_trades": int(len(df)),
        "total_notional_usd": total_notional,
        "weighted_avg_cost_bps": weighted_bps,
        "median_cost_bps": float(df["cost_bps"].median()),
        "p95_cost_bps": float(df["cost_bps"].quantile(0.95)),
        "p99_cost_bps": float(df["cost_bps"].quantile(0.99)),
        "total_cost_usd": float((df["cost_bps"] * df["notional"]).sum() / 10000),
    }


__all__ = [
    "TradeContext",
    "implementation_shortfall",
    "slippage_decomposition",
    "aggregate_trade_costs",
]
