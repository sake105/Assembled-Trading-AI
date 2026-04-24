"""trading_cycle_v2 — decomposed trading cycle (Week 4–6 refactor).

The old trading_cycle.py remains the active implementation until Day 9.
This file holds the 7-function target structure.  Each function below is
a stub that raises NotImplementedError; they will be filled in during
Days 2–8 by migrating steps from _run_trading_cycle_inner.

Target: every function ≤ 500 lines, zero observability-only steps.
A step survives only when ALL three hold:
  1. It changes a value that a downstream step or caller reads.
  2. It has a test asserting concrete output values (not just existence).
  3. It does not have the shape  result.meta["x"] = {"available": True}.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.assembled_core.pipeline.trading_cycle import TradingContext, TradingCycleResult

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Sub-functions (one per pipeline stage)
# ---------------------------------------------------------------------------


def ingest_data(ctx: TradingContext) -> pd.DataFrame:
    """Filter and validate raw price data PIT-safely.

    Returns prices_filtered: one row per symbol <= ctx.as_of (EOD/paper)
    or full history slice <= ctx.as_of (backtest).
    """
    raise NotImplementedError("ingest_data — stub, not yet filled in")


def build_features(prices: pd.DataFrame, ctx: TradingContext) -> pd.DataFrame:
    """Add technical features to the filtered price DataFrame.

    Returns prices_with_features: all columns from prices plus feature
    columns (ma_20, ma_50, atr_14, rsi_14, …).
    """
    raise NotImplementedError("build_features — stub, not yet filled in")


def generate_signals(features: pd.DataFrame, ctx: TradingContext) -> pd.DataFrame:
    """Apply signal_fn to feature-enriched prices.

    Returns signals: columns [timestamp, symbol, direction, score].
    """
    raise NotImplementedError("generate_signals — stub, not yet filled in")


def size_positions(signals: pd.DataFrame, ctx: TradingContext) -> pd.DataFrame:
    """Apply position_sizing_fn to convert signals to target weights/quantities.

    Returns target_positions: columns [symbol, target_weight, target_qty].
    """
    raise NotImplementedError("size_positions — stub, not yet filled in")


def check_risk(targets: pd.DataFrame, ctx: TradingContext) -> pd.DataFrame:
    """Apply pre-trade risk controls (kill switch, exposure caps, vol checks).

    Returns filtered_targets: same schema as target_positions, with
    violating positions removed or scaled down.
    """
    raise NotImplementedError("check_risk — stub, not yet filled in")


def route_orders(checked: pd.DataFrame, ctx: TradingContext) -> pd.DataFrame:
    """Generate and filter orders from approved target positions.

    Returns orders: columns [timestamp, symbol, side, qty, price].
    """
    raise NotImplementedError("route_orders — stub, not yet filled in")


def book_fills(orders: pd.DataFrame, ctx: TradingContext) -> TradingCycleResult:
    """Write outputs (CSV, equity curve, state) and package final result.

    Returns a TradingCycleResult with all intermediate fields populated.
    """
    raise NotImplementedError("book_fills — stub, not yet filled in")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_trading_cycle(ctx: TradingContext) -> TradingCycleResult:
    """Run the full trading cycle via the seven stage functions.

    This replaces _run_trading_cycle_inner once all stubs are filled.
    The old trading_cycle.run_trading_cycle() remains active until Day 9.
    """
    prices = ingest_data(ctx)
    features = build_features(prices, ctx)
    signals = generate_signals(features, ctx)
    targets = size_positions(signals, ctx)
    checked = check_risk(targets, ctx)
    orders = route_orders(checked, ctx)
    return book_fills(orders, ctx)
