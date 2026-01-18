# src/assembled_core/pipeline/portfolio.py
"""Portfolio simulation with cost model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.execution.transaction_costs import (
    SlippageModel,
    SpreadModel,
    add_cost_columns_to_trades,
    commission_model_from_cost_params,
)


def simulate_with_costs(
    orders: pd.DataFrame,
    start_capital: float,
    commission_bps: float,
    spread_w: float,
    impact_w: float,
    freq: str,
    prices: pd.DataFrame | None = None,
    spread_model: SpreadModel | None = None,
    slippage_model: SlippageModel | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Simulate portfolio equity with transaction costs.

    Args:
        orders: DataFrame with columns: timestamp, symbol, side, qty, price
        start_capital: Starting capital
        commission_bps: Commission in basis points
        spread_w: Spread weight (multiplier for bid/ask spread)
        impact_w: Market impact weight (multiplier for price impact)
        freq: Frequency string ("1d" or "5min") for timeline generation

    Returns:
        Tuple of (equity DataFrame, metrics dict)
        equity: DataFrame with columns: timestamp, equity
        metrics: Dictionary with keys: final_pf, sharpe, trades

    Side effects:
        None (pure function)
    """
    # Timeline aus Orders ableiten – falls nur wenige Orders, erweitern wir leicht
    if orders.empty:
        ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=60, freq=freq)
        eq = pd.DataFrame({"timestamp": ts, "equity": float(start_capital)})
        rep = {"final_pf": 1.0, "sharpe": float("nan"), "trades": 0}
        return eq, rep

    t0, t1 = orders["timestamp"].min(), orders["timestamp"].max()
    tl = pd.date_range(start=t0, end=t1, freq=freq)
    equity = np.full(len(tl), start_capital, dtype=np.float64)

    # primitive, aber reproduzierbare Kostenmodellierung:
    # - Kommission in bps * notional-per-trade (now via commission_cash column)
    # - bid/ask via spread_w in bp Äquivalent
    # - 'impact' als zusätzl. Preisabschlag in bp
    # Wir haben keinen Notional im CSV; wir modellieren 1x Preis * qty als notional surrogate.
    s = spread_w * 1e-4
    im = impact_w * 1e-4

    # Apply fill model pipeline (session gate -> limit -> partial)
    # This must happen BEFORE cost calculation, as costs are based on fill_qty
    if not orders.empty and prices is not None:
        from src.assembled_core.execution.fill_model_pipeline import (
            apply_fill_model_pipeline,
        )
        
        # Apply fill model pipeline
        # For now, use default partial fill model (can be made configurable later)
        partial_fill_model = None  # Default: full fills (no ADV cap)
        # TODO: Make partial_fill_model configurable via cost_model or separate parameter
        
        orders = apply_fill_model_pipeline(
            orders,
            prices=prices,
            freq=freq,
            partial_fill_model=partial_fill_model,
            strict_session_gate=True,
        )
    
    # Add cost columns to orders (commission_cash, spread_cash, slippage_cash, total_cost_cash)
    # Costs are now computed based on fill_qty (for partial/rejected fills)
    # Create commission model from legacy parameters
    commission_model = commission_model_from_cost_params(commission_bps=commission_bps)
    
    # Create slippage model from legacy parameters (if impact_w > 0)
    # For now, map impact_w to slippage model (can be refined later)
    if slippage_model is None and impact_w > 0.0:
        # Map impact_w to slippage model (impact_w is a weight, convert to reasonable slippage)
        # Default: k=1.0, min_bps=0.0, max_bps=50.0, fallback=impact_w*100 (convert to bps)
        slippage_model = SlippageModel(
            vol_window=20,
            k=impact_w,  # Use impact_w as scaling factor
            min_bps=0.0,
            max_bps=50.0,
            fallback_slippage_bps=impact_w * 100.0,  # Convert to bps
        )
    
    orders = add_cost_columns_to_trades(
        orders,
        commission_model=commission_model,
        spread_model=spread_model,
        slippage_model=slippage_model,
        prices=prices,
    )

    # Gruppiere Orders je Zeitstempel, summiere Cash-Delta
    orders = orders.copy()
    orders["sign"] = np.where(
        orders["side"].eq("BUY"), +1.0, np.where(orders["side"].eq("SELL"), -1.0, 0.0)
    )
    # Notional is now computed in add_cost_columns_to_trades based on fill_qty
    # For legacy compatibility, keep notional column (but it's not used for cash_delta)
    if "fill_qty" in orders.columns and "fill_price" in orders.columns:
        orders["notional"] = (orders["fill_qty"].abs() * orders["fill_price"].abs()).astype(np.float64)
    else:
        orders["notional"] = (orders["qty"].abs() * orders["price"].abs()).astype(np.float64)

    # Update spread_cash and slippage_cash (for legacy compatibility)
    # Spread_cash and slippage_cash are already computed in add_cost_columns_to_trades()
    # if spread_model/slippage_model are provided. Otherwise, use legacy calculation.
    if spread_model is None:
        # Legacy: use spread_w for spread_cash
        orders["spread_cash"] = orders["notional"] * s
    # Otherwise, spread_cash is already set from add_cost_columns_to_trades()
    
    if slippage_model is None:
        # Legacy: use impact_w for slippage_cash
        orders["slippage_cash"] = orders["notional"] * im
    # Otherwise, slippage_cash is already set from add_cost_columns_to_trades()
    
    orders["total_cost_cash"] = (
        orders["commission_cash"] + orders["spread_cash"] + orders["slippage_cash"]
    )

    # effektiver Preisaufschlag/-abschlag
    # BUY zahlt: price * (1 + s + im) + kommission
    # SELL erhält: price * (1 - s - im) - kommission
    # Use total_cost_cash instead of separate k * notional
    orders["cash_delta"] = np.where(
        orders["sign"] > 0,
        -(orders["qty"] * orders["price"] * (1.0 + s + im) + orders["total_cost_cash"]),
        +(
            orders["qty"].abs() * orders["price"] * (1.0 - s - im)
            - orders["total_cost_cash"]
        ),
    )

    ts_to_delta = (
        orders.groupby(pd.Grouper(key="timestamp", freq=freq))["cash_delta"]
        .sum()
        .reindex(tl, fill_value=0.0)
        .to_numpy()
    )

    # wende Cash-Deltas ab jeweiligem Index an (cum)
    # First equity value should always be start_capital (before any trades)
    # Apply deltas starting from the period AFTER they occur
    # So equity[0] = start_capital, equity[1] = start_capital + delta[0], etc.
    cumsum_deltas = np.cumsum(ts_to_delta)
    # Shift by 1: equity[0] stays at start_capital, equity[1] = start_capital + delta[0], etc.
    equity[1:] = equity[1:] + cumsum_deltas[:-1]

    eq = pd.DataFrame({"timestamp": tl, "equity": equity})
    # simple Kennzahlen
    pf = float(equity[-1] / equity[0]) if equity.size else 1.0
    ret = pd.Series(equity).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = (
        float(ret.mean() / ret.std() * np.sqrt(252 if freq == "1d" else 252 * 78))
        if len(ret) > 5 and ret.std() > 0
        else float("nan")
    )
    rep = {"final_pf": pf, "sharpe": sharpe, "trades": int(len(orders))}
    return eq, rep


def write_portfolio_report(
    equity: pd.DataFrame,
    metrics: dict[str, float | int],
    freq: str,
    output_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write portfolio results to CSV and Markdown report.

    Args:
        equity: DataFrame with columns: timestamp, equity
        metrics: Dictionary with performance metrics (final_pf, sharpe, trades)
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)

    Returns:
        Tuple of (equity_path, report_path)

    Side effects:
        Creates output directory if it doesn't exist
        Writes CSV file: output_dir/portfolio_equity_{freq}.csv
        Writes Markdown file: output_dir/portfolio_report_{freq}.md
    """
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_path = out_dir / f"portfolio_equity_{freq}.csv"
    rep_path = out_dir / f"portfolio_report_{freq}.md"

    try:
        eq_path.parent.mkdir(parents=True, exist_ok=True)
        equity.to_csv(eq_path, index=False)
    except (IOError, OSError) as exc:
        raise RuntimeError(f"Failed to write portfolio equity CSV to {eq_path}") from exc

    try:
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with rep_path.open("w", encoding="utf-8") as f:
            f.write(f"# Portfolio Report ({freq})\n\n")
            f.write(f"- Final PF: {metrics['final_pf']:.4f}\n")
            f.write(f"- Sharpe: {metrics['sharpe']}\n")
    except (IOError, OSError) as exc:
        raise RuntimeError(f"Failed to write portfolio report to {rep_path}") from exc
        f.write(f"- Trades: {metrics['trades']}\n")

    return eq_path, rep_path
