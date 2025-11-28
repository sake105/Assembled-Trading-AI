# src/assembled_core/pipeline/backtest.py
"""Backtest simulation (equity curve without costs)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.assembled_core.config import OUTPUT_DIR


def simulate_equity(prices: pd.DataFrame, orders: pd.DataFrame, start_capital: float) -> pd.DataFrame:
    """Simulate equity curve from prices and orders (without costs).
    
    Args:
        prices: DataFrame with columns: timestamp, symbol, close
        orders: DataFrame with columns: timestamp, symbol, side, qty, price
        start_capital: Starting capital
    
    Returns:
        DataFrame with columns: timestamp, equity
        Sorted by timestamp
    
    Side effects:
        None (pure function)
    """
    # Timeline & Price pivot
    prices = prices.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    timeline = prices["timestamp"].sort_values().drop_duplicates().to_list()
    symbols = sorted(prices["symbol"].unique().tolist())
    px = prices.pivot(index="timestamp", columns="symbol", values="close").sort_index()

    # Orders nach Timestamp gruppieren
    orders = orders.sort_values("timestamp").reset_index(drop=True)
    orders_by_ts = {}
    if len(orders):
        for ts, group in orders.groupby("timestamp"):
            orders_by_ts[ts] = group

    # Simulationszustand
    cash = float(start_capital)
    pos = {s: 0.0 for s in symbols}
    equity_series = []

    for ts in timeline:
        # Orders zum aktuellen Timestamp ausführen (Market zum angegebenen Orderpreis)
        if ts in orders_by_ts:
            g = orders_by_ts[ts]
            for _, row in g.iterrows():
                sym = row.get("symbol", "")
                side = row.get("side", "")
                qty = float(row.get("qty", 0.0))
                price = float(row.get("price", np.nan))
                if not sym or np.isnan(price) or qty == 0.0:
                    continue
                if side == "BUY":
                    cash -= qty * price
                    pos[sym] = pos.get(sym, 0.0) + qty
                elif side == "SELL":
                    cash += qty * price
                    pos[sym] = pos.get(sym, 0.0) - qty

        # Mark-to-Market
        if ts in px.index:
            mtm = 0.0
            row = px.loc[ts]
            for s in symbols:
                pr = float(row.get(s, np.nan))
                if not np.isnan(pr):
                    mtm += pos.get(s, 0.0) * pr
            equity = cash + mtm
        else:
            equity = cash  # falls Lücke

        equity_series.append((ts, float(equity)))

    eq = pd.DataFrame(equity_series, columns=["timestamp", "equity"])
    # Sanitizing
    s = pd.Series(eq["equity"].values, index=eq["timestamp"])
    s = s.replace([np.inf, -np.inf], np.nan).ffill().fillna(start_capital)
    eq["equity"] = s.values
    return eq


def compute_metrics(equity: pd.DataFrame) -> dict[str, float | int]:
    """Compute performance metrics from equity curve.
    
    Args:
        equity: DataFrame with columns: timestamp, equity
    
    Returns:
        Dictionary with keys: final_pf, sharpe, rows, first, last
        final_pf: Final performance factor (equity[-1] / equity[0])
        sharpe: Sharpe ratio (simple calculation)
        rows: Number of rows
        first: First timestamp
        last: Last timestamp
    """
    pf = float(equity["equity"].iloc[-1] / max(equity["equity"].iloc[0], 1e-12))
    ret = equity["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = float(ret.mean() / (ret.std() + 1e-12)) if not ret.empty else float("nan")
    
    return {
        "final_pf": pf,
        "sharpe": sharpe,
        "rows": len(equity),
        "first": equity["timestamp"].iloc[0],
        "last": equity["timestamp"].iloc[-1],
    }


def write_backtest_report(equity: pd.DataFrame, metrics: dict[str, float | int], freq: str, output_dir: Path | str | None = None) -> tuple[Path, Path]:
    """Write backtest results to CSV and Markdown report.
    
    Args:
        equity: DataFrame with columns: timestamp, equity
        metrics: Dictionary with performance metrics
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
    
    Returns:
        Tuple of (equity_curve_path, report_path)
    
    Side effects:
        Creates output directory if it doesn't exist
        Writes CSV file: output_dir/equity_curve_{freq}.csv
        Writes Markdown file: output_dir/performance_report_{freq}.md
    """
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    curve_path = out_dir / f"equity_curve_{freq}.csv"
    rep_path = out_dir / f"performance_report_{freq}.md"

    equity.to_csv(curve_path, index=False)

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"# Performance Report ({freq})\n\n")
        f.write(f"- Final PF: {metrics['final_pf']:.4f}\n")
        f.write(f"- Sharpe: {metrics['sharpe']:.4f}\n")
        f.write(f"- Rows: {metrics['rows']}\n")
        f.write(f"- First: {metrics['first']}\n")
        f.write(f"- Last:  {metrics['last']}\n")

    return curve_path, rep_path

