# src/assembled_core/pipeline/portfolio.py
"""Portfolio simulation with cost model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.assembled_core.config import OUTPUT_DIR


def simulate_with_costs(
    orders: pd.DataFrame,
    start_capital: float,
    commission_bps: float,
    spread_w: float,
    impact_w: float,
    freq: str,
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
    # - Kommission in bps * notional-per-trade
    # - bid/ask via spread_w in bp Äquivalent
    # - 'impact' als zusätzl. Preisabschlag in bp
    # Wir haben keinen Notional im CSV; wir modellieren 1x Preis * qty als notional surrogate.
    k = commission_bps * 1e-4
    s = spread_w * 1e-4
    im = impact_w * 1e-4

    # Gruppiere Orders je Zeitstempel, summiere Cash-Delta
    orders = orders.copy()
    orders["sign"] = np.where(
        orders["side"].eq("BUY"), +1.0,
        np.where(orders["side"].eq("SELL"), -1.0, 0.0)
    )
    orders["notional"] = (orders["qty"].abs() * orders["price"].abs()).astype(np.float64)

    # effektiver Preisaufschlag/-abschlag
    # BUY zahlt: price * (1 + s + im) + kommission
    # SELL erhält: price * (1 - s - im) - kommission
    orders["cash_delta"] = np.where(
        orders["sign"] > 0,
        -(orders["qty"] * orders["price"] * (1.0 + s + im) + k * orders["notional"]),
        +(orders["qty"].abs() * orders["price"] * (1.0 - s - im) - k * orders["notional"])
    )

    ts_to_delta = (
        orders.groupby(pd.Grouper(key="timestamp", freq=freq))["cash_delta"]
        .sum()
        .reindex(tl, fill_value=0.0)
        .to_numpy()
    )

    # wende Cash-Deltas ab jeweiligem Index an (cum)
    equity = equity + np.cumsum(ts_to_delta)

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
    
    equity.to_csv(eq_path, index=False)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"# Portfolio Report ({freq})\n\n")
        f.write(f"- Final PF: {metrics['final_pf']:.4f}\n")
        f.write(f"- Sharpe: {metrics['sharpe']}\n")
        f.write(f"- Trades: {metrics['trades']}\n")
    
    return eq_path, rep_path

