"""Isolated A/B backtest for pairs_trading_v1 (Plan 11/10 §1.1).

Activation thresholds: Sharpe > 0.8, MDD < -15%, Trades > 50.
Results written to output/backtest_results/pairs_trading_v1_<date>.json.

Usage:
    python scripts/backtest_pairs_trading.py --start 2020-01-01 --end 2024-12-31
    python scripts/backtest_pairs_trading.py --start 2020-01-01 --end 2024-12-31 --price-file data/sample/watchlist_2020_2026.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backtest_pairs_trading")

ACTIVATION_THRESHOLDS = {
    "min_sharpe": 0.8,
    "max_mdd_pct": -15.0,
    "min_trades": 50,
}


def _load_prices(price_file: str, start: str, end: str) -> pd.DataFrame:
    path = Path(price_file)
    if not path.exists():
        logger.error("Price file not found: %s", path)
        sys.exit(1)
    df = pd.read_parquet(path)
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    # Normalize date column — support 'date', 'timestamp', DatetimeIndex
    if "date" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "date"})
        elif isinstance(df.index, pd.DatetimeIndex) or df.index.name in (
            "date",
            "timestamp",
        ):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    # Sort by date so window slicing works correctly across all symbols
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d rows from %s (%s to %s)", len(df), path.name, start, end)
    return df


def _run_pairs_backtest(
    prices: pd.DataFrame,
    entry_z: float = 1.8,
    exit_z: float = 0.5,
    lookback_days: int = 252,
    min_cointegration_p: float = 0.05,
    max_pairs: int = 20,
    start_capital: float = 100_000.0,
) -> dict:
    from src.assembled_core.strategies.pairs_trading_v1 import PairsTradingStrategy

    strategy = PairsTradingStrategy(
        {
            "lookback_days": lookback_days,
            "min_cointegration_p": min_cointegration_p,
            "entry_zscore": entry_z,
            "exit_zscore": exit_z,
            "max_pairs": max_pairs,
        }
    )

    dates = sorted(prices["date"].unique())
    if len(dates) < lookback_days + 5:
        logger.error(
            "Not enough data (%d dates) for %d-day lookback", len(dates), lookback_days
        )
        return {}

    equity = start_capital
    equity_curve = [equity]
    trades: list[dict] = []
    open_positions: dict[str, dict] = {}

    # Discover pairs using full historical dataset (cointegration needs max history)
    discovery_prices = prices.copy()
    active_pairs = strategy.discover_pairs(discovery_prices)
    logger.info("Active pairs (%d): %s", len(active_pairs), active_pairs)

    if not active_pairs:
        logger.warning(
            "No pairs discovered — check data coverage and cointegration settings"
        )
        return {"n_trades": 0, "sharpe": 0.0, "mdd_pct": 0.0, "activation": "FAIL"}

    # Walk-forward through dates
    for i, date in enumerate(dates[lookback_days:], start=lookback_days):
        # Slice by date-count, not row-count (long format has n_symbols rows per date)
        window_start = dates[max(0, i - lookback_days * 2)]
        window = prices[(prices["date"] >= window_start) & (prices["date"] <= date)]
        try:
            signals = strategy.generate_signals(window, pairs=active_pairs)
        except Exception as exc:
            logger.debug("Signal error on %s: %s", date, exc)
            equity_curve.append(equity)
            continue

        if signals.empty:
            equity_curve.append(equity)
            continue

        # Simple execution: process signals as market orders, $2k notional per leg
        notional_per_leg = min(2_000.0, equity * 0.05)
        commission_per_trade = 1.0

        for _, sig in signals.iterrows():
            sym = str(sig.get("symbol", ""))
            direction = str(sig.get("direction", "")).upper()
            if not sym or direction not in ("LONG", "SHORT", "EXIT"):
                continue

            if direction == "EXIT" and sym in open_positions:
                pos = open_positions.pop(sym)
                close_row = prices[(prices["date"] == date) & (prices["symbol"] == sym)]
                if close_row.empty:
                    continue
                close_px = float(close_row["close"].iloc[0])
                qty = pos["qty"]
                side = pos["side"]
                entry_px = pos["entry_px"]
                pnl = (
                    qty * (close_px - entry_px) * (1 if side == "LONG" else -1)
                    - commission_per_trade
                )
                equity += pnl
                trades.append(
                    {
                        "date": str(date),
                        "symbol": sym,
                        "side": side,
                        "pnl": round(pnl, 2),
                        "type": "close",
                    }
                )
            elif direction in ("LONG", "SHORT") and sym not in open_positions:
                entry_row = prices[(prices["date"] == date) & (prices["symbol"] == sym)]
                if entry_row.empty:
                    continue
                entry_px = float(entry_row["close"].iloc[0])
                if entry_px <= 0:
                    continue
                qty = notional_per_leg / entry_px
                equity -= commission_per_trade
                open_positions[sym] = {
                    "qty": qty,
                    "side": direction,
                    "entry_px": entry_px,
                    "date": str(date),
                }
                trades.append(
                    {
                        "date": str(date),
                        "symbol": sym,
                        "side": direction,
                        "pnl": -commission_per_trade,
                        "type": "open",
                    }
                )

        equity_curve.append(equity)

    # Close remaining open positions at last price
    last_date = dates[-1]
    for sym, pos in open_positions.items():
        close_row = prices[(prices["date"] == last_date) & (prices["symbol"] == sym)]
        if close_row.empty:
            continue
        close_px = float(close_row["close"].iloc[0])
        qty = pos["qty"]
        pnl = (
            qty * (close_px - pos["entry_px"]) * (1 if pos["side"] == "LONG" else -1)
            - 1.0
        )
        equity += pnl
        trades.append(
            {
                "date": str(last_date),
                "symbol": sym,
                "side": pos["side"],
                "pnl": round(pnl, 2),
                "type": "close_eod",
            }
        )
    equity_curve.append(equity)

    # Compute metrics
    n_trades = sum(1 for t in trades if t["type"].startswith("close"))
    eq = np.array(equity_curve, dtype=float)
    daily_returns = np.diff(eq) / eq[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    if len(daily_returns) > 1:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    n_years = (
        pd.Timestamp(dates[-1]) - pd.Timestamp(dates[lookback_days])
    ).days / 365.25
    cagr = (
        float((equity / start_capital) ** (1 / max(n_years, 0.01)) - 1)
        if n_years > 0
        else 0.0
    )

    peaks = np.maximum.accumulate(eq)
    drawdowns = (eq - peaks) / np.where(peaks > 0, peaks, 1)
    mdd_pct = float(np.min(drawdowns) * 100)

    thresholds = ACTIVATION_THRESHOLDS
    activation = (
        "GO"
        if (
            sharpe >= thresholds["min_sharpe"]
            and mdd_pct >= thresholds["max_mdd_pct"]
            and n_trades >= thresholds["min_trades"]
        )
        else "NO-GO"
    )

    return {
        "n_trades": n_trades,
        "sharpe": round(sharpe, 4),
        "cagr_pct": round(cagr * 100, 2),
        "mdd_pct": round(mdd_pct, 2),
        "final_equity": round(equity, 2),
        "n_pairs": len(active_pairs),
        "activation": activation,
        "thresholds": thresholds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated pairs trading backtest")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--price-file",
        default="data/sample/watchlist_2020_2026.parquet",
    )
    parser.add_argument("--start-capital", type=float, default=100_000.0)
    parser.add_argument("--entry-z", type=float, default=1.8)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--max-pairs", type=int, default=20)
    args = parser.parse_args(argv)

    prices = _load_prices(args.price_file, args.start, args.end)
    if prices.empty:
        logger.error("No data loaded — check --start/--end and price file")
        return 1

    logger.info("Running pairs backtest %s → %s", args.start, args.end)
    results = _run_pairs_backtest(
        prices,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        start_capital=args.start_capital,
        max_pairs=args.max_pairs,
    )

    if not results:
        return 1

    print("\n" + "=" * 50)
    print("PAIRS TRADING BACKTEST RESULTS")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    print(f"  ACTIVATION: {results['activation']}")

    out_dir = ROOT / "output" / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"pairs_trading_v1_{ts}.json"
    payload = {
        "strategy": "pairs_trading_v1",
        "start": args.start,
        "end": args.end,
        "price_file": args.price_file,
        "run_at": datetime.now(timezone.utc).isoformat(),
        **results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_path)

    return 0 if results.get("activation") == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
