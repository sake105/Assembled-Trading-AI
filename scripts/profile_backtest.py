"""B0 — cProfile wrapper for backtest performance investigation.

Runs the reference backtest configuration and writes a .prof file plus a flat
pstats report. Used to identify hotspots before B1-B5 speed work so that each
hotspot fix can be justified against a real profile rather than guesswork.

Usage:
    python scripts/profile_backtest.py
    python scripts/profile_backtest.py --label post_b1
    python scripts/profile_backtest.py --n-symbols 25 --n-days 126

Reference config (plan default): 25 symbols, 126 days (H1 2024-ish), seed=42.
Output: output/profiling/backtest_<label>_<timestamp>.prof + .txt flat report.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.backtest_engine import run_portfolio_backtest  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_reference_prices(n_symbols: int, n_days: int, seed: int) -> pd.DataFrame:
    """Deterministic synthetic prices. Matches the B0 reference config."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2024-01-02", periods=n_days, freq="B", tz="UTC")
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    rows = []
    for sym_idx, symbol in enumerate(symbols):
        base = 50.0 + (sym_idx * 7) % 150  # deterministic base spread
        rets = rng.normal(0.0004, 0.018, len(dates))
        closes = base * np.exp(np.cumsum(rets))
        for i, date in enumerate(dates):
            rows.append(
                {"timestamp": date, "symbol": symbol, "close": float(closes[i])}
            )
    return (
        pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    )


def simple_signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
    grouped = prices_df.groupby("symbol", sort=False)
    frames = []
    for symbol, g in grouped:
        g = g.sort_values("timestamp").copy()
        ma = g["close"].rolling(20, min_periods=5).mean()
        direction = np.where(g["close"] > ma, "LONG", "FLAT")
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": g["timestamp"].values,
                    "symbol": symbol,
                    "direction": direction,
                    "score": np.where(direction == "LONG", 1.0, 0.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def equal_weight_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Equal-weight sizing over LONG signals in the latest bar only.

    ``target_qty`` is treated as **notional** by order_generation; we emit the
    per-symbol dollar target, not an estimated share count.
    """
    if signals_df.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    last_ts = signals_df["timestamp"].max()
    last = signals_df[signals_df["timestamp"] == last_ts]
    long_syms = sorted(last.loc[last["direction"] == "LONG", "symbol"].unique())
    if not long_syms:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    w = 1.0 / len(long_syms)
    return pd.DataFrame(
        {
            "symbol": long_syms,
            "target_weight": [w] * len(long_syms),
            "target_qty": [capital * w] * len(long_syms),
        }
    )


def run_reference_backtest(prices: pd.DataFrame, seed: int) -> dict:
    start = time.perf_counter()
    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=simple_signal_fn,
        position_sizing_fn=equal_weight_sizing_fn,
        start_capital=100_000.0,
        include_costs=True,
        include_trades=False,
        include_ledger=False,
        strict_session_gate=False,
    )
    wall = time.perf_counter() - start
    equity_df = result.equity
    final_equity = float(equity_df["equity"].iloc[-1])
    return {
        "wall_seconds": wall,
        "final_equity": final_equity,
        "n_bars": len(equity_df),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-symbols", type=int, default=25)
    parser.add_argument("--n-days", type=int, default=126)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="baseline", help="label for output file")
    parser.add_argument("--top", type=int, default=30, help="top-N rows in flat report")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "profiling"),
        help="directory for .prof + .txt outputs",
    )
    args = parser.parse_args()

    prices = build_reference_prices(args.n_symbols, args.n_days, args.seed)
    logger.info(
        "[PROFILE] reference fixture: %d symbols, %d days, seed=%d (%d rows)",
        args.n_symbols,
        args.n_days,
        args.seed,
        len(prices),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prof_path = output_dir / f"backtest_{args.label}_{ts}.prof"
    report_path = prof_path.with_suffix(".txt")

    profiler = cProfile.Profile()
    profiler.enable()
    summary = run_reference_backtest(prices, args.seed)
    profiler.disable()

    profiler.dump_stats(str(prof_path))

    buf = io.StringIO()
    ps = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    ps.print_stats(args.top)
    buf.write("\n--- sort=tottime ---\n")
    ps.sort_stats("tottime").print_stats(args.top)
    report = buf.getvalue()
    report_path.write_text(report, encoding="utf-8")

    logger.info(
        "[PROFILE] wall=%.3fs final_equity=%.2f bars=%d",
        summary["wall_seconds"],
        summary["final_equity"],
        summary["n_bars"],
    )
    logger.info("[PROFILE] prof=%s", prof_path)
    logger.info("[PROFILE] report=%s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
