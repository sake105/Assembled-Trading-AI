"""Side-by-side paper trading simulation: V1 (EMA signals) vs V2 (multifactor).

Runs both strategies over N days of historical price data using the same:
  - Seed capital
  - FillModel (half-spread + market impact)
  - Universe

Reports per-strategy: equity curve, Sharpe ratio, MaxDD, turnover.
Outputs JSON comparison report.

Usage
-----
python scripts/comparison/paper_trade_v1_v2.py
python scripts/comparison/paper_trade_v1_v2.py \\
    --n-days 60 \\
    --seed-capital 100000 \\
    --price-dir data/raw/equities_eod/yfinance \\
    --output-path output/comparison/paper_comparison_report.json

Log prefix: [PAPER-V1V2]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Strategy imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from src.assembled_core.strategies.ema_trend_v0 import (
        compute_signals as _v1_compute_signals,
    )
    _V1_AVAILABLE = True
except Exception as _e_v1:
    _V1_AVAILABLE = False

try:
    from src.assembled_core.strategies.multifactor_v2 import (
        compute_signals as _v2_compute_signals,
    )
    _V2_AVAILABLE = True
except Exception as _e_v2:
    _V2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)
_TAG = "[PAPER-V1V2]"


def _log(msg: str) -> None:
    logger.info("%s %s", _TAG, msg)


def _warn(msg: str) -> None:
    logger.warning("%s %s", _TAG, msg)


# ---------------------------------------------------------------------------
# FillModel
# ---------------------------------------------------------------------------

def _simulate_fill_price(
    price: float,
    qty: float,
    adv_proxy: float,
    side: str = "BUY",
    half_spread_bps: float = 5.0,
    impact_coefficient: float = 0.1,
) -> float:
    """Apply half-spread + square-root market impact to a mid price.

    BUY:  fill = price * (1 + slippage_fraction)
    SELL: fill = price * (1 - slippage_fraction)
    """
    pov = min(float(qty) / max(float(adv_proxy), 1.0), 1.0)
    slippage_bps = float(half_spread_bps) + float(impact_coefficient) * np.sqrt(pov) * 10_000.0
    slippage_frac = slippage_bps / 10_000.0
    if side.upper() == "BUY":
        return float(price) * (1.0 + slippage_frac)
    return float(price) * (1.0 - slippage_frac)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DaySnapshot:
    date: str
    equity: float
    cash: float
    n_positions: int
    turnover: float  # total notional traded today


@dataclass
class StrategyResult:
    name: str
    daily_snapshots: list[DaySnapshot] = field(default_factory=list)
    final_equity: float = 0.0
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_turnover: float = 0.0
    annualised_turnover: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)
    available: bool = True
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------

def _load_prices(price_dir: Path, n_days: int) -> pd.DataFrame:
    """Load parquet price files from price_dir, return last n_days of data."""
    parquets = sorted(price_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {price_dir}")

    _log(f"Loading {len(parquets)} parquet files from {price_dir}")
    frames: list[pd.DataFrame] = []
    for fp in parquets:
        try:
            df = pd.read_parquet(fp)
            frames.append(df)
        except Exception as exc:
            _warn(f"Skipping {fp.name}: {exc!r}")

    if not frames:
        raise ValueError(f"No price data loaded from {price_dir}")

    combined = pd.concat(frames, ignore_index=True)

    # Normalise timestamp
    ts_col = "timestamp" if "timestamp" in combined.columns else "date"
    if ts_col not in combined.columns:
        raise ValueError(f"No timestamp/date column in price data; got {combined.columns.tolist()}")

    combined["timestamp"] = pd.to_datetime(combined[ts_col], utc=True, errors="coerce").dt.tz_localize(None)
    combined = combined.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"])

    # Keep only last n_days of calendar dates
    all_dates = sorted(combined["timestamp"].dt.normalize().unique())
    if len(all_dates) > n_days:
        cutoff = all_dates[-n_days]
        combined = combined[combined["timestamp"].dt.normalize() >= cutoff]

    _log(
        f"Prices: {len(combined)} rows, "
        f"{combined['symbol'].nunique()} symbols, "
        f"{combined['timestamp'].dt.date.min()} to "
        f"{combined['timestamp'].dt.date.max()}"
    )
    return combined.reset_index(drop=True)


def _load_prices_from_panel(panel_path: Path, n_days: int) -> pd.DataFrame:
    """Load OHLCV data from a factor panel (fallback data source)."""
    _log(f"Loading prices from factor panel: {panel_path}")
    df = pd.read_parquet(panel_path)

    ts_col = "timestamp" if "timestamp" in df.columns else "date"
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=["timestamp", "close"])

    all_dates = sorted(df["timestamp"].unique())
    if len(all_dates) > n_days:
        cutoff = all_dates[-n_days]
        df = df[df["timestamp"] >= cutoff]

    _log(
        f"Panel prices: {len(df)} rows, "
        f"{df['symbol'].nunique()} symbols"
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _get_longs(signals: pd.DataFrame) -> set[str]:
    """Extract LONG symbols from a signals DataFrame."""
    if signals is None or signals.empty:
        return set()
    if "direction" in signals.columns:
        return set(signals[signals["direction"] == "LONG"]["symbol"].tolist())
    if "signal" in signals.columns:
        return set(signals[signals["signal"] > 0]["symbol"].tolist())
    return set()


def _compute_v1_signals(prices_up_to: pd.DataFrame) -> set[str]:
    """Compute V1 EMA signals for the given price window."""
    if not _V1_AVAILABLE or prices_up_to.empty:
        return set()
    try:
        sig = _v1_compute_signals(prices_up_to, ema_fast=20, ema_slow=60)
        return _get_longs(sig)
    except Exception as exc:
        logger.debug("%s V1 signal error: %s", _TAG, exc)
        return set()


def _compute_v2_signals(prices_up_to: pd.DataFrame) -> set[str]:
    """Compute V2 multifactor signals for the given price window."""
    if not _V2_AVAILABLE or prices_up_to.empty:
        return set()
    try:
        sig = _v2_compute_signals(prices_up_to)
        return _get_longs(sig)
    except Exception as exc:
        logger.debug("%s V2 signal error: %s", _TAG, exc)
        return set()


def _ema_signal_fallback(prices_up_to: pd.DataFrame, fast: int = 20, slow: int = 60) -> set[str]:
    """Pure-pandas EMA crossover fallback (used when V1 import fails)."""
    if prices_up_to.empty or "close" not in prices_up_to.columns:
        return set()

    longs = set()
    for symbol, grp in prices_up_to.groupby("symbol", sort=False):
        grp = grp.sort_values("timestamp")
        if len(grp) < slow:
            continue
        closes = grp["close"].values
        ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
        if ema_fast[-1] > ema_slow[-1]:
            longs.add(str(symbol))
    return longs


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

def _simulate_strategy(
    prices: pd.DataFrame,
    strategy_name: str,
    signal_fn,
    seed_capital: float,
    half_spread_bps: float,
    impact_coefficient: float,
    max_positions: int,
    target_position_size: float,
) -> StrategyResult:
    """Run a single strategy simulation over all dates in prices.

    Uses EOD close prices for mark-to-market and fills.
    """
    result = StrategyResult(name=strategy_name)

    all_dates = sorted(prices["timestamp"].dt.normalize().unique())
    n = len(all_dates)

    cash = seed_capital
    # positions: symbol -> {qty, avg_price}
    positions: dict[str, dict[str, float]] = {}
    equity_curve: list[float] = []

    for i, dt in enumerate(all_dates):
        # Prices available up to (and including) dt -- no look-ahead
        prices_up_to = prices[prices["timestamp"].dt.normalize() <= dt]
        prices_today = prices[prices["timestamp"].dt.normalize() == dt]

        if prices_today.empty:
            equity_curve.append(cash + _mark_positions(positions, prices_up_to))
            continue

        # Build close-price dict for today
        today_closes: dict[str, float] = {}
        for _, row in prices_today.iterrows():
            sym = str(row["symbol"])
            if "close" in prices_today.columns and not pd.isna(row["close"]):
                today_closes[sym] = float(row["close"])

        if not today_closes:
            equity_curve.append(cash + _mark_positions(positions, prices_up_to))
            continue

        # Compute signals
        target_longs = signal_fn(prices_up_to)

        # Estimate ADV proxy as 10x today's close (simplification)
        adv_proxy: dict[str, float] = {sym: close * 10_000 for sym, close in today_closes.items()}

        # Current portfolio value for position sizing
        portfolio_value = cash + _mark_positions(positions, prices_up_to)
        target_position_notional = portfolio_value * target_position_size

        daily_turnover = 0.0

        # --- SELL: exit positions not in target longs ---
        symbols_to_exit = [s for s in list(positions.keys()) if s not in target_longs]
        for sym in symbols_to_exit:
            pos = positions[sym]
            qty = pos["qty"]
            if sym not in today_closes or qty <= 0:
                continue
            mid = today_closes[sym]
            fill = _simulate_fill_price(mid, qty, adv_proxy.get(sym, mid * 1000),
                                        side="SELL",
                                        half_spread_bps=half_spread_bps,
                                        impact_coefficient=impact_coefficient)
            proceeds = fill * qty
            cash += proceeds
            daily_turnover += proceeds
            del positions[sym]

        # --- BUY: enter new target longs ---
        current_positions = set(positions.keys())
        new_longs = [s for s in target_longs if s not in current_positions]
        n_slots = max(0, max_positions - len(current_positions))

        for sym in new_longs[:n_slots]:
            if sym not in today_closes:
                continue
            mid = today_closes[sym]
            if mid <= 0:
                continue
            # Size: target notional / price = qty
            qty = int(target_position_notional / mid)
            if qty <= 0:
                continue
            cost = qty * mid  # approximate notional
            if cost > cash * 0.95:  # keep 5% cash buffer
                qty = int(cash * 0.95 / mid)
            if qty <= 0:
                continue
            fill = _simulate_fill_price(mid, qty, adv_proxy.get(sym, mid * 1000),
                                        side="BUY",
                                        half_spread_bps=half_spread_bps,
                                        impact_coefficient=impact_coefficient)
            cost_actual = fill * qty
            if cost_actual > cash:
                continue
            cash -= cost_actual
            daily_turnover += cost_actual
            positions[sym] = {"qty": float(qty), "avg_price": fill}

        # EOD mark
        mark = _mark_positions(positions, prices_today)
        eod_equity = cash + mark

        equity_curve.append(eod_equity)
        result.daily_snapshots.append(
            DaySnapshot(
                date=str(dt.date()),
                equity=round(eod_equity, 2),
                cash=round(cash, 2),
                n_positions=len(positions),
                turnover=round(daily_turnover, 2),
            )
        )

    result.equity_curve = [round(e, 2) for e in equity_curve]

    # Compute metrics
    if len(equity_curve) >= 2:
        rets = np.diff(equity_curve) / np.array(equity_curve[:-1])
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
        result.daily_returns = [round(float(r), 8) for r in rets]

        std = float(rets.std(ddof=1))
        mean_r = float(rets.mean())
        result.sharpe = round(mean_r / std * np.sqrt(252), 4) if std > 1e-10 else 0.0

        # MaxDD
        eq_arr = np.array(equity_curve)
        roll_max = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - roll_max) / np.where(roll_max > 0, roll_max, 1)
        result.max_drawdown = round(float(dd.min()), 4)

        result.total_return = round(
            float(equity_curve[-1] / equity_curve[0] - 1) if equity_curve[0] != 0 else 0.0, 4
        )
        result.final_equity = round(equity_curve[-1], 2)

        # Turnover
        total_tv = sum(s.turnover for s in result.daily_snapshots)
        avg_equity = float(np.mean(equity_curve)) if equity_curve else seed_capital
        result.total_turnover = round(total_tv, 2)
        if avg_equity > 0 and n > 0:
            result.annualised_turnover = round(
                (total_tv / avg_equity) * (252.0 / max(n, 1)), 4
            )

    return result


def _mark_positions(
    positions: dict[str, dict[str, float]],
    prices_df: pd.DataFrame,
) -> float:
    """Mark open positions to latest available prices."""
    if not positions or prices_df.empty or "close" not in prices_df.columns:
        return 0.0

    # Build last-close lookup from prices_df
    close_map: dict[str, float] = {}
    for sym, grp in prices_df.groupby("symbol", sort=False):
        last_close = grp["close"].dropna().iloc[-1] if not grp["close"].dropna().empty else np.nan
        if not np.isnan(last_close):
            close_map[str(sym)] = float(last_close)

    total = 0.0
    for sym, pos in positions.items():
        price = close_map.get(sym, pos.get("avg_price", 0.0))
        total += float(pos["qty"]) * price
    return total


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def _compare_results(v1: StrategyResult, v2: StrategyResult) -> dict[str, Any]:
    """Compute side-by-side comparison dict."""
    def _s(r: StrategyResult) -> dict[str, Any]:
        return {
            "final_equity": r.final_equity,
            "total_return": r.total_return,
            "sharpe": r.sharpe,
            "max_drawdown": r.max_drawdown,
            "annualised_turnover": r.annualised_turnover,
            "n_days": len(r.equity_curve),
            "available": r.available,
            "skip_reason": r.skip_reason,
        }

    cmp: dict[str, Any] = {
        "v1": _s(v1),
        "v2": _s(v2),
    }

    if v1.available and v2.available and v1.daily_returns and v2.daily_returns:
        min_len = min(len(v1.daily_returns), len(v2.daily_returns))
        r1 = np.array(v1.daily_returns[:min_len])
        r2 = np.array(v2.daily_returns[:min_len])
        if r1.std() > 0 and r2.std() > 0:
            corr = float(np.corrcoef(r1, r2)[0, 1])
        else:
            corr = float("nan")
        te_daily = float((r2 - r1).std(ddof=1)) if min_len >= 2 else float("nan")
        te_ann = te_daily * np.sqrt(252) if not np.isnan(te_daily) else float("nan")

        cmp["comparison"] = {
            "sharpe_improvement": round(v2.sharpe - v1.sharpe, 4),
            "return_improvement": round(v2.total_return - v1.total_return, 4),
            "maxdd_improvement": round(v2.max_drawdown - v1.max_drawdown, 4),
            "return_correlation": round(corr, 4) if not np.isnan(corr) else None,
            "tracking_error_annualised": round(te_ann, 4) if not np.isnan(te_ann) else None,
            "v2_better_sharpe": bool(v2.sharpe > v1.sharpe),
            "v2_better_return": bool(v2.total_return > v1.total_return),
            "v2_lower_maxdd": bool(v2.max_drawdown > v1.max_drawdown),
        }
    else:
        cmp["comparison"] = {"note": "one_or_both_strategies_unavailable"}

    return cmp


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_paper_comparison(
    price_dir: Path | None,
    panel_path: Path | None,
    n_days: int,
    seed_capital: float,
    half_spread_bps: float,
    impact_coefficient: float,
    max_positions: int,
    target_position_size: float,
    output_path: Path,
) -> dict[str, Any]:
    """Main comparison pipeline."""
    _log("=" * 60)
    _log("paper_trade_v1_v2.py -- START")
    _log(f"n_days           : {n_days}")
    _log(f"seed_capital     : {seed_capital}")
    _log(f"half_spread_bps  : {half_spread_bps}")
    _log(f"impact_coeff     : {impact_coefficient}")
    _log(f"V1 available     : {_V1_AVAILABLE}")
    _log(f"V2 available     : {_V2_AVAILABLE}")
    _log("=" * 60)

    # Load prices
    prices: pd.DataFrame | None = None
    if price_dir and price_dir.exists():
        try:
            prices = _load_prices(price_dir, n_days)
        except Exception as exc:
            _warn(f"Cannot load from price_dir: {exc!r}")

    if prices is None and panel_path and panel_path.exists():
        try:
            prices = _load_prices_from_panel(panel_path, n_days)
        except Exception as exc:
            _warn(f"Cannot load from panel: {exc!r}")

    if prices is None or prices.empty:
        _warn("No price data available -- cannot run comparison.")
        report: dict[str, Any] = {
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_days_requested": n_days,
            "error": "no_price_data",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=True)
        return report

    # ---- V1 simulation -------------------------------------------------------
    _log("[START] V1 simulation (EMA signals)")
    if _V1_AVAILABLE:
        v1_fn = _compute_v1_signals
        v1_result = _simulate_strategy(
            prices, "V1_EMA", v1_fn, seed_capital,
            half_spread_bps, impact_coefficient, max_positions, target_position_size,
        )
    else:
        _warn("V1 strategy import failed -- using pure-pandas EMA fallback.")
        v1_result = _simulate_strategy(
            prices, "V1_EMA_fallback",
            lambda p: _ema_signal_fallback(p, fast=20, slow=60),
            seed_capital, half_spread_bps, impact_coefficient,
            max_positions, target_position_size,
        )
        v1_result.skip_reason = "import_failed_used_fallback"

    _log(
        f"[OK] V1 done: return={v1_result.total_return:.2%}, "
        f"sharpe={v1_result.sharpe:.3f}, "
        f"maxdd={v1_result.max_drawdown:.2%}"
    )

    # ---- V2 simulation -------------------------------------------------------
    _log("[START] V2 simulation (multifactor signals)")
    if _V2_AVAILABLE:
        v2_fn = _compute_v2_signals
        v2_result = _simulate_strategy(
            prices, "V2_multifactor", v2_fn, seed_capital,
            half_spread_bps, impact_coefficient, max_positions, target_position_size,
        )
    else:
        _warn("V2 strategy import failed -- V2 result will be empty.")
        v2_result = StrategyResult(name="V2_multifactor", available=False, skip_reason="import_failed")

    if v2_result.available:
        _log(
            f"[OK] V2 done: return={v2_result.total_return:.2%}, "
            f"sharpe={v2_result.sharpe:.3f}, "
            f"maxdd={v2_result.max_drawdown:.2%}"
        )

    # ---- Comparison ----------------------------------------------------------
    comparison = _compare_results(v1_result, v2_result)

    # ---- Build equity curve summary (compact, not every row) ----------------
    def _eq_summary(r: StrategyResult) -> dict[str, Any]:
        n = len(r.equity_curve)
        if n == 0:
            return {}
        # Sample at most 100 points
        step = max(1, n // 100)
        return {
            "sampled_equity": r.equity_curve[::step],
            "snapshots_count": n,
            "daily_return_mean": round(float(np.mean(r.daily_returns)), 6) if r.daily_returns else None,
            "daily_return_std": round(float(np.std(r.daily_returns, ddof=1)), 6) if len(r.daily_returns) > 1 else None,
        }

    # ---- Assemble full report ------------------------------------------------
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "n_days": n_days,
            "seed_capital": seed_capital,
            "half_spread_bps": half_spread_bps,
            "impact_coefficient": impact_coefficient,
            "max_positions": max_positions,
            "target_position_size": target_position_size,
        },
        "v1": comparison["v1"],
        "v2": comparison["v2"],
        "comparison": comparison.get("comparison", {}),
        "v1_equity_curve": _eq_summary(v1_result),
        "v2_equity_curve": _eq_summary(v2_result),
    }

    # ---- Save ---------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=True, default=str)

    _log(f"[OK] Report written to {output_path}")
    cmp = comparison.get("comparison", {})
    _log(
        f"Sharpe improvement: {cmp.get('sharpe_improvement', 'n/a')} | "
        f"V2 better Sharpe: {cmp.get('v2_better_sharpe', 'n/a')}"
    )
    _log("paper_trade_v1_v2.py -- DONE")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Side-by-side paper trading comparison: V1 (EMA) vs V2 (multifactor).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=60,
        help="Number of trading days to simulate.",
    )
    parser.add_argument(
        "--seed-capital",
        type=float,
        default=100_000.0,
        help="Starting capital for both strategies.",
    )
    parser.add_argument(
        "--price-dir",
        type=Path,
        default=Path("data/raw/equities_eod/yfinance"),
        help="Directory of per-symbol OHLCV parquet files.",
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=Path("output/factor_panels/full_panel_7y.parquet"),
        help="Fallback factor panel path (used when price-dir is unavailable).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("output/comparison/paper_comparison_report.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--half-spread-bps",
        type=float,
        default=5.0,
        help="Fixed half-spread in bps for the FillModel.",
    )
    parser.add_argument(
        "--impact-coefficient",
        type=float,
        default=0.1,
        help="Price-impact coefficient for the FillModel.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=20,
        help="Maximum concurrent positions per strategy.",
    )
    parser.add_argument(
        "--target-position-size",
        type=float,
        default=0.05,
        help="Target position size as fraction of portfolio (e.g. 0.05 = 5%%).",
    )
    args = parser.parse_args(argv)

    price_dir = args.price_dir if args.price_dir.exists() else None
    panel_path = args.panel_path if args.panel_path.exists() else None

    if price_dir is None:
        _warn(f"price-dir not found ({args.price_dir}) -- will try panel fallback.")
    if panel_path is None and price_dir is None:
        _warn(f"panel-path not found ({args.panel_path}) -- no price data available.")

    run_paper_comparison(
        price_dir=price_dir,
        panel_path=panel_path,
        n_days=args.n_days,
        seed_capital=args.seed_capital,
        half_spread_bps=args.half_spread_bps,
        impact_coefficient=args.impact_coefficient,
        max_positions=args.max_positions,
        target_position_size=args.target_position_size,
        output_path=args.output_path.resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
