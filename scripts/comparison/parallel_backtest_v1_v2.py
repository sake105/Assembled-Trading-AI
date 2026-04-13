"""Parallel backtest comparison: V1 (EMA baseline) vs V2 (multifactor_v2).

Usage:
    python scripts/comparison/parallel_backtest_v1_v2.py
    python scripts/comparison/parallel_backtest_v1_v2.py --start-date 2022-01-01 --end-date 2024-01-01
    python scripts/comparison/parallel_backtest_v1_v2.py --price-dir data/raw/equities_eod/yfinance --output-dir output/comparison

Log prefix: [V1V2]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
TAG = "[V1V2]"

# ---------------------------------------------------------------------------
# Path setup -- allow running from repo root without pip install
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Strategy imports (V2 is optional -- graceful degradation)
# ---------------------------------------------------------------------------

try:
    from src.assembled_core.strategies.ema_trend_v0 import (
        compute_signals as _v1_compute_signals,
    )
    _V1_AVAILABLE = True
except Exception as _e:
    logger.error("%s Failed to import V1 strategy: %s", TAG, _e)
    _V1_AVAILABLE = False

try:
    from src.assembled_core.strategies.multifactor_v2 import (
        compute_signals as _v2_compute_signals,
    )
    _V2_AVAILABLE = True
except Exception as _e:
    logger.warning("%s Failed to import V2 strategy: %s -- V2 will be skipped.", TAG, _e)
    _V2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    v1_metrics: dict = field(default_factory=dict)
    v2_metrics: dict = field(default_factory=dict)
    improvement: dict = field(default_factory=dict)
    statistical_tests: dict = field(default_factory=dict)
    is_significant: bool = False
    report_path: Path = Path("output/comparison/comparison_report.json")


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------


def load_prices(
    price_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load all parquet files from price_dir into a single DataFrame.

    Normalises timestamps to UTC-naive dates. Filters by date range if given.
    Returns columns: timestamp, symbol, open, high, low, close, volume.
    """
    parquet_files = list(price_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"{TAG} No parquet files found in {price_dir}")

    logger.info("%s Loading %d parquet files from %s", TAG, len(parquet_files), price_dir)
    frames: list[pd.DataFrame] = []
    for fp in parquet_files:
        try:
            df = pd.read_parquet(fp)
            frames.append(df)
        except Exception as exc:
            logger.warning("%s Skipping %s -- %s", TAG, fp.name, exc)

    if not frames:
        raise ValueError(f"{TAG} No price data loaded from {price_dir}")

    combined = pd.concat(frames, ignore_index=True)

    # Normalise timestamp column
    if "timestamp" not in combined.columns:
        raise ValueError(f"{TAG} 'timestamp' column missing in price data")

    ts = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined["timestamp"] = ts.dt.tz_localize(None)  # UTC-naive

    if start_date:
        combined = combined[combined["timestamp"] >= pd.Timestamp(start_date)]
    if end_date:
        combined = combined[combined["timestamp"] <= pd.Timestamp(end_date)]

    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info(
        "%s Loaded %d rows, %d symbols, date range %s - %s",
        TAG,
        len(combined),
        combined["symbol"].nunique(),
        combined["timestamp"].min().date(),
        combined["timestamp"].max().date(),
    )
    return combined


# ---------------------------------------------------------------------------
# Signal computation helpers
# ---------------------------------------------------------------------------


def _get_longs_for_date(signals: pd.DataFrame) -> set[str]:
    """Return symbols with LONG direction from a signal DataFrame."""
    if signals.empty or "direction" not in signals.columns:
        return set()
    longs = signals[signals["direction"] == "LONG"]
    return set(longs["symbol"].tolist())


def _compute_v1_signals_rolling(
    prices: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, set[str]]:
    """Compute V1 signals for each date (look-back uses all data up to that date)."""
    signals_by_date: dict[pd.Timestamp, set[str]] = {}
    prices_sorted = prices.sort_values(["symbol", "timestamp"])

    for dt in all_dates:
        subset = prices_sorted[prices_sorted["timestamp"] <= dt]
        if subset.empty:
            signals_by_date[dt] = set()
            continue
        try:
            sig = _v1_compute_signals(subset, ema_fast=20, ema_slow=60)
            signals_by_date[dt] = _get_longs_for_date(sig)
        except Exception as exc:
            logger.debug("%s V1 signal error on %s: %s", TAG, dt.date(), exc)
            signals_by_date[dt] = set()

    return signals_by_date


def _compute_v2_signals_rolling(
    prices: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, set[str]]:
    """Compute V2 signals for each date with score-based weighting support."""
    signals_by_date: dict[pd.Timestamp, set[str]] = {}
    prices_sorted = prices.sort_values(["symbol", "timestamp"])

    for dt in all_dates:
        subset = prices_sorted[prices_sorted["timestamp"] <= dt]
        if subset.empty:
            signals_by_date[dt] = set()
            continue
        try:
            try:
                from src.assembled_core.features.ta_features import compute_ta_features
                subset_with_features = compute_ta_features(subset)
            except Exception as _feat_exc:
                logger.warning("%s V2 TA features failed on %s, using raw prices: %s", TAG, dt.date(), _feat_exc)
                subset_with_features = subset
            sig = _v2_compute_signals(subset_with_features, strategy_cfg={})
            signals_by_date[dt] = _get_longs_for_date(sig)
        except Exception as exc:
            logger.warning("%s V2 signal error on %s: %s", TAG, dt.date(), exc)
            signals_by_date[dt] = set()

    return signals_by_date


def _compute_v2_scores_rolling(
    prices: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, dict[str, float]]:
    """Compute V2 score-weighted signals (symbol -> score) per date."""
    scores_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    prices_sorted = prices.sort_values(["symbol", "timestamp"])

    for dt in all_dates:
        subset = prices_sorted[prices_sorted["timestamp"] <= dt]
        if subset.empty:
            scores_by_date[dt] = {}
            continue
        try:
            try:
                from src.assembled_core.features.ta_features import compute_ta_features
                subset_with_features = compute_ta_features(subset)
            except Exception as _feat_exc:
                logger.warning("%s V2 TA features failed on %s (scores), using raw prices: %s", TAG, dt.date(), _feat_exc)
                subset_with_features = subset
            sig = _v2_compute_signals(subset_with_features, strategy_cfg={})
            longs = sig[sig["direction"] == "LONG"] if not sig.empty else sig
            if longs.empty or "score" not in longs.columns:
                scores_by_date[dt] = {}
            else:
                scores_by_date[dt] = dict(zip(longs["symbol"], longs["score"]))
        except Exception as exc:
            logger.warning("%s V2 score error on %s: %s", TAG, dt.date(), exc)
            scores_by_date[dt] = {}

    return scores_by_date


# ---------------------------------------------------------------------------
# Portfolio return computation
# ---------------------------------------------------------------------------


def _build_return_pivot(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a date x symbol DataFrame of daily close-to-close returns."""
    piv = prices.pivot_table(
        index="timestamp", columns="symbol", values="close", aggfunc="last"
    )
    piv = piv.sort_index()
    daily_ret = piv.pct_change()
    return daily_ret


def _equal_weight_portfolio_returns(
    signals_by_date: dict[pd.Timestamp, set[str]],
    return_pivot: pd.DataFrame,
) -> pd.Series:
    """Compute equal-weight portfolio daily returns.

    On each day d we form a portfolio from signals as of day d,
    then collect the NEXT day's returns (next-day fill assumption).
    """
    dates = return_pivot.index
    port_returns: list[float] = []
    port_dates: list[pd.Timestamp] = []

    for i in range(len(dates) - 1):
        signal_date = dates[i]
        return_date = dates[i + 1]

        longs = signals_by_date.get(signal_date, set())
        if not longs:
            port_returns.append(0.0)
            port_dates.append(return_date)
            continue

        available = [s for s in longs if s in return_pivot.columns]
        if not available:
            port_returns.append(0.0)
            port_dates.append(return_date)
            continue

        day_rets = return_pivot.loc[return_date, available].dropna()
        if day_rets.empty:
            port_returns.append(0.0)
        else:
            port_returns.append(float(day_rets.mean()))
        port_dates.append(return_date)

    return pd.Series(port_returns, index=pd.DatetimeIndex(port_dates), name="v1_returns")


def _score_weight_portfolio_returns(
    scores_by_date: dict[pd.Timestamp, dict[str, float]],
    return_pivot: pd.DataFrame,
) -> pd.Series:
    """Compute score-weighted portfolio daily returns (V2).

    Weights are proportional to each symbol's signal score, normalised to sum=1.
    """
    dates = return_pivot.index
    port_returns: list[float] = []
    port_dates: list[pd.Timestamp] = []

    for i in range(len(dates) - 1):
        signal_date = dates[i]
        return_date = dates[i + 1]

        scores = scores_by_date.get(signal_date, {})
        if not scores:
            port_returns.append(0.0)
            port_dates.append(return_date)
            continue

        available = {s: v for s, v in scores.items() if s in return_pivot.columns}
        if not available:
            port_returns.append(0.0)
            port_dates.append(return_date)
            continue

        total_score = sum(available.values())
        if total_score <= 0:
            weights = {s: 1.0 / len(available) for s in available}
        else:
            weights = {s: v / total_score for s, v in available.items()}

        syms = list(weights.keys())
        day_rets = return_pivot.loc[return_date, syms].dropna()
        if day_rets.empty:
            port_returns.append(0.0)
        else:
            port_ret = sum(weights.get(s, 0.0) * r for s, r in day_rets.items())
            port_returns.append(float(port_ret))
        port_dates.append(return_date)

    return pd.Series(port_returns, index=pd.DatetimeIndex(port_dates), name="v2_returns")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

TRADING_DAYS = 252


def compute_strategy_metrics(returns: pd.Series) -> dict:
    """Compute: Sharpe, Sortino, MaxDD, Calmar, annual_return, annual_vol, hit_rate, win_loss_ratio."""
    r = returns.dropna()
    n = len(r)

    if n < 10:
        return {
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "annual_return": float("nan"),
            "annual_vol": float("nan"),
            "hit_rate": float("nan"),
            "win_loss_ratio": float("nan"),
            "n_days": n,
        }

    annual_return = float((1 + r).prod() ** (TRADING_DAYS / n) - 1)
    annual_vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = annual_return / annual_vol if annual_vol > 1e-10 else float("nan")

    downside = r[r < 0]
    downside_vol = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else float("nan")
    sortino = annual_return / downside_vol if (downside_vol and downside_vol > 1e-10) else float("nan")

    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = float(dd.min())

    calmar = annual_return / abs(max_dd) if max_dd < -1e-10 else float("nan")

    hit_rate = float((r > 0).mean())
    wins = r[r > 0]
    losses = r[r < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-10 else float("nan")

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "annual_return": round(annual_return, 4),
        "annual_vol": round(annual_vol, 4),
        "hit_rate": round(hit_rate, 4),
        "win_loss_ratio": round(win_loss_ratio, 4),
        "n_days": n,
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def test_strategy_difference(
    v1_returns: pd.Series,
    v2_returns: pd.Series,
    n_bootstrap: int = 10000,
) -> dict:
    """Statistical significance:
    1. Bootstrap CI for Sharpe difference (10k resamples)
    2. Paired t-test on daily returns
    3. Report p-value and confidence interval
    """
    try:
        from scipy import stats as scipy_stats
        _scipy_ok = True
    except ImportError:
        logger.warning("%s scipy not available -- t-test skipped.", TAG)
        _scipy_ok = False

    # Align series on common dates
    aligned = pd.concat([v1_returns, v2_returns], axis=1, join="inner").dropna()
    r1 = aligned.iloc[:, 0].values
    r2 = aligned.iloc[:, 1].values
    n = len(r1)

    if n < 20:
        return {
            "n_paired_obs": n,
            "note": "Insufficient overlapping observations for statistical tests.",
        }

    # --- Paired t-test on daily returns difference ---
    diff = r2 - r1
    t_stat: float = float("nan")
    p_value: float = float("nan")
    if _scipy_ok:
        ttest = scipy_stats.ttest_1samp(diff, popmean=0.0)
        t_stat = float(ttest.statistic)
        p_value = float(ttest.pvalue)

    # --- Bootstrap CI for Sharpe difference ---
    rng = np.random.default_rng(seed=42)
    bs_sharpe_diffs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s1 = r1[idx]
        s2 = r2[idx]
        def _sharpe(x: np.ndarray) -> float:
            mu = x.mean()
            sd = x.std()
            if sd < 1e-10:
                return float("nan")
            return float(mu / sd * np.sqrt(TRADING_DAYS))

        sh1 = _sharpe(s1)
        sh2 = _sharpe(s2)
        if np.isfinite(sh1) and np.isfinite(sh2):
            bs_sharpe_diffs.append(sh2 - sh1)

    if bs_sharpe_diffs:
        ci_low = float(np.percentile(bs_sharpe_diffs, 2.5))
        ci_high = float(np.percentile(bs_sharpe_diffs, 97.5))
        bs_mean = float(np.mean(bs_sharpe_diffs))
    else:
        ci_low = ci_high = bs_mean = float("nan")

    return {
        "n_paired_obs": n,
        "n_bootstrap": n_bootstrap,
        "paired_ttest_statistic": round(t_stat, 4) if np.isfinite(t_stat) else None,
        "paired_ttest_pvalue": round(p_value, 6) if np.isfinite(p_value) else None,
        "bootstrap_sharpe_diff_mean": round(bs_mean, 4) if np.isfinite(bs_mean) else None,
        "bootstrap_sharpe_diff_ci_low": round(ci_low, 4) if np.isfinite(ci_low) else None,
        "bootstrap_sharpe_diff_ci_high": round(ci_high, 4) if np.isfinite(ci_high) else None,
        "ci_excludes_zero": bool(ci_low > 0) if np.isfinite(ci_low) else False,
    }


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _compute_improvement(v1: dict, v2: dict) -> dict:
    improvement: dict = {}
    for key in v1:
        if key == "n_days":
            continue
        a = v1.get(key)
        b = v2.get(key)
        if isinstance(a, float) and isinstance(b, float) and np.isfinite(a) and np.isfinite(b):
            improvement[key] = round(b - a, 4)
        else:
            improvement[key] = None
    return improvement


def _print_summary_table(
    v1_metrics: dict,
    v2_metrics: dict,
    improvement: dict,
) -> None:
    header = f"{'Metric':<22} {'V1 (EMA)':>12} {'V2 (MF)':>12} {'Delta':>12}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(f"{'  V1 vs V2 Strategy Comparison':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    def _fmt(v: object) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "     N/A"
        if isinstance(v, int):
            return f"{v:>12d}"
        return f"{v:>12.4f}"

    metrics_order = [
        "annual_return",
        "annual_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "win_loss_ratio",
        "n_days",
    ]
    for key in metrics_order:
        label = key.replace("_", " ").title()
        v1v = v1_metrics.get(key)
        v2v = v2_metrics.get(key)
        dv = improvement.get(key)
        print(f"{label:<22}{_fmt(v1v)}{_fmt(v2v)}{_fmt(dv)}")

    print(sep)
    print()


def _save_equity_curves(
    v1_returns: pd.Series,
    v2_returns: pd.Series | None,
    output_dir: Path,
) -> Path:
    csv_path = output_dir / "equity_curves.csv"
    v1_cum = (1 + v1_returns.fillna(0)).cumprod()
    v1_cum.name = "v1_equity"

    if v2_returns is not None:
        v2_cum = (1 + v2_returns.fillna(0)).cumprod()
        v2_cum.name = "v2_equity"
        df_out = pd.concat([v1_cum, v2_cum], axis=1)
    else:
        df_out = v1_cum.to_frame()

    df_out.to_csv(csv_path)
    logger.info("%s [OK] Equity curves saved to %s", TAG, csv_path)
    return csv_path


def _save_report(result: ComparisonResult, output_dir: Path) -> Path:
    report = {
        "v1_metrics": result.v1_metrics,
        "v2_metrics": result.v2_metrics,
        "improvement": result.improvement,
        "statistical_tests": result.statistical_tests,
        "is_significant": result.is_significant,
    }
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("%s [OK] Comparison report saved to %s", TAG, report_path)
    result.report_path = report_path
    return report_path


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------


def run_parallel_comparison(
    price_dir: Path = Path("data/raw/equities_eod/yfinance"),
    start_date: str | None = None,
    end_date: str | None = None,
    start_capital: float = 100_000.0,
    output_dir: Path = Path("output/comparison"),
) -> ComparisonResult:
    """Run V1 and V2 strategies on same price data and compare."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("%s [START] Parallel backtest comparison", TAG)
    logger.info(
        "%s price_dir=%s  start=%s  end=%s  capital=%.0f",
        TAG, price_dir, start_date or "earliest", end_date or "latest", start_capital,
    )

    if not _V1_AVAILABLE:
        raise RuntimeError(f"{TAG} V1 strategy (ema_trend_v0) is not available -- cannot continue.")

    # --- Load prices ---
    prices = load_prices(price_dir, start_date=start_date, end_date=end_date)

    # --- Build return pivot ---
    return_pivot = _build_return_pivot(prices)
    all_dates = return_pivot.index

    if len(all_dates) < 5:
        raise ValueError(f"{TAG} Not enough trading days ({len(all_dates)}) after filtering.")

    # --- V1: equal-weight signals ---
    logger.info("%s [START] Computing V1 (EMA) signals for %d dates ...", TAG, len(all_dates))
    v1_signals_by_date = _compute_v1_signals_rolling(prices, all_dates)
    v1_returns = _equal_weight_portfolio_returns(v1_signals_by_date, return_pivot)
    v1_returns.name = "v1_returns"
    logger.info(
        "%s [OK] V1 returns computed. Non-zero days: %d / %d",
        TAG, int((v1_returns != 0).sum()), len(v1_returns),
    )

    # --- V2: score-weighted signals (optional) ---
    v2_returns: pd.Series | None = None
    if _V2_AVAILABLE:
        logger.info("%s [START] Computing V2 (multifactor_v2) signals for %d dates ...", TAG, len(all_dates))
        try:
            v2_scores_by_date = _compute_v2_scores_rolling(prices, all_dates)
            v2_returns = _score_weight_portfolio_returns(v2_scores_by_date, return_pivot)
            v2_returns.name = "v2_returns"
            logger.info(
                "%s [OK] V2 returns computed. Non-zero days: %d / %d",
                TAG, int((v2_returns != 0).sum()), len(v2_returns),
            )
        except Exception as exc:
            logger.error("%s V2 computation failed: %s -- continuing with V1 only.", TAG, exc)
            v2_returns = None
    else:
        logger.warning("%s V2 not available -- comparison will be V1 only.", TAG)

    # --- Metrics ---
    v1_metrics = compute_strategy_metrics(v1_returns)
    v2_metrics = compute_strategy_metrics(v2_returns) if v2_returns is not None else {}

    # --- Statistical tests ---
    stat_tests: dict = {}
    if v2_returns is not None:
        logger.info("%s [START] Running statistical tests ...", TAG)
        stat_tests = test_strategy_difference(v1_returns, v2_returns, n_bootstrap=10_000)
        logger.info("%s [OK] Statistical tests complete.", TAG)

    # --- Improvement delta ---
    improvement = _compute_improvement(v1_metrics, v2_metrics) if v2_metrics else {}

    # --- Significance (bootstrap CI excludes zero AND p < 0.05) ---
    ci_ok = stat_tests.get("ci_excludes_zero", False)
    p_val = stat_tests.get("paired_ttest_pvalue")
    is_significant = bool(ci_ok and p_val is not None and p_val < 0.05)

    # --- Outputs ---
    _save_equity_curves(v1_returns, v2_returns, output_dir)

    result = ComparisonResult(
        v1_metrics=v1_metrics,
        v2_metrics=v2_metrics,
        improvement=improvement,
        statistical_tests=stat_tests,
        is_significant=is_significant,
        report_path=output_dir / "comparison_report.json",
    )
    _save_report(result, output_dir)

    # --- Console table ---
    if v2_metrics:
        _print_summary_table(v1_metrics, v2_metrics, improvement)
        sig_str = "YES (bootstrap CI excludes 0 and p < 0.05)" if is_significant else "NO"
        print(f"  Statistically significant improvement: {sig_str}")
        p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
        print(f"  Paired t-test p-value: {p_str}")
        ci_low = stat_tests.get("bootstrap_sharpe_diff_ci_low")
        ci_high = stat_tests.get("bootstrap_sharpe_diff_ci_high")
        ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]" if ci_low is not None else "N/A"
        print(f"  Bootstrap Sharpe diff 95% CI: {ci_str}")
        print()
    else:
        logger.warning("%s V2 not available -- only V1 metrics computed.", TAG)
        print("\nV1 (EMA baseline) metrics:")
        for k, v in v1_metrics.items():
            print(f"  {k:<22}: {v}")
        print()

    logger.info("%s [OK] Comparison complete. Report: %s", TAG, result.report_path)
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel backtest comparison: V1 (EMA) vs V2 (multifactor_v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--price-dir",
        default="data/raw/equities_eod/yfinance",
        help="Directory containing per-symbol parquet files.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date filter (YYYY-MM-DD). Inclusive.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date filter (YYYY-MM-DD). Inclusive.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/comparison",
        help="Directory for report JSON and equity curves CSV.",
    )
    parser.add_argument(
        "--start-capital",
        type=float,
        default=100_000.0,
        help="Starting capital (used in reporting context).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    result = run_parallel_comparison(
        price_dir=Path(args.price_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        start_capital=args.start_capital,
        output_dir=Path(args.output_dir),
    )
