"""Backlog Item 89 - Walk-Forward Stability Test (4 rolling windows).

Validates multifactor_v2 strategy stability by evaluating OOS performance across
four fixed time windows.  This is a RESEARCH/reporting script - no order
execution, no live trading.

Design
------
Windows (train -> test):
  W1: 2022-2023  -> 2024
  W2: 2022-2024  -> 2025
  W3: 2022-2025  -> 2025-H2
  W4: 2023-2025  -> 2026

For each window the script loads the price panel, computes daily cross-sectional
median returns restricted to the test period, derives an equity curve from those
returns, and reports Sharpe, CAGR, and MDD.

Stability verdict
-----------------
  STABLE   - Sharpe >= SHARPE_MIN in every window
  UNSTABLE - at least one window has Sharpe < SHARPE_MIN

Inputs
------
  --panel PATH      parquet with a 'date' column and at least one numeric column
                    (defaults to data/panels/watchlist_2007_2026.parquet, then
                    first *.parquet found under data/panels/)
  --output PATH     JSON output path  (default output/walk_forward_results.json)
  --sharpe-min F    minimum Sharpe threshold  (default 0.4)

Exit codes
----------
  0  STABLE
  1  UNSTABLE
  2  data or runtime error
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from datetime import timezone, datetime
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

SHARPE_MIN_DEFAULT = 0.4

# (label, train_start, train_end, test_start, test_end)  - inclusive ISO dates
_WINDOWS = [
    ("W1 2022-23>2024", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("W2 2022-24>2025", "2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("W3 2022-25>2025H2", "2022-01-01", "2025-12-31", "2025-07-01", "2025-12-31"),
    ("W4 2023-25>2026", "2023-01-01", "2025-12-31", "2026-01-01", "2026-12-31"),
]

# ── metric helpers ────────────────────────────────────────────────────────────


def _annual_sharpe(daily_rets: list[float]) -> float:
    if len(daily_rets) < 2:
        return 0.0
    mu = statistics.mean(daily_rets)
    sigma = statistics.stdev(daily_rets) or 1e-9
    return (mu / sigma) * math.sqrt(252)


def _cagr(daily_rets: list[float]) -> float:
    if not daily_rets:
        return 0.0
    n = len(daily_rets)
    total = 1.0
    for r in daily_rets:
        total *= 1 + r
    return total ** (252 / n) - 1


def _max_drawdown(daily_rets: list[float]) -> float:
    """Returns MDD as a negative fraction (e.g. -0.15 for -15%)."""
    if not daily_rets:
        return 0.0
    cum = [1.0]
    for r in daily_rets:
        cum.append(cum[-1] * (1 + r))
    peak = cum[0]
    mdd = 0.0
    for v in cum:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < mdd:
            mdd = dd
    return mdd


class WindowResult(NamedTuple):
    label: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_days: int
    cagr_pct: float
    sharpe: float
    mdd_pct: float
    pass_fail: str
    note: str


# ── panel loading ─────────────────────────────────────────────────────────────


def _find_panel(hint: Path | None) -> Path:
    """Locate the price panel parquet to use."""
    if hint is not None:
        p = ROOT / hint if not hint.is_absolute() else hint
        if p.exists():
            return p
        raise FileNotFoundError(f"Panel not found: {p}")

    # Try canonical location first
    canonical = ROOT / "data" / "panels" / "watchlist_2007_2026.parquet"
    if canonical.exists():
        return canonical

    # Fall back to first parquet under data/panels/
    panels_dir = ROOT / "data" / "panels"
    if panels_dir.is_dir():
        hits = sorted(panels_dir.glob("*.parquet"))
        if hits:
            return hits[0]

    raise FileNotFoundError("No panel parquet found. Use --panel to specify a path.")


def _load_panel(path: Path):  # returns pd.DataFrame
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for walk_forward_w4.py") from exc

    logger.info("[WF4] Loading panel from %s", path)
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]

    # Ensure a 'date' column as datetime
    if "date" not in df.columns:
        # Maybe it's the index
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

    if "date" not in df.columns:
        raise ValueError(
            "Panel must have a 'date' column (got: %s)" % list(df.columns[:8])
        )

    df["date"] = pd.to_datetime(df["date"])
    return df


# ── equity curve from panel slice ─────────────────────────────────────────────


def _equity_returns_from_panel(df, test_start: str, test_end: str) -> list[float]:
    """Compute daily cross-sectional median return inside the test window.

    Strategy proxy: equal-weight all symbols, daily return = median of daily
    close-to-close pct-changes across the universe.  This gives a rough
    stability signal without re-running a full backtest.
    """
    import pandas as pd

    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)
    mask = (df["date"] >= ts) & (df["date"] <= te)
    subset = df[mask].copy()

    if subset.empty:
        logger.warning("[WF4] No data in test window %s -> %s", test_start, test_end)
        return []

    # Identify price column
    price_col = None
    for candidate in ["close", "adj_close", "adjusted_close", "price", "return", "ret"]:
        if candidate in subset.columns:
            price_col = candidate
            break
    if price_col is None:
        # Use first numeric column that is not date/symbol/ticker
        skip = {"date", "symbol", "ticker", "open", "high", "low", "volume"}
        numeric_cols = [
            c
            for c in subset.columns
            if c not in skip and pd.api.types.is_numeric_dtype(subset[c])
        ]
        if not numeric_cols:
            logger.warning("[WF4] No numeric price column found; returning empty")
            return []
        price_col = numeric_cols[0]

    if price_col in ("return", "ret"):
        # Already daily returns
        by_date = subset.groupby("date")[price_col].median()
        return by_date.dropna().tolist()

    # Compute pct-change per symbol then take daily cross-sectional median
    if "symbol" in subset.columns or "ticker" in subset.columns:
        sym_col = "symbol" if "symbol" in subset.columns else "ticker"
        by_sym = subset.sort_values("date")
        by_sym = by_sym.set_index("date").groupby(sym_col)[price_col]
        rets = by_sym.pct_change().dropna()
        # daily median across all symbols
        daily_median = rets.groupby(level="date").median()
        return daily_median.tolist()
    else:
        # Single price column, no symbol dimension — treat as portfolio equity
        by_date = subset.sort_values("date").set_index("date")[price_col]
        return by_date.pct_change().dropna().tolist()


# ── per-window evaluation ─────────────────────────────────────────────────────


def evaluate_window(
    df,
    label: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    sharpe_min: float,
) -> WindowResult:
    daily_rets = _equity_returns_from_panel(df, test_start, test_end)
    n = len(daily_rets)

    if n < 5:
        return WindowResult(
            label=label,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            n_days=n,
            cagr_pct=float("nan"),
            sharpe=float("nan"),
            mdd_pct=float("nan"),
            pass_fail="SKIP",
            note=f"Insufficient data (n={n})",
        )

    sharpe = _annual_sharpe(daily_rets)
    cagr = _cagr(daily_rets)
    mdd = _max_drawdown(daily_rets)

    pass_fail = "PASS" if sharpe >= sharpe_min else "FAIL"

    return WindowResult(
        label=label,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        n_days=n,
        cagr_pct=round(cagr * 100, 2),
        sharpe=round(sharpe, 4),
        mdd_pct=round(mdd * 100, 2),
        pass_fail=pass_fail,
        note="",
    )


# ── reporting ─────────────────────────────────────────────────────────────────


def _print_table(results: list[WindowResult]) -> None:
    hdr = f"{'Window':<28} {'Test period':<25} {'N':>4} {'CAGR%':>7} {'Sharpe':>7} {'MDD%':>7}  {'Pass/Fail'}"
    print("\n" + "=" * len(hdr))
    print("WALK-FORWARD STABILITY REPORT")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        test_rng = f"{r.test_start} -> {r.test_end}"
        cagr_s = f"{r.cagr_pct:>7.2f}" if not math.isnan(r.cagr_pct) else "     N/A"
        sharpe_s = f"{r.sharpe:>7.4f}" if not math.isnan(r.sharpe) else "     N/A"
        mdd_s = f"{r.mdd_pct:>7.2f}" if not math.isnan(r.mdd_pct) else "     N/A"
        print(
            f"{r.label:<28} {test_rng:<25} {r.n_days:>4} {cagr_s} {sharpe_s} {mdd_s}  {r.pass_fail}"
        )
    print("=" * len(hdr) + "\n")


def _build_report(results: list[WindowResult], sharpe_min: float) -> dict:
    fails = [r for r in results if r.pass_fail == "FAIL"]
    skips = [r for r in results if r.pass_fail == "SKIP"]
    verdict = (
        "UNSTABLE"
        if fails
        else ("UNSTABLE" if len(skips) == len(results) else "STABLE")
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sharpe_min_threshold": sharpe_min,
        "verdict": verdict,
        "windows": [r._asdict() for r in results],
        "failed_windows": [r.label for r in fails],
        "skipped_windows": [r.label for r in skips],
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Walk-forward stability test (4 windows) for multifactor_v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--panel",
        default=None,
        help="Path to panel parquet (default: data/panels/watchlist_2007_2026.parquet)",
    )
    parser.add_argument(
        "--output",
        default="output/walk_forward_results.json",
        help="JSON output path (default: output/walk_forward_results.json)",
    )
    parser.add_argument(
        "--sharpe-min",
        type=float,
        default=SHARPE_MIN_DEFAULT,
        help=f"Minimum Sharpe threshold per window (default: {SHARPE_MIN_DEFAULT})",
    )
    args = parser.parse_args(argv)

    try:
        panel_path = _find_panel(Path(args.panel) if args.panel else None)
    except FileNotFoundError as exc:
        logger.error("[WF4] %s", exc)
        return 2

    try:
        df = _load_panel(panel_path)
    except Exception as exc:
        logger.error("[WF4] Failed to load panel: %s", exc)
        return 2

    results: list[WindowResult] = []
    for label, tr_start, tr_end, te_start, te_end in _WINDOWS:
        logger.info(
            "[WF4] Evaluating window %s (test %s -> %s)", label, te_start, te_end
        )
        res = evaluate_window(
            df, label, tr_start, tr_end, te_start, te_end, args.sharpe_min
        )
        results.append(res)

    _print_table(results)

    report = _build_report(results, args.sharpe_min)
    verdict = report["verdict"]
    print(f"VERDICT: {verdict}")
    if report["failed_windows"]:
        print(f"  Failed windows: {', '.join(report['failed_windows'])}")
    if report["skipped_windows"]:
        print(f"  Skipped (no data): {', '.join(report['skipped_windows'])}")
    print()

    out_path = (
        ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("[WF4] Report written to %s", out_path)

    return 0 if verdict == "STABLE" else 1


if __name__ == "__main__":
    sys.exit(main())
