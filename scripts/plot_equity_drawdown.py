#!/usr/bin/env python3
"""Equity curve + drawdown visualization.

Loads a backtest equity JSON or Parquet file and produces a two-panel plot:
  - Top panel: equity curve (normalized to 100)
  - Bottom panel: rolling drawdown (%)

Supports the JSON schema produced by run_backtest_strategy.py:
    {"equity": [{"timestamp": "...", "equity": 12345.0}, ...]}

And also a flat CSV/Parquet with columns [timestamp|date, equity].

Usage:
    python scripts/plot_equity_drawdown.py output/backtest_result.json
    python scripts/plot_equity_drawdown.py output/equity.parquet --out output/equity_plot.png
    python scripts/plot_equity_drawdown.py A.json B.json --labels Baseline ML --out compare.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


def _load_equity(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "equity" in raw:
            rows = raw["equity"]
        elif isinstance(raw, list):
            rows = raw
        else:
            raise ValueError(f"{path}: unexpected JSON structure")
        df = pd.DataFrame(rows)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Normalize column names
    ts_col = next(
        (c for c in df.columns if c.lower() in ("timestamp", "date", "datetime")),
        None,
    )
    eq_col = next(
        (c for c in df.columns if c.lower() in ("equity", "value", "portfolio_value")),
        None,
    )

    if ts_col is None or eq_col is None:
        raise ValueError(
            f"{path}: need timestamp+equity columns, found: {list(df.columns)}"
        )

    df = df[[ts_col, eq_col]].rename(columns={ts_col: "timestamp", eq_col: "equity"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    return df


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    """Rolling drawdown from peak, in percent."""
    running_max = equity.cummax()
    return (equity - running_max) / running_max * 100.0


def plot_equity_drawdown(
    paths: list[Path],
    labels: list[str],
    out: Path | None = None,
    title: str = "Equity Curve & Drawdown",
    figsize: tuple[int, int] = (14, 8),
    normalize: bool = True,
) -> None:
    """Render equity + drawdown plot.

    Args:
        paths: One or more equity data files.
        labels: Legend labels (one per path).
        out: Output PNG path. If None, displays interactively.
        title: Plot title.
        figsize: Figure size in inches.
        normalize: If True, normalize each equity curve to 100 at start.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        matplotlib.use("Agg" if out is not None else "TkAgg")
    except ImportError:
        logger.error("[Plot] matplotlib not installed — cannot generate plot")
        sys.exit(1)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax_equity, ax_dd = axes

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (path, label) in enumerate(zip(paths, labels)):
        try:
            df = _load_equity(path)
        except Exception as exc:
            logger.warning("[Plot] Skipping %s: %s", path, exc)
            continue

        eq = df["equity"].copy()
        if normalize and eq.iloc[0] > 0:
            eq = eq / eq.iloc[0] * 100.0

        dd = _compute_drawdown(eq)
        ts = df["timestamp"]
        color = colors[i % len(colors)]

        ax_equity.plot(ts, eq, label=label, color=color, linewidth=1.5)
        ax_dd.fill_between(ts, dd, 0, alpha=0.4, color=color)
        ax_dd.plot(ts, dd, color=color, linewidth=0.8)

        # Annotate final value
        final_eq = eq.iloc[-1]
        final_dd = dd.min()
        ax_equity.annotate(
            f"{final_eq:.1f}",
            xy=(ts.iloc[-1], final_eq),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
        logger.info(
            "[Plot] %s | final=%.1f | max_dd=%.2f%%",
            label,
            final_eq,
            final_dd,
        )

    ax_equity.set_title(title, fontsize=13, fontweight="bold")
    ax_equity.set_ylabel("Equity (normalized to 100)" if normalize else "Equity")
    ax_equity.legend(loc="upper left", fontsize=9)
    ax_equity.grid(True, alpha=0.3)
    ax_equity.axhline(
        100 if normalize else 0, color="grey", linewidth=0.8, linestyle="--"
    )

    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Date")
    ax_dd.axhline(0, color="grey", linewidth=0.8)
    ax_dd.grid(True, alpha=0.3)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info("[OK] Plot saved to %s", out)
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Equity curve + drawdown visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Equity JSON/Parquet/CSV files to plot",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels (one per input; defaults to filenames)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: display interactively)",
    )
    parser.add_argument(
        "--title",
        default="Equity Curve & Drawdown",
        help="Plot title",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not normalize equity curves to 100 at start",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    labels = args.labels or [p.stem for p in args.inputs]
    if len(labels) < len(args.inputs):
        labels += [p.stem for p in args.inputs[len(labels) :]]

    plot_equity_drawdown(
        paths=args.inputs,
        labels=labels,
        out=args.out,
        title=args.title,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
