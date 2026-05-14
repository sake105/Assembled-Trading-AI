"""Event Study CLI Workflow.

Loads events from CSV/JSON, fetches prices via the configured data source,
runs the three core qa/event_study.py functions, and writes a CSV + Markdown
report. The qa engine already exists — this CLI now just wires it up.

Example:
    python scripts/run_event_study.py \\
        --events-file data/events/earnings_2024.csv \\
        --symbols-file config/universe_ai_tech_tickers.txt \\
        --start-date 2020-01-01 \\
        --end-date 2025-12-03 \\
        --window-before 20 \\
        --window-after 40 \\
        --output-dir output/event_studies/earnings_2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.assembled_core.config.settings import get_settings
from src.assembled_core.data.data_source import get_price_data_source
from src.assembled_core.qa.event_study import (
    aggregate_event_study,
    build_event_window_prices,
    compute_event_returns,
)

log = logging.getLogger(__name__)


def _load_events(path: Path) -> pd.DataFrame:
    """Load events from CSV or JSON. Must contain timestamp, symbol, event_type."""
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=(suffix == ".jsonl"))
    else:
        raise ValueError(
            f"Unsupported events file format '{suffix}' (use .csv or .json)"
        )

    required = {"timestamp", "symbol", "event_type"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Events file missing required columns: {sorted(missing)}")
    return df


def _resolve_symbols(args: argparse.Namespace, events_df: pd.DataFrame) -> list[str]:
    """Pick the symbol universe from --symbols, --symbols-file, or events."""
    if args.symbols:
        return list(args.symbols)
    if args.symbols_file:
        path = Path(args.symbols_file)
        if not path.exists():
            raise FileNotFoundError(f"Symbols file not found: {path}")
        with path.open(encoding="utf-8") as f:
            syms = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
        if not syms:
            raise ValueError(f"Symbols file is empty: {path}")
        return syms
    # Fall back to symbols mentioned in the events file.
    return sorted(events_df["symbol"].dropna().unique().tolist())


def _write_markdown_report(
    out_path: Path,
    events_df: pd.DataFrame,
    aggregated: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Write a concise Markdown summary next to the CSV outputs."""
    n_events = len(events_df)
    event_types = sorted(events_df["event_type"].dropna().unique().tolist())
    n_symbols = events_df["symbol"].nunique()

    # aggregate_event_study() returns columns: rel_day, avg_ret, std_ret,
    # n_events, se, ci_lower, ci_upper, cum_ret. cum_ret is the cumulative
    # average return across events (CAR if return_col was abnormal_return).
    car_col = "cum_ret"

    event_day = aggregated.loc[aggregated["rel_day"] == 0]
    post_window = aggregated.loc[aggregated["rel_day"] >= 0]

    car_event_day = (
        float(event_day[car_col].iloc[0]) if not event_day.empty else float("nan")
    )
    car_end = (
        float(post_window[car_col].iloc[-1]) if not post_window.empty else float("nan")
    )

    lines: list[str] = []
    lines.append(f"# Event Study Report — {args.events_file}")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Events: **{n_events}**")
    lines.append(f"- Distinct symbols: **{n_symbols}**")
    lines.append(f"- Event types: {', '.join(event_types) if event_types else '—'}")
    lines.append(
        f"- Window: [-{args.window_before}, +{args.window_after}] trading days"
    )
    lines.append(f"- Return type: `{args.return_type}`")
    lines.append(f"- Abnormal returns: {'yes' if args.use_abnormal else 'no'}")
    if args.benchmark_col:
        lines.append(f"- Benchmark column: `{args.benchmark_col}`")
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- CAR on event day (rel_day=0): **{car_event_day:+.4f}**")
    lines.append(
        f"- CAR at end of window (rel_day=+{args.window_after}): **{car_end:+.4f}**"
    )
    lines.append("")
    lines.append("## Aggregated stats per relative day")
    lines.append("")
    lines.append("First 5 rows:")
    lines.append("")
    lines.extend(_df_to_md_table(aggregated.head(5)))
    lines.append("")
    lines.append("Last 5 rows:")
    lines.append("")
    lines.extend(_df_to_md_table(aggregated.tail(5)))
    lines.append("")
    lines.append("Full table in the accompanying CSV file.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _df_to_md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> list[str]:
    """Render a small DataFrame to a Markdown pipe-table without tabulate.

    Iterates column-wise to preserve per-column dtypes — iterrows() unifies
    mixed columns to a single Series dtype, which casts ints to floats.
    """
    if df.empty:
        return ["(empty)"]
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [header, sep]
    for i in range(len(df)):
        cells: list[str] = []
        for col in cols:
            v = df[col].iloc[i]
            if isinstance(v, (bool,)):
                cells.append(str(v))
            elif isinstance(v, float):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def run_event_study_from_args(args: argparse.Namespace) -> int:
    """Run the full event study workflow end-to-end."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    try:
        events_path = Path(args.events_file)
        log.info("[START] Loading events from %s", events_path)
        events_df = _load_events(events_path)
        log.info("[OK] Loaded %d events", len(events_df))

        symbols = _resolve_symbols(args, events_df)
        log.info("[OK] Resolved %d symbols", len(symbols))

        settings = get_settings()
        price_source = get_price_data_source(settings, data_source=args.data_source)

        log.info(
            "[START] Loading prices from %s for %s..%s",
            args.data_source,
            args.start_date,
            args.end_date,
        )
        prices_df = price_source.get_history(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            freq=args.freq,
        )
        if prices_df.empty:
            log.error("[ERROR] No prices returned from data source")
            return 2
        log.info("[OK] Loaded %d price rows", len(prices_df))

        log.info("[START] Building event windows")
        windows = build_event_window_prices(
            prices_df=prices_df,
            events_df=events_df,
            window_before=args.window_before,
            window_after=args.window_after,
        )
        log.info("[OK] Built %d event-window rows", len(windows))

        log.info("[START] Computing event returns")
        returns = compute_event_returns(
            event_windows_df=windows,
            benchmark_col=args.benchmark_col,
            return_type=args.return_type,
        )
        log.info("[OK] Computed returns for %d rows", len(returns))

        if args.use_abnormal and args.benchmark_col is None:
            log.warning(
                "[WARN] --use-abnormal requested without --benchmark-col; "
                "aggregation will fall back to event_return"
            )

        log.info("[START] Aggregating event study")
        aggregated = aggregate_event_study(
            returns_df=returns,
            use_abnormal=args.use_abnormal and args.benchmark_col is not None,
            confidence_level=args.confidence_level,
        )
        log.info("[OK] Aggregated %d relative-day rows", len(aggregated))

        # Output directory.
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Returns per event-day CSV.
        returns_path = (
            Path(args.output_csv) if args.output_csv else out_dir / "event_returns.csv"
        )
        returns.to_csv(returns_path, index=False)
        log.info("[OK] Wrote returns CSV: %s", returns_path)

        # Aggregated CSV.
        agg_path = out_dir / "event_study_aggregated.csv"
        aggregated.to_csv(agg_path, index=False)
        log.info("[OK] Wrote aggregated CSV: %s", agg_path)

        # Markdown report.
        md_path = (
            Path(args.output_md)
            if args.output_md
            else out_dir / "event_study_report.md"
        )
        _write_markdown_report(md_path, events_df, aggregated, args)
        log.info("[OK] Wrote Markdown report: %s", md_path)

        log.info("[DONE] Event study complete")
        return 0

    except FileNotFoundError as e:
        log.error("[ERROR] %s", e)
        return 2
    except (KeyError, ValueError) as e:
        log.error("[ERROR] %s", e)
        return 1
    except Exception as e:  # noqa: BLE001
        log.exception("[ERROR] Unexpected failure: %s", e)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for analyze_events command."""
    parser = argparse.ArgumentParser(
        description="Run event study analysis on events from CSV/JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze earnings events from CSV
    python scripts/run_event_study.py \\
        --events-file data/events/earnings_2024.csv \\
        --symbols-file config/universe_ai_tech_tickers.txt \\
        --start-date 2020-01-01 \\
        --end-date 2025-12-03 \\
        --window-before 20 \\
        --window-after 40

    # Analyze insider transactions with custom benchmark column
    python scripts/run_event_study.py \\
        --events-file data/events/insider_2024.csv \\
        --start-date 2020-01-01 \\
        --end-date 2025-12-03 \\
        --benchmark-col market_return --use-abnormal \\
        --output-dir output/event_studies/insider
        """,
    )

    parser.add_argument(
        "--events-file",
        type=str,
        required=True,
        help="Path to CSV or JSON file with events (columns: timestamp, symbol, event_type)",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        help="Path to text file with symbol list (one per line)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="List of symbols (alternative to --symbols-file)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="local",
        choices=["local", "yahoo", "finnhub", "twelve_data"],
        help="Data source for price data (default: local)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="1d",
        choices=["1d", "5min"],
        help="Price frequency (default: 1d)",
    )
    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--window-before",
        type=int,
        default=20,
        help="Days before event to include (default: 20)",
    )
    parser.add_argument(
        "--window-after",
        type=int,
        default=40,
        help="Days after event to include (default: 40)",
    )
    parser.add_argument(
        "--benchmark-col",
        type=str,
        help="Column name for benchmark prices (for abnormal returns)",
    )
    parser.add_argument(
        "--return-type",
        type=str,
        default="log",
        choices=["log", "simple"],
        help="Return calculation type (default: log)",
    )
    parser.add_argument(
        "--use-abnormal",
        action="store_true",
        help="Use abnormal returns instead of normal returns",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/event_studies",
        help="Output directory for reports (default: output/event_studies)",
    )
    parser.add_argument(
        "--output-csv", type=str, help="Optional: explicit path for returns CSV output"
    )
    parser.add_argument(
        "--output-md", type=str, help="Optional: explicit path for Markdown report"
    )

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    return run_event_study_from_args(args)


if __name__ == "__main__":
    sys.exit(main())
