"""Backlog Item 29 — Performance Attribution Report.

Trade-level P&L decomposition by:
  - sector
  - signal_source
  - side (long / short)
  - conviction_bucket (low / medium / high)

Outputs a daily Markdown report + JSON to output/qa/.

Usage:
    python scripts/performance_attribution.py \\
        --trades output/trades.csv \\
        --equity output/equity_curve.parquet \\
        --output-dir output/qa

    # Auto-discover latest backtest trades:
    python scripts/performance_attribution.py --auto

Input format (trades CSV / Parquet):
    Required: symbol, entry_date, exit_date, side, pnl
    Optional: sector, signal_source, conviction, qty, entry_price, exit_price
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ─── Sector mapping fallback ────────────────────────────────────────────────

_SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Technology",
    "META": "Technology",
    "AMZN": "Consumer Disc",
    "TSLA": "Consumer Disc",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "SPY": "ETF",
    "QQQ": "ETF",
    "AGG": "ETF",
}


def _get_sector(symbol: str) -> str:
    return _SECTOR_MAP.get(symbol, "Unknown")


def _conviction_bucket(conviction: float | None) -> str:
    if conviction is None:
        return "unknown"
    if conviction >= 0.75:
        return "high"
    if conviction >= 0.50:
        return "medium"
    return "low"


# ─── Trade loading ────────────────────────────────────────────────────────────


def _load_trades(path: Path) -> list[dict]:
    """Load trades from CSV or Parquet into a list of dicts."""
    try:
        import pandas as pd

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        records = df.to_dict("records")
        logger.info("[attr] loaded %d trades from %s", len(records), path.name)
        return records
    except Exception as exc:
        logger.error("[attr] failed to load trades: %s", exc)
        return []


def _discover_latest_trades() -> Path | None:
    """Auto-discover the most recent trades file in output/."""
    candidates = sorted(
        list(ROOT.glob("output/**/trades*.csv"))
        + list(ROOT.glob("output/**/trades*.parquet"))
        + list(ROOT.glob("output/**/*trade_journal*.csv")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        logger.info("[attr] auto-discovered trades: %s", candidates[0])
        return candidates[0]
    return None


# ─── Attribution computation ──────────────────────────────────────────────────


def _safe_get(record: dict, *keys, default=None):
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return default


def _compute_attribution(trades: list[dict]) -> dict:
    """Decompose P&L by sector, signal_source, side, conviction_bucket."""
    buckets: dict[str, dict] = {
        "sector": defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0}),
        "signal_source": defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0}),
        "side": defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0}),
        "conviction_bucket": defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0}),
    }
    total_pnl = 0.0
    total_trades = 0

    for t in trades:
        pnl = float(_safe_get(t, "pnl", "realized_pnl", "gross_pnl", default=0.0))
        symbol = str(_safe_get(t, "symbol", default=""))
        side = str(_safe_get(t, "side", default="long")).lower()
        signal_source = str(_safe_get(t, "signal_source", "signal", default="unknown"))
        sector = str(_safe_get(t, "sector", default=_get_sector(symbol)))
        conviction = _safe_get(t, "conviction", "conviction_score")
        if conviction is not None:
            try:
                conviction = float(conviction)
            except (ValueError, TypeError):
                conviction = None
        cb = _conviction_bucket(conviction)

        is_win = pnl > 0
        total_pnl += pnl
        total_trades += 1

        for dim, key in [
            ("sector", sector),
            ("signal_source", signal_source),
            ("side", side),
            ("conviction_bucket", cb),
        ]:
            buckets[dim][key]["pnl"] += pnl
            buckets[dim][key]["trades"] += 1
            if is_win:
                buckets[dim][key]["wins"] += 1

    # Convert to sorted lists with win-rate and % contribution
    result: dict = {
        "total_pnl": round(total_pnl, 2),
        "total_trades": total_trades,
        "dimensions": {},
    }

    for dim, groups in buckets.items():
        rows = []
        for key, stats in sorted(groups.items(), key=lambda x: -abs(x[1]["pnl"])):
            n = stats["trades"]
            win_rate = stats["wins"] / n if n > 0 else 0.0
            pct = (stats["pnl"] / total_pnl * 100) if abs(total_pnl) > 1e-8 else 0.0
            rows.append(
                {
                    "group": key,
                    "pnl": round(stats["pnl"], 2),
                    "pnl_pct": round(pct, 1),
                    "trades": n,
                    "win_rate_pct": round(win_rate * 100, 1),
                }
            )
        result["dimensions"][dim] = rows

    return result


# ─── Report formatting ────────────────────────────────────────────────────────


def _format_dim_table(rows: list[dict], title: str) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append("| Group | P&L ($) | P&L % | Trades | Win Rate |")
    lines.append("|-------|--------:|------:|-------:|---------:|")
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['pnl']:,.2f} | {r['pnl_pct']:.1f}% "
            f"| {r['trades']} | {r['win_rate_pct']:.1f}% |"
        )
    return lines + [""]


def _format_report(attr: dict, trades_path: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_pnl = attr["total_pnl"]
    total_trades = attr["total_trades"]

    lines = [
        "# Performance Attribution Report",
        f"Generated: {now}",
        f"Source: {trades_path}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total P&L | ${total_pnl:,.2f} |",
        f"| Total Trades | {total_trades} |",
        "",
    ]

    dim_titles = {
        "sector": "By Sector",
        "signal_source": "By Signal Source",
        "side": "By Side (Long / Short)",
        "conviction_bucket": "By Conviction Bucket",
    }
    for dim, title in dim_titles.items():
        rows = attr["dimensions"].get(dim, [])
        if rows:
            lines += _format_dim_table(rows, title)

    lines += [
        "## Notes",
        "- P&L % = contribution to total P&L (signed).",
        "- Win Rate = fraction of trades that were profitable.",
        "- Conviction bucket: low < 0.50, medium 0.50–0.74, high ≥ 0.75.",
        "- Sector: symbol-level lookup (unknown = not in reference map).",
        "- Survivorship bias may be present if symbols were pre-filtered.",
    ]
    return "\n".join(lines)


# ─── Top-trade spotlight ──────────────────────────────────────────────────────


def _top_trades(trades: list[dict], n: int = 10) -> list[dict]:
    """Return top-N and bottom-N trades by P&L."""
    sortable = []
    for t in trades:
        pnl = float(_safe_get(t, "pnl", "realized_pnl", "gross_pnl", default=0.0))
        sortable.append((pnl, t))
    sortable.sort(key=lambda x: x[0], reverse=True)
    top = [x[1] for x in sortable[:n]]
    bottom = [x[1] for x in sortable[-n:]]
    return top, bottom


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Performance Attribution Report")
    parser.add_argument("--trades", help="Path to trades CSV or Parquet")
    parser.add_argument("--equity", help="Path to equity curve (optional)")
    parser.add_argument("--output-dir", default="output/qa")
    parser.add_argument(
        "--auto", action="store_true", help="Auto-discover latest trades file"
    )
    parser.add_argument(
        "--top-n", type=int, default=5, help="Number of top/bottom trades to highlight"
    )
    args = parser.parse_args(argv)

    # Resolve trades path
    if args.auto:
        trades_path = _discover_latest_trades()
        if not trades_path:
            logger.error("[attr] no trades file found in output/")
            return 2
    elif args.trades:
        trades_path = ROOT / args.trades
    else:
        logger.error("[attr] --trades or --auto required")
        return 2

    if not trades_path.exists():
        logger.error("[attr] trades file not found: %s", trades_path)
        return 2

    trades = _load_trades(trades_path)
    if not trades:
        logger.error("[attr] no trades loaded")
        return 1

    attr = _compute_attribution(trades)

    # Output
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    md_path = out_dir / f"performance_attribution_{ts}.md"
    json_path = out_dir / f"performance_attribution_{ts}.json"

    md_content = _format_report(attr, str(trades_path.name))
    md_path.write_text(md_content, encoding="utf-8")

    json_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(trades_path),
        **attr,
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    # Console summary
    print(f"\n{'=' * 65}")
    print(f"{'PERFORMANCE ATTRIBUTION':^65}")
    print(f"{'=' * 65}")
    print(f"Total P&L: ${attr['total_pnl']:,.2f}   Trades: {attr['total_trades']}")
    print(f"{'=' * 65}")

    for dim, title in [
        ("sector", "Sector"),
        ("signal_source", "Signal"),
        ("side", "Side"),
        ("conviction_bucket", "Conviction"),
    ]:
        rows = attr["dimensions"].get(dim, [])
        if not rows:
            continue
        print(f"\n  {title}:")
        for r in rows[:6]:
            bar = "#" * max(0, int(abs(r["pnl_pct"]) / 2))
            sign = "+" if r["pnl"] >= 0 else ""
            print(
                f"    {r['group']:<18} {sign}{r['pnl']:>10,.2f}  "
                f"{r['pnl_pct']:>+5.1f}%  wr={r['win_rate_pct']:.0f}%  {bar}"
            )

    # Top/Bottom trades
    top, bottom = _top_trades(trades, args.top_n)
    print(f"\n  Top {args.top_n} trades:")
    for t in top:
        pnl = float(_safe_get(t, "pnl", "realized_pnl", "gross_pnl", default=0.0))
        sym = _safe_get(t, "symbol", default="?")
        side = _safe_get(t, "side", default="?")
        src = _safe_get(t, "signal_source", "signal", default="?")
        print(f"    {sym:<6} {side:<5} pnl={pnl:>+10,.2f}  src={src}")

    print(f"\n  Bottom {args.top_n} trades:")
    for t in bottom:
        pnl = float(_safe_get(t, "pnl", "realized_pnl", "gross_pnl", default=0.0))
        sym = _safe_get(t, "symbol", default="?")
        side = _safe_get(t, "side", default="?")
        src = _safe_get(t, "signal_source", "signal", default="?")
        print(f"    {sym:<6} {side:<5} pnl={pnl:>+10,.2f}  src={src}")

    print(f"\n{'=' * 65}")
    print(f"Reports: {md_path.name}, {json_path.name}")
    print(f"{'=' * 65}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
