"""Survivorship-Bias-Check (audit §8.7 / C3-063).

Checks a watchlist CSV against survivorship-bias indicators without
requiring external SP500-constituent data. The full audit requires CRSP
or similar historical constituent data; this script delivers the
**in-repo** indicators that ARE actionable.

Indicators (in order of strength):

1. **Active/delisted ratio** — if 100% of the watchlist is ``status=active``
   over a 10+ year window, that is the canonical survivorship-bias signal.
   Real US universes over 2007-2026 should contain ~5-10% delisted names
   (Lehman 2008, Bear Stearns 2008, Worldcom 2002 legacy, AIG split-spinoff,
   Sears 2018, JCPenney 2020, etc.).

2. **Cross-check against known major delistings** — a hard-coded sample of
   well-known US-large-cap delisting events. If the watchlist's date range
   covers an event but the bankrupt/delisted name is missing, that is
   prima facie evidence the universe was selected post-hoc.

3. **Start-date clustering** — if all symbols share the same start_date
   (e.g. all "2008-09-02"), the universe was likely sampled at a single
   snapshot rather than reconstructed point-in-time. Real PIT universes
   have varied start_dates as companies IPO over time.

The script's verdict is a **risk level** (low / medium / high), NOT a
binary pass/fail. A real C3-063 closure requires CRSP-quality constituent
history — see ``KNOWN_ISSUES.md §8.7``.

Usage::

    python scripts/forensic/survivorship_bias_check.py
    python scripts/forensic/survivorship_bias_check.py \\
        --watchlist data/universe/watchlist_2007_2026.csv

Output: JSON + Markdown under ``output/qa/survivorship_<run_id>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known US-large-cap delisting / bankruptcy events (sample, not exhaustive)
# ---------------------------------------------------------------------------
#
# Each entry is (symbol, event_date, reason). Event-date is the LAST date
# the stock traded; a watchlist with date_range covering this date but NOT
# including the symbol is a survivorship-bias smoking gun.


@dataclass(frozen=True)
class KnownDelisting:
    symbol: str
    event_date: str  # ISO YYYY-MM-DD
    reason: str


KNOWN_US_DELISTINGS: tuple[KnownDelisting, ...] = (
    # F-S2-SBC-2: only DEFINITE US-listed equity delistings. AIG (continued
    # trading post-bailout, ticker survived) and FTX (private crypto exchange,
    # never US-listed under that ticker on regular equity exchanges) were
    # removed because they produce false positives — a strategy watchlist
    # legitimately excluding them does not signal survivorship bias.
    KnownDelisting("LEH", "2008-09-15", "Lehman Brothers bankruptcy"),
    KnownDelisting("BSC", "2008-03-17", "Bear Stearns / JPM acquisition"),
    KnownDelisting("WAMUQ", "2008-09-26", "Washington Mutual bankruptcy"),
    KnownDelisting("WB", "2008-12-31", "Wachovia / Wells Fargo merger"),
    KnownDelisting("CFC", "2008-07-01", "Countrywide / BAC acquisition"),
    KnownDelisting("GM_OLD", "2009-06-01", "Old General Motors bankruptcy"),
    KnownDelisting("CIT", "2009-11-01", "CIT Group bankruptcy"),
    KnownDelisting("SHLD", "2018-10-15", "Sears Holdings bankruptcy"),
    KnownDelisting("JCP", "2020-05-15", "JCPenney Chapter 11"),
    KnownDelisting("HTZ_OLD", "2020-05-22", "Hertz Chapter 11 (later relisted as HTZ)"),
    KnownDelisting("SVB", "2023-03-10", "Silicon Valley Bank FDIC takeover"),
    KnownDelisting("SI", "2023-03-08", "Silvergate Capital wind-down"),
    KnownDelisting("FRC", "2023-05-01", "First Republic Bank FDIC takeover"),
)


# ---------------------------------------------------------------------------
# Indicator computations
# ---------------------------------------------------------------------------


def compute_active_delisted_ratio(df: pd.DataFrame) -> dict[str, Any]:
    """Indicator 1: ratio of active vs delisted symbols.

    Real 2007-2026 US universes should have ~5-10% delisted/inactive names.
    100% active over 18+ years is a clear survivorship signal.
    """
    n = int(len(df))
    if n == 0:
        return {"n_total": 0, "n_active": 0, "n_delisted": 0, "pct_active": 0.0}
    if "status" not in df.columns:
        return {
            "n_total": n,
            "n_active": None,
            "n_delisted": None,
            "pct_active": None,
            "warning": "no 'status' column — cannot compute ratio",
        }
    # F-S2-SBC-1: distinguish 'active' / 'delisted' / 'unknown' explicitly.
    # NaN / "" / "unknown" / other typos must not silently inflate
    # n_delisted (which would mask the survivorship signal).
    status_norm = df["status"].fillna("unknown").astype(str).str.lower().str.strip()
    n_active = int((status_norm == "active").sum())
    n_delisted = int((status_norm == "delisted").sum())
    n_unknown = n - n_active - n_delisted
    out = {
        "n_total": n,
        "n_active": n_active,
        "n_delisted": n_delisted,
        "n_unknown": n_unknown,
        "pct_active": round(100.0 * n_active / n, 2),
    }
    if n_unknown > 0:
        out["warning"] = (
            f"{n_unknown} symbol(s) have status not in {{active, delisted}} "
            "(NaN or unrecognised value). pct_active is computed against the "
            "n_total denominator so unknowns are NOT counted as either."
        )
    return out


def cross_check_known_delistings(
    df: pd.DataFrame,
    watchlist_start: pd.Timestamp,
    watchlist_end: pd.Timestamp,
) -> dict[str, Any]:
    """Indicator 2: known major delistings within watchlist date range.

    For each KNOWN_US_DELISTINGS entry whose event_date is in the watchlist
    window, check if the symbol is in the watchlist. Missing = bias signal.
    """
    symbols_in_list = (
        set(df["symbol"].astype(str).str.upper()) if "symbol" in df.columns else set()
    )
    in_window: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for d in KNOWN_US_DELISTINGS:
        event_ts = pd.Timestamp(d.event_date, tz="UTC")
        if not (watchlist_start <= event_ts <= watchlist_end):
            continue
        present = d.symbol in symbols_in_list
        entry = {
            "symbol": d.symbol,
            "event_date": d.event_date,
            "reason": d.reason,
            "present_in_watchlist": present,
        }
        in_window.append(entry)
        if not present:
            missing.append(entry)
    return {
        "n_events_in_window": len(in_window),
        "n_missing": len(missing),
        "missing_delistings": missing,
        "events_in_window": in_window,
    }


def check_start_date_clustering(df: pd.DataFrame) -> dict[str, Any]:
    """Indicator 3: are start_dates clustered at a single timestamp?

    A real PIT universe has varied start_dates (IPO dates, index inclusions).
    A single shared start_date suggests snapshot sampling.

    F-S2-SBC-3: clustering signal triggers when ``unique <= 2 and len > 5``.
    Threshold chosen because watchlist_2007_2026 has 19/19 at one date.
    For production use across varying universe sizes, a ratio-based threshold
    (``unique/len < 0.2``) would generalise better — current absolute
    threshold is conservative on small universes (won't false-positive on
    6 IPOs that happen to share 2 dates).
    """
    if "start_date" not in df.columns:
        return {"warning": "no 'start_date' column"}
    starts = pd.to_datetime(df["start_date"], errors="coerce", utc=True).dropna()
    if len(starts) == 0:
        return {"n_unique_start_dates": 0, "clustering_signal": False}
    unique = starts.nunique()
    return {
        "n_total": int(len(starts)),
        "n_unique_start_dates": int(unique),
        "most_common_start_date": str(starts.value_counts().index[0]),
        "most_common_count": int(starts.value_counts().iloc[0]),
        "clustering_signal": unique <= 2 and len(starts) > 5,
    }


# ---------------------------------------------------------------------------
# Verdict aggregation
# ---------------------------------------------------------------------------


def assign_risk_level(
    ratio: dict[str, Any],
    cross_check: dict[str, Any],
    clustering: dict[str, Any],
) -> dict[str, Any]:
    """Combine the three indicators into low / medium / high risk verdict."""
    flags: list[str] = []
    # Indicator 1: 100% active → strong signal
    pct_active = ratio.get("pct_active")
    if pct_active is not None and pct_active >= 99.0:
        flags.append(
            f"100%-active (pct_active={pct_active}): no delisted names in "
            "list over multi-year window"
        )
    # Indicator 2: missing known delistings in window
    if cross_check.get("n_missing", 0) > 0:
        flags.append(
            f"{cross_check['n_missing']} known delistings in watchlist date "
            "range but absent from watchlist"
        )
    # Indicator 3: start-date clustering
    if clustering.get("clustering_signal"):
        flags.append(
            "start_date clustering: "
            f"{clustering.get('most_common_count', 0)} symbols share "
            f"start_date {clustering.get('most_common_start_date', '?')}"
        )
    n_flags = len(flags)
    if n_flags >= 2:
        verdict = "high"
    elif n_flags == 1:
        verdict = "medium"
    else:
        verdict = "low"
    return {
        "risk_level": verdict,
        "n_flags": n_flags,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_survivorship_check(
    watchlist_path: Path,
    expected_window_start: str = "2007-01-01",
    expected_window_end: str = "2026-12-31",
) -> dict[str, Any]:
    """Run all three indicators on a watchlist CSV.

    Args:
        watchlist_path: CSV with at least a 'symbol' column. Optionally
            'status', 'start_date', 'end_date' columns add to the indicators.
        expected_window_start: ISO date; assumed start of the universe window.
            Used for cross-check against known delistings.
        expected_window_end: ISO date; assumed end of the universe window.

    Returns:
        Dict with indicators + risk verdict. JSON-serialisable.
    """
    if not watchlist_path.exists():
        raise FileNotFoundError(f"watchlist not found: {watchlist_path}")
    df = pd.read_csv(watchlist_path)
    if "symbol" not in df.columns:
        raise ValueError(
            f"watchlist {watchlist_path} missing 'symbol' column. "
            f"Got: {list(df.columns)}"
        )
    win_start = pd.Timestamp(expected_window_start, tz="UTC")
    win_end = pd.Timestamp(expected_window_end, tz="UTC")
    ratio = compute_active_delisted_ratio(df)
    cross_check = cross_check_known_delistings(df, win_start, win_end)
    clustering = check_start_date_clustering(df)
    verdict = assign_risk_level(ratio, cross_check, clustering)
    return {
        "input_path": str(watchlist_path),
        "window_start": expected_window_start,
        "window_end": expected_window_end,
        "indicators": {
            "active_delisted_ratio": ratio,
            "known_delistings_cross_check": cross_check,
            "start_date_clustering": clustering,
        },
        "verdict": verdict,
        "limitations": (
            "This is an in-repo bias-indicator scan only. The full audit "
            "C3-063 closure requires CRSP-quality historical constituent "
            "data to cross-check against the watchlist. KNOWN_US_DELISTINGS "
            "is a sample, not exhaustive — see KNOWN_ISSUES.md §8.7."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Survivorship-Bias-Check (§8.7 / C3-063)",
        "",
        f"**Input:** `{report['input_path']}`",
        f"**Assumed window:** {report['window_start']} → {report['window_end']}",
        "",
        f"## Verdict: `{report['verdict']['risk_level']}` "
        f"({report['verdict']['n_flags']} flag(s))",
        "",
    ]
    for f in report["verdict"]["flags"]:
        lines.append(f"- {f}")
    if not report["verdict"]["flags"]:
        lines.append("(no survivorship indicators triggered)")
    lines.append("")
    lines.append("## Indicator 1: Active/Delisted Ratio")
    ratio = report["indicators"]["active_delisted_ratio"]
    lines.append(f"- **Total symbols:** {ratio.get('n_total', '?')}")
    lines.append(f"- **Active:** {ratio.get('n_active', '?')}")
    lines.append(f"- **Delisted:** {ratio.get('n_delisted', '?')}")
    lines.append(f"- **% Active:** {ratio.get('pct_active', '?')}")
    lines.append("")
    lines.append("## Indicator 2: Known Delistings Cross-Check")
    cc = report["indicators"]["known_delistings_cross_check"]
    lines.append(f"- **Events in window:** {cc.get('n_events_in_window', '?')}")
    lines.append(f"- **Missing from watchlist:** {cc.get('n_missing', '?')}")
    if cc.get("missing_delistings"):
        lines.append("- **Missing symbols:**")
        for entry in cc["missing_delistings"]:
            lines.append(
                f"  - `{entry['symbol']}` ({entry['event_date']}): {entry['reason']}"
            )
    lines.append("")
    lines.append("## Indicator 3: Start-Date Clustering")
    cl = report["indicators"]["start_date_clustering"]
    if "warning" in cl:
        lines.append(f"- {cl['warning']}")
    else:
        lines.append(f"- **Unique start_dates:** {cl.get('n_unique_start_dates', '?')}")
        lines.append(
            f"- **Most common:** {cl.get('most_common_start_date', '?')} "
            f"({cl.get('most_common_count', '?')} symbols)"
        )
        lines.append(f"- **Clustering signal:** {cl.get('clustering_signal', False)}")
    lines.append("")
    lines.append("## Limitations")
    lines.append(report["limitations"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watchlist",
        type=Path,
        default=Path("data/universe/watchlist_2007_2026.csv"),
        help="Path to watchlist CSV (default: data/universe/watchlist_2007_2026.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/qa"),
        help="Output directory for JSON + Markdown",
    )
    parser.add_argument(
        "--window-start",
        default="2007-01-01",
        help="Assumed universe window start (ISO date)",
    )
    parser.add_argument(
        "--window-end",
        default="2026-12-31",
        help="Assumed universe window end (ISO date)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run_survivorship_check(
        watchlist_path=args.watchlist,
        expected_window_start=args.window_start,
        expected_window_end=args.window_end,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    run_id = args.watchlist.stem
    json_path = args.out / f"survivorship_{run_id}.json"
    md_path = args.out / f"survivorship_{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("[survivorship] JSON: %s", json_path)
    logger.info("[survivorship] Markdown: %s", md_path)
    logger.info(
        "[survivorship] verdict=%s flags=%d",
        report["verdict"]["risk_level"],
        report["verdict"]["n_flags"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
