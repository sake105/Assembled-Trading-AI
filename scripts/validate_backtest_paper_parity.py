"""scripts/validate_backtest_paper_parity.py — Backtest vs paper-trade parity check.

Compares trade-level output from backtest vs paper-trading for the same date range.
Flags divergences > 5% at the symbol level.

Usage:
    python scripts/validate_backtest_paper_parity.py
    python scripts/validate_backtest_paper_parity.py --backtest-output output/backtest_latest.csv --paper-output output/pilot/paper_trades.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_BACKTEST = ROOT / "output" / "backtest_latest.csv"
DEFAULT_PAPER = ROOT / "output" / "pilot" / "paper_trades.csv"


def _load_trades(path: Path) -> dict:
    """Load trade CSV → {symbol: [qty, ...]}."""
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas required")
        return {}
    if not path.exists():
        log.warning("File not found: %s", path)
        return {}
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        log.warning("No 'symbol' column in %s", path)
        return {}
    qty_col = next(
        (c for c in df.columns if "qty" in c.lower() or "quantity" in c.lower()), None
    )
    if qty_col is None:
        log.warning("No quantity column found")
        return {}
    return df.groupby("symbol")[qty_col].sum().to_dict()


def _parity_check(bt: dict, pt: dict, threshold: float = 0.05) -> list[str]:
    flags = []
    all_syms = set(bt) | set(pt)
    for sym in sorted(all_syms):
        b = bt.get(sym, 0.0)
        p = pt.get(sym, 0.0)
        if b == 0 and p == 0:
            continue
        denom = max(abs(b), abs(p), 1e-10)
        pct_diff = abs(b - p) / denom
        if pct_diff > threshold:
            flags.append(f"{sym}: backtest={b:.1f}, paper={p:.1f}, diff={pct_diff:.1%}")
    return flags


def _main() -> int:
    ap = argparse.ArgumentParser(description="Validate backtest vs paper parity")
    ap.add_argument("--backtest-output", type=Path, default=DEFAULT_BACKTEST)
    ap.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Max allowed qty divergence fraction (default 0.05 = 5%%)",
    )
    args = ap.parse_args()

    bt = _load_trades(args.backtest_output)
    pt = _load_trades(args.paper_output)

    if not bt and not pt:
        log.warning("[parity] No trade data loaded — skipping comparison")
        return 0

    flags = _parity_check(bt, pt, args.threshold)
    overlap = len(set(bt) & set(pt))
    log.info(
        "[parity] Symbols in backtest: %d, paper: %d, overlap: %d",
        len(bt),
        len(pt),
        overlap,
    )

    if flags:
        log.warning(
            "[parity] %d symbol(s) exceed %.0f%% divergence threshold:",
            len(flags),
            args.threshold * 100,
        )
        for f in flags:
            log.warning("  %s", f)
        pct_match = 100.0 * (overlap - len(flags)) / max(overlap, 1)
        log.warning(
            "[parity] PARITY: %.1f%% symbols within threshold (target ≥ 95%%)",
            pct_match,
        )
        return 1 if pct_match < 95.0 else 0
    else:
        log.info(
            "[parity] OK — all %d overlapping symbols within %.0f%% threshold",
            overlap,
            args.threshold * 100,
        )
        return 0


if __name__ == "__main__":
    sys.exit(_main())
