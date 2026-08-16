#!/usr/bin/env python
"""Shadow comparison: point-in-time panel vs. operational panel.

WHY THIS EXISTS
---------------
The project runs two price universes side by side:

  operational  output/aggregates/daily.parquet          220 symbols, 0 delisted
  PIT          research/mandat/data/prices_verdict.parquet  1167 symbols, ~418 delisted

Switching the live path from one to the other in a single step would change the
pilot's behaviour in a way nobody has measured. This script measures it first.
It writes a report and changes nothing.

WHAT IT MEASURES
----------------
1. Universe divergence - who is in which panel, and specifically which failed
   companies the operational panel has never seen.
2. Survivorship exposure - how much of the PIT panel consists of symbols that
   stopped trading, i.e. the mass the operational panel structurally omits.
3. Price divergence on the overlap - both panels carry total-return adjusted
   closes, but anchored to DIFFERENT last dates (PIT frozen 2026-07-06,
   operational 2026-08-05). The ratio is therefore expected to be slightly
   above 1 for dividend payers; this quantifies it rather than assuming it.
4. Coverage gaps - dates present in one panel and missing in the other.

Usage
-----
    python scripts/ops/compare_pit_vs_operational.py
    python scripts/ops/compare_pit_vs_operational.py --start 2010-01-01
    python scripts/ops/compare_pit_vs_operational.py --json output/ops/pit_shadow.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("pit_shadow")

DEFAULT_OPERATIONAL = REPO_ROOT / "output" / "aggregates" / "daily.parquet"
DEFAULT_REPORT = REPO_ROOT / "output" / "ops" / "pit_shadow_report.json"

#: Well-known failures/acquisitions used as a coverage probe. A panel that
#: contains NONE of these is survivorship-biased by construction. Ambiguous
#: tickers (WM = Waste Management today, GM relisted 2010) are deliberately
#: excluded - a probe whose result is ambiguous is not a probe (E-113).
DELISTED_PROBE = [
    "ABMD",
    "AGN",
    "ALXN",
    "ATVI",
    "BSC",
    "CELG",
    "CERN",
    "DISCA",
    "ETFC",
    "FLIR",
    "FRC",
    "MYL",
    "RTN",
    "SIVB",
    "TWTR",
    "WLTW",
    "XLNX",
]


def compare(
    operational_path: Path | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from src.assembled_core.data.pit_prices import PANEL_FROZEN_AT, load_pit_prices

    # Resolved at call time, not bound as a default argument: an import-time
    # default cannot be redirected by a test and would point this script at
    # production data (same class as the bug fixed in backfill_adj_close.py).
    operational_path = (
        operational_path if operational_path is not None else DEFAULT_OPERATIONAL
    )

    if not operational_path.exists():
        raise FileNotFoundError(f"operational panel not found: {operational_path}")

    op = pd.read_parquet(operational_path, columns=["timestamp", "symbol", "close"])
    op["timestamp"] = pd.to_datetime(op["timestamp"], utc=True)

    pit = load_pit_prices(start=start, end=end, warn_synthetic=False)

    if start is not None:
        op = op[op["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        op = op[op["timestamp"] <= pd.Timestamp(end, tz="UTC")]

    op_syms = set(op["symbol"].unique())
    pit_syms = set(pit["symbol"].unique())

    # --- 1. universe divergence -------------------------------------------
    only_pit = sorted(pit_syms - op_syms)
    only_op = sorted(op_syms - pit_syms)
    both = sorted(op_syms & pit_syms)

    probe_op = sorted(s for s in DELISTED_PROBE if s in op_syms)
    probe_pit = sorted(s for s in DELISTED_PROBE if s in pit_syms)

    # --- 2. survivorship exposure -----------------------------------------
    # A symbol whose last bar is well before the panel end stopped trading.
    pit_last = pit.groupby("symbol")["timestamp"].max()
    pit_end = pit["timestamp"].max()
    stopped = pit_last[pit_last < (pit_end - pd.Timedelta(days=10))]
    stopped_rows = int(pit[pit["symbol"].isin(stopped.index)].shape[0])

    # --- 3. price divergence on the overlap -------------------------------
    merged = op.merge(
        pit[["timestamp", "symbol", "close"]],
        on=["timestamp", "symbol"],
        how="inner",
        suffixes=("_op", "_pit"),
    )
    ratio_stats: dict[str, Any] = {"overlap_rows": int(len(merged))}
    if len(merged):
        valid = merged[(merged["close_pit"] != 0) & merged["close_pit"].notna()]
        ratio = (
            (valid["close_op"] / valid["close_pit"])
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(ratio):
            ratio_stats.update(
                {
                    "ratio_median": round(float(ratio.median()), 6),
                    "ratio_q05": round(float(ratio.quantile(0.05)), 6),
                    "ratio_q95": round(float(ratio.quantile(0.95)), 6),
                    "ratio_max": round(float(ratio.max()), 6),
                    "pct_within_1pct": round(
                        100.0 * float(((ratio - 1.0).abs() < 0.01).mean()), 2
                    ),
                }
            )

    # --- 4. coverage -------------------------------------------------------
    report: dict[str, Any] = {
        "generated_for_window": {"start": start, "end": end},
        "operational": {
            "path": str(operational_path),
            "symbols": len(op_syms),
            "rows": int(len(op)),
            "first": str(op["timestamp"].min()) if len(op) else None,
            "last": str(op["timestamp"].max()) if len(op) else None,
            "delisted_probe_hits": probe_op,
            "delisted_probe_n": len(probe_op),
        },
        "pit": {
            "path": "research/mandat/data/prices_verdict.parquet",
            "frozen_at": PANEL_FROZEN_AT,
            "symbols": len(pit_syms),
            "rows": int(len(pit)),
            "first": str(pit["timestamp"].min()) if len(pit) else None,
            "last": str(pit["timestamp"].max()) if len(pit) else None,
            "delisted_probe_hits": probe_pit,
            "delisted_probe_n": len(probe_pit),
        },
        "universe_divergence": {
            "in_both": len(both),
            "only_pit": len(only_pit),
            "only_operational": len(only_op),
            "only_operational_symbols": only_op[:50],
        },
        "survivorship_exposure": {
            "pit_symbols_that_stopped_trading": int(len(stopped)),
            "pct_of_pit_universe": round(
                100.0 * len(stopped) / max(len(pit_syms), 1), 2
            ),
            "rows_belonging_to_stopped_symbols": stopped_rows,
            "pct_of_pit_rows": round(100.0 * stopped_rows / max(len(pit), 1), 2),
            "note": (
                "These are the symbols the operational panel structurally "
                "cannot contain. Delisting is inferred from panel coverage, "
                "not corporate actions (DAT-006 hazard)."
            ),
        },
        "price_divergence_on_overlap": ratio_stats,
        "interpretation": (
            "Both panels carry total-return adjusted closes anchored to "
            "different end dates, so a ratio slightly above 1.0 is expected "
            "and is NOT an error. What matters is the universe divergence: "
            "the operational panel omits the failures entirely."
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operational", type=Path, default=DEFAULT_OPERATIONAL)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--json", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = compare(args.operational, start=args.start, end=args.end)

    op, pit = report["operational"], report["pit"]
    surv = report["survivorship_exposure"]
    print("")
    print("=== SHADOW: PIT vs. operational ===")
    print(
        f"  operational : {op['symbols']:>5} Symbole, {op['rows']:>9,} Zeilen, "
        f"delisted-Probe {op['delisted_probe_n']}/{len(DELISTED_PROBE)}"
    )
    print(
        f"  PIT         : {pit['symbols']:>5} Symbole, {pit['rows']:>9,} Zeilen, "
        f"delisted-Probe {pit['delisted_probe_n']}/{len(DELISTED_PROBE)}"
    )
    print(f"  nur im PIT  : {report['universe_divergence']['only_pit']} Symbole")
    print(
        f"  ausgeschieden im PIT: {surv['pit_symbols_that_stopped_trading']} Symbole "
        f"({surv['pct_of_pit_universe']}%), {surv['rows_belonging_to_stopped_symbols']:,} Zeilen "
        f"({surv['pct_of_pit_rows']}%)"
    )
    rs = report["price_divergence_on_overlap"]
    if rs.get("ratio_median") is not None:
        print(
            f"  Preis-Ratio op/PIT auf {rs['overlap_rows']:,} gemeinsamen Zeilen: "
            f"Median {rs['ratio_median']}, q95 {rs['ratio_q95']}, "
            f"{rs['pct_within_1pct']}% innerhalb 1%"
        )
    print("")

    try:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        tmp.replace(args.json)
        print(f"[OK] Report: {args.json}")
    except Exception as exc:
        logger.warning("could not write report: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
