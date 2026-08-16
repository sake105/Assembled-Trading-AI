#!/usr/bin/env python
"""Producer for output/corporate_actions.csv.

WHY THIS EXISTS
---------------
``src/assembled_core/data/corporate_actions.py`` is ~530 lines of fully
implemented logic — splits, dividends, delisting exits, spinoffs — and until
2026-08-15 **nothing in the repository ever wrote the file it reads**. A grep
for ``.to_csv``/``.to_parquet`` against ``corporate_actions`` returned only the
two read sites. The delisting-exit path was therefore dead code that could
never execute.

That is the E-135/E-143 pattern: a feature built on a mechanism that never
fires, and an exit with no consumer at the other end. This script is the
missing producer.

SCHEMA
------
One CSV, union of all four action types (unused columns stay empty):

    symbol, action_type, effective_date, split_ratio, dividend_cash,
    child_symbol, spinoff_ratio

``action_type`` is matched case-sensitively by the consumer:
``SPLIT`` | ``DIVIDEND`` | ``DELISTING`` | ``SPINOFF``.

WHAT IS PRODUCED, AND FROM WHAT
-------------------------------
``DIVIDEND``
    From ``research/mandat/data/dividends.parquet`` (64,493 ex-dates, 896
    symbols, 1995-2026) and ``output/dividends.parquet`` (3,286 rows, 103
    symbols, 2018-2026), unioned and de-duplicated on (symbol, date).

    CAVEAT, carried into the file header: the mandat pull took EODHD's
    ``value`` field, not ``unadjustedValue`` — i.e. the amounts are adjusted to
    today's share count, not the historically declared dividend
    (``research/mandat/pull_dividends.py:45``). For a ledger that also carries
    split-adjusted quantities this is consistent; for one that carries raw
    historical quantities it is not.

``DELISTING``
    Derived from the point-in-time universe history
    (``data/universe/verdict_sp500.csv``), which carries a real ``end_date``
    for 418 of 1,167 symbols. This is a large improvement over the previous
    best available source — a hardcoded 13-entry list in
    ``scripts/forensic/survivorship_bias_check.py``.

    DAT-006 HAZARD, and it is not a footnote: those end dates are inferred
    from PANEL COVERAGE, not from corporate actions. A feed gap, an ingestion
    failure or a ticker rename is indistinguishable here from a real
    delisting. Worse, ticker recycling (E-114/E-117) means some symbols carry
    two companies, so an "end date" can be the seam between them rather than
    anyone's delisting. Every row produced here is marked
    ``source=universe_coverage_inferred`` for that reason.

``SPLIT`` and ``SPINOFF``
    **Deliberately empty.** There is no split data anywhere in this repository
    and no way to derive it:
      - no splits.parquet / splits.csv exists; no puller fetches one
      - every local price panel is ALREADY adjusted, so the classic
        raw-vs-adjusted ratio trick has no raw series to work from
      - ``dataquality/checks/splits.py`` only offers a >40%-single-bar-drop
        heuristic, which on an already-adjusted panel produces false positives
        and nothing else
    Emitting guessed split ratios would be worse than emitting none: the
    consumer would divide correct prices by invented factors. Getting real
    split data is a procurement question (EODHD ``/splits``, Norgate), not a
    coding one — see docs/DATENZUGANG_STATUS.md.

DO NOT POINT THE PAPER ENGINE AT THIS FILE
------------------------------------------
Producing this file makes a previously unreachable code path reachable, and the
two consumers are not equally protected:

  - ``qa/backtest_engine.py`` got a ``prices_are_total_return_adjusted`` guard
    (default True) on 2026-08-15, so it SKIPS split adjustment rather than
    double-adjusting an already-adjusted panel.
  - ``execution/unified_paper_engine.py`` has ``enable_corporate_actions=True``
    by default, calls ``adjust_prices_for_splits``, has NO such guard, and keeps
    a broad ``except Exception -> warning`` around it.

Until now that mattered to nobody because the file did not exist. It does now.
Setting ``corporate_actions_path`` in a paper/live config would silently
double-adjust prices in the LIVE path. The guard sits on the safer of the two
consumers; extending it to the paper engine touches a protected execution path
and is deliberately left as a separate, explicitly-scoped step.

This file currently contains zero SPLIT rows, so the hazard is latent rather
than active — but that is a property of today's data, not a safeguard.

Usage
-----
    python scripts/ops/build_corporate_actions.py                  # dry-run
    python scripts/ops/build_corporate_actions.py --apply
    python scripts/ops/build_corporate_actions.py --apply --universe-limit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("build_corporate_actions")

OUT_PATH = REPO_ROOT / "output" / "corporate_actions.csv"
MANDAT_DIVIDENDS = REPO_ROOT / "research" / "mandat" / "data" / "dividends.parquet"
OUTPUT_DIVIDENDS = REPO_ROOT / "output" / "dividends.parquet"
PIT_UNIVERSE = REPO_ROOT / "data" / "universe" / "verdict_sp500.csv"
DAILY_PANEL = REPO_ROOT / "output" / "aggregates" / "daily.parquet"

SCHEMA = [
    "symbol",
    "action_type",
    "effective_date",
    "split_ratio",
    "dividend_cash",
    "child_symbol",
    "spinoff_ratio",
    "source",
]


def _operational_symbols() -> set[str] | None:
    """Symbols in the operational panel, or None if it is unavailable."""
    import pandas as pd

    if not DAILY_PANEL.exists():
        return None
    try:
        return set(pd.read_parquet(DAILY_PANEL, columns=["symbol"])["symbol"].unique())
    except Exception as exc:
        logger.warning("[corp-actions] could not read %s: %s", DAILY_PANEL, exc)
        return None


def build_dividends(limit_symbols: set[str] | None = None):
    import pandas as pd

    frames = []

    if MANDAT_DIVIDENDS.exists():
        d = pd.read_parquet(MANDAT_DIVIDENDS)
        d = d.rename(columns={"ex_date": "effective_date", "dividend": "dividend_cash"})
        d = d[["symbol", "effective_date", "dividend_cash"]].copy()
        d["source"] = "mandat_eodhd_adjusted_value"
        frames.append(d)
        logger.info("[corp-actions] mandat dividends: %d rows", len(d))
    else:
        logger.warning("[corp-actions] missing %s", MANDAT_DIVIDENDS)

    if OUTPUT_DIVIDENDS.exists():
        d2 = pd.read_parquet(OUTPUT_DIVIDENDS)
        rename = {}
        if "timestamp" in d2.columns:
            rename["timestamp"] = "effective_date"
        if "dividend_amount" in d2.columns:
            rename["dividend_amount"] = "dividend_cash"
        d2 = d2.rename(columns=rename)
        keep = [
            c for c in ("symbol", "effective_date", "dividend_cash") if c in d2.columns
        ]
        if len(keep) == 3:
            d2 = d2[keep].copy()
            d2["source"] = "output_dividends"
            frames.append(d2)
            logger.info("[corp-actions] output dividends: %d rows", len(d2))
        else:
            logger.warning(
                "[corp-actions] %s lacks required columns (have %s) - skipped",
                OUTPUT_DIVIDENDS,
                list(d2.columns),
            )
    else:
        logger.info("[corp-actions] no %s (optional)", OUTPUT_DIVIDENDS)

    if not frames:
        return pd.DataFrame(columns=SCHEMA)

    div = pd.concat(frames, ignore_index=True)
    div["effective_date"] = pd.to_datetime(div["effective_date"], utc=True)
    div = div.dropna(subset=["symbol", "effective_date", "dividend_cash"])
    div = div[div["dividend_cash"] > 0]

    if limit_symbols is not None:
        before = len(div)
        div = div[div["symbol"].isin(limit_symbols)]
        logger.info(
            "[corp-actions] dividends limited to operational universe: %d -> %d rows",
            before,
            len(div),
        )

    # mandat wins on collision: longer history, single consistent vendor.
    div = div.sort_values(["symbol", "effective_date", "source"]).drop_duplicates(
        subset=["symbol", "effective_date"], keep="first"
    )

    div["action_type"] = "DIVIDEND"
    div["split_ratio"] = float("nan")
    div["child_symbol"] = ""
    div["spinoff_ratio"] = float("nan")
    return div[SCHEMA]


def build_delistings(limit_symbols: set[str] | None = None):
    import pandas as pd

    if not PIT_UNIVERSE.exists():
        logger.warning(
            "[corp-actions] missing %s - no DELISTING rows. Generate it via "
            "src.assembled_core.data.pit_prices.build_pit_universe_history().",
            PIT_UNIVERSE,
        )
        return pd.DataFrame(columns=SCHEMA)

    u = pd.read_csv(PIT_UNIVERSE)
    if "end_date" not in u.columns:
        logger.warning("[corp-actions] %s lacks end_date", PIT_UNIVERSE)
        return pd.DataFrame(columns=SCHEMA)

    u["end_date"] = pd.to_datetime(u["end_date"], utc=True, errors="coerce")
    d = u[u["end_date"].notna()][["symbol", "end_date"]].copy()
    d = d.rename(columns={"end_date": "effective_date"})

    if limit_symbols is not None:
        d = d[d["symbol"].isin(limit_symbols)]

    d["action_type"] = "DELISTING"
    d["split_ratio"] = float("nan")
    d["dividend_cash"] = float("nan")
    d["child_symbol"] = ""
    d["spinoff_ratio"] = float("nan")
    # Never let this look like authoritative corporate-action data (DAT-006).
    d["source"] = "universe_coverage_inferred"

    logger.info("[corp-actions] delistings (coverage-inferred): %d rows", len(d))
    return d[SCHEMA]


def build(apply: bool = False, universe_limit: bool = False) -> int:
    import pandas as pd

    limit = _operational_symbols() if universe_limit else None
    if universe_limit and limit is None:
        logger.warning("[corp-actions] --universe-limit requested but panel unreadable")

    div = build_dividends(limit)
    del_ = build_delistings(limit)

    actions = pd.concat([div, del_], ignore_index=True)
    if actions.empty:
        logger.error("[corp-actions] nothing to write - no sources available")
        return -1

    actions = actions.sort_values(["symbol", "effective_date", "action_type"])
    actions["effective_date"] = pd.to_datetime(
        actions["effective_date"], utc=True
    ).dt.strftime("%Y-%m-%d")
    actions = actions.reset_index(drop=True)

    counts = dict(actions["action_type"].value_counts())
    logger.info(
        "[corp-actions] built %d rows: %s | symbols=%d | %s .. %s",
        len(actions),
        counts,
        actions["symbol"].nunique(),
        actions["effective_date"].min(),
        actions["effective_date"].max(),
    )
    logger.info(
        "[corp-actions] SPLIT=0 and SPINOFF=0 by design - no split data exists "
        "locally and guessing ratios would corrupt correct prices."
    )

    if not apply:
        logger.info(
            "[corp-actions] [SKIP] dry-run - pass --apply to write %s", OUT_PATH
        )
        return len(actions)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by scripts/ops/build_corporate_actions.py\n"
        "# DIVIDEND amounts are EODHD 'value' (adjusted to current share count),\n"
        "#   NOT the historically declared amount (research/mandat/pull_dividends.py:45).\n"
        "# DELISTING dates are INFERRED FROM PANEL COVERAGE, not corporate actions\n"
        "#   (DAT-006). A feed gap or ticker rename is indistinguishable from a real\n"
        "#   delisting here; ticker recycling (E-114/E-117) can put a seam between two\n"
        "#   companies where this file claims a delisting.\n"
        "# SPLIT / SPINOFF are intentionally absent - no local split data exists.\n"
        "# Do NOT feed this into split adjustment of an already-adjusted panel.\n"
        "# WARNING: execution/unified_paper_engine.py has enable_corporate_actions=True\n"
        "#   by default and NO double-adjustment guard. Setting corporate_actions_path\n"
        "#   in a paper/live config would double-adjust prices in the LIVE path.\n"
        "#   Only qa/backtest_engine.py is guarded.\n"
    )
    tmp = OUT_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        actions.to_csv(fh, index=False)
    tmp.replace(OUT_PATH)
    logger.info("[corp-actions] [OK] wrote %s (%d rows)", OUT_PATH, len(actions))
    return len(actions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--universe-limit",
        action="store_true",
        help="restrict to symbols present in output/aggregates/daily.parquet",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rc = build(apply=args.apply, universe_limit=args.universe_limit)
    return 0 if rc >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
