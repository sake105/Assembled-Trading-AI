#!/usr/bin/env python
"""One-time backfill of the adj_close NaN sentinel in the daily price cache.

WHY THIS EXISTS
---------------
``output/aggregates/daily.parquet`` HAD accumulated 98,279 NaN values in ``adj_close``
(35.22% of 279,013 rows, measured 2026-08-15 BEFORE the repair). This script was
applied with ``--apply`` on 2026-08-15 and the live cache now holds 0 NaN; the
pre-repair state is preserved as
``archive/orphaned_data_2026-08-15/daily.parquet.PRE_ADJCLOSE_BACKFILL.bak``.
Past tense on purpose: a docstring in the present tense would contradict the file
it describes on the next measurement, which is the drift this whole step is about. They were written deliberately as a
"loud sentinel" by two writers whenever the upstream source carried no separate
adjusted-close column:

  - scripts/ops/refresh_daily_cache_from_panel.py
  - scripts/ops/refresh_sector_etf_cache.py

The sentinel rested on the premise that ``adj_close = close`` "would silently
mis-handle ex-dividend dates". That premise was measured and REFUTED:

  1. ``close`` in this cache is already TOTAL-RETURN adjusted (split + dividend).
     - documented: src/assembled_core/pipeline/io.py:33-41,
       docs/CORPORATE_ACTIONS.md (correction box, 2026-07-23)
     - measured: AAPL's 4:1 split on 2020-08-31 shows no jump
       (121.699 -> 125.829); only 15 of 279,013 rows have close outside [low, high] and all 15 are float epsilon (max 2.8e-14 absolute, 1.5e-16 relative), i.e. the whole OHLC tuple is adjusted consistently
  2. Wherever ``adj_close`` was populated it equalled ``close`` EXACTLY:
     180,734 of 180,734 rows, max abs difference 0.0.

So the column never carried information distinct from ``close``, and the sentinel
protected against nothing while poisoning a third of the cache.

Both writers were fixed to mirror ``close`` going forward. This script repairs the
rows that were already written.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does NOT backfill from ``research/mandat/data/prices_verdict.parquet``. That was
evaluated and rejected: it would cover only 76.4% of the NaN rows (75,110 of 98,279),
and both series are total-return adjusted to DIFFERENT anchor dates (verdict
2026-07-06 vs. daily 2026-08-05). The ratio daily.close / verdict.close has a
median of 1.00191 and a q95 of 1.0595 — a 1:1 copy would be up to ~6% wrong for
dividend payers, and correcting it would need per-symbol re-anchoring. More moving
parts, more failure modes, zero information gained over mirroring ``close``.

SAFETY
------
Dry-run by default. ``--apply`` is required to write. The write is atomic
(tmp + replace) and refuses to run if any invariant below fails:

  - ``close`` must have no NaN (nothing to mirror from otherwise)
  - every already-populated ``adj_close`` must equal ``close``; if that is ever
    violated the column DOES carry independent information and this script is
    the wrong tool -> hard abort, no write

Usage
-----
    python scripts/ops/backfill_adj_close.py                 # dry-run report
    python scripts/ops/backfill_adj_close.py --apply         # write
    python scripts/ops/backfill_adj_close.py --apply --no-backup
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("backfill_adj_close")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "output" / "aggregates" / "daily.parquet"
STATUS_PATH = REPO_ROOT / "output" / "ops" / "backfill_adj_close_status.json"

# Tolerance for the "adj_close == close where populated" invariant. The measured
# max abs diff is exactly 0.0, so anything above float noise is a real violation.
_EQ_ATOL = 1e-9
_EQ_RTOL = 1e-9


def _write_status(payload: dict[str, Any], status_path: Path | None = None) -> None:
    """Write the run report atomically (tmp + replace), never raising.

    ``status_path`` is resolved at CALL time, not bound as a default argument.
    A default of ``STATUS_PATH`` would be evaluated at import and could not be
    monkeypatched afterwards — so a test that redirects the module attribute
    would still write into the real ``output/ops/``. That is the same
    contamination class the test suite's autouse isolation exists to prevent
    (E-139); a repair script must not be the thing that reintroduces it.
    """
    target = status_path if status_path is not None else STATUS_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(target)
    except Exception as exc:  # pragma: no cover - reporting must never break the run
        logger.warning("[backfill-adj-close] could not write status: %s", exc)


def backfill(
    cache_path: Path | None = None,
    *,
    apply: bool = False,
    backup: bool = True,
) -> int:
    """Mirror ``close`` into NaN ``adj_close`` rows.

    Returns the number of rows repaired (>= 0), or -1 on a hard error.
    In dry-run mode it returns the number of rows that WOULD be repaired.
    """
    import numpy as np
    import pandas as pd

    # Resolved at CALL time, not bound as a default argument - same reason
    # as _write_status above: an import-time default cannot be redirected
    # by a test and would point a repair script at production data.
    cache_path = cache_path if cache_path is not None else DEFAULT_CACHE

    if not cache_path.exists():
        logger.error("[backfill-adj-close] cache not found: %s", cache_path)
        _write_status({"rc": -1, "error": f"cache not found: {cache_path}"})
        return -1

    df = pd.read_parquet(cache_path)

    for col in ("close", "adj_close", "symbol", "timestamp"):
        if col not in df.columns:
            logger.error("[backfill-adj-close] cache lacks required column %r", col)
            _write_status({"rc": -1, "error": f"missing column: {col}"})
            return -1

    n_rows = len(df)
    nan_mask = df["adj_close"].isna()
    n_nan = int(nan_mask.sum())

    # --- Invariant 1: close must be complete -------------------------------
    n_close_nan = int(df["close"].isna().sum())
    if n_close_nan:
        logger.error(
            "[backfill-adj-close] ABORT - close has %d NaN; there is nothing "
            "to mirror from for those rows.",
            n_close_nan,
        )
        _write_status({"rc": -1, "error": f"close has {n_close_nan} NaN"})
        return -1

    # --- Invariant 2: populated adj_close must equal close ------------------
    populated = df.loc[~nan_mask]
    if len(populated):
        eq = np.isclose(
            populated["adj_close"].to_numpy(dtype="float64"),
            populated["close"].to_numpy(dtype="float64"),
            rtol=_EQ_RTOL,
            atol=_EQ_ATOL,
        )
        n_mismatch = int((~eq).sum())
        if n_mismatch:
            max_diff = float((populated["adj_close"] - populated["close"]).abs().max())
            logger.error(
                "[backfill-adj-close] ABORT - %d of %d populated adj_close rows "
                "DIFFER from close (max abs diff %.6g). That means adj_close "
                "carries independent information and mirroring close would "
                "destroy it. Investigate before rerunning; do NOT --apply.",
                n_mismatch,
                len(populated),
                max_diff,
            )
            _write_status(
                {
                    "rc": -1,
                    "error": "adj_close differs from close where populated",
                    "n_mismatch": n_mismatch,
                    "n_populated": int(len(populated)),
                    "max_abs_diff": max_diff,
                }
            )
            return -1

    affected_symbols = sorted(df.loc[nan_mask, "symbol"].unique().tolist())
    ts = df.loc[nan_mask, "timestamp"]
    report: dict[str, Any] = {
        "rc": n_nan,
        "cache_path": str(cache_path),
        "n_rows_total": n_rows,
        "n_adj_close_nan": n_nan,
        "pct_adj_close_nan": round(100.0 * n_nan / n_rows, 4) if n_rows else 0.0,
        "n_symbols_affected": len(affected_symbols),
        "nan_range_first": str(ts.min()) if n_nan else None,
        "nan_range_last": str(ts.max()) if n_nan else None,
        "n_populated_verified_equal": int(len(populated)),
        "applied": bool(apply),
        "method": "mirror close (close is total-return adjusted)",
    }

    if n_nan == 0:
        logger.info("[backfill-adj-close] [OK] nothing to do - no NaN in adj_close.")
        _write_status(report)
        return 0

    logger.info(
        "[backfill-adj-close] %d of %d rows have adj_close=NaN (%.2f%%), "
        "%d symbols affected, range %s .. %s",
        n_nan,
        n_rows,
        report["pct_adj_close_nan"],
        len(affected_symbols),
        report["nan_range_first"],
        report["nan_range_last"],
    )
    logger.info(
        "[backfill-adj-close] invariant verified: all %d populated adj_close "
        "values equal close exactly.",
        len(populated),
    )

    if not apply:
        logger.info(
            "[backfill-adj-close] [SKIP] dry-run - pass --apply to write. "
            "No file was modified."
        )
        _write_status(report)
        return n_nan

    if backup:
        backup_path = cache_path.with_suffix(".parquet.bak")
        shutil.copy2(cache_path, backup_path)
        report["backup_path"] = str(backup_path)
        logger.info("[backfill-adj-close] backup written: %s", backup_path)

    df.loc[nan_mask, "adj_close"] = df.loc[nan_mask, "close"]

    remaining = int(df["adj_close"].isna().sum())
    if remaining:  # pragma: no cover - defensive
        logger.error(
            "[backfill-adj-close] ABORT - %d NaN remain after fill; not writing.",
            remaining,
        )
        report.update({"rc": -1, "error": f"{remaining} NaN remain after fill"})
        _write_status(report)
        return -1

    tmp = cache_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(cache_path)

    report["remaining_nan"] = 0
    logger.info(
        "[backfill-adj-close] [OK] repaired %d rows, wrote %s", n_nan, cache_path
    )
    _write_status(report)
    return n_nan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write; without it the script only reports",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="skip the .parquet.bak copy (only with --apply)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    rc = backfill(args.cache, apply=args.apply, backup=not args.no_backup)
    return 0 if rc >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
