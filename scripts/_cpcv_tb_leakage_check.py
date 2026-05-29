"""B2 CPCV Leakage Check — trend_baseline walk-forward fold structure.

GO_LIVE_CHECKLIST B2: verify no information leakage between train/test folds
in the _oos_wf_trend_baseline.py walk-forward design.

Does NOT re-run the full backtest (no Alpaca API required).
Uses generate_walk_forward_splits() — the same production function called by
_oos_wf_trend_baseline.py — to reconstruct the exact fold boundaries, then
runs four leakage checks on those boundaries.

Output: docs/results/2026_05_cpcv_tb_leakage_check.md
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.walk_forward import (
    WalkForwardConfig,
    WalkForwardWindow,
    generate_walk_forward_splits,
)

# ---------------------------------------------------------------------------
# Mirror _oos_wf_trend_baseline.py constants exactly.
# Any change here must also be reflected in the production script.
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW_DAYS = 252  # calendar days
TEST_WINDOW_DAYS = 252  # calendar days
STEP_SIZE_DAYS = 252  # calendar days
MIN_TRAIN_PERIODS = 200  # _oos_wf_trend_baseline.py L313
MA_WARMUP_BARS = 90  # trading-day warmup prepended to test slice
# Conservative calendar-day equivalent of MA_WARMUP_BARS trading days:
# 90 trading days * (365 / 252) ≈ 130 calendar days + 5 buffer
MA_WARMUP_CAL_APPROX = int(MA_WARMUP_BARS * 365 / 252) + 5

OUTPUT = ROOT / "docs" / "results" / "2026_05_cpcv_tb_leakage_check.md"


def _build_folds() -> list[WalkForwardWindow]:
    """Call the production fold-generation function with the same config."""
    config = WalkForwardConfig(
        start_date=PERIOD_START,
        end_date=PERIOD_END,
        train_window_days=TRAIN_WINDOW_DAYS,
        test_window_days=TEST_WINDOW_DAYS,
        mode="rolling",
        step_size_days=STEP_SIZE_DAYS,
        min_train_periods=MIN_TRAIN_PERIODS,
    )
    return generate_walk_forward_splits(PERIOD_START, PERIOD_END, config)


def _check_no_overlap(folds: list[WalkForwardWindow]) -> list[str]:
    """Check 1: test_start >= train_end in every fold.

    WalkForwardWindow convention: train_end is EXCLUSIVE, test_start is INCLUSIVE.
    Overlap exists only if test_start < train_end (test period starts inside
    the training window). With purge_days=0 (default), test_start == train_end
    is the normal boundary — contiguous, no overlap.
    """
    failures: list[str] = []
    for f in folds:
        if f.test_start < f.train_end:
            failures.append(
                f"Fold {f.split_index + 1}: test_start={f.test_start.date()} "
                f"< train_end={f.train_end.date()} (train_end exclusive) — OVERLAP"
            )
    return failures


def _check_warmup_in_train(folds: list[WalkForwardWindow]) -> list[str]:
    """Check 2: MA warmup buffer is sourced from the training window.

    The _oos_wf_trend_baseline.py prepends MA_WARMUP_BARS (90) trading-day
    bars before test_start to initialize the moving averages. These bars
    must fall within [train_start, train_end) — i.e., strictly within the
    training period.

    We use MA_WARMUP_CAL_APPROX (130 calendar days) as a conservative upper
    bound on 90 trading days (~126 calendar days). Warmup_start must be
    >= train_start to ensure the training window actually contains those bars.
    """
    failures: list[str] = []
    for f in folds:
        warmup_start = f.test_start - pd.Timedelta(days=MA_WARMUP_CAL_APPROX)
        if warmup_start < f.train_start:
            failures.append(
                f"Fold {f.split_index + 1}: warmup_start={warmup_start.date()} "
                f"< train_start={f.train_start.date()} — "
                f"MA warmup extends beyond training window boundary"
            )
        if warmup_start >= f.test_start:
            failures.append(
                f"Fold {f.split_index + 1}: warmup_start={warmup_start.date()} "
                f">= test_start={f.test_start.date()} — warmup overlaps test period"
            )
    return failures


def _check_cross_fold_test_independence(folds: list[WalkForwardWindow]) -> list[str]:
    """Check 3: test windows of different folds do not overlap each other.

    With STEP_SIZE_DAYS == TEST_WINDOW_DAYS, this is always satisfied under
    the current config. This check acts as a CONFIG DRIFT DETECTOR — it would
    fire if STEP_SIZE_DAYS were ever reduced below TEST_WINDOW_DAYS, enabling
    overlapping test windows. Both values are explicit in this script, so any
    future modification would be caught here before running a backtest.

    WalkForwardWindow: test_end is EXCLUSIVE. Overlap: test_start_a < test_end_b
    AND test_start_b < test_end_a.
    """
    failures: list[str] = []
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            a, b = folds[i], folds[j]
            if a.test_start < b.test_end and b.test_start < a.test_end:
                failures.append(
                    f"Folds {a.split_index + 1} and {b.split_index + 1}: "
                    f"test windows overlap — CROSS-FOLD LEAKAGE"
                )
    return failures


def _check_purge_boundary(folds: list[WalkForwardWindow]) -> list[str]:
    """Check 4: test_start >= train_end in all folds (explicit purge gap assertion).

    With purge_days=0 (default, as used in _oos_wf_trend_baseline.py):
    test_start == train_end — continuous boundary, no purge gap, no overlap.
    This check would also catch any regression where generate_walk_forward_splits
    produces a negative gap (e.g. due to purge_days implementation change).
    """
    failures: list[str] = []
    for f in folds:
        gap_days = (f.test_start - f.train_end).days
        if gap_days < 0:
            failures.append(
                f"Fold {f.split_index + 1}: test_start={f.test_start.date()} "
                f"is {-gap_days} days BEFORE train_end={f.train_end.date()} — "
                f"negative purge gap (train labels leak into test)"
            )
    return failures


def _format_report(
    folds: list[WalkForwardWindow],
    failures_overlap: list[str],
    failures_warmup: list[str],
    failures_cross: list[str],
    failures_purge: list[str],
    run_date: str,
) -> str:
    lines = [
        "# B2 CPCV Leakage Check — trend_baseline Walk-Forward",
        "",
        f"Run date: {run_date}  ",
        f"Period: {PERIOD_START.date()} → {PERIOD_END.date()}  ",
        f"Config: train={TRAIN_WINDOW_DAYS}d / test={TEST_WINDOW_DAYS}d / "
        f"step={STEP_SIZE_DAYS}d / min_train={MIN_TRAIN_PERIODS}d / purge=0d  ",
        f"MA warmup buffer: {MA_WARMUP_BARS} trading-day bars "
        f"(≈{MA_WARMUP_CAL_APPROX} calendar days, conservative upper bound)  ",
        f"Folds generated: **{len(folds)}**  ",
        "",
        "> **Purpose:** Verify GO_LIVE_CHECKLIST B2 — no information leakage between",
        "> train and test folds in the `_oos_wf_trend_baseline.py` walk-forward design.",
        "> Folds are built by calling `generate_walk_forward_splits()` — the same",
        "> production function used by `_oos_wf_trend_baseline.py`.",
        "> Does not re-run the backtest. Checks fold boundary geometry only.",
        "",
        "---",
        "",
        "## 1. Fold Boundary Summary",
        "",
        "WalkForwardWindow convention: `train_end` is EXCLUSIVE; `test_start` is INCLUSIVE;",
        "`test_end` is EXCLUSIVE.",
        "",
        "| Fold | Train Start | Train End (excl) | Test Start (incl) | Test End (excl) | n_train_days | n_test_days |",
        "|------|------------|-----------------|-------------------|-----------------|-------------|-------------|",
    ]
    for f in folds:
        lines.append(
            f"| {f.split_index + 1} | {f.train_start.date()} | {f.train_end.date()} "
            f"| {f.test_start.date()} | {f.test_end.date()} "
            f"| {f.n_train} | {f.n_test} |"
        )

    lines += ["", "---", "", "## 2. Leakage Checks", ""]

    def _section(failures: list[str], label: str, note: str = "") -> list[str]:
        out: list[str] = []
        if not failures:
            out.append(f"**{label}: PASS**")
        else:
            out.append(f"**{label}: FAIL** — {len(failures)} violation(s)")
            for err in failures:
                out.append(f"- {err}")
        if note:
            out.append(f"  _{note}_")
        return out

    lines += _section(failures_overlap, "Check 1 — No Train/Test Overlap")
    lines += [""]
    lines += _section(
        failures_warmup,
        "Check 2 — MA Warmup Sourced from Training Period",
        f"Using conservative {MA_WARMUP_CAL_APPROX}-calendar-day bound for "
        f"{MA_WARMUP_BARS} trading-day warmup.",
    )
    lines += [""]
    lines += _section(
        failures_cross,
        "Check 3 — Cross-Fold Test Window Independence",
        "Config drift detector: trivially satisfied when STEP_SIZE_DAYS == "
        "TEST_WINDOW_DAYS. Would fire if step_size < test_window_days.",
    )
    lines += [""]
    lines += _section(
        failures_purge,
        "Check 4 — Non-Negative Purge Gap (test_start >= train_end)",
        "Regression guard: fires if generate_walk_forward_splits ever produces "
        "a negative purge gap.",
    )

    all_failures = failures_overlap + failures_warmup + failures_cross + failures_purge
    all_pass = not all_failures
    verdict = "PASS" if all_pass else "FAIL"

    lines += [
        "",
        "---",
        "",
        "## 3. Overall Verdict",
        "",
        f"**{verdict}**",
        "",
    ]
    if all_pass:
        lines += [
            f"All {len(folds)} production folds pass all four leakage checks.",
            "",
            "- **No train/test overlap**: test_start >= train_end in all folds.",
            f"- **MA warmup in training**: {MA_WARMUP_BARS} trading-day warmup "
            f"(≈{MA_WARMUP_CAL_APPROX} cal days) fits within {TRAIN_WINDOW_DAYS}-day "
            "training window in all folds.",
            "- **Cross-fold independence**: test windows are non-overlapping "
            "(STEP_SIZE_DAYS == TEST_WINDOW_DAYS).",
            "- **Purge gap non-negative**: no calendar-day regression in fold generator.",
            "",
            "Combined with A3 PIT regression tests (signal-level look-ahead verification),",
            "the walk-forward design of `_oos_wf_trend_baseline.py` is leak-free.",
            "**GO_LIVE B2 criterion satisfied.**",
        ]
    else:
        lines += [
            f"**{len(all_failures)} violation(s) found.** Review above. "
            "B2 cannot be marked ERFÜLLT until all violations are resolved.",
        ]

    lines += [
        "",
        "---",
        "",
        "_Script: `scripts/_cpcv_tb_leakage_check.py`_  ",
        "_Production WF: `scripts/_oos_wf_trend_baseline.py`_  ",
        "_Fold generator: `src/assembled_core/qa/walk_forward.py::generate_walk_forward_splits`_  ",
        "_CPCV module: `src/assembled_core/qa/cpcv_validation.py`_  ",
        "_GO_LIVE B2: `docs/GO_LIVE_CHECKLIST.md`_  ",
    ]
    return "\n".join(lines)


def main() -> None:
    folds = _build_folds()
    print(f"[CPCV-B2] Built {len(folds)} folds via generate_walk_forward_splits()")

    failures_overlap = _check_no_overlap(folds)
    failures_warmup = _check_warmup_in_train(folds)
    failures_cross = _check_cross_fold_test_independence(folds)
    failures_purge = _check_purge_boundary(folds)

    for label, failures in [
        ("Check 1 (no overlap)", failures_overlap),
        ("Check 2 (warmup in train)", failures_warmup),
        ("Check 3 (cross-fold independence)", failures_cross),
        ("Check 4 (purge gap >= 0)", failures_purge),
    ]:
        status = "PASS" if not failures else f"FAIL ({len(failures)} violations)"
        print(f"[CPCV-B2] {label}: {status}")
        for err in failures:
            print(f"  ERROR: {err}")

    run_date = date.today().isoformat()
    report = _format_report(
        folds,
        failures_overlap,
        failures_warmup,
        failures_cross,
        failures_purge,
        run_date,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")
    print(f"[CPCV-B2] Report: {OUTPUT}")

    all_pass = not (
        failures_overlap or failures_warmup or failures_cross or failures_purge
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
