"""Crisis-overlay falsification comparison.

Runs the same backtest twice — once with the default crisis overlay active and once
with `--no-crisis-overlay` — then prints a side-by-side performance comparison.

The comparison addresses the audit question:
  "Does GPR-Z-spike as a pure Exposure-Cut-Trigger reduce drawdowns without harming Sharpe?"

Usage:
    python scripts/compare_crisis_overlay_impact.py \\
        --bundle output/mfv2_bundle.parquet \\
        --universe watchlist_200.csv \\
        --start 2015-01-01 --end 2024-12-31 \\
        --start-capital 100000 \\
        [--out-dir output/crisis_falsification]   # parent dir for the two run dirs

All other arguments are forwarded to run_backtest_strategy.py as-is.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Metric fields to compare
# ---------------------------------------------------------------------------

_COMPARE_FIELDS: list[tuple[str, str, str]] = [
    # (json_key, display_label, format_spec)
    # cagr/total_return are stored as fractions (0.17 = 17%) → .2% correct
    # max_drawdown_pct is stored as percent-points (-15.0 = 15% DD) → .2f correct
    ("cagr", "CAGR", ".2%"),
    ("sharpe_ratio", "Sharpe", ".3f"),
    ("max_drawdown_pct", "MaxDD (pp)", ".2f"),
    ("total_return", "Total Return", ".2%"),
    ("calmar_ratio", "Calmar", ".3f"),
    ("sortino_ratio", "Sortino", ".3f"),
]

_TRADE_FIELDS: list[tuple[str, str, str]] = [
    ("total_trades", "# Trades", ".0f"),
    ("hit_rate", "Hit Rate", ".2%"),
]


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    m = run_dir / "reports" / "metrics.json"
    if not m.exists():
        raise FileNotFoundError(f"metrics.json not found at {m}")
    with m.open(encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(val: Any, spec: str) -> str:
    if val is None or val != val:  # None or NaN
        return "—"
    try:
        return format(float(val), spec)
    except (TypeError, ValueError):
        return str(val)


def _delta_arrow(
    base: float | None, alt: float | None, higher_is_better: bool = True
) -> str:
    if base is None or alt is None:
        return ""
    try:
        d = float(alt) - float(base)
        if abs(d) < 1e-9:
            return "  ="
        up = d > 0
        good = up if higher_is_better else not up
        symbol = "▲" if up else "▼"
        return f"  {symbol}{abs(d):.3f} {'✓' if good else '✗'}"
    except (TypeError, ValueError):
        return ""


def _print_comparison(
    baseline: dict[str, Any],
    no_overlay: dict[str, Any],
    baseline_label: str = "with crisis overlay",
    alt_label: str = "no-crisis-overlay",
    rc_base: int = 0,
    rc_alt: int = 0,
) -> None:
    col = 22
    header = f"{'Metric':<{col}}  {'Baseline':>14}  {'No-Overlay':>14}  {'Delta':>18}"
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  Crisis-Overlay Falsification Comparison")
    print("=" * len(header))
    rc_base_str = f"  [exit={rc_base}]" if rc_base != 0 else ""
    rc_alt_str = f"  [exit={rc_alt}]" if rc_alt != 0 else ""
    print(f"  Baseline  : {baseline_label}{rc_base_str}")
    print(f"  No-overlay: {alt_label}{rc_alt_str}")
    if rc_base != 0 or rc_alt != 0:
        print("  WARNING   : One or both runs failed — metrics may be stale.")
    print(sep)
    print(header)
    print(sep)

    for key, label, fmt in _COMPARE_FIELDS:
        bv = baseline.get(key)
        av = no_overlay.get(key)
        # max_drawdown_pct is stored as a negative float (e.g. -15.0 = 15% drawdown).
        # A less-negative value is better, so higher_is_better=True is correct here.
        # max_drawdown (absolute) uses the same convention.
        higher_better = key not in ("max_drawdown",)
        delta = _delta_arrow(bv, av, higher_is_better=higher_better)
        print(
            f"  {label:<{col - 2}}  {_fmt(bv, fmt):>14}  {_fmt(av, fmt):>14}  {delta:>18}"
        )

    print(sep)
    for key, label, fmt in _TRADE_FIELDS:
        bv = baseline.get(key)
        av = no_overlay.get(key)
        print(f"  {label:<{col - 2}}  {_fmt(bv, fmt):>14}  {_fmt(av, fmt):>14}")

    print(sep)
    # Quick interpretation
    b_sharpe = baseline.get("sharpe_ratio")
    a_sharpe = no_overlay.get("sharpe_ratio")
    b_dd = baseline.get("max_drawdown_pct")
    a_dd = no_overlay.get("max_drawdown_pct")

    if b_sharpe is not None and a_sharpe is not None:
        sharpe_change = float(a_sharpe) - float(b_sharpe)
        dd_change = (
            (float(a_dd) - float(b_dd))
            if b_dd is not None and a_dd is not None
            else None
        )
        print()
        print("  INTERPRETATION:")
        if dd_change is not None:
            # max_drawdown_pct is stored as a negative percent value (e.g. -15.0 = 15% drawdown).
            # dd_change < 0 means no-overlay drawdown is more negative → overlay was helping.
            direction = "increases" if dd_change < 0 else "reduces"
            print(
                f"    - Removing crisis overlay {direction} MaxDD by {abs(dd_change):.2f}pp"
            )
        if abs(sharpe_change) < 0.05:
            print(f"    - Sharpe impact is negligible ({sharpe_change:+.3f})")
        elif sharpe_change > 0:
            print(f"    - Removing overlay IMPROVES Sharpe by {sharpe_change:.3f}")
        else:
            print(f"    - Removing overlay HURTS Sharpe by {abs(sharpe_change):.3f}")
        if dd_change is not None and dd_change < 0 and sharpe_change >= -0.05:
            print(
                "    → Crisis overlay reduces drawdown without meaningful Sharpe cost (SUPPORTED)"
            )
        elif dd_change is not None and dd_change > 0:
            print(
                "    → Crisis overlay did NOT reduce drawdown — removing it improved MaxDD (NOT SUPPORTED)"
            )
        else:
            print("    → Mixed — inspect equity curves manually")
    print(sep)
    print()


def _run_backtest(
    backtest_args: list[str],
    out_dir: Path,
    label: str,
    env: dict[str, str] | None = None,
) -> int:
    """Run run_backtest_strategy.py as subprocess with given extra args."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_backtest_strategy.py"),
        "--out",
        str(out_dir),
        *backtest_args,
    ]
    print(f"\n[compare] Running {label} ...")
    print(f"[compare] CMD: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        print(f"[compare] WARNING: {label} exited with code {result.returncode}")
    return result.returncode


def main() -> int:
    # Ensure unicode arrows/checkmarks render on Windows terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/crisis_falsification"),
        help="Parent output directory for the two backtest runs (default: output/crisis_falsification)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        default=False,
        help="Skip running backtests — only compare existing metrics.json files in --out-dir",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Write comparison table to this JSON file (optional)",
    )

    # Collect remaining args to pass through to run_backtest_strategy.py
    parsed, passthrough = parser.parse_known_args()

    # Strip any user-supplied --out / -o to prevent collision with the injected dirs.
    _clean: list[str] = []
    _i = 0
    while _i < len(passthrough):
        if passthrough[_i] in ("--out", "-o"):
            _i += 2  # skip flag + value
        else:
            _clean.append(passthrough[_i])
            _i += 1
    passthrough = _clean

    baseline_dir = parsed.out_dir / "baseline"
    no_overlay_dir = parsed.out_dir / "no_crisis_overlay"

    rc_base: int = 0
    rc_alt: int = 0

    if not parsed.compare_only:
        baseline_dir.mkdir(parents=True, exist_ok=True)
        no_overlay_dir.mkdir(parents=True, exist_ok=True)

        # Baseline: explicitly unset ASSEMBLED_NO_CRISIS_OVERLAY so an inherited env var
        # from the parent process cannot silently disable the overlay in the baseline run.
        baseline_env = {
            k: v for k, v in os.environ.items() if k != "ASSEMBLED_NO_CRISIS_OVERLAY"
        }

        rc_base = _run_backtest(
            passthrough,
            baseline_dir,
            "baseline (with crisis overlay)",
            env=baseline_env,
        )
        rc_alt = _run_backtest(
            passthrough + ["--no-crisis-overlay"], no_overlay_dir, "no-crisis-overlay"
        )

        if rc_base != 0 or rc_alt != 0:
            print(
                f"[compare] One or both runs failed (baseline={rc_base}, alt={rc_alt}). "
                "Attempting to compare partial results..."
            )

    # Load and compare metrics
    try:
        baseline_m = _load_metrics(baseline_dir)
    except FileNotFoundError as e:
        print(f"[compare] Cannot load baseline metrics: {e}")
        return 1

    try:
        no_overlay_m = _load_metrics(no_overlay_dir)
    except FileNotFoundError as e:
        print(f"[compare] Cannot load no-overlay metrics: {e}")
        return 1

    _print_comparison(
        baseline_m,
        no_overlay_m,
        baseline_label=str(baseline_dir),
        alt_label=str(no_overlay_dir),
        rc_base=rc_base,
        rc_alt=rc_alt,
    )

    if parsed.write_json:
        comparison = {
            "baseline_dir": str(baseline_dir),
            "no_overlay_dir": str(no_overlay_dir),
            "run_exit_codes": {"baseline": rc_base, "no_overlay": rc_alt},
            "baseline": {
                k: baseline_m.get(k) for k, _, _ in _COMPARE_FIELDS + _TRADE_FIELDS
            },
            "no_overlay": {
                k: no_overlay_m.get(k) for k, _, _ in _COMPARE_FIELDS + _TRADE_FIELDS
            },
            "delta": {
                k: (
                    (float(no_overlay_m[k]) - float(baseline_m[k]))
                    if (
                        no_overlay_m.get(k) is not None
                        and baseline_m.get(k) is not None
                        and isinstance(no_overlay_m.get(k), (int, float))
                        and isinstance(baseline_m.get(k), (int, float))
                    )
                    else None
                )
                for k, _, _ in _COMPARE_FIELDS + _TRADE_FIELDS
            },
        }
        parsed.write_json.parent.mkdir(parents=True, exist_ok=True)
        with parsed.write_json.open("w", encoding="utf-8") as fh:
            json.dump(comparison, fh, indent=2)
        print(f"[compare] Comparison written to {parsed.write_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
