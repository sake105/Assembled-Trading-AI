"""E5 — Real-vs-synthetic fill calibration (Plan v3 Part E5).

After Part A flows real Alpaca executions into ``output/executions/``,
this script walks the last N days of fills and compares them to what the
paper engine would have predicted at the same arrival time.

Per the plan gate::

    if p95(|delta_bps|) > 2.0:
        recalibrate config.paper.half_spread_bps + impact_coefficient

Output
------

- ``output/qa/real_vs_synthetic_fills.json`` — metrics + per-order rows.
- ``output/qa/real_vs_synthetic_fills.md`` — human summary.
- Exit non-zero only when ``--enforce`` is passed (matches E3/E4 grace
  convention).

This is a *read-side* calibration helper. It does not touch
``config.paper`` — it only surfaces the delta so the user can decide when
to re-tune. Automatic re-tuning is explicitly out-of-scope per the plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("compare_real_vs_synthetic_fills")


def _load_fills(root: Path) -> list[dict[str, Any]]:
    fills: list[dict[str, Any]] = []
    if not root.exists():
        return fills
    for path in sorted(root.glob("*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                fills.append(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[calib] unreadable %s: %s", path, exc)
    return fills


def _delta_bps(row: dict[str, Any]) -> float | None:
    # Tolerant of several field names used across the execution stack.
    real = row.get("fill_price") or row.get("avg_fill_price") or row.get("price")
    synth = (
        row.get("synthetic_fill_price")
        or row.get("paper_fill_price")
        or row.get("expected_fill_price")
    )
    ref = row.get("arrival_price") or row.get("mid_at_submit")
    if real is None or synth is None or ref in (None, 0):
        return None
    try:
        return (float(real) - float(synth)) / float(ref) * 10_000.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[k]


def build_calibration_report(
    fills_dir: Path,
    *,
    p95_threshold_bps: float = 2.0,
) -> dict[str, Any]:
    fills = _load_fills(fills_dir)
    deltas: list[float] = []
    rows: list[dict[str, Any]] = []
    for f in fills:
        d = _delta_bps(f)
        if d is None:
            continue
        deltas.append(d)
        rows.append(
            {
                "order_id": f.get("order_id") or f.get("id"),
                "symbol": f.get("symbol"),
                "side": f.get("side"),
                "delta_bps": round(d, 4),
            }
        )

    abs_deltas = [abs(d) for d in deltas]
    if abs_deltas:
        p50 = median(abs_deltas)
        p95 = _percentile(abs_deltas, 95.0)
        p99 = _percentile(abs_deltas, 99.0)
        mean_abs = sum(abs_deltas) / len(abs_deltas)
    else:
        p50 = p95 = p99 = mean_abs = float("nan")

    passes = bool(abs_deltas) and p95 <= p95_threshold_bps

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fills_considered": len(fills),
        "deltas_computed": len(abs_deltas),
        "p50_abs_bps": p50,
        "p95_abs_bps": p95,
        "p99_abs_bps": p99,
        "mean_abs_bps": mean_abs,
        "threshold_bps": p95_threshold_bps,
        "passes": passes,
        "rows": rows[:1000],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Real-vs-Synthetic Fill Calibration (E5)",
        "",
        f"Generated: {report['generated_at']}",
        f"Fills considered: {report['fills_considered']}",
        f"Deltas computed: {report['deltas_computed']}",
        "",
        "## Summary",
        "",
        "| Statistic | |bps| |",
        "| --- | ---: |",
        f"| p50 | {report['p50_abs_bps']:.3f} |",
        f"| p95 | {report['p95_abs_bps']:.3f} |",
        f"| p99 | {report['p99_abs_bps']:.3f} |",
        f"| mean | {report['mean_abs_bps']:.3f} |",
        "",
        f"Plan threshold: **p95 ≤ {report['threshold_bps']} bps**",
        f"Passes: **{report['passes']}**",
        "",
        "## Next action",
        "",
        "If this gate fails, recalibrate the paper engine:",
        "",
        "- `config.paper.half_spread_bps` should shift by roughly half of",
        "  the mean absolute delta (signed).",
        "- `config.paper.impact_coefficient` should be adjusted only after",
        "  confirming the excess is not attributable to spread.",
        "- Commit the change with explicit evidence: before/after p95 plus a",
        "  reference to this report JSON.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fills-dir",
        default=str(ROOT / "output" / "executions"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "output" / "qa"),
    )
    parser.add_argument("--p95-threshold-bps", type=float, default=2.0)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    fills_dir = Path(args.fills_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_calibration_report(
        fills_dir, p95_threshold_bps=args.p95_threshold_bps
    )
    md_path = out_dir / "real_vs_synthetic_fills.md"
    json_path = out_dir / "real_vs_synthetic_fills.json"
    write_markdown(report, md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    status = "PASS" if report["passes"] else "FAIL"
    print(
        f"[CALIB] {status} — "
        f"p95={report['p95_abs_bps']:.3f} bps "
        f"(<= {report['threshold_bps']} bps) "
        f"over {report['deltas_computed']} fills — "
        f"md={md_path}"
    )

    if report["passes"]:
        return 0
    if report["deltas_computed"] == 0:
        logger.warning("[CALIB] no real fills available — skipping gate")
        return 0
    if args.enforce:
        return 1
    logger.warning("[CALIB] grace period — fail NOT blocking")
    return 0


if __name__ == "__main__":
    sys.exit(main())
