#!/usr/bin/env python3
"""Dev-only: run strategy benchmark (quick) and write top-level SUMMARY.md.

Best variant by stability_score, return, sharpe, calmar; warnings/anomalies count.
No new dependencies. ASCII-only.

Usage:
  py -3 scripts/dev/run_full_analysis_suite.py --output-root output/system_run
  py -3 scripts/dev/run_full_analysis_suite.py --output-root output/system_run --quick --max-variants 6 --include-synthetic
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = ROOT / "scripts" / "dev" / "run_strategy_benchmark.py"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run benchmark and write SUMMARY.md")
    ap.add_argument("--output-root", type=Path, default=Path("output/system_run"))
    ap.add_argument("--quick", action="store_true", default=True, help="Quick run (1y only)")
    ap.add_argument("--max-variants", type=int, default=6)
    ap.add_argument("--include-synthetic", action="store_true", help="Use synthetic if no real data")
    ap.add_argument("--skip-run", action="store_true", help="Only write SUMMARY from existing benchmark output")
    args = ap.parse_args()
    out_root = args.output_root.resolve()
    if not out_root.is_absolute():
        out_root = (ROOT / out_root).resolve()
    bench_root = out_root / "benchmark"

    if not args.skip_run:
        cmd = [
            sys.executable, str(BENCH_SCRIPT),
            "--output-root", str(out_root),
            "--quick", "--max-variants", str(args.max_variants),
        ]
        if args.include_synthetic:
            cmd.append("--include-synthetic")
        ret = subprocess.run(cmd, cwd=str(ROOT), timeout=600)
        if ret.returncode != 0:
            print("Benchmark run failed.", file=sys.stderr)
            return ret.returncode

    lines = ["# Analysis Suite Summary", ""]
    scoreboard_path = bench_root / "scoreboard.json"
    anomalies_path = bench_root / "anomalies.json"
    if not scoreboard_path.exists():
        lines.append("No scoreboard.json found. Run benchmark first.")
        (out_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {out_root / 'SUMMARY.md'}")
        return 0

    with scoreboard_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    valid = [r for r in rows if r.get("total_return") is not None]
    if not valid:
        lines.append("No valid runs in scoreboard.")
    else:
        by_stability = sorted(valid, key=lambda x: (x.get("stability_score") or 0), reverse=True)[:1]
        by_return = sorted(valid, key=lambda x: (x.get("total_return") or 0), reverse=True)[:1]
        by_sharpe = sorted([r for r in valid if r.get("sharpe_ratio") is not None], key=lambda x: (x.get("sharpe_ratio") or 0), reverse=True)[:1]
        by_calmar = sorted([r for r in valid if r.get("calmar_ratio") is not None], key=lambda x: (x.get("calmar_ratio") or 0), reverse=True)[:1]
        lines.append("## Best variant by stability_score")
        lines.append("")
        for r in by_stability:
            lines.append(f"- {r.get('variant_id')} / {r.get('horizon')}: stability_score={r.get('stability_score')}")
        lines.append("")
        lines.append("## Best by total return")
        lines.append("")
        for r in by_return:
            lines.append(f"- {r.get('variant_id')} / {r.get('horizon')}: return={r.get('total_return')}")
        lines.append("")
        lines.append("## Best by Sharpe")
        lines.append("")
        for r in by_sharpe:
            lines.append(f"- {r.get('variant_id')} / {r.get('horizon')}: sharpe={r.get('sharpe_ratio')}")
        lines.append("")
        lines.append("## Best by Calmar")
        lines.append("")
        for r in by_calmar:
            lines.append(f"- {r.get('variant_id')} / {r.get('horizon')}: calmar={r.get('calmar_ratio')}")
        lines.append("")

    anom_count = 0
    if anomalies_path.exists():
        try:
            anom = json.loads(anomalies_path.read_text(encoding="utf-8"))
            anom_count = len(anom)
        except Exception:
            pass
    lines.append("## Warnings / Anomalies")
    lines.append("")
    lines.append(f"Total anomalies: {anom_count}")
    lines.append("")
    lines.append("See benchmark/BENCHMARK_REPORT.md and benchmark/anomalies.json for details.")
    lines.append("")

    (out_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_root / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
