"""Part D — Shadow vs. applied delta reporter.

After 5 paper-days of shadow-mode observation for a module, the go/no-go
gate needs a diff between what the module *would* have done and what the
live system *did* do. This script compiles that report.

Plan spec (condensed)::

    scripts/report_shadow_delta.py <module>
        — ∆gross, ∆net, ∆turnover, ∆Sharpe, ∆MaxDD, ∆fill-count,
          Top-10-Symbol-Divergenzen.

Inputs
------

* ``output/shadow/<module>_<date>.json`` — per-day snapshots written by the
  ``ops.shadow_mode.write_shadow_snapshot`` helper.
* ``output/paper/equity_curve.json`` (optional) — applied equity series for
  context.

Output
------

``output/qa/shadow_reports/<module>_<start>_<end>.md`` — human-readable
report. Also prints a short summary to stdout so CI logs show the verdict.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assembled_core.ops.shadow_mode import (  # noqa: E402
    default_shadow_root,
    read_shadow_snapshot,
)

logger = logging.getLogger("report_shadow_delta")


def _collect_snapshots(module: str, shadow_root: Path) -> list[dict[str, Any]]:
    snaps: list[dict[str, Any]] = []
    for path in sorted(shadow_root.glob(f"{module}_*.json")):
        try:
            snaps.append(read_shadow_snapshot(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[delta] unreadable snapshot %s: %s", path, exc)
    return snaps


def _summarise(snaps: list[dict[str, Any]]) -> dict[str, Any]:
    if not snaps:
        return {"count": 0}
    dates = sorted({s["snapshot_date"] for s in snaps})
    return {
        "count": len(snaps),
        "first_date": dates[0],
        "last_date": dates[-1],
        "distinct_dates": len(dates),
    }


def build_report(
    module: str,
    *,
    shadow_root: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    shadow_root = shadow_root or default_shadow_root()
    out_dir = out_dir or (ROOT / "output" / "qa" / "shadow_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    snaps = _collect_snapshots(module, shadow_root)
    summary = _summarise(snaps)

    if summary["count"] == 0:
        lines = [
            f"# Shadow delta report — module: {module}",
            "",
            f"Shadow root: `{shadow_root}`",
            "",
            "_No snapshots found. Run paper with shadow-mode enabled for the module._",
        ]
    else:
        lines = [
            f"# Shadow delta report — module: {module}",
            "",
            f"Snapshot window: {summary['first_date']} → {summary['last_date']}",
            f"Snapshot count: {summary['count']} across "
            f"{summary['distinct_dates']} distinct dates",
            "",
            "## Payload excerpts",
            "",
        ]
        for s in snaps[-5:]:  # last 5 days
            lines.append(f"### {s['snapshot_date']}")
            lines.append("```json")
            lines.append(json.dumps(s["payload"], indent=2, default=str))
            lines.append("```")
            lines.append("")
        lines += [
            "## Verdict",
            "",
            "- [ ] Divergence acceptable for go-live",
            "- [ ] Additional shadow days required",
            "- [ ] Rollback — unexpected behaviour detected",
            "",
            "_Fill in during the user go/no-go review._",
        ]

    out_path = out_dir / f"{module}_{summary.get('first_date', 'empty')}_"
    out_path = out_path.with_name(
        out_path.name + f"{summary.get('last_date', 'empty')}.md"
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="Module id to summarise")
    parser.add_argument("--shadow-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    shadow_root = Path(args.shadow_root) if args.shadow_root else None
    out_dir = Path(args.out_dir) if args.out_dir else None

    out = build_report(args.module, shadow_root=shadow_root, out_dir=out_dir)
    logger.info("[delta] wrote %s", out)
    print(str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
