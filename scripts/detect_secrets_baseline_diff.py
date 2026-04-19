"""Diff detect-secrets scan output against the committed baseline.

Reads the current `detect-secrets scan` JSON (path via --current) and the
baseline `.secrets.baseline` (path via --baseline). Any finding in the
current scan whose `hashed_secret` is not present in the baseline for the
same file is treated as a NEW finding and causes exit 1.

Moved out of `.github/workflows/secrets-scan.yml` because the inline
Python heredoc there used unindented source that terminated the YAML
block scalar on parse — the workflow failed to load entirely and the
detect-secrets job never ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True, help="detect-secrets scan JSON")
    ap.add_argument("--baseline", required=True, help=".secrets.baseline path")
    args = ap.parse_args()

    current = json.loads(Path(args.current).read_text(encoding="utf-8")).get("results", {})
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")).get("results", {})

    new_findings: dict[str, list[dict]] = {}
    for path, items in current.items():
        baseline_items = baseline.get(path, [])
        baseline_hashes = {i.get("hashed_secret") for i in baseline_items}
        new = [i for i in items if i.get("hashed_secret") not in baseline_hashes]
        if new:
            new_findings[path] = new

    if new_findings:
        print("NEW secret findings detected (not in baseline):")
        for path, items in new_findings.items():
            print(f"  {path}: {len(items)} new finding(s)")
            for item in items:
                print(f"    line {item.get('line_number')}: {item.get('type')}")
        return 1

    print("No new secret findings. Baseline-aligned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
