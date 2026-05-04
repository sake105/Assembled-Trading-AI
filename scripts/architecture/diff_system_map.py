"""
diff_system_map.py — Compare two system_map.json files

Usage:
  python scripts/architecture/diff_system_map.py [OLD] [NEW]

  OLD defaults to data/changelog.json last entry baseline
  NEW defaults to current system_map.json

Exit: 0 no changes, 1 changes found
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MAP = REPO_ROOT / "docs/architecture/system_map/data/system_map.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def diff(old_path: Path, new_path: Path) -> int:
    try:
        old = load(old_path)
        new = load(new_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    old_nodes = {n["id"]: n for n in old.get("nodes", [])}
    new_nodes = {n["id"]: n for n in new.get("nodes", [])}

    added = sorted(new_nodes.keys() - old_nodes.keys())
    removed = sorted(old_nodes.keys() - new_nodes.keys())
    changed_status = sorted(
        nid
        for nid in new_nodes.keys() & old_nodes.keys()
        if old_nodes[nid].get("status") != new_nodes[nid].get("status")
    )

    if not added and not removed and not changed_status:
        print("[OK] No changes detected")
        return 0

    if added:
        print(
            f"ADDED   ({len(added)}): {', '.join(added[:5])}{'...' if len(added) > 5 else ''}"
        )
    if removed:
        print(
            f"REMOVED ({len(removed)}): {', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}"
        )
    if changed_status:
        print(f"STATUS  ({len(changed_status)}):")
        for nid in changed_status[:10]:
            old_s = old_nodes[nid].get("status", "?")
            new_s = new_nodes[nid].get("status", "?")
            print(f"  {nid:60s}  {old_s} → {new_s}")
        if len(changed_status) > 10:
            print(f"  ... and {len(changed_status) - 10} more")

    print(
        f"\nSummary: +{len(added)} -{len(removed)} ~{len(changed_status)} status changes"
    )
    return 1


def main() -> int:
    if len(sys.argv) == 3:
        old_p = Path(sys.argv[1])
        new_p = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        old_p = Path(sys.argv[1])
        new_p = DEFAULT_MAP
    else:
        print("Usage: diff_system_map.py [OLD] [NEW]")
        print("       Both default to current system_map.json when only one given")
        return 2
    return diff(old_p, new_p)


if __name__ == "__main__":
    sys.exit(main())
