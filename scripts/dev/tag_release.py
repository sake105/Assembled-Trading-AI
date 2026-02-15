#!/usr/bin/env python3
"""Local release tag helper: create annotated tag, optional push. No network required except for --push.

Usage: py -3 scripts/dev/tag_release.py --tag vX.Y.Z [--dry-run] [--push]
Tag must match assembled_core.__version__ (e.g. v0.1.0 <-> 0.1.0). --dry-run skips version check.
Output: ASCII-only. Single-line OK or ERROR: ...
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ascii(s: str) -> str:
    return s.encode("ascii", errors="ignore").decode("ascii")


def tag_version_matches_package(tag: str) -> bool:
    """Return True if tag (e.g. v0.1.0) matches assembled_core.__version__ (e.g. 0.1.0)."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from src.assembled_core import __version__ as pkg_version
    except Exception:
        return False
    tag_stripped = tag.lstrip("vV")
    if not re.match(r"^\d+\.\d+\.\d+", tag_stripped):
        return False
    return tag_stripped == pkg_version


def _run(cmd: list[str], capture: bool = True) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=capture,
            text=True,
        )
        out = (r.stdout or "") + (r.stderr or "") if capture else ""
        return r.returncode, _ascii(out)
    except Exception as e:
        return 1, _ascii(str(e))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create annotated release tag (Sprint 13). Optional --push to origin."
    )
    parser.add_argument("--tag", type=str, required=True, metavar="vX.Y.Z", help="Tag name (e.g. v0.1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not run")
    parser.add_argument("--push", action="store_true", help="Run git push origin <tag> after creating tag")
    args = parser.parse_args()
    tag = args.tag.strip()
    dry_run = args.dry_run
    do_push = args.push

    if dry_run:
        print(_ascii(f"[dry-run] git tag -a {tag} -m \"Release {tag} (Sprint 13)\""))
        if do_push:
            print(_ascii(f"[dry-run] git push origin {tag}"))
        print(_ascii(f"OK: tag_created={tag} pushed={str(do_push).lower()}"))
        return 0

    # Tag must match package version (e.g. v0.1.0 <-> 0.0.1)
    if not tag_version_matches_package(tag):
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from src.assembled_core import __version__ as pkg_version
        except Exception:
            pkg_version = "?"
        print(_ascii(f"ERROR: tag {tag} does not match assembled_core.__version__ ({pkg_version})"))
        return 1

    # Check git available
    code, _ = _run(["git", "--version"])
    if code != 0:
        print(_ascii("ERROR: git not available"))
        return 1

    # Working tree clean
    code, out = _run(["git", "status", "--porcelain"])
    if code != 0:
        print(_ascii(f"ERROR: git status failed: {out}"))
        return 1
    if out.strip():
        print(_ascii("ERROR: working tree not clean (git status --porcelain must be empty)"))
        return 1

    # Tag must not exist
    code, _ = _run(["git", "rev-parse", "--verify", tag], capture=True)
    if code == 0:
        print(_ascii(f"ERROR: tag already exists: {tag}"))
        return 1

    # Create annotated tag
    code, err = _run(["git", "tag", "-a", tag, "-m", f"Release {tag} (Sprint 13)"])
    if code != 0:
        print(_ascii(f"ERROR: git tag failed: {err}"))
        return 1

    pushed = False
    if do_push:
        code, err = _run(["git", "push", "origin", tag])
        if code != 0:
            print(_ascii(f"ERROR: git push failed: {err}"))
            return 1
        pushed = True

    print(_ascii(f"OK: tag_created={tag} pushed={str(pushed).lower()}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
