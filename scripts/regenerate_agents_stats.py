"""Regenerate AGENTS.md repo statistics (C5).

Updates the summary line in AGENTS.md with current module/script/test/CI counts.
Run after adding new modules, scripts, tests, or workflows.

Usage:
    python scripts/regenerate_agents_stats.py [--dry-run]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def collect_stats(root: Path) -> dict[str, int]:
    return {
        "core_modules": len(list((root / "src" / "assembled_core").glob("*/"))),
        "scripts": len(list((root / "scripts").glob("*.py"))),
        "test_files": len(list((root / "tests").rglob("test_*.py"))),
        "ci_workflows": len(list((root / ".github" / "workflows").glob("*.yml"))),
    }


def update_agents_md(root: Path, stats: dict[str, int], dry_run: bool = False) -> bool:
    agents_path = root / "AGENTS.md"
    if not agents_path.exists():
        print(f"[WARN] {agents_path} not found", file=sys.stderr)
        return False

    content = agents_path.read_text(encoding="utf-8")
    pattern = r"Es umfasst \*\*\d+ Kernmodule\*\*.*?\(Stand \d{4}-\d{2}-\d{2}.*?\)"
    replacement = (
        f"Es umfasst **{stats['core_modules']} Kernmodule** in `src/assembled_core/`, "
        f"**~{stats['scripts']} Scripts**, **~{stats['test_files']} Testdateien** "
        f"und **{stats['ci_workflows']} CI-Workflows**. "
        f"(Stand 2026-04-26 — `scripts/regenerate_agents_stats.py` für aktuelle Zahlen.)"
    )

    new_content, n = re.subn(pattern, replacement, content)
    if n == 0:
        print("[WARN] Pattern not found in AGENTS.md — no update made", file=sys.stderr)
        return False

    if dry_run:
        print("[DRY-RUN] Would update AGENTS.md:")
        print(f"  core_modules: {stats['core_modules']}")
        print(f"  scripts: {stats['scripts']}")
        print(f"  test_files: {stats['test_files']}")
        print(f"  ci_workflows: {stats['ci_workflows']}")
        return True

    agents_path.write_text(new_content, encoding="utf-8")
    print(f"[OK] AGENTS.md updated: {stats}")
    return True


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    root = Path(__file__).parent.parent
    stats = collect_stats(root)
    print(f"[INFO] Stats: {stats}")
    update_agents_md(root, stats, dry_run=dry_run)


if __name__ == "__main__":
    main()
