#!/usr/bin/env python
"""
Generate Review Bundle - Creates a single text file with project structure and file contents.

This script collects the structure and contents of important files for code review.
Large files and excluded directories are skipped.

Usage:
    python scripts/generate_review_bundle.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("review_bundle.txt")

INCLUDE_DIRS = [
    "src/assembled_core",
    "scripts",
    "tests",
    "docs",
    "configs",
    ".github/workflows",
]

INCLUDE_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "watchlist.txt",
]

EXCLUDE_SUBSTR = [
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "data/raw",
    "output",
    "logs",
    "experiments",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
]

MAX_FILE_KB = 400  # große Dateien abschneiden


def is_excluded(p: Path) -> bool:
    """Check if path should be excluded."""
    s = str(p).replace("\\", "/")
    return any(x in s for x in EXCLUDE_SUBSTR)


def read_text_file(p: Path) -> str:
    """Read text file with error handling."""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[Could not read: {e}]"


def main() -> None:
    """Generate review bundle."""
    lines = []
    lines.append("=" * 80 + "\n")
    lines.append("REVIEW BUNDLE - Assembled Trading AI\n")
    lines.append("=" * 80 + "\n")
    lines.append(f"Generated: {Path(__file__).stat().st_mtime}\n")
    lines.append(f"Project Root: {ROOT.resolve()}\n")
    lines.append("=" * 80 + "\n\n")

    lines.append("=== PROJECT TREE (selected) ===\n\n")

    def add_tree(base: Path) -> None:
        """Add directory tree to lines."""
        for p in sorted(base.rglob("*")):
            if p.is_dir() or is_excluded(p):
                continue
            rel = p.relative_to(ROOT)
            lines.append(str(rel) + "\n")

    for d in INCLUDE_DIRS:
        dp = ROOT / d
        if dp.exists():
            add_tree(dp)

    lines.append("\n" + "=" * 80 + "\n")
    lines.append("=== FILE CONTENTS ===\n")
    lines.append("=" * 80 + "\n\n")

    # Include root files
    for f in INCLUDE_FILES:
        fp = ROOT / f
        if fp.exists() and fp.is_file() and not is_excluded(fp):
            lines.append(f"\n{'=' * 80}\n")
            lines.append(f"--- {f} ---\n")
            lines.append(f"{'=' * 80}\n\n")
            lines.append(read_text_file(fp) + "\n")

    # Include directory file contents
    for d in INCLUDE_DIRS:
        dp = ROOT / d
        if not dp.exists():
            continue
        for p in sorted(dp.rglob("*")):
            if p.is_dir() or is_excluded(p):
                continue
            if p.suffix.lower() not in {
                ".py",
                ".md",
                ".yml",
                ".yaml",
                ".toml",
                ".txt",
                ".json",
                ".ini",
                ".ps1",
            }:
                continue
            kb = p.stat().st_size / 1024
            rel_path = p.relative_to(ROOT)
            lines.append(f"\n{'=' * 80}\n")
            lines.append(f"--- {rel_path} ({kb:.1f} KB) ---\n")
            lines.append(f"{'=' * 80}\n\n")
            if kb > MAX_FILE_KB:
                lines.append(
                    f"[SKIPPED: file too large > {MAX_FILE_KB}KB, showing first 100 lines]\n\n"
                )
                # Show first 100 lines for large files
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    first_lines = content.split("\n")[:100]
                    lines.append("\n".join(first_lines) + "\n")
                    lines.append(f"\n[... {len(content.split(chr(10))) - 100} more lines ...]\n")
                except Exception as e:
                    lines.append(f"[Could not read: {e}]\n")
            else:
                lines.append(read_text_file(p) + "\n")

    # Write output
    OUT.write_text("".join(lines), encoding="utf-8")
    file_size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"OK: Wrote: {OUT.resolve()}")
    print(f"OK: Size: {file_size_mb:.2f} MB")
    print(f"OK: Lines: {len(lines)}")


if __name__ == "__main__":
    main()

