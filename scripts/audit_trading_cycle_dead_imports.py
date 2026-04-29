"""Audit dead imports in trading_cycle.py.

For every 'from src.assembled_core.X.Y import ...' line, checks whether
the source file exists under src/ or has been archived. Outputs a CSV.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARCHIVE_DIRS = list(ROOT.glob("archive/**")) + list(ROOT.glob("*graveyard*/**"))

TARGET = SRC / "assembled_core" / "pipeline" / "trading_cycle.py"
OUT_CSV = ROOT / "docs" / "audit" / "trading_cycle_dead_imports.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

IMPORT_RE = re.compile(r"^\s*(from|import)\s+(src\.assembled_core\S+)")


def module_to_path(module: str) -> Path:
    # module starts with "src.assembled_core..." — map relative to ROOT
    parts = module.split(".")
    as_file = ROOT / Path(*parts).with_suffix(".py")
    as_pkg = ROOT / Path(*parts) / "__init__.py"
    # Return whichever exists; default to as_file for archive checks
    return as_file if as_file.exists() or not as_pkg.exists() else as_pkg


def is_in_try_block(lines: list[str], lineno: int) -> bool:
    """Heuristic: scan backwards for 'try:' before hitting a non-indented line."""
    indent = len(lines[lineno]) - len(lines[lineno].lstrip())
    for i in range(lineno - 1, max(0, lineno - 30), -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        cur_indent = len(lines[i]) - len(lines[i].lstrip())
        if cur_indent < indent and stripped.startswith("try:"):
            return True
        if cur_indent == 0 and not stripped.startswith("#"):
            break
    return False


def check_archive(module: str) -> str:
    path = module_to_path(module)
    rel = path.relative_to(ROOT) if ROOT in path.parents else path
    for d in ARCHIVE_DIRS:
        candidate = d / path.name
        if candidate.exists():
            return f"ARCHIVED({candidate.relative_to(ROOT)})"
    return "MISSING"


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    lines = source.splitlines()

    rows = []
    for i, line in enumerate(lines, start=1):
        m = IMPORT_RE.match(line)
        if not m:
            continue
        module = m.group(2).split()[0].rstrip(",")
        path = module_to_path(module)
        if path.exists():
            status = "OK"
        else:
            status = check_archive(module)
        in_try = is_in_try_block(lines, i - 1)
        rows.append({
            "line_number": i,
            "module": module,
            "status": status,
            "in_try_block": in_try,
            "line": line.strip(),
        })

    dead = [r for r in rows if r["status"] != "OK"]
    ok = [r for r in rows if r["status"] == "OK"]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["line_number", "module", "status", "in_try_block", "line"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total imports scanned : {len(rows)}")
    print(f"  OK (file exists)    : {len(ok)}")
    print(f"  Dead (missing/arch) : {len(dead)}")
    print(f"CSV written to        : {OUT_CSV}")

    if dead:
        print("\nDead imports sample (first 20):")
        for r in dead[:20]:
            print(f"  L{r['line_number']:5d}  try={r['in_try_block']}  {r['status'][:40]}  {r['module']}")


if __name__ == "__main__":
    sys.exit(main())
