"""Dependency Audit Script (Plan 11.10).

Compares pyproject.toml ranges vs requirements.txt pins.
Warns on drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_requirements_txt(path: Path) -> dict[str, str]:
    """Parse requirements.txt into {package: version_spec}."""
    reqs: dict[str, str] = {}
    if not path.exists():
        return reqs
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Match package==version or package>=version
        match = re.match(r"^([a-zA-Z0-9_-]+)\s*([><=!~]+.+)?", line)
        if match:
            pkg = match.group(1).lower().replace("-", "_")
            ver = match.group(2) or ""
            reqs[pkg] = ver.strip()
    return reqs


def parse_pyproject_deps(path: Path) -> dict[str, str]:
    """Parse pyproject.toml dependencies into {package: version_spec}."""
    deps: dict[str, str] = {}
    if not path.exists():
        return deps

    in_deps = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps and stripped == "]":
            break
        if in_deps and stripped.startswith('"'):
            dep = stripped.strip('",')
            match = re.match(r"^([a-zA-Z0-9_-]+)\s*([><=!~]+.+)?", dep)
            if match:
                pkg = match.group(1).lower().replace("-", "_")
                ver = match.group(2) or ""
                deps[pkg] = ver.strip()
    return deps


def audit_dependencies(
    project_root: Path | None = None,
) -> dict:
    """Compare pyproject.toml and requirements.txt for drift.

    Returns:
        Dict with matching, drifted, only_pyproject, only_requirements.
    """
    root = project_root or Path(".")
    pyproject = parse_pyproject_deps(root / "pyproject.toml")
    requirements = parse_requirements_txt(root / "requirements.txt")

    all_pkgs = set(pyproject) | set(requirements)
    matching = []
    drifted = []
    only_pyproject = []
    only_requirements = []

    for pkg in sorted(all_pkgs):
        in_pp = pkg in pyproject
        in_req = pkg in requirements

        if in_pp and in_req:
            if pyproject[pkg] == requirements[pkg]:
                matching.append(pkg)
            else:
                drifted.append({
                    "package": pkg,
                    "pyproject": pyproject[pkg],
                    "requirements": requirements[pkg],
                })
        elif in_pp:
            only_pyproject.append(pkg)
        else:
            only_requirements.append(pkg)

    return {
        "n_matching": len(matching),
        "n_drifted": len(drifted),
        "drifted": drifted,
        "only_pyproject": only_pyproject,
        "only_requirements": only_requirements,
    }


if __name__ == "__main__":
    result = audit_dependencies()
    print(f"Matching: {result['n_matching']}")
    print(f"Drifted: {result['n_drifted']}")
    for d in result["drifted"]:
        print(f"  {d['package']}: pyproject={d['pyproject']} vs requirements={d['requirements']}")
    if result["only_pyproject"]:
        print(f"Only in pyproject.toml: {result['only_pyproject']}")
    if result["only_requirements"]:
        print(f"Only in requirements.txt: {result['only_requirements']}")
    sys.exit(1 if result["n_drifted"] > 0 else 0)
