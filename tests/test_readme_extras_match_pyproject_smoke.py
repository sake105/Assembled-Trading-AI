"""Smoke test: README pip install extras match pyproject.toml optional-dependencies (no drift)."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


def _parse_readme_extras_install(readme_text: str) -> set[str]:
    """Parse README for pip install -e \".[extra1,extra2,...]\" or '.[...]'; return set of extra names."""
    # Match .[dev,scipy,ml] or .[dev, scipy, ml] (optional quotes around the spec)
    pattern = re.compile(
        r"pip\s+install\s+(?:-e\s+)?[\"']?\s*\.\s*\[([^\]]+)\]\s*[\"']?",
        re.IGNORECASE,
    )
    found: set[str] = set()
    for m in pattern.finditer(readme_text):
        names = [x.strip() for x in m.group(1).split(",") if x.strip()]
        found.update(names)
    return found


def test_readme_extras_install_matches_pyproject() -> None:
    """README pip install -e \".[...]\" extras set must equal pyproject.toml optional-dependencies keys."""
    repo_root = Path(__file__).resolve().parents[1]
    pyproject_path = repo_root / "pyproject.toml"
    readme_path = repo_root / "README.md"

    assert pyproject_path.exists(), "pyproject.toml not found"
    assert readme_path.exists(), "README.md not found"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    optional = data.get("project", {}).get("optional-dependencies", {})
    pyproject_extras = set(optional.keys())
    assert pyproject_extras, (
        "pyproject.toml should define at least one optional-dependency extra"
    )

    readme_text = readme_path.read_text(encoding="utf-8")
    readme_extras = _parse_readme_extras_install(readme_text)

    assert readme_extras, (
        'README should contain at least one pip install -e ".[extra1,extra2,...]" line'
    )
    assert readme_extras == pyproject_extras, (
        f"README extras in install line {sorted(readme_extras)} must equal "
        f"pyproject.toml optional-dependencies keys {sorted(pyproject_extras)}"
    )
