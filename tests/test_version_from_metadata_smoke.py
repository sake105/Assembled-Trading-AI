"""Smoke test: assembled_core.__version__ from package metadata (ASCII, non-empty). Optional: match pyproject.toml."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core import __version__


def test_version_is_string_non_empty() -> None:
    """__version__ is a non-empty string (from importlib.metadata or fallback)."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0
    assert __version__.encode("ascii", errors="ignore").decode("ascii") == __version__


def test_version_matches_pyproject_when_available() -> None:
    """When pyproject.toml exists, parse project.version and assert equality if parseable (stdlib)."""
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        return
    try:
        import tomllib
    except ImportError:
        return  # Python < 3.11
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    proj = data.get("project") or {}
    raw = proj.get("version")
    if not isinstance(raw, str) or not raw.strip():
        return
    expected = raw.strip()
    if __version__ == "0.0.0+unknown":
        return  # Metadata not available (e.g. not installed); skip equality
    assert (
        __version__ == expected
    ), f"__version__ {__version__!r} should match pyproject.toml project.version {expected!r}"
