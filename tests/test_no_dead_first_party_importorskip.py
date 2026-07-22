# tests/test_no_dead_first_party_importorskip.py
"""Guard: no importorskip() on deleted first-party modules.

``pytest.importorskip("src.assembled_core....")`` on a module that no longer
exists silently skips tests forever, making module deletions invisible instead
of red (this happened at scale in the 2026 archive sweeps: ~280 phantom skips).

This meta-test scans every file under tests/ for first-party importorskip
targets and FAILS if any target module is not importable. A future module
deletion therefore turns the suite red, forcing an explicit decision about the
guarded tests (delete or repoint) instead of a silent shrink of coverage.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
FIRST_PARTY_IMPORTORSKIP = re.compile(
    r"importorskip\(\s*[\"'](src\.[A-Za-z0-9_\.]+)[\"']"
)


def _find_spec_safe(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def test_no_dead_first_party_importorskip_targets() -> None:
    """Every first-party importorskip target in tests/ must be importable."""
    targets: dict[str, list[str]] = {}
    for path in TESTS_DIR.rglob("*.py"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in FIRST_PARTY_IMPORTORSKIP.finditer(text):
            targets.setdefault(match.group(1), []).append(
                str(path.relative_to(TESTS_DIR))
            )

    dead = {
        module: sorted(set(files))
        for module, files in sorted(targets.items())
        if not _find_spec_safe(module)
    }

    assert not dead, (
        "Dead first-party importorskip targets found (module deleted but tests "
        "still skip-guard on it — delete or repoint the guarded tests):\n"
        + "\n".join(f"  {mod} <- {', '.join(files)}" for mod, files in dead.items())
    )
