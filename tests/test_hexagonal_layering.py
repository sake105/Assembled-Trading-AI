"""Hexagonal-architecture invariant tests (audit C-001).

These tests pin the dependency rules so the skeleton cannot silently
rot back into a spaghetti import graph:

- Domain modules MUST NOT import from ``adapters.*``.
- Domain modules MUST NOT import third-party I/O libs (httpx, fastapi,
  alpaca-py, polygon, yfinance, sqlalchemy).
- Ports modules MUST NOT import from ``adapters.*`` or ``application.*``.
- Bootstrap is the ONLY package allowed to import from both ports and
  adapters.

The checks are static — we parse the import graph, we don't execute
the modules — so adding a forbidden import surfaces immediately at
test time, not at runtime.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "assembled_core"


_FORBIDDEN_IN_DOMAIN_PREFIXES = (
    "src.assembled_core.adapters",
    "assembled_core.adapters",
    "httpx",
    "fastapi",
    "alpaca",
    "polygon",
    "yfinance",
    "sqlalchemy",
)

_FORBIDDEN_IN_PORTS_PREFIXES = (
    "src.assembled_core.adapters",
    "src.assembled_core.application",
    "assembled_core.adapters",
    "assembled_core.application",
)


def _iter_imports(py_file: pathlib.Path) -> Iterable[str]:
    """Yield every dotted module name imported by ``py_file``."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module


def _files_under(*parts: str) -> Iterable[pathlib.Path]:
    base = ROOT.joinpath(*parts)
    if not base.exists():
        return
    for f in base.rglob("*.py"):
        if "__pycache__" in f.parts:
            continue
        yield f


def test_domain_does_not_import_adapters_or_third_party_io() -> None:
    """Audit C-001 invariant — domain stays pure."""
    offenders: list[tuple[str, str]] = []
    for f in _files_under("domain"):
        for imp in _iter_imports(f):
            if imp.startswith(_FORBIDDEN_IN_DOMAIN_PREFIXES):
                offenders.append((str(f.relative_to(ROOT)), imp))
    msg = "domain layer imported forbidden modules:\n  " + "\n  ".join(
        f"{path}  ->  {imp}" for path, imp in offenders
    )
    assert not offenders, msg


def test_ports_do_not_import_adapters_or_application() -> None:
    """Audit C-001 invariant — ports are pure typing surfaces."""
    offenders: list[tuple[str, str]] = []
    for f in _files_under("ports"):
        for imp in _iter_imports(f):
            if imp.startswith(_FORBIDDEN_IN_PORTS_PREFIXES):
                offenders.append((str(f.relative_to(ROOT)), imp))
    msg = "ports layer imported forbidden modules:\n  " + "\n  ".join(
        f"{path}  ->  {imp}" for path, imp in offenders
    )
    assert not offenders, msg


def test_bootstrap_is_only_layer_importing_both_ports_and_adapters() -> None:
    """The composition root is allowed both. No other module is."""
    seen_both: list[str] = []
    for f in _files_under():
        rel = str(f.relative_to(ROOT))
        if rel.startswith(("bootstrap", "__init__")) or rel == "__init__.py":
            continue
        imports = list(_iter_imports(f))
        has_ports = any(
            i.startswith(("src.assembled_core.ports", "assembled_core.ports"))
            for i in imports
        )
        has_adapters = any(
            i.startswith(("src.assembled_core.adapters", "assembled_core.adapters"))
            for i in imports
        )
        if has_ports and has_adapters:
            seen_both.append(rel)
    # Tests in tests/ are also allowed both (they build containers).
    # Anything else is suspect.
    msg = "non-bootstrap modules importing BOTH ports and adapters:\n  " + "\n  ".join(
        seen_both
    )
    assert not seen_both, msg
