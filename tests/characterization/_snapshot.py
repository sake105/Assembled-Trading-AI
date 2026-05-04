"""Lightweight snapshot helper for golden-master tests.

Replaces the `approvaltests` package dependency with a self-contained
implementation that stores approved files as plain text.

Usage::

    from tests.characterization._snapshot import verify_snapshot

    verify_snapshot("my_test_name", actual_string, approved_dir)

On first run or when UPDATE_SNAPSHOTS=1:
  - Writes the actual string to approved_dir/name.approved.txt
  - Test passes (auto-approves).

On subsequent runs:
  - Compares actual against stored approved.
  - Raises AssertionError with a unified diff if they differ.

To update a snapshot after an intentional change::

    UPDATE_SNAPSHOTS=1 pytest tests/characterization/test_golden_equity.py
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path


def verify_snapshot(name: str, actual: str, approved_dir: Path) -> None:
    """Assert that *actual* matches the stored snapshot for *name*.

    Args:
        name: Identifier for the snapshot (used as filename stem).
        actual: The string to compare / store.
        approved_dir: Directory that holds `<name>.approved.txt` files.
    """
    approved_dir = Path(approved_dir)
    approved_dir.mkdir(parents=True, exist_ok=True)
    approved_path = approved_dir / f"{name}.approved.txt"
    received_path = approved_dir / f"{name}.received.txt"

    # Always write received for debugging
    received_path.write_text(actual, encoding="utf-8")

    update = os.environ.get("UPDATE_SNAPSHOTS", "").lower() in ("1", "true", "yes")

    if not approved_path.exists() or update:
        approved_path.write_text(actual, encoding="utf-8")
        return  # first run / explicit update — auto-approve

    approved = approved_path.read_text(encoding="utf-8")
    if actual == approved:
        received_path.unlink(missing_ok=True)
        return

    diff = "\n".join(
        difflib.unified_diff(
            approved.splitlines(),
            actual.splitlines(),
            fromfile=f"{name}.approved.txt",
            tofile=f"{name}.received.txt",
            lineterm="",
        )
    )
    raise AssertionError(
        f"Snapshot mismatch for '{name}'.\n"
        f"Run with UPDATE_SNAPSHOTS=1 to approve intentional changes.\n\n{diff}"
    )
