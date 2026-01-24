"""Candidate gate for strategy validation (Sprint 12 Final + Sprint 13).

This module provides a gate function to check if a strategy is allowed to be
marked as "candidate" based on robustness pack and reconciliation results.

A strategy can only be marked as "candidate" if:
- robustness_ok == True (Sprint 12)
- reconciliation_ok != False (Sprint 13: blocks if False, allows if None/True)

If robustness pack was not run, candidate_allowed = False and a warning is logged.
If reconciliation failed (reconciliation_ok=False), candidate_allowed = False.
If reconciliation was not run (reconciliation_ok=None), candidate_allowed = True (backward compatible) with warning.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def check_candidate_allowed(
    robustness_ok: bool | None,
    robustness_pack_path: str | Path | None = None,
    reconciliation_ok: bool | None = None,
    reconcile_report_path: str | Path | None = None,
) -> tuple[bool, str]:
    """Check if strategy is allowed to be marked as "candidate" (Sprint 12 Final + Sprint 13).

    Args:
        robustness_ok: Robustness pack result (True/False/None)
            - True: All enabled robustness tests passed
            - False: At least one robustness test failed
            - None: Robustness pack was not run
        robustness_pack_path: Optional path to robustness pack directory (for logging)
        reconciliation_ok: Reconciliation result (True/False/None) (Sprint 13 extension)
            - True: Reconciliation passed (or not performed)
            - False: Reconciliation failed
            - None: Reconciliation was not run (backward compatible)
        reconcile_report_path: Optional path to reconciliation report (for logging)

    Returns:
        Tuple of (candidate_allowed: bool, message: str):
        - candidate_allowed: True if all gates pass, False otherwise
        - message: Human-readable message explaining the decision

    Note:
        - Robustness gate: If robustness_ok is None, candidate_allowed = False
        - Reconciliation gate (Sprint 13): If reconciliation_ok is False, candidate_allowed = False
        - Reconciliation gate: If reconciliation_ok is None, candidate_allowed = True (backward compatible)
          but a warning message is included
        - This gate is minimal-invasive: it only checks gates, does not modify any existing logic
    """
    messages = []
    candidate_allowed = True

    # Check robustness gate
    if robustness_ok is None:
        # Robustness pack was not run
        pack_path_str = str(robustness_pack_path) if robustness_pack_path else "unknown"
        message = (
            f"Robustness pack not run (path: {pack_path_str}). "
            "Candidate status requires robustness pack to be executed."
        )
        logger.warning(message)
        messages.append(message)
        candidate_allowed = False
    elif robustness_ok is False:
        # Robustness pack failed
        pack_path_str = str(robustness_pack_path) if robustness_pack_path else "unknown"
        message = (
            f"Robustness pack failed (path: {pack_path_str}). "
            "Candidate status requires all enabled robustness tests to pass."
        )
        logger.warning(message)
        messages.append(message)
        candidate_allowed = False
    else:
        # robustness_ok is True
        messages.append("Robustness pack passed")

    # Check reconciliation gate (Sprint 13 extension)
    if reconciliation_ok is False:
        # Reconciliation failed - block candidate
        report_path_str = str(reconcile_report_path) if reconcile_report_path else "unknown"
        message = (
            f"Reconciliation failed (report: {report_path_str}). "
            "Candidate status requires reconciliation to pass."
        )
        logger.warning(message)
        messages.append(message)
        candidate_allowed = False
    elif reconciliation_ok is None:
        # Reconciliation not run - allow but warn (backward compatible)
        message = "Reconciliation not run (missing reconciliation_ok in manifest). Candidate allowed (backward compatible)."
        logger.warning(message)
        messages.append(message)
        # candidate_allowed remains True (backward compatible)
    else:
        # reconciliation_ok is True
        messages.append("Reconciliation passed")

    # Combine messages
    if candidate_allowed:
        combined_message = " - ".join(messages) + " - candidate allowed"
        logger.info(combined_message)
        return True, combined_message
    else:
        combined_message = " | ".join(messages) + " - candidate NOT allowed"
        logger.warning(combined_message)
        return False, combined_message


def read_robustness_ok_from_manifest(manifest_path: Path) -> bool | None:
    """Read robustness_ok from run manifest (helper function).

    Args:
        manifest_path: Path to run_manifest.json

    Returns:
        robustness_ok value (True/False/None) or None if manifest not found/invalid
    """
    import json

    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest.get("robustness_ok")
    except Exception as exc:
        logger.warning(f"Failed to read robustness_ok from manifest {manifest_path}: {exc}")
        return None


def read_reconciliation_ok_from_manifest(manifest_path: Path) -> bool | None:
    """Read reconciliation_ok from run manifest (helper function, Sprint 13).

    Args:
        manifest_path: Path to run_manifest.json

    Returns:
        reconciliation_ok value (True/False/None) or None if manifest not found/invalid
    """
    import json

    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest.get("reconciliation_ok")
    except Exception as exc:
        logger.warning(f"Failed to read reconciliation_ok from manifest {manifest_path}: {exc}")
        return None
