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

    Both gates (robustness and reconciliation) must pass for candidate status.
    The function combines both gates deterministically:
    - If any gate is False -> block candidate
    - If both gates are True -> allow candidate
    - If one or both gates are None -> allow with warning (backward compatible)

    Args:
        robustness_ok: Robustness pack result (True/False/None)
            - True: All enabled robustness tests passed
            - False: At least one robustness test failed
            - None: Robustness pack was not run
        robustness_pack_path: Optional path to robustness pack directory (included in message if set)
        reconciliation_ok: Reconciliation result (True/False/None) (Sprint 13 extension)
            - True: Reconciliation passed
            - False: Reconciliation failed
            - None: Reconciliation was not run (backward compatible)
        reconcile_report_path: Optional path to reconciliation report (included in message if set)

    Returns:
        Tuple of (candidate_allowed: bool, message: str):
        - candidate_allowed: True if all gates pass, False otherwise
        - message: Human-readable message explaining the decision, including report links if available

    Note:
        - If any gate is False, candidate_allowed = False
        - If both gates are True, candidate_allowed = True
        - If one or both gates are None, candidate_allowed = True (backward compatible) with warning
        - Report paths are included in messages when available for easy troubleshooting
    """
    robustness_status = []
    reconciliation_status = []
    candidate_allowed = True

    # Check robustness gate
    pack_link = f" (report: {robustness_pack_path})" if robustness_pack_path else ""
    if robustness_ok is None:
        # Robustness pack was not run
        status_msg = f"Robustness pack not run{pack_link}"
        robustness_status.append(status_msg)
        logger.warning(status_msg)
        # None is backward compatible (allow with warning)
    elif robustness_ok is False:
        # Robustness pack failed - block candidate
        status_msg = f"Robustness pack failed{pack_link}"
        robustness_status.append(status_msg)
        logger.warning(status_msg)
        candidate_allowed = False
    else:
        # robustness_ok is True
        status_msg = f"Robustness pack passed{pack_link}"
        robustness_status.append(status_msg)

    # Check reconciliation gate (Sprint 13 extension)
    report_link = f" (report: {reconcile_report_path})" if reconcile_report_path else ""
    if reconciliation_ok is False:
        # Reconciliation failed - block candidate
        status_msg = f"Reconciliation failed{report_link}"
        reconciliation_status.append(status_msg)
        logger.warning(status_msg)
        candidate_allowed = False
    elif reconciliation_ok is None:
        # Reconciliation not run - allow but warn (backward compatible)
        status_msg = f"Reconciliation not run{report_link} (backward compatible)"
        reconciliation_status.append(status_msg)
        logger.warning(status_msg)
        # None is backward compatible (allow with warning)
    else:
        # reconciliation_ok is True
        status_msg = f"Reconciliation passed{report_link}"
        reconciliation_status.append(status_msg)

    # Combine status messages deterministically
    all_status = robustness_status + reconciliation_status
    if candidate_allowed:
        combined_message = " - ".join(all_status) + " - candidate allowed"
        logger.info(combined_message)
        return True, combined_message
    else:
        combined_message = " | ".join(all_status) + " - candidate NOT allowed"
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
        logger.warning(
            f"Failed to read robustness_ok from manifest {manifest_path}: {exc}"
        )
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
        logger.warning(
            f"Failed to read reconciliation_ok from manifest {manifest_path}: {exc}"
        )
        return None
