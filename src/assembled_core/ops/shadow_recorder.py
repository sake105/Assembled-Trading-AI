"""Shadow-mode helper — thin wrapper over ``shadow_mode.write_shadow_snapshot``.

Provides the ergonomic ``record_shadow(module, would_apply, ...)`` surface
used by Part D wire-in sites in ``pipeline/trading_cycle.py`` and the
``is_shadow_only(policy, module_key)`` policy gate helper.

Delegates actual persistence to :mod:`shadow_mode` so every D-module writes
through the same atomic writer and shares one snapshot envelope format.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.assembled_core.ops.shadow_mode import write_shadow_snapshot


logger = logging.getLogger(__name__)


def record_shadow(
    module: str,
    would_apply: Mapping[str, Any],
    *,
    as_of: str | None = None,
    meta: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> Path | None:
    """Write one shadow-mode observation.

    Silently swallows I/O errors — a shadow log failure must never break the
    trading cycle.
    """
    try:
        snap_date: date | None = None
        if as_of:
            try:
                snap_date = datetime.fromisoformat(str(as_of)[:19]).date()
            except Exception:
                try:
                    snap_date = datetime.fromisoformat(str(as_of)[:10]).date()
                except Exception:
                    snap_date = None

        payload = {
            "would_apply": dict(would_apply),
            "meta": dict(meta or {}),
            "as_of": as_of,
            "recorded_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        return write_shadow_snapshot(
            module,
            payload,
            snapshot_date=snap_date,
            shadow_root=root,
        )
    except Exception as exc:
        logger.debug("record_shadow(%s) swallowed exception: %s", module, exc)
        return None


def is_shadow_only(policy: Mapping[str, Any], module_key: str) -> bool:
    """Return True if ``policy.<module_key>.shadow_only`` is truthy.

    Default for new-Part-D modules: ``True`` (opt-out into live application).
    """
    cfg = (policy or {}).get(module_key, {}) or {}
    return bool(cfg.get("shadow_only", True))
