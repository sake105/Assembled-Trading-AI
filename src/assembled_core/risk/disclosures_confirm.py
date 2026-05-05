from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DISCLOSURES_PATH = Path("output/disclosures_latest.json")


def apply_disclosures_confirm(ctx: object, policy: dict) -> None:
    """Apply disclosures confirmation overlay to ctx based on policy config."""
    try:
        cfg = (policy.get("disclosures_confirm") or {})
        if not cfg.get("enabled", False):
            return

        if not _DISCLOSURES_PATH.exists():
            logger.debug("[SKIP] disclosures_confirm: %s not found", _DISCLOSURES_PATH)
            return

        try:
            with open(_DISCLOSURES_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as load_exc:
            logger.debug("[WARN] disclosures_confirm: failed to load file: %s", load_exc)
            return

        universe = getattr(ctx, "universe", []) or []
        universe_set = set(universe)

        disclosures = data if isinstance(data, list) else data.get("disclosures", [])
        confirmed = False
        for item in disclosures:
            try:
                sym = item.get("symbol") or item.get("ticker", "")
                severity = int(item.get("severity", 0))
                if severity >= 1 and (not universe_set or sym in universe_set):
                    confirmed = True
                    break
            except Exception:
                continue

        if confirmed:
            ctx.disclosures_confirmed = True
            logger.debug("[OK] disclosures_confirm: confirmed=True for ctx")

    except Exception as exc:
        logger.debug("[ERROR] apply_disclosures_confirm: %s", exc)
