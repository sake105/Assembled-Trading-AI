from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DISCLOSURES_PATH = Path("output/disclosures_latest.json")


def apply_disclosures_confirm(ctx: object, policy: dict) -> None:
    """Apply disclosures confirmation overlay to ctx.

    Two paths:
    1. If ctx.disclosures_triggers is set (DisclosuresTriggerSnapshot), read
       max_severity from its summary and apply boost to ctx.news_geo.
    2. Otherwise fall back to reading output/disclosures_latest.json.

    Boost config (policy.disclosures_confirm.boost):
        min_severity: int  — minimum severity to trigger boost (default 1)
        add_confidence: float — amount to add to news_geo["geo_confidence"]
        max_confidence: float — ceiling for geo_confidence after boost
    """
    try:
        cfg = (policy.get("disclosures_confirm") or {})
        if not cfg.get("enabled", False):
            return

        boost_cfg = cfg.get("boost") or {}
        min_sev = int(boost_cfg.get("min_severity", 1))
        add_conf = float(boost_cfg.get("add_confidence", 0.0))
        max_conf = float(boost_cfg.get("max_confidence", 1.0))

        # Degraded intel health blocks all paths
        _degraded = (getattr(ctx, "intel_health_flags", {}) or {}).get(
            "intel_disclosures_triggers"
        )
        if _degraded == "DEGRADED":
            logger.debug("[SKIP] disclosures_confirm: intel_health DEGRADED")
            return

        max_severity: int | None = None

        # Path 1: use ctx.disclosures_triggers if available
        dt = getattr(ctx, "disclosures_triggers", None)
        if dt is not None:
            summary = getattr(dt, "summary", {}) or {}
            max_severity = int(summary.get("max_severity", 0))
        else:
            # Path 2: read from file

            if not _DISCLOSURES_PATH.exists():
                logger.debug("[SKIP] disclosures_confirm: %s not found", _DISCLOSURES_PATH)
                return
            try:
                with open(_DISCLOSURES_PATH, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as load_exc:
                logger.debug("[WARN] disclosures_confirm: load failed: %s", load_exc)
                return

            universe = getattr(ctx, "universe", []) or []
            universe_set = set(universe)
            disclosures = data if isinstance(data, list) else data.get("disclosures", [])
            for item in disclosures:
                try:
                    sym = item.get("symbol") or item.get("ticker", "")
                    sev = int(item.get("severity", 0))
                    if sev >= min_sev and (not universe_set or sym in universe_set):
                        max_severity = max(max_severity or 0, sev)
                except Exception:
                    continue

        if max_severity is None or max_severity < min_sev:
            return

        # Apply boost to ctx.news_geo if present
        news_geo = getattr(ctx, "news_geo", None)
        if isinstance(news_geo, dict) and add_conf > 0.0:
            old_conf = float(news_geo.get("geo_confidence", 0.0))
            new_conf = min(old_conf + add_conf, max_conf)
            news_geo["geo_confidence"] = round(new_conf, 10)
            news_geo["boost"] = {
                "source": "disclosures",
                "added": add_conf,
                "max_discl_sev": max_severity,
            }
            logger.debug(
                "[OK] disclosures_confirm: geo_confidence %.2f → %.2f (sev=%d)",
                old_conf, new_conf, max_severity,
            )

        ctx.disclosures_confirmed = True  # type: ignore[attr-defined]
        logger.debug("[OK] disclosures_confirm: confirmed=True")

    except Exception as exc:
        logger.debug("[ERROR] apply_disclosures_confirm: %s", exc)
