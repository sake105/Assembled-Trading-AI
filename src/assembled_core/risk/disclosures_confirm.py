"""Disclosures confirmation boost for NEWS geo_confidence (DISCL-4.2).

When disclosures triggers have severity >= min_severity, boost ctx.news_geo.geo_confidence
by add_confidence (capped at max_confidence). No severity or trade changes; transparent via
ctx.news_geo["boost"] and reasons artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from src.assembled_core.pipeline.trading_cycle import TradingContext


def apply_disclosures_confirm(ctx: "TradingContext", policy: Dict[str, Any]) -> None:
    """Apply disclosures-based confidence boost to ctx.news_geo if policy enabled and conditions met.

    Rules:
    1) If disabled or ctx.news_geo is None -> no-op.
    2) If disclosures triggers missing or degraded (intel_health_flags) -> no boost.
    3) max_discl_sev from ctx.disclosures_triggers.summary or triggers.
    4) If max_discl_sev >= min_severity: new_conf = min(max_confidence, old_conf + add_confidence);
       set ctx.news_geo["geo_confidence"] and ctx.news_geo["boost"] = {source, added, max_discl_sev}.
    """
    cfg = (policy or {}).get("disclosures_confirm") or {}
    if not cfg.get("enabled", False):
        return

    news_geo = getattr(ctx, "news_geo", None)
    if news_geo is None:
        return

    # QC: no boost if disclosures triggers missing or degraded
    intel_flags = getattr(ctx, "intel_health_flags", None) or {}
    if intel_flags.get("intel_disclosures_triggers") == "DEGRADED":
        return

    disc_triggers = getattr(ctx, "disclosures_triggers", None)
    if disc_triggers is None:
        return

    # max_discl_sev from summary or from triggers
    max_discl_sev = 0
    if hasattr(disc_triggers, "summary") and isinstance(getattr(disc_triggers, "summary"), dict):
        max_discl_sev = int((disc_triggers.summary or {}).get("max_severity", 0))
    if max_discl_sev == 0 and hasattr(disc_triggers, "triggers"):
        triggers_list = getattr(disc_triggers, "triggers") or []
        for t in triggers_list:
            if isinstance(t, dict):
                max_discl_sev = max(max_discl_sev, int(t.get("severity", 0)))

    boost_cfg = cfg.get("boost") or {}
    min_severity = int(boost_cfg.get("min_severity", 1))
    add_confidence = float(boost_cfg.get("add_confidence", 0.10))
    max_confidence = float(boost_cfg.get("max_confidence", 0.95))

    if max_discl_sev < min_severity:
        return

    # Ensure news_geo is mutable dict (support both dict and object with .get)
    if isinstance(news_geo, dict):
        old_conf = float(news_geo.get("geo_confidence", 0.0))
        new_conf = min(max_confidence, old_conf + add_confidence)
        news_geo["geo_confidence"] = new_conf
        news_geo["boost"] = {
            "source": "disclosures",
            "added": add_confidence,
            "max_discl_sev": max_discl_sev,
        }
    else:
        old_conf = float(getattr(news_geo, "geo_confidence", 0.0) or 0.0)
        new_conf = min(max_confidence, old_conf + add_confidence)
        if hasattr(news_geo, "__setitem__"):
            try:
                news_geo["geo_confidence"] = new_conf
                news_geo["boost"] = {
                    "source": "disclosures",
                    "added": add_confidence,
                    "max_discl_sev": max_discl_sev,
                }
            except Exception:
                pass
        else:
            try:
                setattr(news_geo, "geo_confidence", new_conf)
                setattr(
                    news_geo,
                    "boost",
                    {"source": "disclosures", "added": add_confidence, "max_discl_sev": max_discl_sev},
                )
            except Exception:
                pass


__all__ = ["apply_disclosures_confirm"]
