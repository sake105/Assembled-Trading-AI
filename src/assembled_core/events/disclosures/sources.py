"""Load disclosures source registry and pipeline config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore[import]
except Exception:
    yaml = None


@dataclass
class DisclosureSource:
    """Configuration for a single disclosure source."""

    source_id: str
    name: str
    domain: str
    type: str  # house_ptr | edgar
    tier: str
    weight: float
    active: bool
    config: Dict[str, Any]


def load_sources_registry(config_path: str | Path) -> List[DisclosureSource]:
    """Load disclosures sources from YAML (configs/disclosures/sources.yaml)."""
    path = Path(config_path)
    if not path.exists() or yaml is None:
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[disclosures] Failed to load sources config %s: %s", path, exc
        )
        return []

    sources_cfg = data.get("sources") or []
    sources: List[DisclosureSource] = []
    for entry in sources_cfg:
        if not isinstance(entry, dict):
            continue
        source_id = str(entry.get("source_id") or "").strip()
        name = str(entry.get("name") or "").strip()
        domain = str(entry.get("domain") or "").strip().lower()
        src_type = str(entry.get("type") or "").strip().lower()
        if not source_id or not name or not domain or not src_type:
            continue
        tier = str(entry.get("tier") or "B").strip().upper()
        weight = float(entry.get("weight", 1.0))
        active = bool(entry.get("active", True))
        config = {
            k: v
            for k, v in entry.items()
            if k
            not in {
                "source_id",
                "name",
                "domain",
                "type",
                "tier",
                "weight",
                "active",
                "notes",
            }
        }
        sources.append(
            DisclosureSource(
                source_id=source_id,
                name=name,
                domain=domain,
                type=src_type,
                tier=tier,
                weight=weight,
                active=active,
                config=config,
            )
        )
    return sources


def load_disclosures_params(config_path: str | Path) -> Dict[str, Any]:
    """Load pipeline params from configs/disclosures/disclosures.yaml."""
    path = Path(config_path)
    if not path.exists() or yaml is None:
        data = {}
    else:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}

    if not isinstance(data, dict):
        data = {}

    fetch = data.get("fetch") or {}
    if not isinstance(fetch, dict):
        fetch = {}
    fetch.setdefault("timeout_s", 15.0)
    fetch.setdefault("retries", 2)
    fetch.setdefault("user_agent", "Assembled-Trading-AI/Disclosures-v1")

    health = data.get("health") or {}
    if not isinstance(health, dict):
        health = {}
    health.setdefault("min_sources_ok", 1)

    dedupe = data.get("dedupe") or {}
    if not isinstance(dedupe, dict):
        dedupe = {}
    dedupe.setdefault("enabled", True)
    dedupe.setdefault("window_days", 30)

    edgar = data.get("edgar") or {}
    if not isinstance(edgar, dict):
        edgar = {}
    form4 = edgar.get("form4") or {}
    if not isinstance(form4, dict):
        form4 = {}
    form4.setdefault(
        "feed_url",
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&count=100&output=atom",
    )
    form4.setdefault(
        "user_agent",
        str(fetch.get("user_agent", "Assembled-Trading-AI/Disclosures-v1")),
    )
    form4.setdefault("cache_minutes", 10)
    form4.setdefault("stale_on_error_minutes", 60)
    form4.setdefault("timeout_s", float(fetch.get("timeout_s", 15.0)))
    edgar["form4"] = form4
    edgar.setdefault("enabled", True)

    house_ptr = data.get("house_ptr") or {}
    if not isinstance(house_ptr, dict):
        house_ptr = {}
    house_ptr.setdefault("enabled", True)
    house_ptr.setdefault("index_url", "https://<HOUSE_PTR_RSS_OR_INDEX>")
    house_ptr.setdefault(
        "user_agent",
        str(fetch.get("user_agent", "Assembled-Trading-AI/Disclosures-v1")),
    )
    house_ptr.setdefault("cache_minutes", 60)
    house_ptr.setdefault("stale_on_error_minutes", 240)
    house_ptr.setdefault("download_pdfs", False)
    house_ptr.setdefault("download_dir", "output/intel/disclosures/raw/house_ptr")
    house_ptr.setdefault("max_items", 50)
    house_ptr.setdefault("timeout_s", float(fetch.get("timeout_s", 15.0)))
    pdf_meta = house_ptr.get("pdf_meta") or {}
    if not isinstance(pdf_meta, dict):
        pdf_meta = {}
    pdf_meta.setdefault("enabled", True)
    pdf_meta.setdefault("compute_sha256", True)
    pdf_meta.setdefault("max_mb", 25)
    house_ptr["pdf_meta"] = pdf_meta

    trigger_scoring = data.get("trigger_scoring") or {}
    if not isinstance(trigger_scoring, dict):
        trigger_scoring = {}
    trigger_scoring.setdefault("enabled", True)
    sev = trigger_scoring.get("severity") or {}
    if not isinstance(sev, dict):
        sev = {}
    sev.setdefault("base_by_action", {"FORM4_FILED": 1, "HOUSE_PTR_FILED": 1})
    sev.setdefault("max", 3)
    trigger_scoring["severity"] = sev
    conf = trigger_scoring.get("confidence") or {}
    if not isinstance(conf, dict):
        conf = {}
    conf.setdefault("tierA_alone", 0.85)
    conf.setdefault("tierB_two_domains", 0.70)
    conf.setdefault("otherwise", 0.40)
    trigger_scoring["confidence"] = conf
    gating = trigger_scoring.get("gating") or {}
    if not isinstance(gating, dict):
        gating = {}
    gating.setdefault("require_evidence_ok", True)
    trigger_scoring["gating"] = gating
    ttl = trigger_scoring.get("ttl") or {}
    if not isinstance(ttl, dict):
        ttl = {}
    ttl.setdefault("default_hours", 168)
    ttl.setdefault("by_action", {"FORM4_FILED": 96, "HOUSE_PTR_FILED": 168})
    trigger_scoring["ttl"] = ttl
    decay = trigger_scoring.get("decay") or {}
    if not isinstance(decay, dict):
        decay = {}
    decay.setdefault("half_life_hours", 72)
    decay.setdefault("min_confidence_floor", 0.25)
    decay.setdefault("severity_floor", 0)
    trigger_scoring["decay"] = decay
    qc = trigger_scoring.get("qc_gates") or {}
    if not isinstance(qc, dict):
        qc = {}
    qc.setdefault("degraded_max_severity", 1)
    qc.setdefault("error_max_severity", 0)
    trigger_scoring["qc_gates"] = qc

    return {
        "fetch": fetch,
        "health": health,
        "dedupe": dedupe,
        "edgar": edgar,
        "house_ptr": house_ptr,
        "trigger_scoring": trigger_scoring,
        "cadence": data.get("cadence") or "hourly",
    }
