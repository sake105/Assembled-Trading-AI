"""Pydantic v2 models for the Intel/Crisis Alpha Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceTier(str, Enum):
    T0 = "T0"  # Government sanctions lists (OFAC, UN)
    T1 = "T1"  # Licensed newswires (AP, Reuters)
    T2 = "T2"  # Open/aggregator (GDELT, ACLED, World Bank)
    T3 = "T3"  # Scrapes/social (restricted)


class TriggerType(str, Enum):
    CHOKEPOINT_STRESS = "CHOKEPOINT_STRESS"
    WAR_ESCALATION = "WAR_ESCALATION"
    SANCTIONS_ESCALATION = "SANCTIONS_ESCALATION"
    ENERGY_SUPPLY_RISK = "ENERGY_SUPPLY_RISK"
    SHIPPING_DISRUPTION = "SHIPPING_DISRUPTION"
    CYBER_ESCALATION = "CYBER_ESCALATION"
    COUP_RISK = "COUP_RISK"
    POLICY_SHIFT = "POLICY_SHIFT"


class CrisisMode(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"


class NodeType(str, Enum):
    COUNTRY = "country"
    REGION = "region"
    CHOKEPOINT = "chokepoint"
    COMMODITY = "commodity"
    SECTOR = "sector"
    COMPANY = "company"
    ASSET = "asset"
    MACRO_INDEX = "macro_index"


class EdgeType(str, Enum):
    IMPORTS_FROM = "IMPORTS_FROM"
    EXPORTS_TO = "EXPORTS_TO"
    TRANSITS_THROUGH = "TRANSITS_THROUGH"
    DEPENDS_ON_ENERGY = "DEPENDS_ON_ENERGY"
    SENSITIVE_TO_PRICE = "SENSITIVE_TO_PRICE"
    SANCTIONS_EXPOSED = "SANCTIONS_EXPOSED"
    LOGISTICS_ROUTE = "LOGISTICS_ROUTE"
    SAFE_HAVEN_FOR = "SAFE_HAVEN_FOR"
    BENEFITS_FROM = "BENEFITS_FROM"
    HURT_BY = "HURT_BY"


class ShockType(str, Enum):
    OIL_SUPPLY_RISK = "oil_supply_risk"
    SHIPPING_COST_RISK = "shipping_cost_risk"
    INSURANCE_COST_RISK = "insurance_cost_risk"
    GLOBAL_RISK_OFF = "global_risk_off"
    ENERGY_PRICE_SPIKE = "energy_price_spike"
    DEFENSE_DEMAND_SURGE = "defense_demand_surge"
    SANCTIONS_EXPOSURE = "sanctions_exposure"
    CYBER_RISK = "cyber_risk"


# ---------------------------------------------------------------------------
# Core Pydantic models
# ---------------------------------------------------------------------------


class NewsEvent(BaseModel):
    event_id: str  # sha256 hash of (title+source+published_at)
    source_id: str
    source_tier: SourceTier
    title: str
    url: str
    published_at: datetime
    ingested_at: datetime
    geo_tags: list[str] = Field(default_factory=list)  # country/region codes
    entities: list[str] = Field(default_factory=list)  # named entities
    keywords: list[str] = Field(default_factory=list)  # extracted keywords
    content_hash: str  # for deduplication

    model_config = {"frozen": False}


class EvidenceCluster(BaseModel):
    cluster_id: str
    trigger_type: TriggerType
    summary: str
    supporting_events: list[str] = Field(default_factory=list)  # event_ids
    confidence: float  # 0-1
    max_tier: SourceTier
    created_at: datetime
    expires_at: datetime

    model_config = {"frozen": False}


class GeoTrigger(BaseModel):
    trigger_id: str
    trigger_type: TriggerType
    trigger_score: int  # 0-3
    confidence: float
    evidence_cluster_id: str
    ttl_minutes: int
    decay_half_life_minutes: int
    created_at: datetime
    expires_at: datetime
    source_breakdown: dict[str, int] = Field(default_factory=dict)  # tier -> count

    model_config = {"frozen": False}

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


class DependencyNode(BaseModel):
    node_id: str
    node_type: NodeType
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class DependencyEdge(BaseModel):
    from_node: str
    to_node: str
    edge_type: EdgeType
    weight: float  # 0-1
    direction: str  # "+" or "-"
    lag_hours: int  # expected response lag
    confidence: float
    source_refs: list[str] = Field(default_factory=list)

    model_config = {"frozen": False}


class TransmissionHop(BaseModel):
    node_id: str
    node_type: str
    impact_direction: str  # "+" or "-"
    weight: float

    model_config = {"frozen": False}


class ShockTransmission(BaseModel):
    shock_id: str
    origin_trigger_id: str
    shock_type: ShockType
    path: list[TransmissionHop] = Field(default_factory=list)
    expected_impact_direction: str
    expected_horizon: str  # "intraday", "short" (1-3d), "medium" (1-4w)
    path_confidence: float

    model_config = {"frozen": False}


class DependencySignal(BaseModel):
    signal_id: str
    trigger_id: str
    beneficiaries: list[str] = Field(default_factory=list)  # asset symbols
    losers: list[str] = Field(default_factory=list)  # asset symbols
    severity: int  # 0-3
    confidence: float
    time_horizon: str
    ttl_expires_ts: datetime
    basket_overrides: dict[str, list[str]] = Field(default_factory=dict)

    model_config = {"frozen": False}


class CrisisState(BaseModel):
    mode: CrisisMode
    geo_score: int  # 0-3
    active_triggers: list[str] = Field(default_factory=list)  # trigger_ids
    dependency_signal_id: str | None = None
    entered_at: datetime
    risk_posture: dict[str, Any] = Field(default_factory=dict)
    basket_overrides: dict[str, list[str]] = Field(default_factory=dict)
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# Dataclass helpers (lightweight internal structs)
# ---------------------------------------------------------------------------


@dataclass
class ComponentHealth:
    component_name: str
    last_updated: datetime | None
    status: str  # "OK", "STALE", "ERROR"
    stale_threshold_minutes: int = 60

    def is_stale(self, now: datetime) -> bool:
        if self.last_updated is None:
            return True
        elapsed = (now - self.last_updated).total_seconds() / 60
        return elapsed > self.stale_threshold_minutes or self.status != "OK"
