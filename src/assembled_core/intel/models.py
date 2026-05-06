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
    # --- Original 8 ---
    CHOKEPOINT_STRESS = "CHOKEPOINT_STRESS"
    WAR_ESCALATION = "WAR_ESCALATION"
    SANCTIONS_ESCALATION = "SANCTIONS_ESCALATION"
    ENERGY_SUPPLY_RISK = "ENERGY_SUPPLY_RISK"
    SHIPPING_DISRUPTION = "SHIPPING_DISRUPTION"
    CYBER_ESCALATION = "CYBER_ESCALATION"
    COUP_RISK = "COUP_RISK"
    POLICY_SHIFT = "POLICY_SHIFT"
    # --- Geopolitik ---
    TRADE_WAR_ESCALATION = "TRADE_WAR_ESCALATION"
    ALLIANCE_SHIFT = "ALLIANCE_SHIFT"
    RESOURCE_NATIONALIZATION = "RESOURCE_NATIONALIZATION"
    STRAIT_BLOCKADE = "STRAIT_BLOCKADE"
    HEGEMONIC_CHALLENGE = "HEGEMONIC_CHALLENGE"
    DIPLOMATIC_CRISIS = "DIPLOMATIC_CRISIS"
    PROXY_WAR_EXPANSION = "PROXY_WAR_EXPANSION"
    # --- Finanzen ---
    PEG_STRESS = "PEG_STRESS"
    RESERVE_DRAIN = "RESERVE_DRAIN"
    RATE_SURPRISE = "RATE_SURPRISE"
    FISCAL_CLIFF = "FISCAL_CLIFF"
    CREDIT_DOWNGRADE = "CREDIT_DOWNGRADE"
    BANKING_CRISIS = "BANKING_CRISIS"
    # --- Militaer ---
    MILITARY_BUILDUP = "MILITARY_BUILDUP"
    CAPABILITY_SHIFT = "CAPABILITY_SHIFT"
    CASUALTY_SPIKE = "CASUALTY_SPIKE"
    TERRITORIAL_ESCALATION = "TERRITORIAL_ESCALATION"
    NUCLEAR_THREAT = "NUCLEAR_THREAT"
    # --- Technologie ---
    NEW_EXPORT_CONTROL = "NEW_EXPORT_CONTROL"
    ENTITY_LISTING = "ENTITY_LISTING"
    TECHNOLOGY_GAP_WIDENING = "TECHNOLOGY_GAP_WIDENING"
    # --- Cyber ---
    ZERO_DAY_DISCLOSURE = "ZERO_DAY_DISCLOSURE"
    MAJOR_BREACH_DETECTED = "MAJOR_BREACH_DETECTED"
    STATE_ACTOR_ACTIVITY = "STATE_ACTOR_ACTIVITY"
    # --- Klima ---
    SEVERE_WEATHER_ALERT = "SEVERE_WEATHER_ALERT"
    SUPPLY_CHAIN_BREAK = "SUPPLY_CHAIN_BREAK"
    LOGISTICS_DISRUPTION = "LOGISTICS_DISRUPTION"


class CrisisMode(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"


class NodeType(str, Enum):
    # --- Original 8 ---
    COUNTRY = "country"
    REGION = "region"
    CHOKEPOINT = "chokepoint"
    COMMODITY = "commodity"
    SECTOR = "sector"
    COMPANY = "company"
    ASSET = "asset"
    MACRO_INDEX = "macro_index"
    # --- M15 Erweiterung ---
    SUPPLY_CHAIN = "supply_chain"
    ALLIANCE = "alliance"
    SHIPPING_LANE = "shipping_lane"
    CURRENCY = "currency"
    CENTRAL_BANK = "central_bank"
    SOVEREIGN = "sovereign"
    MILITARY_FORCE = "military_force"
    CONFLICT_ZONE = "conflict_zone"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    SANCTIONING_AUTHORITY = "sanctioning_authority"
    ENTITY = "entity"
    TECHNOLOGY_REGIME = "technology_regime"
    WEATHER_ZONE = "weather_zone"
    NATURAL_HAZARD = "natural_hazard"
    FISCAL_METRIC = "fiscal_metric"
    TRADE_BLOC = "trade_bloc"
    REFINERY = "refinery"


class EdgeType(str, Enum):
    # --- Original 10 ---
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
    # --- Geopolitik & Allianzen ---
    ALLY_OF = "ALLY_OF"
    RIVAL_OF = "RIVAL_OF"
    SANCTIONS_TARGET = "SANCTIONS_TARGET"
    SECONDARY_SANCTIONS = "SECONDARY_SANCTIONS"
    SANCTIONED_BY = "SANCTIONED_BY"
    # --- Ressourcen & Lieferketten ---
    TRANSIT_DEPENDENT = "TRANSIT_DEPENDENT"
    TECH_DEPENDENT = "TECH_DEPENDENT"
    RESOURCE_MONOPOLY = "RESOURCE_MONOPOLY"
    REFINES_FOR = "REFINES_FOR"
    MANUFACTURES_FOR = "MANUFACTURES_FOR"
    IMPORTS_RESOURCE = "IMPORTS_RESOURCE"
    EXPORTS_RESOURCE = "EXPORTS_RESOURCE"
    SUPPLIES_DUAL_USE = "SUPPLIES_DUAL_USE"
    SUBSTITUTABLE_BY = "SUBSTITUTABLE_BY"
    RESTRICTED_TO = "RESTRICTED_TO"
    # --- Waehrung & Finanzen ---
    CURRENCY_PEGGED = "CURRENCY_PEGGED"
    RESERVES_IN = "RESERVES_IN"
    CAPITAL_FLOW_TO = "CAPITAL_FLOW_TO"
    TRADE_IMBALANCE = "TRADE_IMBALANCE"
    INVESTS_IN = "INVESTS_IN"
    SETS_POLICY_FOR = "SETS_POLICY_FOR"
    DIVERGES_FROM = "DIVERGES_FROM"
    # --- Militaer & Konflikt ---
    SUPPLIES_MILITARY = "SUPPLIES_MILITARY"
    BACKED_BY = "BACKED_BY"
    ATTACKED_BY = "ATTACKED_BY"
    OCCUPIES = "OCCUPIES"
    # --- Klima & Infrastruktur ---
    VULNERABLE_TO_WEATHER = "VULNERABLE_TO_WEATHER"
    DEPENDENT_ON_LOGISTICS = "DEPENDENT_ON_LOGISTICS"
    ENABLES_TRANSACTION = "ENABLES_TRANSACTION"


class ShockType(str, Enum):
    # --- Original 8 ---
    OIL_SUPPLY_RISK = "oil_supply_risk"
    SHIPPING_COST_RISK = "shipping_cost_risk"
    INSURANCE_COST_RISK = "insurance_cost_risk"
    GLOBAL_RISK_OFF = "global_risk_off"
    ENERGY_PRICE_SPIKE = "energy_price_spike"
    DEFENSE_DEMAND_SURGE = "defense_demand_surge"
    SANCTIONS_EXPOSURE = "sanctions_exposure"
    CYBER_RISK = "cyber_risk"
    # --- Ressourcen ---
    RARE_EARTH_SUPPLY_RISK = "rare_earth_supply_risk"
    SEMICONDUCTOR_SUPPLY_RISK = "semiconductor_supply_risk"
    FOOD_SUPPLY_RISK = "food_supply_risk"
    LITHIUM_SUPPLY_RISK = "lithium_supply_risk"
    URANIUM_SUPPLY_RISK = "uranium_supply_risk"
    LNG_SUPPLY_RISK = "lng_supply_risk"
    COPPER_SUPPLY_RISK = "copper_supply_risk"
    NEON_GAS_RISK = "neon_gas_risk"
    # --- Waehrung & Fiskal ---
    CURRENCY_CRISIS = "currency_crisis"
    RESERVE_DEPLETION = "reserve_depletion"
    CAPITAL_FLIGHT = "capital_flight"
    FISCAL_SHOCK = "fiscal_shock"
    INFLATION_SPIKE = "inflation_spike"
    RATE_SHOCK = "rate_shock"
    POLICY_DIVERGENCE = "policy_divergence"
    TAPER_SHOCK = "taper_shock"
    SOVEREIGN_DEFAULT = "sovereign_default"
    # --- Geopolitik ---
    ALLIANCE_SHIFT = "alliance_shift"
    HEGEMONIC_DECOUPLING = "hegemonic_decoupling"
    SHIPPING_LANE_DISRUPTION = "shipping_lane_disruption"
    SECONDARY_SANCTIONS_RISK = "secondary_sanctions_risk"
    DELISTING_RISK = "delisting_risk"
    BANKING_ISOLATION = "banking_isolation"
    # --- Militaer ---
    MILITARY_LOSS_SURGE = "military_loss_surge"
    SUPPLY_LINE_THREAT = "supply_line_threat"
    WEAPONS_TECH_ADVANTAGE = "weapons_tech_advantage"
    NUCLEAR_ESCALATION_RISK = "nuclear_escalation_risk"
    # --- Cyber ---
    DATA_BREACH_SYSTEMIC = "data_breach_systemic"
    LOGISTICS_VISIBILITY_LOSS = "logistics_visibility_loss"
    FINANCIAL_SYSTEM_STRESS = "financial_system_stress"
    # --- Klima ---
    CLIMATE_DISRUPTION = "climate_disruption"
    PORT_CLOSURE = "port_closure"
    CROP_FAILURE = "crop_failure"
    SUPPLY_CHAIN_BREAK = "supply_chain_break"
    # --- Technologie ---
    TECH_RESTRICTION_SHOCK = "tech_restriction_shock"
    CHIP_SHORTAGE = "chip_shortage"
    INNOVATION_GAP = "innovation_gap"


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
    urgency: float = 0.0  # 0-1; 1.0 = Breaking/Flash/Urgent
    tickers: list[str] = Field(default_factory=list)  # resolved equity tickers
    sentiment_score: float = 0.0  # -1.0 (very negative) to +1.0 (very positive)
    # --- Classification fields (Batch 2) ---
    event_types: list[str] = Field(
        default_factory=list
    )  # multi-label: ["sanctions", "military_strike"]
    severity: float = 0.0  # 0-10 impact severity
    market_direction: str = "neutral"  # "bearish"/"bullish"/"neutral"/"mixed"
    time_horizon: str = "short"  # "intraday"/"short"/"medium"/"long"
    affected_sectors: list[str] = Field(default_factory=list)
    affected_assets: list[str] = Field(
        default_factory=list
    )  # ETFs/tickers derived from classification
    news_confidence: float = 0.0  # overall classification confidence
    language: str = "en"  # detected language code
    is_noise: bool = False  # true if classified as non-relevant noise
    # --- Impact-estimator fields (persisted so archive replay retains them) ---
    impact_bps: float = 0.0  # estimated return impact in basis points (neg=bearish)
    impact_horizon_days: int = 0  # expected days to mean-reversion
    impact_confidence: float = 0.0  # confidence of the BPS estimate [0,1]
    impact_geo_premium_bps: float = 0.0  # additional geo-risk component
    impact_dominant_event_type: str = ""  # most impactful event type
    # Decay bookkeeping (K3): fraction of original impact still active at enrich time
    impact_decay_fraction: float = 1.0  # 1.0 = fresh; 0.0 = fully decayed
    impact_decay_minutes: float = 0.0  # minutes between publish and enrichment
    # --- Cross-source corroboration (higher = more independent confirmations) ---
    corroboration_score: float = 0.0  # [0,1]; 1.0 = fully corroborated
    corroboration_n_sources: int = 0  # number of distinct sources reporting this story
    # --- Taxonomy category (Wave 2) ---
    category: str = "SONSTIGE"  # FINANZEN/KONFLIKTE/GEOPOLITIK/ROHSTOFFE/TECHNOLOGIE/POLITIK/SONSTIGE

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
    # M15 extended fields
    sentiment_score: float = 0.0  # FinBERT integration (-1 to +1)
    escalation_level: int = 0  # 0-10 escalation ladder

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
    # M15 extended fields
    resource_profile: dict[str, float] = Field(default_factory=dict)
    maritime_dependency_pct: float = 0.0
    fiscal_health: dict[str, float] = Field(default_factory=dict)
    military_capability_index: float = 0.0
    cyber_resilience_score: float = 0.5
    sanctions_exposure_score: float = 0.0
    tech_self_sufficiency: dict[str, float] = Field(default_factory=dict)

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
    # M15 continuous magnitude
    magnitude: float = 1.0  # initial shock strength
    dampened_magnitude: float = 1.0  # after per-hop dampening
    # M16: estimated days until impact materializes
    time_to_impact_days: float = 0.0

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


# ---------------------------------------------------------------------------
# M15 extended dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EscalationLevel:
    """Single rung on a military/geopolitical escalation ladder (0-10)."""

    level: int
    name: str
    description: str
    market_impact_multiplier: float = 1.0
    expected_duration_days: int = 30


@dataclass
class ConflictState:
    """Tracks an active geopolitical conflict."""

    conflict_id: str
    parties: list[str] = field(default_factory=list)
    current_level: int = 0  # 0-10 on the escalation ladder
    escalation_probability: dict[int, float] = field(default_factory=dict)
    affected_commodities: list[str] = field(default_factory=list)
    affected_sectors: list[str] = field(default_factory=list)


@dataclass
class SanctionPackage:
    """Models a sanctions package issued against a target."""

    package_id: str
    issuer: str  # OFAC, EU, UN, UK
    target_nation: str
    target_entities: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)  # finance, energy, tech, military
    severity: int = 1  # 1-5
    secondary_sanctions: bool = False
    swift_exclusion: bool = False
    affected_sectors: list[str] = field(default_factory=list)
    estimated_gdp_impact_pct: float = 0.0
    evasion_difficulty: float = 0.5  # 0=easy to evade, 1=impossible


@dataclass
class CurrencyProfile:
    """Currency crisis indicator for a sovereign currency."""

    currency: str
    nation: str
    reserve_months_import: float = 6.0
    real_interest_rate: float = 0.0
    inflation_rate: float = 2.0
    current_account_gdp_pct: float = 0.0
    peg_type: str = "free_float"  # free_float, managed_float, managed_band, hard_peg
    dollarization_pct: float = 0.0
    crisis_probability_12m: float = 0.05


@dataclass
class ShippingLane:
    """A major maritime shipping lane with chokepoint dependencies."""

    lane_id: str
    name: str
    chokepoints: list[str] = field(default_factory=list)
    daily_traffic_value_bn: float = 0.0
    oil_transit_mbpd: float = 0.0
    lng_transit_bcm_year: float = 0.0
    container_teu_year: float = 0.0
    alternative_route: str | None = None
    reroute_cost_multiplier: float = 1.0
    reroute_time_days: float = 0.0
    insurance_surge_pct: float = 0.0
    nations_dependent: list[str] = field(default_factory=list)


@dataclass
class NationProfile:
    """Comprehensive resource/vulnerability profile for a nation."""

    nation_id: str
    name: str
    imports: dict[str, float] = field(default_factory=dict)
    exports: dict[str, float] = field(default_factory=dict)
    transit_dependencies: dict[str, float] = field(default_factory=dict)
    fiscal: dict[str, float] = field(default_factory=dict)
    military: dict[str, Any] = field(default_factory=dict)
    tech_sovereignty: dict[str, float] = field(default_factory=dict)
    vulnerabilities: dict[str, float] = field(default_factory=dict)
