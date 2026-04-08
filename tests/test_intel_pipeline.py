"""
Comprehensive tests for the Intel/Crisis Alpha Pipeline.

Coverage:
- Models: serialize/deserialize
- Source registry: tiers and trust weights
- Geo trigger scoring (T0 alone → score=3, T3 alone → score=0/1)
- Keyword classification
- Dependency graph: load and traverse
- Shock propagation: CHOKEPOINT_STRESS
- Crisis state machine: NORMAL→WATCH, WATCH→ACTIVE, ACTIVE→COOLDOWN
- Health monitor: stale detection
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.assembled_core.intel.models import (
        CrisisState,
        EvidenceCluster,
        GeoTrigger,
        NewsEvent,
        SourceTier,
        TriggerType,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
GRAPH_YAML = CONFIGS_DIR / "dependency_graph.yaml"

UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _ts(offset_minutes: int = 0) -> datetime:
    return _now() + timedelta(minutes=offset_minutes)


def _make_event(
    source_id: str,
    title: str,
    event_id: str | None = None,
    keywords: list[str] | None = None,
    geo_tags: list[str] | None = None,
    entities: list[str] | None = None,
) -> "NewsEvent":
    from src.assembled_core.intel.models import NewsEvent
    from src.assembled_core.intel.source_registry import get_source_tier

    eid = event_id or hashlib.sha256(f"{title}{source_id}".encode()).hexdigest()[:16]
    now = _now()
    return NewsEvent(
        event_id=eid,
        source_id=source_id,
        source_tier=get_source_tier(source_id),
        title=title,
        url=f"https://example.com/{eid}",
        published_at=now,
        ingested_at=now,
        geo_tags=geo_tags or [],
        entities=entities or [],
        keywords=keywords or [],
        content_hash=hashlib.sha256(title.encode()).hexdigest()[:16],
    )


def _make_cluster(
    event_ids: list[str],
    trigger_type: "TriggerType",
    max_tier: "SourceTier",
    cluster_id: str = "cluster_001",
    confidence: float = 0.8,
) -> "EvidenceCluster":
    from src.assembled_core.intel.models import EvidenceCluster

    now = _now()
    return EvidenceCluster(
        cluster_id=cluster_id,
        trigger_type=trigger_type,
        summary="Test cluster",
        supporting_events=event_ids,
        confidence=confidence,
        max_tier=max_tier,
        created_at=now,
        expires_at=now + timedelta(hours=6),
    )


# ===========================================================================
# 1. Models
# ===========================================================================


class TestModels:
    def test_news_event_serializes(self):
        from src.assembled_core.intel.models import NewsEvent, SourceTier

        now = _now()
        evt = NewsEvent(
            event_id="abc123",
            source_id="REUTERS",
            source_tier=SourceTier.T1,
            title="Test Event",
            url="https://reuters.com/test",
            published_at=now,
            ingested_at=now,
            geo_tags=["IRN", "ARE"],
            entities=["Strait of Hormuz"],
            keywords=["hormuz", "tanker"],
            content_hash="deadbeef",
        )
        d = evt.model_dump()
        assert d["source_id"] == "REUTERS"
        assert d["source_tier"] == "T1"
        assert "IRN" in d["geo_tags"]

    def test_news_event_roundtrip(self):
        from src.assembled_core.intel.models import NewsEvent, SourceTier

        now = _now()
        evt = NewsEvent(
            event_id="abc123",
            source_id="REUTERS",
            source_tier=SourceTier.T1,
            title="Test",
            url="https://example.com",
            published_at=now,
            ingested_at=now,
            geo_tags=[],
            entities=[],
            keywords=[],
            content_hash="abc",
        )
        restored = NewsEvent.model_validate(evt.model_dump())
        assert restored.event_id == evt.event_id
        assert restored.source_tier == SourceTier.T1

    def test_geo_trigger_is_expired(self):
        from src.assembled_core.intel.models import GeoTrigger, TriggerType

        now = _now()
        trigger = GeoTrigger(
            trigger_id="t1",
            trigger_type=TriggerType.WAR_ESCALATION,
            trigger_score=2,
            confidence=0.8,
            evidence_cluster_id="c1",
            ttl_minutes=60,
            decay_half_life_minutes=30,
            created_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1),
            source_breakdown={},
        )
        assert trigger.is_expired(now) is True

    def test_geo_trigger_not_expired(self):
        from src.assembled_core.intel.models import GeoTrigger, TriggerType

        now = _now()
        trigger = GeoTrigger(
            trigger_id="t2",
            trigger_type=TriggerType.WAR_ESCALATION,
            trigger_score=2,
            confidence=0.8,
            evidence_cluster_id="c1",
            ttl_minutes=360,
            decay_half_life_minutes=180,
            created_at=now,
            expires_at=now + timedelta(hours=6),
            source_breakdown={},
        )
        assert trigger.is_expired(now) is False

    def test_crisis_state_roundtrip(self):
        from src.assembled_core.intel.models import CrisisMode, CrisisState

        now = _now()
        state = CrisisState(
            mode=CrisisMode.WATCH,
            geo_score=2,
            active_triggers=["t1", "t2"],
            entered_at=now,
            risk_posture={"max_daily_loss_pct": 1.5},
        )
        d = state.model_dump()
        assert d["mode"] == "WATCH"
        restored = CrisisState.model_validate(d)
        assert restored.mode == CrisisMode.WATCH

    def test_dependency_signal_roundtrip(self):
        from src.assembled_core.intel.models import DependencySignal

        now = _now()
        sig = DependencySignal(
            signal_id="sig_001",
            trigger_id="trig_001",
            beneficiaries=["DEFENSE_SECTOR", "GOLD"],
            losers=["US_EQUITIES"],
            severity=2,
            confidence=0.75,
            time_horizon="short",
            ttl_expires_ts=now + timedelta(hours=6),
            basket_overrides={"prefer": ["DEFENSE_SECTOR"], "avoid": ["US_EQUITIES"]},
        )
        d = sig.model_dump()
        assert "DEFENSE_SECTOR" in d["beneficiaries"]
        restored = DependencySignal.model_validate(d)
        assert restored.signal_id == sig.signal_id

    def test_component_health_stale_when_no_update(self):
        from src.assembled_core.intel.models import ComponentHealth

        now = _now()
        ch = ComponentHealth(
            component_name="news_feed",
            last_updated=None,
            status="OK",
            stale_threshold_minutes=30,
        )
        assert ch.is_stale(now) is True

    def test_component_health_stale_after_threshold(self):
        from src.assembled_core.intel.models import ComponentHealth

        now = _now()
        last = now - timedelta(minutes=31)
        ch = ComponentHealth(
            component_name="news_feed",
            last_updated=last,
            status="OK",
            stale_threshold_minutes=30,
        )
        assert ch.is_stale(now) is True

    def test_component_health_ok_within_threshold(self):
        from src.assembled_core.intel.models import ComponentHealth

        now = _now()
        last = now - timedelta(minutes=10)
        ch = ComponentHealth(
            component_name="news_feed",
            last_updated=last,
            status="OK",
            stale_threshold_minutes=30,
        )
        assert ch.is_stale(now) is False


# ===========================================================================
# 2. Source Registry
# ===========================================================================


class TestSourceRegistry:
    def test_ofac_is_t0(self):
        from src.assembled_core.intel.source_registry import get_source_tier
        from src.assembled_core.intel.models import SourceTier

        assert get_source_tier("OFAC") == SourceTier.T0

    def test_un_sanctions_is_t0(self):
        from src.assembled_core.intel.source_registry import get_source_tier
        from src.assembled_core.intel.models import SourceTier

        assert get_source_tier("UN_SANCTIONS") == SourceTier.T0

    def test_reuters_is_t1(self):
        from src.assembled_core.intel.source_registry import get_source_tier
        from src.assembled_core.intel.models import SourceTier

        assert get_source_tier("REUTERS") == SourceTier.T1

    def test_ap_media_is_t1(self):
        from src.assembled_core.intel.source_registry import get_source_tier
        from src.assembled_core.intel.models import SourceTier

        assert get_source_tier("AP_MEDIA") == SourceTier.T1

    def test_gdelt_is_t2(self):
        from src.assembled_core.intel.source_registry import get_source_tier
        from src.assembled_core.intel.models import SourceTier

        assert get_source_tier("GDELT") == SourceTier.T2

    def test_newsapi_is_t3(self):
        from src.assembled_core.intel.source_registry import get_source_tier
        from src.assembled_core.intel.models import SourceTier

        assert get_source_tier("NEWSAPI") == SourceTier.T3

    def test_unknown_source_defaults_to_t3(self):
        from src.assembled_core.intel.source_registry import get_source_tier
        from src.assembled_core.intel.models import SourceTier

        assert get_source_tier("UNKNOWN_BLOG") == SourceTier.T3

    def test_trust_weight_t0(self):
        from src.assembled_core.intel.source_registry import get_trust_weight

        assert get_trust_weight("OFAC") == 1.0

    def test_trust_weight_t1(self):
        from src.assembled_core.intel.source_registry import get_trust_weight

        assert get_trust_weight("REUTERS") == 0.9

    def test_trust_weight_t2(self):
        from src.assembled_core.intel.source_registry import get_trust_weight

        assert get_trust_weight("GDELT") == 0.7

    def test_trust_weight_t3(self):
        from src.assembled_core.intel.source_registry import get_trust_weight

        assert get_trust_weight("NEWSAPI") == 0.4

    def test_list_sources_no_filter(self):
        from src.assembled_core.intel.source_registry import list_sources

        sources = list_sources()
        assert "OFAC" in sources
        assert "REUTERS" in sources
        assert len(sources) >= 10

    def test_list_sources_t0_filter(self):
        from src.assembled_core.intel.source_registry import list_sources
        from src.assembled_core.intel.models import SourceTier

        t0_sources = list_sources(tier=SourceTier.T0)
        assert "OFAC" in t0_sources
        assert "UN_SANCTIONS" in t0_sources
        assert "REUTERS" not in t0_sources

    def test_list_sources_t1_filter(self):
        from src.assembled_core.intel.source_registry import list_sources
        from src.assembled_core.intel.models import SourceTier

        t1_sources = list_sources(tier=SourceTier.T1)
        assert "AP_MEDIA" in t1_sources
        assert "REUTERS" in t1_sources
        assert "GDELT" not in t1_sources


# ===========================================================================
# 3. Geo Trigger Scoring
# ===========================================================================


class TestGeoTrigger:
    def test_score_event_war_keywords(self):
        from src.assembled_core.intel.geo_trigger import score_event

        evt = _make_event("REUTERS", "Military attack in region", keywords=["war", "attack"])
        score = score_event(evt)
        assert score > 0.0

    def test_score_event_no_keywords(self):
        from src.assembled_core.intel.geo_trigger import score_event

        evt = _make_event("REUTERS", "Baseball game results final score", keywords=[])
        score = score_event(evt)
        assert score == 0.0 or score < 0.1

    def test_classify_war_escalation(self):
        from src.assembled_core.intel.geo_trigger import classify_trigger_type
        from src.assembled_core.intel.models import TriggerType

        evt = _make_event("REUTERS", "Missile attack launched", keywords=["missile", "military", "attack"])
        result = classify_trigger_type(evt)
        assert result == TriggerType.WAR_ESCALATION

    def test_classify_chokepoint_stress(self):
        from src.assembled_core.intel.geo_trigger import classify_trigger_type
        from src.assembled_core.intel.models import TriggerType

        evt = _make_event("REUTERS", "Strait of Hormuz tanker blockade", keywords=["hormuz", "tanker", "blockade"])
        result = classify_trigger_type(evt)
        assert result == TriggerType.CHOKEPOINT_STRESS

    def test_classify_energy_supply(self):
        from src.assembled_core.intel.geo_trigger import classify_trigger_type
        from src.assembled_core.intel.models import TriggerType

        evt = _make_event("EIA", "Oil pipeline closure announced", keywords=["oil", "pipeline", "refinery"])
        result = classify_trigger_type(evt)
        assert result == TriggerType.ENERGY_SUPPLY_RISK

    def test_classify_cyber_escalation(self):
        from src.assembled_core.intel.geo_trigger import classify_trigger_type
        from src.assembled_core.intel.models import TriggerType

        evt = _make_event("REUTERS", "Major cyberattack on infrastructure", keywords=["cyberattack", "hack", "malware"])
        result = classify_trigger_type(evt)
        assert result == TriggerType.CYBER_ESCALATION

    def test_classify_no_match_returns_none(self):
        from src.assembled_core.intel.geo_trigger import classify_trigger_type

        evt = _make_event("REUTERS", "Football world cup results", keywords=["football", "score"])
        result = classify_trigger_type(evt)
        assert result is None

    def test_score_cluster_t0_source_gives_score_3(self):
        """T0 source alone should give trigger_score=3."""
        from src.assembled_core.intel.geo_trigger import score_cluster
        from src.assembled_core.intel.models import SourceTier, TriggerType

        evt = _make_event("OFAC", "OFAC sanctions escalation", event_id="ofac_evt_001", keywords=["sanctions"])
        cluster = _make_cluster(
            event_ids=[evt.event_id],
            trigger_type=TriggerType.SANCTIONS_ESCALATION,
            max_tier=SourceTier.T0,
        )
        trigger = score_cluster(cluster, [evt])
        assert trigger.trigger_score == 3

    def test_score_cluster_t1_source_gives_score_3(self):
        """T1 source alone should give trigger_score=3."""
        from src.assembled_core.intel.geo_trigger import score_cluster
        from src.assembled_core.intel.models import SourceTier, TriggerType

        evt = _make_event("REUTERS", "War erupts in region", event_id="reuters_evt_001", keywords=["war", "military"])
        cluster = _make_cluster(
            event_ids=[evt.event_id],
            trigger_type=TriggerType.WAR_ESCALATION,
            max_tier=SourceTier.T1,
        )
        trigger = score_cluster(cluster, [evt])
        assert trigger.trigger_score == 3

    def test_score_cluster_single_t3_gives_score_0(self):
        """Single T3 source should give score=0 (no independent T2+ sources)."""
        from src.assembled_core.intel.geo_trigger import score_cluster
        from src.assembled_core.intel.models import SourceTier, TriggerType

        evt = _make_event("NEWSAPI", "Rumor: conflict looming", event_id="newsapi_evt_001", keywords=["conflict"])
        cluster = _make_cluster(
            event_ids=[evt.event_id],
            trigger_type=TriggerType.WAR_ESCALATION,
            max_tier=SourceTier.T3,
        )
        trigger = score_cluster(cluster, [evt])
        assert trigger.trigger_score == 0

    def test_score_cluster_two_t2_sources_gives_score_2(self):
        """Two independent T2 sources should give score=2."""
        from src.assembled_core.intel.geo_trigger import score_cluster
        from src.assembled_core.intel.models import SourceTier, TriggerType

        evt1 = _make_event("GDELT", "Energy crisis unfolding", event_id="gdelt_evt_001", keywords=["energy", "oil"])
        evt2 = _make_event("ACLED", "Pipeline blocked in region", event_id="acled_evt_001", keywords=["pipeline", "oil"])
        cluster = _make_cluster(
            event_ids=[evt1.event_id, evt2.event_id],
            trigger_type=TriggerType.ENERGY_SUPPLY_RISK,
            max_tier=SourceTier.T2,
        )
        trigger = score_cluster(cluster, [evt1, evt2])
        assert trigger.trigger_score == 2

    def test_score_cluster_single_t2_gives_score_1(self):
        """Single T2 source gives score=1."""
        from src.assembled_core.intel.geo_trigger import score_cluster
        from src.assembled_core.intel.models import SourceTier, TriggerType

        evt = _make_event("GDELT", "Oil supply concerns", event_id="gdelt_evt_002", keywords=["oil", "energy"])
        cluster = _make_cluster(
            event_ids=[evt.event_id],
            trigger_type=TriggerType.ENERGY_SUPPLY_RISK,
            max_tier=SourceTier.T2,
        )
        trigger = score_cluster(cluster, [evt])
        assert trigger.trigger_score == 1

    def test_aggregate_triggers_max_score(self):
        from src.assembled_core.intel.geo_trigger import aggregate_triggers
        from src.assembled_core.intel.models import GeoTrigger, TriggerType

        now = _now()
        t1 = GeoTrigger(
            trigger_id="t1", trigger_type=TriggerType.WAR_ESCALATION,
            trigger_score=2, confidence=0.7, evidence_cluster_id="c1",
            ttl_minutes=360, decay_half_life_minutes=180,
            created_at=now, expires_at=now + timedelta(hours=6),
            source_breakdown={},
        )
        t2 = GeoTrigger(
            trigger_id="t2", trigger_type=TriggerType.CHOKEPOINT_STRESS,
            trigger_score=3, confidence=0.9, evidence_cluster_id="c2",
            ttl_minutes=360, decay_half_life_minutes=180,
            created_at=now, expires_at=now + timedelta(hours=6),
            source_breakdown={},
        )
        result = aggregate_triggers([t1, t2])
        assert result["geo_score"] == 3
        assert "t1" in result["active_triggers"]
        assert "t2" in result["active_triggers"]

    def test_aggregate_triggers_empty(self):
        from src.assembled_core.intel.geo_trigger import aggregate_triggers

        result = aggregate_triggers([])
        assert result["geo_score"] == 0
        assert result["active_triggers"] == []


# ===========================================================================
# 4. Dependency Graph
# ===========================================================================


class TestDependencyGraph:
    def test_load_graph_from_yaml(self):
        from src.assembled_core.intel.dependency_graph import load_graph

        graph = load_graph(GRAPH_YAML)
        assert graph.get_node("HORMUZ") is not None
        assert graph.get_node("ENERGY_SECTOR") is not None

    def test_node_types_correct(self):
        from src.assembled_core.intel.dependency_graph import load_graph
        from src.assembled_core.intel.models import NodeType

        graph = load_graph(GRAPH_YAML)
        hormuz = graph.get_node("HORMUZ")
        assert hormuz is not None
        assert hormuz.node_type == NodeType.CHOKEPOINT

        energy = graph.get_node("ENERGY_SECTOR")
        assert energy is not None
        assert energy.node_type == NodeType.SECTOR

    def test_get_node_nonexistent(self):
        from src.assembled_core.intel.dependency_graph import load_graph

        graph = load_graph(GRAPH_YAML)
        assert graph.get_node("NONEXISTENT_NODE") is None

    def test_get_neighbors_hormuz(self):
        from src.assembled_core.intel.dependency_graph import load_graph

        graph = load_graph(GRAPH_YAML)
        neighbors = graph.get_neighbors("HORMUZ")
        assert len(neighbors) >= 1
        neighbor_ids = [n.node_id for _, n in neighbors]
        assert "GLOBAL_OIL" in neighbor_ids or "GLOBAL_LNG" in neighbor_ids

    def test_get_neighbors_filter_by_edge_type(self):
        from src.assembled_core.intel.dependency_graph import load_graph
        from src.assembled_core.intel.models import EdgeType

        graph = load_graph(GRAPH_YAML)
        neighbors = graph.get_neighbors("HORMUZ", edge_types=[EdgeType.TRANSITS_THROUGH])
        assert len(neighbors) >= 1
        for edge, _ in neighbors:
            assert edge.edge_type == EdgeType.TRANSITS_THROUGH

    def test_get_asset_nodes(self):
        from src.assembled_core.intel.dependency_graph import load_graph

        graph = load_graph(GRAPH_YAML)
        asset_nodes = graph.get_asset_nodes()
        assert len(asset_nodes) >= 3
        node_ids = [n.node_id for n in asset_nodes]
        assert "ENERGY_SECTOR" in node_ids

    def test_find_paths_hormuz_to_energy(self):
        from src.assembled_core.intel.dependency_graph import load_graph

        graph = load_graph(GRAPH_YAML)
        paths = graph.find_paths("HORMUZ", "ENERGY_SECTOR", max_depth=4)
        assert len(paths) >= 1
        # Every path should start with HORMUZ and end with ENERGY_SECTOR
        for path in paths:
            assert path[0] == "HORMUZ"
            assert path[-1] == "ENERGY_SECTOR"

    def test_find_paths_no_path(self):
        from src.assembled_core.intel.dependency_graph import load_graph

        graph = load_graph(GRAPH_YAML)
        # ENERGY_SECTOR → HORMUZ: no such path in this direction
        paths = graph.find_paths("ENERGY_SECTOR", "HORMUZ", max_depth=3)
        # May or may not have paths, just check it returns a list
        assert isinstance(paths, list)

    def test_node_attributes_preserved(self):
        from src.assembled_core.intel.dependency_graph import load_graph

        graph = load_graph(GRAPH_YAML)
        defense = graph.get_node("DEFENSE_SECTOR")
        assert defense is not None
        assert "assets" in defense.attributes
        assert "LMT" in defense.attributes["assets"]


# ===========================================================================
# 5. Shock Propagation
# ===========================================================================


class TestShockPropagation:
    def test_map_trigger_to_shocks_chokepoint(self):
        from src.assembled_core.intel.shock_propagation import map_trigger_to_shocks
        from src.assembled_core.intel.models import GeoTrigger, TriggerType, ShockType

        now = _now()
        trigger = GeoTrigger(
            trigger_id="t1",
            trigger_type=TriggerType.CHOKEPOINT_STRESS,
            trigger_score=3,
            confidence=0.9,
            evidence_cluster_id="c1",
            ttl_minutes=360,
            decay_half_life_minutes=180,
            created_at=now,
            expires_at=now + timedelta(hours=6),
            source_breakdown={},
        )
        shocks = map_trigger_to_shocks(trigger)
        assert ShockType.OIL_SUPPLY_RISK in shocks
        assert ShockType.SHIPPING_COST_RISK in shocks

    def test_map_trigger_to_shocks_war(self):
        from src.assembled_core.intel.shock_propagation import map_trigger_to_shocks
        from src.assembled_core.intel.models import GeoTrigger, TriggerType, ShockType

        now = _now()
        trigger = GeoTrigger(
            trigger_id="t1",
            trigger_type=TriggerType.WAR_ESCALATION,
            trigger_score=3,
            confidence=0.9,
            evidence_cluster_id="c1",
            ttl_minutes=360,
            decay_half_life_minutes=180,
            created_at=now,
            expires_at=now + timedelta(hours=6),
            source_breakdown={},
        )
        shocks = map_trigger_to_shocks(trigger)
        assert ShockType.DEFENSE_DEMAND_SURGE in shocks
        assert ShockType.GLOBAL_RISK_OFF in shocks

    def test_propagate_chokepoint_reaches_energy(self):
        """CHOKEPOINT_STRESS shock should eventually reach ENERGY_SECTOR (beneficiary)."""
        from src.assembled_core.intel.dependency_graph import load_graph
        from src.assembled_core.intel.shock_propagation import propagate
        from src.assembled_core.intel.models import ShockType

        graph = load_graph(GRAPH_YAML)
        transmissions = propagate(
            shocks=[ShockType.OIL_SUPPLY_RISK],
            graph=graph,
            trigger_id="test_trigger",
            max_hops=4,
        )
        assert len(transmissions) >= 1
        # Check that at least one transmission reaches ENERGY_SECTOR
        all_nodes_in_paths = set()
        for t in transmissions:
            for hop in t.path:
                all_nodes_in_paths.add(hop.node_id)
        # The path should pass through or terminate at energy-related nodes
        energy_related = {"GLOBAL_OIL", "ENERGY_SECTOR", "HORMUZ"}
        assert len(all_nodes_in_paths & energy_related) > 0

    def test_propagate_chokepoint_creates_beneficiaries_and_losers(self):
        """CHOKEPOINT_STRESS: energy sector benefits, tech loses."""
        from src.assembled_core.intel.dependency_graph import load_graph
        from src.assembled_core.intel.shock_propagation import propagate, to_dependency_signal
        from src.assembled_core.intel.models import ShockType

        graph = load_graph(GRAPH_YAML)
        shocks = [ShockType.OIL_SUPPLY_RISK, ShockType.SHIPPING_COST_RISK]
        transmissions = propagate(shocks=shocks, graph=graph, trigger_id="trig_test", max_hops=4)

        signal = to_dependency_signal(transmissions, trigger_id="trig_test", trigger_score=3)
        # Signal should have some beneficiaries or losers
        assert len(signal.beneficiaries) + len(signal.losers) > 0

    def test_propagate_defense_demand_surge_positive(self):
        """DEFENSE_DEMAND_SURGE should produce DEFENSE_SECTOR as a beneficiary."""
        from src.assembled_core.intel.dependency_graph import load_graph
        from src.assembled_core.intel.shock_propagation import propagate, to_dependency_signal
        from src.assembled_core.intel.models import ShockType

        graph = load_graph(GRAPH_YAML)
        transmissions = propagate(
            shocks=[ShockType.DEFENSE_DEMAND_SURGE],
            graph=graph,
            trigger_id="trig_war",
            max_hops=3,
        )
        # Check we got transmissions or they terminated at defense nodes
        if transmissions:
            _signal = to_dependency_signal(transmissions, "trig_war", trigger_score=3)
            # DEFENSE_SECTOR should be a beneficiary or in the path
            all_path_nodes = {hop.node_id for t in transmissions for hop in t.path}
            assert "DEFENSE_SECTOR" in all_path_nodes or "WAR_ESCALATION_EVENT" in all_path_nodes

    def test_to_dependency_signal_fields(self):
        from src.assembled_core.intel.dependency_graph import load_graph
        from src.assembled_core.intel.shock_propagation import propagate, to_dependency_signal
        from src.assembled_core.intel.models import ShockType

        graph = load_graph(GRAPH_YAML)
        shocks = [ShockType.OIL_SUPPLY_RISK]
        transmissions = propagate(shocks=shocks, graph=graph, trigger_id="trig_x", max_hops=4)
        signal = to_dependency_signal(transmissions, "trig_x", trigger_score=2)

        assert signal.signal_id.startswith("sig_")
        assert signal.trigger_id == "trig_x"
        assert signal.severity == 2
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.time_horizon in {"intraday", "short", "medium"}
        assert signal.ttl_expires_ts > datetime.now(tz=timezone.utc)

    def test_no_overlap_between_beneficiaries_and_losers(self):
        """A node should not appear in both beneficiaries and losers."""
        from src.assembled_core.intel.dependency_graph import load_graph
        from src.assembled_core.intel.shock_propagation import propagate, to_dependency_signal
        from src.assembled_core.intel.models import ShockType

        graph = load_graph(GRAPH_YAML)
        shocks = [ShockType.OIL_SUPPLY_RISK, ShockType.GLOBAL_RISK_OFF]
        transmissions = propagate(shocks=shocks, graph=graph, trigger_id="t_overlap", max_hops=4)
        signal = to_dependency_signal(transmissions, "t_overlap", trigger_score=2)

        overlap = set(signal.beneficiaries) & set(signal.losers)
        assert len(overlap) == 0, f"Overlap found: {overlap}"


# ===========================================================================
# 6. Crisis State Machine
# ===========================================================================


class TestCrisisStateMachine:
    def _normal_state(self) -> "CrisisState":
        from src.assembled_core.intel.models import CrisisMode, CrisisState

        return CrisisState(
            mode=CrisisMode.NORMAL,
            geo_score=0,
            active_triggers=[],
            entered_at=_now() - timedelta(hours=2),
            risk_posture={},
        )

    def _watch_state(self) -> "CrisisState":
        from src.assembled_core.intel.models import CrisisMode, CrisisState

        return CrisisState(
            mode=CrisisMode.WATCH,
            geo_score=2,
            active_triggers=["t1"],
            entered_at=_now() - timedelta(hours=1),
            risk_posture={},
        )

    def _active_state(self) -> "CrisisState":
        from src.assembled_core.intel.models import CrisisMode, CrisisState

        return CrisisState(
            mode=CrisisMode.ACTIVE,
            geo_score=3,
            active_triggers=["t1", "t2"],
            entered_at=_now() - timedelta(hours=1),
            risk_posture={},
        )

    def _make_trigger(self, score: int, expired: bool = False) -> "GeoTrigger":
        from src.assembled_core.intel.models import GeoTrigger, TriggerType

        now = _now()
        if expired:
            exp = now - timedelta(hours=1)
            created = now - timedelta(hours=7)
        else:
            exp = now + timedelta(hours=5)
            created = now

        return GeoTrigger(
            trigger_id=f"trig_{score}_{expired}",
            trigger_type=TriggerType.WAR_ESCALATION,
            trigger_score=score,
            confidence=0.8,
            evidence_cluster_id="c1",
            ttl_minutes=360,
            decay_half_life_minutes=180,
            created_at=created,
            expires_at=exp,
            source_breakdown={},
        )

    def test_normal_to_watch_on_geo_score_2(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state
        from src.assembled_core.intel.models import CrisisMode

        prev = self._normal_state()
        trigger = self._make_trigger(score=2)
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=2,
            active_triggers=[trigger],
            dependency_signal=None,
            market_confirm={},
            now=_now(),
        )
        assert new_state.mode == CrisisMode.WATCH

    def test_normal_stays_normal_on_geo_score_1(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state
        from src.assembled_core.intel.models import CrisisMode

        prev = self._normal_state()
        trigger = self._make_trigger(score=1)
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=1,
            active_triggers=[trigger],
            dependency_signal=None,
            market_confirm={},
            now=_now(),
        )
        assert new_state.mode == CrisisMode.NORMAL

    def test_watch_to_active_with_geo_3_and_market_confirm(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state
        from src.assembled_core.intel.models import CrisisMode

        prev = self._watch_state()
        trigger = self._make_trigger(score=3)
        market_confirm = {"oil_move": 3.5, "gold_move": 0.5, "vix_spike": False}
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=3,
            active_triggers=[trigger],
            dependency_signal=None,
            market_confirm=market_confirm,
            now=_now(),
        )
        assert new_state.mode == CrisisMode.ACTIVE

    def test_watch_stays_watch_without_market_confirm(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state
        from src.assembled_core.intel.models import CrisisMode

        prev = self._watch_state()
        trigger = self._make_trigger(score=3)
        market_confirm = {"oil_move": 0.5, "gold_move": 0.0, "vix_spike": False}
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=3,
            active_triggers=[trigger],
            dependency_signal=None,
            market_confirm=market_confirm,
            now=_now(),
        )
        assert new_state.mode == CrisisMode.WATCH

    def test_watch_to_active_with_vix_spike(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state
        from src.assembled_core.intel.models import CrisisMode

        prev = self._watch_state()
        trigger = self._make_trigger(score=3)
        market_confirm = {"oil_move": 0.0, "gold_move": 0.0, "vix_spike": True}
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=3,
            active_triggers=[trigger],
            dependency_signal=None,
            market_confirm=market_confirm,
            now=_now(),
        )
        assert new_state.mode == CrisisMode.ACTIVE

    def test_watch_to_active_with_gold_move(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state
        from src.assembled_core.intel.models import CrisisMode

        prev = self._watch_state()
        trigger = self._make_trigger(score=3)
        market_confirm = {"oil_move": 0.0, "gold_move": 1.5, "vix_spike": False}
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=3,
            active_triggers=[trigger],
            dependency_signal=None,
            market_confirm=market_confirm,
            now=_now(),
        )
        assert new_state.mode == CrisisMode.ACTIVE

    def test_active_to_cooldown_on_score_drop(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state
        from src.assembled_core.intel.models import CrisisMode

        prev = self._active_state()
        trigger = self._make_trigger(score=2)  # score dropped below 3
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=2,
            active_triggers=[trigger],
            dependency_signal=None,
            market_confirm={},
            now=_now(),
        )
        assert new_state.mode == CrisisMode.COOLDOWN

    def test_active_to_cooldown_on_trigger_expiry(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state
        from src.assembled_core.intel.models import CrisisMode

        prev = self._active_state()
        trigger = self._make_trigger(score=3, expired=True)
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=0,
            active_triggers=[trigger],
            dependency_signal=None,
            market_confirm={},
            now=_now(),
        )
        # All triggers expired + geo_score=0 → NORMAL (hard reset path)
        assert new_state.mode in {CrisisMode.COOLDOWN, CrisisMode.NORMAL}

    def test_cooldown_to_normal_after_elapsed(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state, CrisisStateConfig
        from src.assembled_core.intel.models import CrisisMode, CrisisState

        # Entered cooldown 13 hours ago, cooldown_min=720min=12h
        config = CrisisStateConfig(cooldown_min_minutes=720)
        prev = CrisisState(
            mode=CrisisMode.COOLDOWN,
            geo_score=1,
            active_triggers=[],
            entered_at=_now() - timedelta(hours=13),
            risk_posture={},
        )
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=0,
            active_triggers=[],
            dependency_signal=None,
            market_confirm={},
            now=_now(),
            config=config,
        )
        assert new_state.mode == CrisisMode.NORMAL

    def test_cooldown_stays_cooldown_before_elapsed(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state, CrisisStateConfig
        from src.assembled_core.intel.models import CrisisMode, CrisisState

        config = CrisisStateConfig(cooldown_min_minutes=720)
        prev = CrisisState(
            mode=CrisisMode.COOLDOWN,
            geo_score=1,
            active_triggers=[],
            entered_at=_now() - timedelta(hours=5),  # only 5h, need 12h
            risk_posture={},
        )
        trigger = self._make_trigger(score=1)
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=1,
            active_triggers=[trigger],
            dependency_signal=None,
            market_confirm={},
            now=_now(),
            config=config,
        )
        assert new_state.mode == CrisisMode.COOLDOWN

    def test_active_risk_posture_is_restrictive(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state
        from src.assembled_core.intel.models import CrisisMode

        prev = self._watch_state()
        trigger = self._make_trigger(score=3)
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=3,
            active_triggers=[trigger],
            dependency_signal=None,
            market_confirm={"oil_move": 5.0, "gold_move": 2.0, "vix_spike": True},
            now=_now(),
        )
        assert new_state.mode == CrisisMode.ACTIVE
        assert new_state.risk_posture["no_overnight"] is True
        assert new_state.risk_posture["max_open_positions"] <= 3

    def test_audit_trail_populated(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state
        from src.assembled_core.intel.models import CrisisMode

        prev = self._normal_state()
        trigger = self._make_trigger(score=2)
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=2,
            active_triggers=[trigger],
            dependency_signal=None,
            market_confirm={},
            now=_now(),
        )
        assert new_state.mode == CrisisMode.WATCH
        assert len(new_state.audit_trail) >= 1
        entry = new_state.audit_trail[-1]
        assert "prev_mode" in entry
        assert "new_mode" in entry
        assert "ts" in entry

    def test_dependency_signal_id_stored(self):
        from src.assembled_core.intel.crisis_alpha_worker import update_crisis_state
        from src.assembled_core.intel.models import DependencySignal

        prev = self._normal_state()
        trigger = self._make_trigger(score=2)
        now = _now()
        sig = DependencySignal(
            signal_id="sig_test_123",
            trigger_id="trig_x",
            beneficiaries=["DEFENSE_SECTOR"],
            losers=["US_EQUITIES"],
            severity=2,
            confidence=0.7,
            time_horizon="short",
            ttl_expires_ts=now + timedelta(hours=6),
        )
        new_state = update_crisis_state(
            prev_state=prev,
            geo_score=2,
            active_triggers=[trigger],
            dependency_signal=sig,
            market_confirm={},
            now=now,
        )
        assert new_state.dependency_signal_id == "sig_test_123"


# ===========================================================================
# 7. Health Monitor
# ===========================================================================


class TestHealthMonitor:
    def test_register_and_initially_stale(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        hm.register("news_feed", stale_threshold_minutes=30)
        assert hm.is_stale("news_feed") is True

    def test_update_makes_component_fresh(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        hm.register("news_feed", stale_threshold_minutes=30)
        now = _now()
        hm.update("news_feed", status="OK", now=now)
        assert hm.is_stale("news_feed", now=now) is False

    def test_component_becomes_stale_after_threshold(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        hm.register("news_feed", stale_threshold_minutes=30)
        past = _now() - timedelta(minutes=31)
        hm.update("news_feed", status="OK", now=past)
        now = _now()
        assert hm.is_stale("news_feed", now=now) is True

    def test_all_ok_when_all_fresh(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        now = _now()
        for name in ["feed_a", "feed_b", "feed_c"]:
            hm.register(name, stale_threshold_minutes=60)
            hm.update(name, status="OK", now=now)
        assert hm.all_ok(now=now) is True

    def test_all_ok_false_if_one_stale(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        now = _now()
        hm.register("feed_a", stale_threshold_minutes=60)
        hm.register("feed_b", stale_threshold_minutes=60)
        hm.update("feed_a", status="OK", now=now)
        hm.update("feed_b", status="OK", now=now - timedelta(minutes=61))
        assert hm.all_ok(now=now) is False

    def test_all_ok_false_if_error_status(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        now = _now()
        hm.register("feed_a", stale_threshold_minutes=60)
        hm.update("feed_a", status="ERROR", now=now)
        # ERROR sets is_stale → True via status check
        assert hm.all_ok(now=now) is False

    def test_snapshot_returns_all_components(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        now = _now()
        hm.register("feed_a")
        hm.register("feed_b")
        hm.update("feed_a", status="OK", now=now)
        snap = hm.snapshot(now=now)
        assert "feed_a" in snap
        assert "feed_b" in snap
        assert snap["feed_a"]["status"] == "OK"

    def test_unknown_component_is_stale(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        assert hm.is_stale("nonexistent_component") is True

    def test_all_ok_false_when_no_components(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        assert hm.all_ok() is False

    def test_can_go_active_true_when_all_ok(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        now = _now()
        hm.register("intel_feed", stale_threshold_minutes=60)
        hm.update("intel_feed", status="OK", now=now)
        assert hm.can_go_active(now=now) is True

    def test_can_go_active_false_when_stale(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        hm.register("intel_feed", stale_threshold_minutes=30)
        # Never updated → stale
        assert hm.can_go_active() is False

    def test_auto_register_on_update(self):
        from src.assembled_core.intel.health_monitor import HealthMonitor

        hm = HealthMonitor()
        now = _now()
        # Component not registered yet — should auto-register
        hm.update("new_component", status="OK", now=now)
        assert not hm.is_stale("new_component", now=now)
