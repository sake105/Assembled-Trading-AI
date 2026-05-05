"""Intel loaders and geopolitical/macro intel modules."""

from __future__ import annotations

from .disclosures_triggers_loader import (
    DisclosuresTriggerSnapshot,
    load_disclosures_triggers,
)
from .news_triggers_loader import (
    NewsTriggerSnapshot,
    load_news_triggers,
)
from .news_alerts import AlertEngine, NewsAlert  # noqa: F401
from .news_archiver import NewsArchiver  # noqa: F401
from .news_entity_graph import EntityCoGraph, EntityStat  # noqa: F401
from .news_macro_calendar import MacroCalendar, MacroEvent, Proximity  # noqa: F401
from .news_semantic_dedup import SemanticDedup  # noqa: F401
from .news_sentiment_drift import DriftEntry, SentimentDriftTracker  # noqa: F401
from .news_ticker_velocity import TickerSignal, TickerVelocityTracker  # noqa: F401
from .news_velocity import VelocityResult, VelocityTracker  # noqa: F401
from .sector_news_overlay import SectorNewsOverlay  # noqa: F401
from .trigger_basket import (  # noqa: F401
    TriggerBasket,
    build_trigger_basket,
    compute_basket_score,
)
from .conviction_engine import (  # noqa: F401
    compute_conviction_score,
    compute_event_beta,
    compute_edcl_position_size,
)
from .tail_hunting import (  # noqa: F401
    TailHuntSignal,
    load_tail_plans,
    match_tail_plans,
    tail_signals_to_targets,
)
from .geo_event_logger import (  # noqa: F401
    log_basket_event,
    read_geo_event_log,
)

__all__ = [
    "DisclosuresTriggerSnapshot",
    "load_disclosures_triggers",
    "NewsTriggerSnapshot",
    "load_news_triggers",
    # EDCL modules
    "TriggerBasket",
    "build_trigger_basket",
    "compute_basket_score",
    "compute_conviction_score",
    "compute_event_beta",
    "compute_edcl_position_size",
    "TailHuntSignal",
    "load_tail_plans",
    "match_tail_plans",
    "tail_signals_to_targets",
    "log_basket_event",
    "read_geo_event_log",
]
