"""Signal generation modules."""

from src.assembled_core.signals import regime as regime  # noqa: F401
from src.assembled_core.signals.multifactor_signal import (
    MultiFactorSignalResult,
    build_multifactor_signal,
    select_top_bottom,
)
from src.assembled_core.signals.composite_score import (  # noqa: F401
    composite_score,
    generate_composite_score_signals,
)
from src.assembled_core.signals.pairs_trading import (  # noqa: F401
    generate_pairs_signals,
    generate_pairs_signals_from_panel,
    cointegration_score,
)
from src.assembled_core.signals.buyback_drift import (  # noqa: F401
    detect_buyback_announcement,
    buyback_signal_score,
)
from src.assembled_core.signals.pead_sue import (  # noqa: F401
    compute_sue,
    batch_sue,
    pre_trade_earnings_check,
)
from src.assembled_core.signals.tail_risk_hedge import (  # noqa: F401
    TailHedgeConfig,
    TailHedgeOrder,
    tail_hedge_rules,
    should_buy_hedge,
    should_roll_hedge,
)
from src.assembled_core.signals.etf_flows import (  # noqa: F401
    compute_etf_flow,
    sector_rotation_signal,
    etf_flow_summary,
)
from src.assembled_core.signals.insider_cluster import (  # noqa: F401
    cluster_buy_score,
    net_officer_usd,
    insider_cluster_signal,
    batch_insider_signals,
)
from src.assembled_core.signals.cross_asset_carry import (  # noqa: F401
    equity_carry,
    bond_carry,
    cross_asset_carry_score,
    carry_exposure_multiplier,
)

__all__ = [
    "MultiFactorSignalResult",
    "build_multifactor_signal",
    "select_top_bottom",
    # Composite scoring
    "composite_score",
    "generate_composite_score_signals",
    # Pairs trading
    "generate_pairs_signals",
    "generate_pairs_signals_from_panel",
    "cointegration_score",
    # Alt-data signals
    "detect_buyback_announcement",
    "buyback_signal_score",
    "compute_sue",
    "batch_sue",
    "pre_trade_earnings_check",
    # Risk overlay
    "TailHedgeConfig",
    "TailHedgeOrder",
    "tail_hedge_rules",
    "should_buy_hedge",
    "should_roll_hedge",
    # Market structure / flow signals
    "compute_etf_flow",
    "sector_rotation_signal",
    "etf_flow_summary",
    # Insider cluster (alt-data)
    "cluster_buy_score",
    "net_officer_usd",
    "insider_cluster_signal",
    "batch_insider_signals",
    # Cross-asset carry
    "equity_carry",
    "bond_carry",
    "cross_asset_carry_score",
    "carry_exposure_multiplier",
]
