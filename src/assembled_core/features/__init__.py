"""Technical analysis features and feature engineering modules."""

from src.assembled_core.features.altdata_earnings_insider_factors import (
    build_earnings_surprise_factors,
    build_insider_activity_factors,
)
from src.assembled_core.features.altdata_news_macro_factors import (
    build_macro_regime_factors,
    build_news_sentiment_factors,
)
from src.assembled_core.features.correlation_features import (  # noqa: F401
    build_correlation_features_panel,
    compute_avg_pairwise_correlation,
    compute_correlation_regime_features,
    compute_correlation_to_benchmark,
    compute_return_dispersion,
    compute_sector_dispersion,
)
from src.assembled_core.features.factor_store_integration import build_or_load_factors
from src.assembled_core.features.geopolitical_features import (  # noqa: F401
    compute_gpr_from_fred,
    compute_gpr_proxy,
)
from src.assembled_core.features.incremental_updates import (  # noqa: F401
    compute_last_N_sessions,
    compute_only_last_session,
    filter_prices_for_incremental,
)
from src.assembled_core.features.index_rebal_features import (
    build_index_rebal_features,
    compute_predicted_demand,
    get_index_rebal_feature_names,
)
from src.assembled_core.features.market_breadth import (
    compute_advance_decline_line,
    compute_market_breadth_ma,
    compute_risk_on_off_indicator,
)
from src.assembled_core.features.supply_chain_features import (  # noqa: F401
    build_supply_chain_features,
    compute_chokepoint_exposure,
    compute_network_centrality,
    compute_sanctions_vulnerability,
    compute_single_source_dependency,
    compute_supply_chain_depth,
    propagate_returns_through_chain,
)
from src.assembled_core.features.ta_factors_core import build_core_ta_factors
from src.assembled_core.features.ta_liquidity_vol_factors import (
    add_realized_volatility,
    add_turnover_and_liquidity_proxies,
    add_vol_of_vol,
)
from src.assembled_core.features.triple_barrier import (  # noqa: F401
    cusum_filter,
    triple_barrier_labels,
    fractional_diff,
    meta_label,
    compute_sample_weights,
)
from src.assembled_core.features.change_point_detection import (  # noqa: F401
    ChangePointResult,
    detect_change_points_pelt,
    detect_change_points_binseg,
    change_point_regime_feature,
    recent_break_flag,
)
from src.assembled_core.features.residual_momentum import (  # noqa: F401
    compute_residual_momentum,
    cross_sectional_residual_momentum,
)

__all__ = [
    "build_or_load_factors",
    "build_core_ta_factors",
    "add_realized_volatility",
    "add_vol_of_vol",
    "add_turnover_and_liquidity_proxies",
    "compute_market_breadth_ma",
    "compute_advance_decline_line",
    "compute_risk_on_off_indicator",
    "build_earnings_surprise_factors",
    "build_insider_activity_factors",
    "build_news_sentiment_factors",
    "build_macro_regime_factors",
    "compute_predicted_demand",
    "build_index_rebal_features",
    "get_index_rebal_feature_names",
    # Lopez de Prado labeling
    "cusum_filter",
    "triple_barrier_labels",
    "fractional_diff",
    "meta_label",
    "compute_sample_weights",
    # Regime / structural break detection
    "ChangePointResult",
    "detect_change_points_pelt",
    "detect_change_points_binseg",
    "change_point_regime_feature",
    "recent_break_flag",
    # Residual momentum
    "compute_residual_momentum",
    "cross_sectional_residual_momentum",
]
