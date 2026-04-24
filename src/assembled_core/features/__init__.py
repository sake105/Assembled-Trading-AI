"""Technical analysis features and feature engineering modules.

This package handles:
- Technical indicators (EMA, SMA, RSI, MACD, etc.)
- Feature engineering for machine learning
- Feature normalization and scaling
- Feature selection
- Core TA/Price factors (Phase A, Sprint A1)
- Liquidity & Volatility factors (Phase A, Sprint A2)
- Market Breadth & Risk-On/Risk-Off indicators (Phase A, Sprint A3)
- Factor store integration (build_or_load_factors for caching)
- Behavioral, buyback, cross-asset, index-rebalance, institutional,
  interaction, intraday, seasonal, short-interest and VPIN features
  (wired 2026-04-22 from previously orphan modules).

Note: Current EMA logic is in pipeline.signals.compute_ema_signals.
This package will provide a broader set of technical indicators.
"""

from src.assembled_core.features.factor_store_integration import build_or_load_factors
from src.assembled_core.features.ta_factors_core import build_core_ta_factors
from src.assembled_core.features.ta_liquidity_vol_factors import (
    add_realized_volatility,
    add_turnover_and_liquidity_proxies,
    add_vol_of_vol,
)
from src.assembled_core.features.market_breadth import (
    compute_advance_decline_line,
    compute_market_breadth_ma,
    compute_risk_on_off_indicator,
)
from src.assembled_core.features.altdata_earnings_insider_factors import (
    build_earnings_surprise_factors,
    build_insider_activity_factors,
)
from src.assembled_core.features.altdata_news_macro_factors import (
    build_macro_regime_factors,
    build_news_sentiment_factors,
)

from src.assembled_core.features.behavioral_features import (
    abnormal_turnover,
    abnormal_volume,
    anchoring_52w_high,
    capital_gains_overhang,
    compute_behavioral_composite,
    max_effect,
    round_number_proximity,
)
from src.assembled_core.features.buyback_features import (
    build_buyback_features,
    compute_buyback_completion_rate,
    compute_buyback_yield,
    detect_buyback_from_shares,
    post_buyback_drift,
)
from src.assembled_core.features.cross_asset_leads import (
    CrossAssetSignal,
    build_cross_asset_signals,
    compute_bond_equity_signal,
    compute_commodity_sector_signal,
    compute_fx_adr_signal,
)
from src.assembled_core.features.index_rebal_features import (
    build_index_rebal_features,
    compute_predicted_demand,
    get_index_rebal_feature_names,
)
from src.assembled_core.features.institutional_features import (
    InstitutionalSignal,
    build_institutional_features,
    compute_institutional_ownership,
    compute_ownership_changes,
    compute_smart_money_flow,
)
from src.assembled_core.features.intraday_features import (
    IntradayFeatureResult,
    build_intraday_features,
    compute_intraday_volatility_ratio,
    compute_last_hour_momentum,
    compute_opening_range_breakout,
    compute_overnight_return,
    compute_vwap_deviation,
)
from src.assembled_core.features.seasonal_features import (
    build_seasonal_features,
    get_seasonal_feature_names,
)
from src.assembled_core.features.short_interest_features import (
    build_short_interest_features,
    compute_short_pct_float,
    compute_short_ratio,
    compute_short_squeeze_score,
    get_short_interest_feature_names,
)
from src.assembled_core.features.vpin import (
    VPINResult,
    classify_volume_bulk,
    compute_vpin,
    compute_vpin_panel,
)
from src.assembled_core.features.correlation_features import (  # noqa: F401
    build_correlation_features_panel,
    compute_avg_pairwise_correlation,
    compute_correlation_regime_features,
    compute_correlation_to_benchmark,
    compute_return_dispersion,
    compute_sector_dispersion,
)
from src.assembled_core.features.cross_sectional import (  # noqa: F401
    neutralize_cross_sectional,
    rank_cross_sectional,
    zscore_cross_sectional,
)
from src.assembled_core.features.fractional_diff import (  # noqa: F401
    adf_stationarity_test,
    apply_ffd_to_panel,
    find_optimal_d,
    frac_diff_ffd,
    frac_diff_weights,
)
from src.assembled_core.features.geopolitical_features import (  # noqa: F401
    compute_gpr_from_fred,
    compute_gpr_proxy,
)
from src.assembled_core.features.incremental_updates import (  # noqa: F401
    compute_last_N_sessions,
    compute_only_last_session,
    filter_prices_for_incremental,
)
from src.assembled_core.features.registry import (  # noqa: F401
    get_feature_metadata,
    list_all_feature_names,
    list_features_by_namespace,
    validate_registry_documented,
    validate_registry_namespaced,
    validate_registry_unique,
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
    "capital_gains_overhang",
    "anchoring_52w_high",
    "round_number_proximity",
    "abnormal_volume",
    "abnormal_turnover",
    "max_effect",
    "compute_behavioral_composite",
    "detect_buyback_from_shares",
    "compute_buyback_yield",
    "compute_buyback_completion_rate",
    "post_buyback_drift",
    "build_buyback_features",
    "CrossAssetSignal",
    "compute_bond_equity_signal",
    "compute_commodity_sector_signal",
    "compute_fx_adr_signal",
    "build_cross_asset_signals",
    "compute_predicted_demand",
    "build_index_rebal_features",
    "get_index_rebal_feature_names",
    "InstitutionalSignal",
    "compute_institutional_ownership",
    "compute_ownership_changes",
    "compute_smart_money_flow",
    "build_institutional_features",
    "IntradayFeatureResult",
    "compute_last_hour_momentum",
    "compute_overnight_return",
    "compute_opening_range_breakout",
    "compute_vwap_deviation",
    "compute_intraday_volatility_ratio",
    "build_intraday_features",
    "build_seasonal_features",
    "get_seasonal_feature_names",
    "compute_short_pct_float",
    "compute_short_ratio",
    "compute_short_squeeze_score",
    "build_short_interest_features",
    "get_short_interest_feature_names",
    "VPINResult",
    "classify_volume_bulk",
    "compute_vpin",
    "compute_vpin_panel",
]
