"""Technical analysis features and feature engineering modules.

This package handles:
- Technical indicators (EMA, SMA, RSI, MACD, etc.)
- Feature engineering for machine learning
- Feature normalization and scaling
- Feature selection
- Core TA/Price factors (Phase A, Sprint A1)
- Liquidity & Volatility factors (Phase A, Sprint A2)
- Market Breadth & Risk-On/Risk-Off indicators (Phase A, Sprint A3)

Note: Current EMA logic is in pipeline.signals.compute_ema_signals.
This package will provide a broader set of technical indicators.
"""

# Core TA/Price Factors (Phase A, Sprint A1)
from src.assembled_core.features.ta_factors_core import build_core_ta_factors

# Liquidity & Volatility Factors (Phase A, Sprint A2)
from src.assembled_core.features.ta_liquidity_vol_factors import (
    add_realized_volatility,
    add_turnover_and_liquidity_proxies,
    add_vol_of_vol,
)

# Market Breadth & Risk-On/Risk-Off Indicators (Phase A, Sprint A3)
from src.assembled_core.features.market_breadth import (
    compute_advance_decline_line,
    compute_market_breadth_ma,
    compute_risk_on_off_indicator,
)

# Alt-Data Factors: Earnings & Insider (Phase B1)
from src.assembled_core.features.altdata_earnings_insider_factors import (
    build_earnings_surprise_factors,
    build_insider_activity_factors,
)
from src.assembled_core.features.altdata_news_macro_factors import (
    build_macro_regime_factors,
    build_news_sentiment_factors,
)

__all__ = [
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
]

