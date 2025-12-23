"""Central constants for the trading system.

This module contains all magic numbers and default values used throughout the system.
This ensures consistency and makes it easy to adjust parameters in one place.
"""

# Trading calendar constants
TRADING_DAYS_PER_YEAR = 252  # Trading days per year
PERIODS_PER_DAY_5MIN = 78  # Five-minute periods per trading day (9:30-16:00 ET)
PERIODS_PER_YEAR_5MIN = TRADING_DAYS_PER_YEAR * PERIODS_PER_DAY_5MIN  # ~19,656 periods

# Default TA (Technical Analysis) feature parameters
DEFAULT_ATR_WINDOW = 14  # Average True Range window
DEFAULT_RSI_WINDOW = 14  # Relative Strength Index window
DEFAULT_MA_WINDOWS = (20, 50)  # Default moving average windows (fast, slow)

# Default capital constants
DEFAULT_START_CAPITAL = 10000.0  # Default starting capital for backtests
DEFAULT_SEED_CAPITAL = 100000.0  # Default seed capital for paper trading

# Default cost model parameters
DEFAULT_COMMISSION_BPS = 0.5  # Default commission in basis points
DEFAULT_SPREAD_W = 0.25  # Default spread weight (25% of spread)
DEFAULT_IMPACT_W = 0.5  # Default market impact weight (50% of impact)

# API limits (DoS protection)
MAX_ORDERS_PER_RESPONSE = 10000  # Maximum number of orders in API response

# Paper track constants
PAPER_TRACK_STATE_VERSION = "2.0"  # Current state format version

