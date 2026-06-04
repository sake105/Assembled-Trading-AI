"""Seasonal and Calendar Effect Features.

Implements well-documented calendar anomalies as trading signals:
    - Turn-of-Month effect (day -1 to +3)
    - January effect (small-cap outperformance)
    - Sell-in-May / Halloween indicator
    - Pre-holiday rally effect
    - Russell Reconstitution window (June)
    - S&P quarterly rebalancing (Mar/Jun/Sep/Dec)
    - Day-of-week effect
    - Month-of-year seasonality

All features are computed from date alone — no look-ahead bias possible.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _is_turn_of_month(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return 1 if date is in turn-of-month window (last day to +3 of next month)."""
    day = dates.day
    days_in_month = dates.days_in_month
    # Last trading day of month or first 3 days of next month
    result: np.ndarray = ((day >= days_in_month - 1) | (day <= 3)).astype(float)
    return result


def _is_january(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return 1 if date is in January (January effect)."""
    result: np.ndarray = (dates.month == 1).astype(float)
    return result


def _is_sell_in_may_period(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return 1 if date is in May-October (Sell in May period).

    The 'Halloween Indicator' suggests that Nov-Apr outperforms May-Oct.
    Signal: -1 during May-Oct, +1 during Nov-Apr.
    """
    month = dates.month
    return np.where((month >= 5) & (month <= 10), -1.0, 1.0)


def _is_pre_holiday(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return 1 if date is the trading day before a market holiday.

    Uses NYSE calendar if available, otherwise approximates with
    known US holidays.
    """
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=dates.min(), end=dates.max())
        # Day before holiday
        pre_holiday_dates = holidays - pd.Timedelta(days=1)
        result = np.isin(dates.normalize(), pre_holiday_dates.normalize()).astype(float)
    except Exception:
        result = np.zeros(len(dates))
    return result


def _russell_reconstitution_window(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return 1 if date is in Russell reconstitution window (June 15-30)."""
    result: np.ndarray = ((dates.month == 6) & (dates.day >= 15)).astype(float)
    return result


def _sp_rebalancing_window(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return 1 if date is near S&P quarterly rebalancing (3rd Friday of Mar/Jun/Sep/Dec)."""
    month = dates.month
    day = dates.day
    is_rebal_month = np.isin(month, [3, 6, 9, 12])
    # Approximate: 3rd week of month
    is_rebal_week = (day >= 14) & (day <= 21)
    result: np.ndarray = (is_rebal_month & is_rebal_week).astype(float)
    return result


def _day_of_week_signal(dates: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-week effect: Monday tends negative, Friday tends positive.

    Returns value in [-1, 1].
    """
    dow = dates.dayofweek  # 0=Mon, 4=Fri
    # Monday effect: -0.5, Tuesday-Thursday: 0, Friday: +0.5
    mapping = {0: -0.5, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.5}
    return np.array([mapping.get(d, 0.0) for d in dow])


def _month_of_year_signal(dates: pd.DatetimeIndex) -> np.ndarray:
    """Month-of-year seasonality score.

    Based on historical monthly return averages:
    Strong: Jan, Apr, Jul, Nov, Dec
    Weak: Feb, May, Jun, Sep
    """
    month_scores = {
        1: 0.6,
        2: -0.2,
        3: 0.1,
        4: 0.5,
        5: -0.3,
        6: -0.2,
        7: 0.3,
        8: 0.0,
        9: -0.5,
        10: 0.1,
        11: 0.5,
        12: 0.6,
    }
    return np.array([month_scores.get(m, 0.0) for m in dates.month])


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_seasonal_features(
    dates: pd.DatetimeIndex | pd.Series,
) -> pd.DataFrame:
    """Build all seasonal features from a date index.

    Args:
        dates: DatetimeIndex or Series of timestamps.

    Returns:
        DataFrame with seasonal feature columns, indexed by the input dates.
    """
    if isinstance(dates, pd.Series):
        idx = pd.DatetimeIndex(dates)
    else:
        idx = dates

    result = pd.DataFrame(index=idx)
    result["seasonal_turn_of_month"] = _is_turn_of_month(idx)
    result["seasonal_january"] = _is_january(idx)
    result["seasonal_sell_in_may"] = _is_sell_in_may_period(idx)
    result["seasonal_pre_holiday"] = _is_pre_holiday(idx)
    result["seasonal_russell_recon"] = _russell_reconstitution_window(idx)
    result["seasonal_sp_rebal"] = _sp_rebalancing_window(idx)
    result["seasonal_day_of_week"] = _day_of_week_signal(idx)
    result["seasonal_month_score"] = _month_of_year_signal(idx)

    logger.info(
        "[Seasonal] Built %d features for %d dates",
        len(get_seasonal_feature_names()),
        len(idx),
    )
    return result


def get_seasonal_feature_names() -> list[str]:
    """Return list of seasonal feature column names."""
    return [
        "seasonal_turn_of_month",
        "seasonal_january",
        "seasonal_sell_in_may",
        "seasonal_pre_holiday",
        "seasonal_russell_recon",
        "seasonal_sp_rebal",
        "seasonal_day_of_week",
        "seasonal_month_score",
    ]
