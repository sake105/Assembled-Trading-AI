"""Extended Regime Analysis for Performance Evaluation (B3).

This module provides extended regime-based analysis functions beyond the
basic regime detection in `regime_models.py`. It focuses on:

1. Simplified regime classification from index returns
2. Extended metrics summarization by regime (including trade-level metrics)
3. Regime transition analysis

See [Walk-Forward and Regime B3 Design](docs/WALK_FORWARD_AND_REGIME_B3_DESIGN.md)
for detailed design and usage examples.

Example:
    from src.assembled_core.risk.regime_analysis import (
        classify_regimes_from_index,
        summarize_metrics_by_regime,
        compute_regime_transitions,
    )

    # Classify regimes from index returns
    regimes = classify_regimes_from_index(
        index_returns=sp500_returns,
        config=RegimeConfig(),
    )

    # Summarize metrics by regime
    metrics_by_regime = summarize_metrics_by_regime(
        equity=equity_curve["equity"],
        regimes=regimes,
        trades=trades_df,
    )

    # Analyze regime transitions
    transitions = compute_regime_transitions(regime_state_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """Configuration for simplified regime classification from index returns.

    Attributes:
        vol_window: Window for realized volatility calculation (default: 20)
        trend_ma_window: Moving average window for trend detection (default: 200)
        drawdown_threshold: Drawdown threshold for crisis detection (default: -0.20, i.e., -20%)
        vol_threshold_high: High volatility threshold (default: 0.30, i.e., 30% annualized)
        vol_threshold_low: Low volatility threshold (default: 0.15, i.e., 15% annualized)
        trend_threshold: Trend strength threshold (default: 0.05, i.e., 5% above/below MA)
        lookback_days_vol: Alias for vol_window (for consistency, default: None, uses vol_window)
        lookback_days_return: Alias for trend_ma_window (default: None, uses trend_ma_window)
        min_periods: Minimum number of periods required for classification (default: 20)
        bench_symbol: Optional benchmark symbol (not used in current implementation, default: None)
    """

    vol_window: int = 20
    trend_ma_window: int = 200
    drawdown_threshold: float = -0.20  # -20% for crisis
    vol_threshold_high: float = 0.30  # 30% annualized volatility
    vol_threshold_low: float = 0.15  # 15% annualized volatility
    trend_threshold: float = 0.05  # 5% above/below MA for trend
    lookback_days_vol: int | None = None  # Alias for vol_window
    lookback_days_return: int | None = None  # Alias for trend_ma_window
    min_periods: int = 20
    bench_symbol: str | None = None


def classify_regimes_from_index(
    index_returns: pd.Series,
    config: RegimeConfig | None = None,
) -> pd.Series:
    """Classify market regimes from index returns using simplified rules.

    This is a simpler alternative to `build_regime_state()` that only
    requires index returns (no macro factors, breadth, etc.).

    Classification rules (applied in priority order):
    1. Crisis: Drawdown < drawdown_threshold OR volatility > vol_threshold_high
    2. Bear: Negative trend (price < MA) AND volatility > vol_threshold_low
    3. Bull: Positive trend (price > MA) AND volatility < vol_threshold_high
    4. Sideways: Moderate trend AND moderate volatility
    5. Neutral: Default (ambiguous signals)
    6. Reflation: Strong positive trend after a crisis period (recovering)

    Args:
        index_returns: Series of index returns (daily or other frequency)
            Index should be timestamps, values should be returns (not prices)
        config: Optional RegimeConfig (default: RegimeConfig())

    Returns:
        Series with regime labels ("bull", "bear", "sideways", "crisis", "reflation", "neutral")
        Index matches index_returns index

    Note:
        - Returns are assumed to be daily (or other frequency) percentage returns
        - Prices are reconstructed as cumulative product: prices = (1 + returns).cumprod()
        - Volatility is annualized (assuming 252 trading days per year)
    """
    if config is None:
        config = RegimeConfig()

    # Use aliases if provided, otherwise use main fields
    vol_window = (
        config.lookback_days_vol
        if config.lookback_days_vol is not None
        else config.vol_window
    )
    trend_window = (
        config.lookback_days_return
        if config.lookback_days_return is not None
        else config.trend_ma_window
    )

    if len(index_returns) < config.min_periods:
        logger.warning(
            f"Insufficient data for regime classification: {len(index_returns)} < {config.min_periods}. "
            "Returning all 'neutral'."
        )
        return pd.Series(["neutral"] * len(index_returns), index=index_returns.index)

    # Reconstruct prices from returns (normalized to start at 1.0)
    prices = (1.0 + index_returns).cumprod()

    # Compute realized volatility (rolling, annualized)
    # Assuming daily returns, annualize by sqrt(252)
    vol = index_returns.rolling(
        window=vol_window, min_periods=config.min_periods
    ).std() * np.sqrt(252)

    # Compute trend: price relative to moving average
    ma = prices.rolling(window=trend_window, min_periods=config.min_periods).mean()
    trend = (prices - ma) / ma  # Relative deviation from MA

    # Compute drawdown: current price relative to running maximum
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max  # Negative values

    # Initialize regime labels
    regimes = pd.Series(
        ["neutral"] * len(index_returns), index=index_returns.index, dtype=object
    )

    # Classification rules (applied in priority order)
    # Note: We use iterative approach because Reflation requires sequential logic
    # (checking if recent crisis occurred)
    for i in range(len(index_returns)):
        if pd.isna(vol.iloc[i]) or pd.isna(trend.iloc[i]) or pd.isna(drawdown.iloc[i]):
            regimes.iloc[i] = "neutral"
            continue

        vol_val = vol.iloc[i]
        trend_val = trend.iloc[i]
        dd_val = drawdown.iloc[i]

        # Rule 1: Crisis (highest priority)
        if dd_val < config.drawdown_threshold or vol_val > config.vol_threshold_high:
            regimes.iloc[i] = "crisis"
        # Rule 2: Bear
        elif trend_val < -config.trend_threshold and vol_val > config.vol_threshold_low:
            regimes.iloc[i] = "bear"
        # Rule 3: Bull
        elif trend_val > config.trend_threshold and vol_val < config.vol_threshold_high:
            regimes.iloc[i] = "bull"
        # Rule 4: Sideways
        elif (
            abs(trend_val) <= config.trend_threshold
            and config.vol_threshold_low <= vol_val <= config.vol_threshold_high
        ):
            regimes.iloc[i] = "sideways"
        # Rule 5: Reflation (strong positive trend after crisis)
        elif trend_val > config.trend_threshold * 1.5 and i > 0:
            # Check if we're recovering from a recent crisis
            recent_crisis = (regimes.iloc[max(0, i - 20) : i] == "crisis").any()
            if recent_crisis:
                regimes.iloc[i] = "reflation"
            else:
                regimes.iloc[i] = "bull"  # Strong bull if not recovering
        # Rule 6: Default to neutral
        else:
            regimes.iloc[i] = "neutral"

    # Log regime distribution
    regime_counts = regimes.value_counts()
    logger.info(f"Regime classification complete. Distribution: {dict(regime_counts)}")

    return regimes


def summarize_metrics_by_regime(
    equity: pd.Series,
    regimes: pd.Series,
    trades: pd.DataFrame | None = None,
    factor_panel: pd.DataFrame | None = None,
    timestamp_col: str = "timestamp",
    freq: Literal["1d", "5min"] = "1d",
) -> pd.DataFrame:
    """Summarize performance metrics by regime with extended trade-level analysis.

    This extends `compute_risk_by_regime()` with additional metrics:
    - Trade-level metrics (win rate, avg trade duration, avg profit/loss)
    - Factor performance by regime (if factor_panel provided)
    - Regime-specific statistics

    Args:
        equity: Series of equity values (index should be timestamps)
        regimes: Series of regime labels (index should match equity index)
        trades: Optional DataFrame with trades (columns: timestamp, symbol, side, qty, price)
        factor_panel: Optional DataFrame with factor values for factor performance analysis
        timestamp_col: Name of timestamp column in trades/factor_panel (default: "timestamp")
        freq: Trading frequency for annualization (default: "1d")

    Returns:
        DataFrame with one row per regime:
        - regime_label: Regime name
        - n_periods: Number of periods in this regime
        - sharpe, sortino, volatility, max_drawdown, cagr: Standard risk metrics
        - win_rate: Trade win rate (if trades provided)
        - avg_trade_duration: Average trade duration in days (if trades provided)
        - avg_profit_per_trade: Average profit per trade (if trades provided)
        - factor_ic_mean: Mean IC for factors (if factor_panel provided)
        - Additional metrics as needed
    """
    from src.assembled_core.qa.metrics import (
        _get_periods_per_year,
        compute_cagr,
        compute_sortino_ratio,
    )

    # Validate inputs
    if equity.empty or regimes.empty:
        logger.warning("equity or regimes is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=["regime_label", "n_periods"])

    # Ensure indices match
    if not equity.index.equals(regimes.index):
        # Try to align
        common_index = equity.index.intersection(regimes.index)
        if len(common_index) == 0:
            logger.warning(
                "No overlapping indices between equity and regimes. Returning empty DataFrame."
            )
            return pd.DataFrame(columns=["regime_label", "n_periods"])
        equity = equity.loc[common_index]
        regimes = regimes.loc[common_index]

    # Compute returns from equity
    returns = equity.pct_change(fill_method=None).dropna()

    # Align returns with regimes (drop first period where return is NaN)
    returns = returns.loc[returns.index.intersection(regimes.index)]
    regimes_aligned = regimes.loc[returns.index]

    if returns.empty:
        logger.warning("No valid returns after alignment. Returning empty DataFrame.")
        return pd.DataFrame(columns=["regime_label", "n_periods"])

    # Compute metrics directly per regime
    periods_per_year = _get_periods_per_year(freq)
    results = []

    for regime in regimes_aligned.unique():
        regime_mask = regimes_aligned == regime
        # Get indices where regime matches
        regime_indices = regime_mask[regime_mask].index
        # Filter equity and returns by these indices
        regime_equity = equity.loc[equity.index.intersection(regime_indices)]
        regime_returns = returns.loc[returns.index.intersection(regime_indices)]

        if len(regime_equity) < 2 or len(regime_returns) < 2:
            continue

        # Compute basic metrics for this regime
        from src.assembled_core.risk.risk_metrics import compute_basic_risk_metrics

        basic_metrics = compute_basic_risk_metrics(
            returns=regime_returns,
            freq=freq,
        )

        # Total return
        total_return = (
            float((1.0 + regime_returns).prod() - 1.0)
            if len(regime_returns) > 0
            else 0.0
        )

        result_row = {
            "regime_label": regime,
            "n_periods": len(regime_equity),
            "sharpe": basic_metrics.get("sharpe"),
            "volatility": basic_metrics.get("vol_annualized"),
            "max_drawdown": basic_metrics.get("max_drawdown"),
            "total_return": total_return,
        }

        # Compute additional metrics
        # Sortino
        if len(regime_returns) > 1 and regime_returns.std() > 0:
            # compute_sortino_ratio uses freq string, not periods_per_year
            freq_str = "1d" if periods_per_year == 252 else "5min"
            result_row["sortino"] = compute_sortino_ratio(regime_returns, freq=freq_str)
        else:
            result_row["sortino"] = None

        # CAGR
        if len(regime_equity) > 1:
            start_value = float(regime_equity.iloc[0])
            end_value = float(regime_equity.iloc[-1])
            n_periods = len(regime_equity)
            freq_str = "1d" if periods_per_year == 252 else "5min"
            result_row["cagr"] = compute_cagr(
                start_value=start_value,
                end_value=end_value,
                periods=n_periods,
                freq=freq_str,
            )
        else:
            result_row["cagr"] = None

        # Trade-level metrics (if trades provided)
        if trades is not None and not trades.empty and timestamp_col in trades.columns:
            regime_trades = trades[trades[timestamp_col].isin(regime_equity.index)]
            if not regime_trades.empty:
                result_row["n_trades"] = len(regime_trades)
                # Win rate: fraction of positive-return days in this regime
                result_row["win_rate"] = (
                    float((regime_returns > 0).mean()) if len(regime_returns) > 0 else None
                )
                # Pair BUY/SELL per symbol to compute duration and P&L
                has_side = "side" in regime_trades.columns
                has_sym = "symbol" in regime_trades.columns
                has_price = "price" in regime_trades.columns
                if has_side and has_sym:
                    durations: list[float] = []
                    pnls: list[float] = []
                    for sym, sym_trades in regime_trades.groupby("symbol"):
                        sym_trades = sym_trades.sort_values(timestamp_col)
                        buys = sym_trades[sym_trades["side"].astype(str).str.upper() == "BUY"]
                        sells = sym_trades[sym_trades["side"].astype(str).str.upper() == "SELL"]
                        for buy_row in buys.itertuples(index=False):
                            buy_ts = getattr(buy_row, timestamp_col)
                            next_sells = sells[sells[timestamp_col] > buy_ts]
                            if next_sells.empty:
                                continue
                            sell_row = next_sells.iloc[0]
                            try:
                                d = (pd.Timestamp(sell_row[timestamp_col]) - pd.Timestamp(buy_ts)).days
                                durations.append(float(d))
                            except Exception as _exc:
                                logger.debug("[regime_analysis] trade duration calc skipped: %s", _exc)
                            if has_price:
                                bp = float(getattr(buy_row, "price", None) or 0)
                                sp = float(sell_row.get("price") or 0)
                                qty = float(getattr(buy_row, "qty", None) or 1)
                                if bp > 0:
                                    pnls.append((sp - bp) / bp * qty)
                    result_row["avg_trade_duration"] = float(np.mean(durations)) if durations else None
                    result_row["avg_profit_per_trade"] = float(np.mean(pnls)) if pnls else None
                else:
                    result_row["avg_trade_duration"] = None
                    result_row["avg_profit_per_trade"] = None
            else:
                result_row["n_trades"] = 0
                result_row["win_rate"] = None
                result_row["avg_trade_duration"] = None
                result_row["avg_profit_per_trade"] = None
        else:
            result_row["n_trades"] = None
            result_row["win_rate"] = None
            result_row["avg_trade_duration"] = None
            result_row["avg_profit_per_trade"] = None

        # Factor IC by regime: correlation between factor values and next-period returns
        if factor_panel is not None and not factor_panel.empty and len(regime_returns) > 2:
            try:
                if timestamp_col in factor_panel.columns:
                    regime_factors = factor_panel[factor_panel[timestamp_col].isin(regime_equity.index)]
                else:
                    regime_factors = factor_panel.loc[factor_panel.index.intersection(regime_equity.index)]
                if not regime_factors.empty:
                    numeric_cols = regime_factors.select_dtypes(include="number").columns.tolist()
                    ics: list[float] = []
                    for col in numeric_cols:
                        fv = regime_factors[col].dropna()
                        rv = regime_returns.reindex(fv.index).dropna()
                        fv = fv.reindex(rv.index).dropna()
                        rv = rv.reindex(fv.index)
                        if len(fv) > 2:
                            ic = float(fv.corr(rv))
                            if not np.isnan(ic):
                                ics.append(ic)
                    result_row["factor_ic_mean"] = float(np.mean(ics)) if ics else None
                else:
                    result_row["factor_ic_mean"] = None
            except Exception as exc:
                logger.warning("[regime_analysis] factor_ic_mean failed: %s", exc)
                result_row["factor_ic_mean"] = None
        else:
            result_row["factor_ic_mean"] = None

        results.append(result_row)

    if not results:
        logger.warning("No valid regime metrics computed. Returning empty DataFrame.")
        return pd.DataFrame(columns=["regime_label", "n_periods"])

    result_df = pd.DataFrame(results)

    logger.info(f"Computed metrics for {len(result_df)} regimes")

    return result_df


def compute_regime_transitions(
    regime_state: pd.DataFrame,
    timestamp_col: str = "timestamp",
    regime_col: str = "regime_label",
) -> pd.DataFrame:
    """Analyze regime transitions (from one regime to another).

    Computes:
    - Transition matrix (probability of transitioning from regime A to B)
    - Average regime duration
    - Transition frequencies
    - Strategy performance around transitions (if equity/trades provided)

    Args:
        regime_state: DataFrame with columns: timestamp_col, regime_col
        timestamp_col: Name of timestamp column (default: "timestamp")
        regime_col: Name of regime label column (default: "regime_label")

    Returns:
        DataFrame with transition analysis:
        - from_regime: Source regime
        - to_regime: Target regime
        - transition_count: Number of transitions
        - transition_probability: Probability of this transition
        - avg_duration_days: Average duration of source regime before transition
        - Additional transition statistics

    """
    if regime_state.empty or regime_col not in regime_state.columns:
        return pd.DataFrame(
            columns=["from_regime", "to_regime", "transition_count", "transition_probability", "avg_duration_days"]
        )

    df = regime_state.copy()
    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    regimes = df[regime_col].tolist()
    if len(regimes) < 2:
        return pd.DataFrame(
            columns=["from_regime", "to_regime", "transition_count", "transition_probability", "avg_duration_days"]
        )

    # Identify consecutive regime runs and transitions between them
    transitions: list[dict] = []
    current = regimes[0]
    run_start = 0

    for i in range(1, len(regimes)):
        if regimes[i] != current:
            transitions.append({
                "from_regime": current,
                "to_regime": regimes[i],
                "duration_before": i - run_start,
            })
            current = regimes[i]
            run_start = i

    if not transitions:
        return pd.DataFrame(
            columns=["from_regime", "to_regime", "transition_count", "transition_probability", "avg_duration_days"]
        )

    trans_df = pd.DataFrame(transitions)
    counts = (
        trans_df.groupby(["from_regime", "to_regime"])
        .agg(
            transition_count=("duration_before", "count"),
            avg_duration_days=("duration_before", "mean"),
        )
        .reset_index()
    )

    totals = counts.groupby("from_regime")["transition_count"].sum().rename("_total")
    counts = counts.join(totals, on="from_regime")
    counts["transition_probability"] = counts["transition_count"] / counts["_total"]
    counts = counts.drop(columns=["_total"])

    return (
        counts
        .sort_values(["from_regime", "transition_probability"], ascending=[True, False])
        .reset_index(drop=True)
    )


def summarize_factor_ic_by_regime(
    ic_series: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Summarize factor IC (Information Coefficient) metrics by regime.

    Groups IC values by regime and computes summary statistics (mean, std, count).

    Args:
        ic_series: Series of IC values (index should be timestamps)
        regimes: Series of regime labels (index should match ic_series index)

    Returns:
        DataFrame with one row per regime:
        - regime_label: Regime name
        - ic_mean: Mean IC for this regime
        - ic_std: Standard deviation of IC for this regime
        - ic_count: Number of IC observations in this regime
        - ic_ir: IC Information Ratio (mean / std) if std > 0, else None

    Note:
        This is a helper function for factor performance analysis by regime.
        IC values should be computed separately (e.g., from factor ranking or ML validation).
    """
    # Validate inputs
    if ic_series.empty or regimes.empty:
        logger.warning("ic_series or regimes is empty. Returning empty DataFrame.")
        return pd.DataFrame(
            columns=["regime_label", "ic_mean", "ic_std", "ic_count", "ic_ir"]
        )

    # Align indices
    common_index = ic_series.index.intersection(regimes.index)
    if len(common_index) == 0:
        logger.warning(
            "No overlapping indices between ic_series and regimes. Returning empty DataFrame."
        )
        return pd.DataFrame(
            columns=["regime_label", "ic_mean", "ic_std", "ic_count", "ic_ir"]
        )

    ic_aligned = ic_series.loc[common_index].dropna()
    regimes_aligned = regimes.loc[ic_aligned.index]

    if ic_aligned.empty:
        logger.warning("No valid IC values after alignment. Returning empty DataFrame.")
        return pd.DataFrame(
            columns=["regime_label", "ic_mean", "ic_std", "ic_count", "ic_ir"]
        )

    # Group by regime and compute statistics
    results = []

    for regime in regimes_aligned.unique():
        regime_ic = ic_aligned[regimes_aligned == regime].dropna()

        if len(regime_ic) == 0:
            continue

        ic_mean = float(regime_ic.mean())
        ic_std = float(regime_ic.std()) if len(regime_ic) > 1 else 0.0
        ic_count = len(regime_ic)
        ic_ir = ic_mean / ic_std if ic_std > 0 else None

        results.append(
            {
                "regime_label": regime,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_count": ic_count,
                "ic_ir": ic_ir,
            }
        )

    if not results:
        logger.warning("No valid IC metrics computed. Returning empty DataFrame.")
        return pd.DataFrame(
            columns=["regime_label", "ic_mean", "ic_std", "ic_count", "ic_ir"]
        )

    result_df = pd.DataFrame(results)

    logger.info(f"Computed IC metrics for {len(result_df)} regimes")

    return result_df
