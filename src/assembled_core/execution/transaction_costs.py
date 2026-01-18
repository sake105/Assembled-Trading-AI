# src/assembled_core/execution/transaction_costs.py
"""Transaction costs computation for fills and trades.

This module provides deterministic commission models for backtest fill simulation.
Costs are computed per trade/fill and added as columns to trade DataFrames.

**Layering:** This module belongs to the execution layer (not qa/).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class CommissionModel:
    """Commission model configuration.

    Attributes:
        mode: Commission calculation mode:
            - "bps": Commission as basis points of notional
            - "fixed": Fixed commission per trade
            - "bps_plus_fixed": Both bps and fixed commission
        commission_bps: Commission in basis points (1 bps = 0.01%)
            Only used if mode is "bps" or "bps_plus_fixed"
        fixed_per_trade: Fixed commission per trade (in cash units)
            Only used if mode is "fixed" or "bps_plus_fixed"
    """

    mode: Literal["bps", "fixed", "bps_plus_fixed"] = "bps"
    commission_bps: float = 0.0
    fixed_per_trade: float = 0.0

    def __post_init__(self) -> None:
        """Validate commission model parameters."""
        if self.mode not in ("bps", "fixed", "bps_plus_fixed"):
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'bps', 'fixed', or 'bps_plus_fixed'")
        if self.commission_bps < 0.0:
            raise ValueError(f"commission_bps must be >= 0.0, got {self.commission_bps}")
        if self.fixed_per_trade < 0.0:
            raise ValueError(f"fixed_per_trade must be >= 0.0, got {self.fixed_per_trade}")


def compute_commission_cash(
    notional: np.ndarray,
    n_trades: int,
    model: CommissionModel,
) -> np.ndarray:
    """Compute commission cash costs per trade.

    Args:
        notional: Array of notional values (abs(qty) * price) for each trade
        n_trades: Number of trades (must match len(notional))
        model: CommissionModel instance

    Returns:
        Array of commission cash costs (positive values, same length as notional)

    Raises:
        ValueError: If n_trades doesn't match len(notional) or model is invalid
    """
    if len(notional) != n_trades:
        raise ValueError(f"notional length ({len(notional)}) must match n_trades ({n_trades})")

    # Handle empty arrays
    if n_trades == 0:
        return np.array([], dtype=np.float64)

    # Initialize with zeros
    commission_cash = np.zeros(n_trades, dtype=np.float64)

    # Compute bps component
    if model.mode in ("bps", "bps_plus_fixed"):
        commission_cash += notional * (model.commission_bps / 10000.0)

    # Compute fixed component
    if model.mode in ("fixed", "bps_plus_fixed"):
        commission_cash += model.fixed_per_trade

    # Ensure non-negative (costs are always positive)
    commission_cash = np.maximum(commission_cash, 0.0)

    return commission_cash


def add_cost_columns_to_trades(
    trades: pd.DataFrame,
    commission_model: CommissionModel | None = None,
    spread_model: SpreadModel | None = None,
    slippage_model: SlippageModel | None = None,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add cost columns to trades DataFrame.

    Adds columns: commission_cash, spread_cash, slippage_cash, total_cost_cash.
    All cost columns default to 0.0 (never NaN).

    Args:
        trades: DataFrame with columns: timestamp, symbol, side, qty, price
            (may have additional columns)
        commission_model: Optional CommissionModel instance.
            If None, uses default (commission_bps=0.0, fixed_per_trade=0.0)
        spread_model: Optional SpreadModel instance.
            If None, spread_cash defaults to 0.0
        slippage_model: Optional SlippageModel instance.
            If None, slippage_cash defaults to 0.0
        prices: Optional prices DataFrame with columns: timestamp, symbol, close, volume (optional)
            Required for spread and slippage calculation if spread_model or slippage_model is provided.
            If None and spread_model/slippage_model is provided, costs default to 0.0

    Returns:
        DataFrame with added cost columns:
        - commission_cash: Commission costs (computed from model)
        - spread_cash: Spread costs (computed from spread_model if provided, else 0.0)
        - slippage_cash: Slippage costs (computed from slippage_model if provided, else 0.0)
        - total_cost_cash: Total costs (commission_cash + spread_cash + slippage_cash)

    Raises:
        ValueError: If required columns are missing in trades DataFrame
    """
    # Validate required columns
    required_cols = ["timestamp", "symbol", "side", "qty", "price"]
    missing_cols = [col for col in required_cols if col not in trades.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in trades DataFrame: {missing_cols}")

    # Make a copy to avoid modifying original
    trades_with_costs = trades.copy()

    # Compute notional based on fill_qty if available, else use original qty
    # For partial fills, costs should be based on filled notional (fill_qty * fill_price)
    if "fill_qty" in trades_with_costs.columns and "fill_price" in trades_with_costs.columns:
        # Use filled notional for cost calculation
        notional = (trades_with_costs["fill_qty"].abs() * trades_with_costs["fill_price"].abs()).values
    else:
        # Fallback: use original qty * price (for backward compatibility)
        notional = (trades_with_costs["qty"].abs() * trades_with_costs["price"].abs()).values

    # Compute commission cash
    if commission_model is None:
        commission_model = CommissionModel(mode="bps", commission_bps=0.0, fixed_per_trade=0.0)

    n_trades = len(trades_with_costs)
    commission_cash = compute_commission_cash(notional, n_trades, commission_model)

    # Compute spread cash if spread_model and prices are provided
    if spread_model is not None and prices is not None and not prices.empty:
        try:
            # Compute ADV proxy
            adv_df = compute_adv_proxy(prices, adv_window=spread_model.adv_window)
            
            # Merge ADV with trades (on timestamp and symbol)
            trades_with_adv = trades_with_costs.merge(
                adv_df,
                on=["timestamp", "symbol"],
                how="left",
                suffixes=("", "_adv"),
            )
            
            # Assign spread_bps based on ADV buckets
            adv_usd = trades_with_adv["adv_usd"].values
            spread_bps = assign_spread_bps(adv_usd, spread_model)
            
            # Compute spread cash
            spread_cash = compute_spread_cash(notional, spread_bps)
        except Exception:
            # Fallback: use fallback_spread_bps for all trades
            spread_bps = np.full(n_trades, spread_model.fallback_spread_bps, dtype=np.float64)
            spread_cash = compute_spread_cash(notional, spread_bps)
    else:
        # No spread model or prices: default to 0.0
        spread_cash = np.zeros(n_trades, dtype=np.float64)

    # Compute slippage cash if slippage_model and prices are provided
    if slippage_model is not None and prices is not None and not prices.empty:
        try:
            # Compute ADV proxy (reuse from spread calculation if available)
            if spread_model is not None:
                # ADV already computed above
                adv_df = compute_adv_proxy(prices, adv_window=spread_model.adv_window)
            else:
                # Compute ADV for slippage (use slippage_model.vol_window as fallback, but ADV needs its own window)
                # For now, use vol_window as adv_window (can be refined later)
                adv_df = compute_adv_proxy(prices, adv_window=slippage_model.vol_window)
            
            # Compute volatility proxy
            vol_df = compute_volatility_proxy(prices, vol_window=slippage_model.vol_window)
            
            # Merge ADV and volatility with trades
            trades_with_adv_vol = trades_with_costs.merge(
                adv_df,
                on=["timestamp", "symbol"],
                how="left",
                suffixes=("", "_adv"),
            )
            trades_with_adv_vol = trades_with_adv_vol.merge(
                vol_df,
                on=["timestamp", "symbol"],
                how="left",
                suffixes=("", "_vol"),
            )
            
            # Assign slippage_bps based on volatility and participation
            adv_usd = trades_with_adv_vol["adv_usd"].values
            volatility = trades_with_adv_vol["volatility"].values
            slippage_bps = compute_slippage_bps(notional, adv_usd, volatility, slippage_model)
            
            # Compute slippage cash
            slippage_cash = compute_slippage_cash(notional, slippage_bps)
        except Exception:
            # Fallback: use fallback_slippage_bps for all trades
            slippage_bps = np.full(n_trades, slippage_model.fallback_slippage_bps, dtype=np.float64)
            slippage_cash = compute_slippage_cash(notional, slippage_bps)
    else:
        # No slippage model or prices: default to 0.0
        slippage_cash = np.zeros(n_trades, dtype=np.float64)

    # Add cost columns (default: 0.0, never NaN)
    trades_with_costs["commission_cash"] = commission_cash.astype(np.float64)
    trades_with_costs["spread_cash"] = spread_cash.astype(np.float64)
    trades_with_costs["slippage_cash"] = slippage_cash.astype(np.float64)
    trades_with_costs["total_cost_cash"] = (
        trades_with_costs["commission_cash"]
        + trades_with_costs["spread_cash"]
        + trades_with_costs["slippage_cash"]
    )

    # Ensure fill schema compliance (add fill_qty, fill_price, status, remaining_qty if missing)
    # For backward compatibility, assume full fills
    from src.assembled_core.execution.fill_model import ensure_fill_schema
    
    trades_with_costs = ensure_fill_schema(trades_with_costs, default_full_fill=True)
    
    # Adjust costs for rejected fills: costs should be 0 if fill_qty == 0
    # Note: Costs are already computed based on fill_qty (via notional = fill_qty * fill_price)
    # But we need to explicitly set costs to 0 for rejected fills (status == "rejected")
    if "status" in trades_with_costs.columns:
        rejected_mask = trades_with_costs["status"] == "rejected"
        if rejected_mask.any():
            trades_with_costs.loc[rejected_mask, "commission_cash"] = 0.0
            trades_with_costs.loc[rejected_mask, "spread_cash"] = 0.0
            trades_with_costs.loc[rejected_mask, "slippage_cash"] = 0.0
            trades_with_costs.loc[rejected_mask, "total_cost_cash"] = 0.0
    if "fill_qty" in trades_with_costs.columns:
        rejected_mask = trades_with_costs["status"] == "rejected"
        if rejected_mask.any():
            trades_with_costs.loc[rejected_mask, "commission_cash"] = 0.0
            trades_with_costs.loc[rejected_mask, "spread_cash"] = 0.0
            trades_with_costs.loc[rejected_mask, "slippage_cash"] = 0.0
            trades_with_costs.loc[rejected_mask, "total_cost_cash"] = 0.0

    # Ensure deterministic sorting (by timestamp, symbol)
    if not trades_with_costs.empty:
        trades_with_costs = trades_with_costs.sort_values(
            ["timestamp", "symbol"], ignore_index=True
        )

    return trades_with_costs


def commission_model_from_cost_params(
    commission_bps: float | None = None,
    fixed_per_trade: float | None = None,
    mode: Literal["bps", "fixed", "bps_plus_fixed"] | None = None,
) -> CommissionModel:
    """Create CommissionModel from legacy cost parameters.

    Maps legacy parameters (commission_bps, spread_w, impact_w) to CommissionModel.
    This is a compatibility function for existing code.

    Args:
        commission_bps: Commission in basis points (default: 0.0)
        fixed_per_trade: Fixed commission per trade (default: 0.0)
        mode: Commission mode (default: "bps" if commission_bps > 0, else "fixed" if fixed_per_trade > 0, else "bps")

    Returns:
        CommissionModel instance
    """
    if commission_bps is None:
        commission_bps = 0.0
    if fixed_per_trade is None:
        fixed_per_trade = 0.0

    # Determine mode if not specified
    if mode is None:
        if commission_bps > 0.0 and fixed_per_trade > 0.0:
            mode = "bps_plus_fixed"
        elif fixed_per_trade > 0.0:
            mode = "fixed"
        else:
            mode = "bps"

    return CommissionModel(
        mode=mode,
        commission_bps=commission_bps,
        fixed_per_trade=fixed_per_trade,
    )


@dataclass
class SpreadModel:
    """Spread model configuration based on ADV (Average Daily Volume) buckets.

    Attributes:
        adv_window: Rolling window for ADV calculation (default: 20 days)
        buckets: List of tuples (adv_threshold, spread_bps) in ascending order.
            ADV values are compared against thresholds to assign spread_bps.
            Example: [(1e6, 5.0), (1e7, 3.0), (1e8, 1.0)] means:
            - ADV < 1e6: 5.0 bps
            - 1e6 <= ADV < 1e7: 3.0 bps
            - 1e7 <= ADV < 1e8: 1.0 bps
            - ADV >= 1e8: fallback_spread_bps
        fallback_spread_bps: Spread in basis points when ADV cannot be computed
            (e.g., volume missing) or ADV exceeds highest bucket (default: 5.0)
    """

    adv_window: int = 20
    buckets: list[tuple[float, float]] | None = None
    fallback_spread_bps: float = 5.0

    def __post_init__(self) -> None:
        """Validate spread model parameters."""
        if self.adv_window < 1:
            raise ValueError(f"adv_window must be >= 1, got {self.adv_window}")
        if self.fallback_spread_bps < 0.0:
            raise ValueError(f"fallback_spread_bps must be >= 0.0, got {self.fallback_spread_bps}")
        if self.buckets is not None:
            # Validate buckets are sorted by threshold (ascending)
            thresholds = [b[0] for b in self.buckets]
            if thresholds != sorted(thresholds):
                raise ValueError(f"buckets must be sorted by threshold (ascending), got {thresholds}")
            # Validate spread_bps are non-negative
            for threshold, spread_bps in self.buckets:
                if spread_bps < 0.0:
                    raise ValueError(f"spread_bps in buckets must be >= 0.0, got {spread_bps} at threshold {threshold}")


def compute_adv_proxy(
    prices: pd.DataFrame,
    adv_window: int = 20,
) -> pd.DataFrame:
    """Compute ADV (Average Daily Volume) proxy from prices DataFrame.

    ADV proxy = rolling mean of (close * volume) over adv_window days.
    If volume is missing, returns NaN for that symbol/timestamp.

    Args:
        prices: DataFrame with columns: timestamp, symbol, close, volume (optional)
            Must be sorted by (symbol, timestamp)
        adv_window: Rolling window for ADV calculation (default: 20)

    Returns:
        DataFrame with columns: timestamp, symbol, adv_usd
            ADV in USD (close * volume, rolling mean)
            NaN if volume is missing or insufficient history

    Note:
        ADV is computed per symbol using groupby + rolling.
        Requires at least adv_window rows per symbol for non-NaN values.
    """
    if prices.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "adv_usd"])

    # Validate required columns
    required_cols = ["timestamp", "symbol", "close"]
    missing_cols = [col for col in required_cols if col not in prices.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in prices DataFrame: {missing_cols}")

    # Make a copy to avoid modifying original
    prices_copy = prices.copy()

    # Ensure sorted by (symbol, timestamp) for rolling window
    if not prices_copy.empty:
        prices_copy = prices_copy.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Compute ADV proxy: rolling mean of (close * volume)
    if "volume" in prices_copy.columns:
        # Compute dollar volume: close * volume
        prices_copy["dollar_volume"] = prices_copy["close"] * prices_copy["volume"]
        
        # Rolling mean per symbol
        adv = (
            prices_copy.groupby("symbol")["dollar_volume"]
            .rolling(window=adv_window, min_periods=adv_window)
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        # Volume missing: return NaN for all rows
        adv = pd.Series(np.nan, index=prices_copy.index)

    # Create result DataFrame
    result = pd.DataFrame({
        "timestamp": prices_copy["timestamp"],
        "symbol": prices_copy["symbol"],
        "adv_usd": adv.values,
    })

    return result


def assign_spread_bps(
    adv_usd: pd.Series | np.ndarray,
    model: SpreadModel,
) -> np.ndarray:
    """Assign spread_bps based on ADV buckets.

    Args:
        adv_usd: Array/Series of ADV values in USD (may contain NaN)
        model: SpreadModel instance with buckets configuration

    Returns:
        Array of spread_bps values (same length as adv_usd)
        Uses fallback_spread_bps for NaN values or ADV exceeding highest bucket

    Note:
        Buckets are applied in ascending order (lowest threshold first).
        First bucket where ADV >= threshold is used.
        If ADV exceeds all thresholds, fallback_spread_bps is used.
    """
    adv_array = np.asarray(adv_usd, dtype=np.float64)
    n = len(adv_array)
    spread_bps = np.full(n, model.fallback_spread_bps, dtype=np.float64)

    # If no buckets defined, use fallback for all
    if model.buckets is None or len(model.buckets) == 0:
        return spread_bps

    # Apply buckets: find highest threshold <= ADV for each value
    # Buckets are sorted by threshold (ascending), so we iterate in reverse
    # to find the highest matching threshold first
    # For each ADV value, we want the highest threshold that is <= ADV
    for threshold, bucket_spread_bps in reversed(model.buckets):
        # For values >= this threshold, assign this bucket's spread_bps
        # (only if not already assigned to a higher bucket)
        mask = (adv_array >= threshold) & (spread_bps == model.fallback_spread_bps)
        spread_bps[mask] = bucket_spread_bps

    # Handle NaN values: use fallback
    nan_mask = np.isnan(adv_array)
    spread_bps[nan_mask] = model.fallback_spread_bps

    return spread_bps


@dataclass
class SlippageModel:
    """Slippage model configuration based on volatility and liquidity.

    Attributes:
        vol_window: Rolling window for volatility calculation (default: 20 days)
        k: Scaling factor for slippage calculation (default: 1.0)
        min_bps: Minimum slippage in basis points (clamp, default: 0.0)
        max_bps: Maximum slippage in basis points (clamp, default: 50.0)
        participation_rate_cap: Maximum participation rate (notional / ADV, default: 1.0)
            Used to cap participation for cost estimation (no partial fills yet)
        fallback_slippage_bps: Slippage in basis points when volatility/ADV cannot be computed
            (e.g., volume missing, insufficient history, default: 5.0)
    """

    vol_window: int = 20
    k: float = 1.0
    min_bps: float = 0.0
    max_bps: float = 50.0
    participation_rate_cap: float = 1.0
    fallback_slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        """Validate slippage model parameters."""
        if self.vol_window < 1:
            raise ValueError(f"vol_window must be >= 1, got {self.vol_window}")
        if self.k < 0.0:
            raise ValueError(f"k must be >= 0.0, got {self.k}")
        if self.min_bps < 0.0:
            raise ValueError(f"min_bps must be >= 0.0, got {self.min_bps}")
        if self.max_bps < self.min_bps:
            raise ValueError(f"max_bps ({self.max_bps}) must be >= min_bps ({self.min_bps})")
        if self.participation_rate_cap <= 0.0:
            raise ValueError(f"participation_rate_cap must be > 0.0, got {self.participation_rate_cap}")
        if self.fallback_slippage_bps < 0.0:
            raise ValueError(f"fallback_slippage_bps must be >= 0.0, got {self.fallback_slippage_bps}")


def compute_volatility_proxy(
    prices: pd.DataFrame,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Compute volatility proxy from prices DataFrame.

    Volatility proxy = rolling standard deviation of log returns over vol_window days.
    Requires at least vol_window rows per symbol for non-NaN values.

    Args:
        prices: DataFrame with columns: timestamp, symbol, close
            Must be sorted by (symbol, timestamp)
        vol_window: Rolling window for volatility calculation (default: 20)

    Returns:
        DataFrame with columns: timestamp, symbol, volatility
            Volatility (rolling std of log returns)
            NaN if insufficient history or missing close prices

    Note:
        Volatility is computed per symbol using groupby + rolling.
        Log returns are computed as log(close / close.shift(1)).
    """
    if prices.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "volatility"])

    # Validate required columns
    required_cols = ["timestamp", "symbol", "close"]
    missing_cols = [col for col in required_cols if col not in prices.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in prices DataFrame: {missing_cols}")

    # Make a copy to avoid modifying original
    prices_copy = prices.copy()

    # Ensure sorted by (symbol, timestamp) for rolling window
    if not prices_copy.empty:
        prices_copy = prices_copy.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Compute log returns per symbol
    prices_copy["log_return"] = (
        prices_copy.groupby("symbol")["close"]
        .apply(lambda x: np.log(x / x.shift(1)))
        .reset_index(level=0, drop=True)
    )

    # Rolling standard deviation per symbol
    volatility = (
        prices_copy.groupby("symbol")["log_return"]
        .rolling(window=vol_window, min_periods=vol_window)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Create result DataFrame
    result = pd.DataFrame({
        "timestamp": prices_copy["timestamp"],
        "symbol": prices_copy["symbol"],
        "volatility": volatility.values,
    })

    return result


def compute_slippage_bps(
    notional: np.ndarray,
    adv_usd: np.ndarray,
    volatility: np.ndarray,
    model: SlippageModel,
) -> np.ndarray:
    """Compute slippage in basis points based on volatility and participation rate.

    Formula: slippage_bps = clip(k * sigma * sqrt(participation) * 10000, min_bps, max_bps)
    where:
        sigma = volatility (rolling std of log returns)
        participation = notional / adv_usd (clipped to participation_rate_cap)

    Args:
        notional: Array of notional values (abs(qty) * price) for each trade
        adv_usd: Array of ADV values in USD (may contain NaN)
        volatility: Array of volatility values (rolling std of log returns, may contain NaN)
        model: SlippageModel instance

    Returns:
        Array of slippage_bps values (same length as notional)
        Uses fallback_slippage_bps for NaN values or when calculation fails

    Note:
        Participation rate is clipped to participation_rate_cap to prevent unrealistic costs.
    """
    if len(notional) != len(adv_usd) or len(notional) != len(volatility):
        raise ValueError(
            f"notional ({len(notional)}), adv_usd ({len(adv_usd)}), "
            f"and volatility ({len(volatility)}) must have same length"
        )

    # Handle empty arrays
    if len(notional) == 0:
        return np.array([], dtype=np.float64)

    n = len(notional)
    slippage_bps = np.full(n, model.fallback_slippage_bps, dtype=np.float64)

    # Compute participation rate: notional / adv_usd
    # Clip to participation_rate_cap and handle division by zero
    participation = np.zeros(n, dtype=np.float64)
    valid_adv_mask = (adv_usd > 0.0) & ~np.isnan(adv_usd)
    participation[valid_adv_mask] = notional[valid_adv_mask] / adv_usd[valid_adv_mask]
    participation = np.clip(participation, 0.0, model.participation_rate_cap)

    # Compute slippage_bps: k * sigma * sqrt(participation) * 10000
    # Only compute where volatility and participation are valid
    valid_mask = (
        ~np.isnan(volatility)
        & (volatility >= 0.0)
        & (participation > 0.0)
        & valid_adv_mask
    )

    if np.any(valid_mask):
        slippage_bps[valid_mask] = (
            model.k * volatility[valid_mask] * np.sqrt(participation[valid_mask]) * 10000.0
        )
        # Clamp to [min_bps, max_bps]
        slippage_bps[valid_mask] = np.clip(
            slippage_bps[valid_mask],
            model.min_bps,
            model.max_bps,
        )

    # Handle NaN/invalid values: use fallback
    invalid_mask = ~valid_mask
    slippage_bps[invalid_mask] = model.fallback_slippage_bps

    return slippage_bps


def compute_slippage_cash(
    notional: np.ndarray,
    slippage_bps: np.ndarray,
) -> np.ndarray:
    """Compute slippage cash costs per trade.

    Slippage cost = notional * (slippage_bps / 10000)

    Args:
        notional: Array of notional values (abs(qty) * price) for each trade
        slippage_bps: Array of slippage in basis points for each trade

    Returns:
        Array of slippage cash costs (positive values, same length as notional)

    Raises:
        ValueError: If notional and slippage_bps have different lengths
    """
    if len(notional) != len(slippage_bps):
        raise ValueError(
            f"notional length ({len(notional)}) must match slippage_bps length ({len(slippage_bps)})"
        )

    # Handle empty arrays
    if len(notional) == 0:
        return np.array([], dtype=np.float64)

    # Slippage cost = notional * (slippage_bps / 10000)
    slippage_cash = notional * (slippage_bps / 10000.0)

    # Ensure non-negative (costs are always positive)
    slippage_cash = np.maximum(slippage_cash, 0.0)

    return slippage_cash


def compute_spread_cash(
    notional: np.ndarray,
    spread_bps: np.ndarray,
) -> np.ndarray:
    """Compute spread cash costs per trade.

    Spread cost = notional * (spread_bps / 10000) * 0.5
    The 0.5 factor represents half-spread (paid cost when crossing bid/ask spread).

    Args:
        notional: Array of notional values (abs(qty) * price) for each trade
        spread_bps: Array of spread in basis points for each trade

    Returns:
        Array of spread cash costs (positive values, same length as notional)

    Raises:
        ValueError: If notional and spread_bps have different lengths
    """
    if len(notional) != len(spread_bps):
        raise ValueError(
            f"notional length ({len(notional)}) must match spread_bps length ({len(spread_bps)})"
        )

    # Handle empty arrays
    if len(notional) == 0:
        return np.array([], dtype=np.float64)

    # Spread cost = notional * (spread_bps / 10000) * 0.5
    # 0.5 = half-spread (paid cost when crossing bid/ask spread)
    spread_cash = notional * (spread_bps / 10000.0) * 0.5

    # Ensure non-negative (costs are always positive)
    spread_cash = np.maximum(spread_cash, 0.0)

    return spread_cash
