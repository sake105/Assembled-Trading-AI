# src/assembled_core/risk/group_exposures.py
"""Group Exposure Calculation: Sector/Region/Currency aggregation (Sprint 9).

Computes gross/net exposure and weights aggregated by sector, region, or currency.

Layering: risk layer (can import from data, but not from pipeline/qa).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

# Valid group types
GROUP_TYPES = ["sector", "region", "currency"]


@dataclass
class GroupExposureSummary:
    """Summary of group exposures.

    Attributes:
        total_groups: Total number of groups
        max_gross_weight: Maximum gross weight across all groups
        max_net_weight: Maximum net weight across all groups
        total_gross_exposure: Sum of gross exposures across all groups
        total_net_exposure: Sum of net exposures across all groups
    """

    total_groups: int
    max_gross_weight: float
    max_net_weight: float
    total_gross_exposure: float
    total_net_exposure: float


def compute_group_exposures(
    exposures_df: pd.DataFrame,
    security_meta_df: pd.DataFrame,
    group_col: Literal["sector", "region", "currency"],
) -> tuple[pd.DataFrame, GroupExposureSummary]:
    """Compute group exposures for a specific group type.

    Args:
        exposures_df: Per-symbol exposures DataFrame (from compute_exposures)
            Required columns: symbol, notional, weight
        security_meta_df: Security metadata DataFrame (from resolve_security_meta)
            Required columns: symbol, <group_col>
        group_col: Group type to aggregate by ("sector", "region", or "currency")

    Returns:
        Tuple of (group_exposures_df, summary)
        - group_exposures_df: DataFrame with columns:
            * group_type: str (e.g., "sector")
            * group_value: str (e.g., "Technology")
            * net_exposure: float (sum of notional for group)
            * gross_exposure: float (sum of abs(notional) for group)
            * net_weight: float (net_exposure / equity)
            * gross_weight: float (gross_exposure / equity)
            * n_symbols: int (number of symbols in group)
          Sorted by group_type, group_value (ascending)
        - summary: GroupExposureSummary with aggregated metrics

    Raises:
        ValueError: If group_col is invalid or if symbols are missing from security_meta_df
    """
    if group_col not in GROUP_TYPES:
        raise ValueError(
            f"Invalid group_col: {group_col}. Must be one of {GROUP_TYPES}"
        )

    # Validate required columns in exposures_df
    required_exposure_cols = ["symbol", "notional", "weight"]
    missing_cols = [
        col for col in required_exposure_cols if col not in exposures_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"exposures_df missing required columns: {missing_cols}. "
            f"Required: {required_exposure_cols}"
        )

    # Validate required columns in security_meta_df
    if "symbol" not in security_meta_df.columns:
        raise ValueError("security_meta_df missing required column: symbol")
    if group_col not in security_meta_df.columns:
        raise ValueError(
            f"security_meta_df missing required column: {group_col}. "
            f"Required columns: symbol, {group_col}"
        )

    # Merge exposures with security metadata
    merged_df = exposures_df.merge(
        security_meta_df[["symbol", group_col]],
        on="symbol",
        how="left",
    )

    # Check for missing mappings (symbols without group value)
    missing_mask = merged_df[group_col].isna()
    if missing_mask.any():
        missing_symbols = sorted(merged_df[missing_mask]["symbol"].unique().tolist())
        raise ValueError(
            f"Missing {group_col} mapping for symbols: {missing_symbols}. "
            f"Total symbols in exposures: {len(exposures_df)}, "
            f"missing: {len(missing_symbols)}"
        )

    # Compute equity from exposures (for weight calculation)
    # We need equity to compute group weights. Infer from exposures_df.
    # Use weight and notional: equity = notional / weight (for non-zero weights)
    non_zero_mask = exposures_df["weight"].abs() > 1e-10
    if non_zero_mask.any():
        # Use first non-zero weight/notional pair to infer equity
        # This is deterministic (first match in sorted order)
        first_non_zero_idx = exposures_df[non_zero_mask].index[0]
        equity = (
            exposures_df.loc[first_non_zero_idx, "notional"]
            / exposures_df.loc[first_non_zero_idx, "weight"]
        )
        # Ensure equity is positive (use absolute value)
        equity = abs(equity)
    else:
        # If all weights are zero, use sum of abs(notional) as equity proxy
        equity = exposures_df["notional"].abs().sum()
        if equity == 0:
            equity = 1.0  # Default fallback to avoid division by zero

    # Aggregate by group
    group_agg = merged_df.groupby(group_col, as_index=False).agg(
        {
            "notional": ["sum", lambda x: x.abs().sum()],  # net, gross
            "symbol": "count",  # n_symbols
        }
    )

    # Flatten column names
    group_agg.columns = ["group_value", "net_exposure", "gross_exposure", "n_symbols"]

    # Compute weights
    group_agg["net_weight"] = group_agg["net_exposure"] / equity if equity > 0 else 0.0
    group_agg["gross_weight"] = (
        group_agg["gross_exposure"] / equity if equity > 0 else 0.0
    )

    # Add group_type column
    group_agg["group_type"] = group_col

    # Reorder columns: group_type, group_value, then metrics
    group_exposures_df = group_agg[
        [
            "group_type",
            "group_value",
            "net_exposure",
            "gross_exposure",
            "net_weight",
            "gross_weight",
            "n_symbols",
        ]
    ].copy()

    # Deterministic sorting: group_type, group_value (ascending)
    group_exposures_df = group_exposures_df.sort_values(
        ["group_type", "group_value"], ascending=True
    ).reset_index(drop=True)

    # Compute summary
    summary = GroupExposureSummary(
        total_groups=len(group_exposures_df),
        max_gross_weight=(
            group_exposures_df["gross_weight"].max()
            if len(group_exposures_df) > 0
            else 0.0
        ),
        max_net_weight=(
            group_exposures_df["net_weight"].abs().max()
            if len(group_exposures_df) > 0
            else 0.0
        ),
        total_gross_exposure=group_exposures_df["gross_exposure"].sum(),
        total_net_exposure=group_exposures_df["net_exposure"].sum(),
    )

    return group_exposures_df, summary


def compute_all_group_exposures(
    exposures_df: pd.DataFrame,
    security_meta_df: pd.DataFrame,
    *,
    group_types: list[Literal["sector", "region", "currency"]] | None = None,
) -> dict[str, tuple[pd.DataFrame, GroupExposureSummary]]:
    """Compute group exposures for all group types.

    Args:
        exposures_df: Per-symbol exposures DataFrame (from compute_exposures)
        security_meta_df: Security metadata DataFrame (from resolve_security_meta)
        group_types: List of group types to compute (default: all ["sector", "region", "currency"])

    Returns:
        Dictionary mapping group_type to (group_exposures_df, summary)
        Example: {
            "sector": (sector_df, sector_summary),
            "region": (region_df, region_summary),
            "currency": (currency_df, currency_summary),
        }

    Raises:
        ValueError: If group_types contains invalid values or if symbols are missing
    """
    if group_types is None:
        group_types = ["sector", "region", "currency"]

    # Validate group_types
    invalid_types = [gt for gt in group_types if gt not in GROUP_TYPES]
    if invalid_types:
        raise ValueError(
            f"Invalid group_types: {invalid_types}. Must be subset of {GROUP_TYPES}"
        )

    result = {}
    for group_type in group_types:
        group_df, summary = compute_group_exposures(
            exposures_df,
            security_meta_df,
            group_type,
        )
        result[group_type] = (group_df, summary)

    return result


# ---------------------------------------------------------------------------
# Net Market Exposure Tracker (inverse ETF aware)
# ---------------------------------------------------------------------------


def compute_net_market_exposure(
    weights: dict[str, float],
    inverse_etf_map: dict[str, str] | None = None,
) -> dict:
    """Compute net and gross market exposure, accounting for inverse ETF positions.

    Inverse ETFs held LONG effectively create short exposure to the underlying.
    This function adjusts gross/net calculations accordingly.

    Args:
        weights: Dict mapping symbol → portfolio weight (positive = long).
        inverse_etf_map: Optional dict mapping long ETF → inverse ETF symbol.
            When an inverse ETF appears in weights, its contribution to net
            market exposure is negated (e.g., long SH = short SPY exposure).
            Default: uses INVERSE_ETF_MAP from universe_etf.

    Returns:
        Dict with keys:
            gross_long: Sum of positive weights.
            gross_short: Sum of abs(negative weights).
            net_exposure: gross_long - gross_short - inverse_etf_exposure.
            inverse_etf_exposure: Total weight in inverse ETFs (as short proxy).
            leverage: gross_long + gross_short.
    """
    if inverse_etf_map is None:
        try:
            from src.assembled_core.data.universe_etf import INVERSE_ETF_MAP
            inverse_etf_map = INVERSE_ETF_MAP
        except ImportError:
            inverse_etf_map = {}

    # Build set of inverse ETF symbols
    inverse_symbols = set(inverse_etf_map.values())

    gross_long = 0.0
    gross_short = 0.0
    inverse_etf_exposure = 0.0

    for sym, w in weights.items():
        if sym in inverse_symbols:
            # Holding an inverse ETF creates synthetic short exposure
            inverse_etf_exposure += abs(w)
        elif w > 0:
            gross_long += w
        else:
            gross_short += abs(w)

    net_exposure = gross_long - gross_short - inverse_etf_exposure
    leverage = gross_long + gross_short + inverse_etf_exposure

    return {
        "gross_long": round(gross_long, 6),
        "gross_short": round(gross_short, 6),
        "inverse_etf_exposure": round(inverse_etf_exposure, 6),
        "net_exposure": round(net_exposure, 6),
        "leverage": round(leverage, 6),
    }
