"""Pre-trade checks for order validation and risk limits.

This module provides pre-trade validation checks to ensure orders comply with
risk limits and trading rules before execution. It validates position sizes,
exposure limits, and integrates with QA gates.

Key features:
- Position size limits (per symbol, per order)
- Gross exposure limits
- QA gate integration
- Sector/Region exposure limits (optional, for future use)
- Robust handling of missing information

Usage:
    >>> from src.assembled_core.execution.pre_trade_checks import (
    ...     PreTradeConfig,
    ...     PreTradeCheckResult,
    ...     run_pre_trade_checks
    ... )
    >>> import pandas as pd
    >>>
    >>> config = PreTradeConfig(
    ...     max_notional_per_symbol=10000.0,
    ...     max_gross_exposure=50000.0
    ... )
    >>> orders = pd.DataFrame({
    ...     "symbol": ["AAPL", "GOOGL"],
    ...     "side": ["BUY", "BUY"],
    ...     "qty": [100, 50],
    ...     "price": [150.0, 2500.0]
    ... })
    >>>
    >>> result = run_pre_trade_checks(orders, portfolio=None, config=config)
    >>> if result.is_ok:
    ...     print("All orders passed pre-trade checks")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    from src.assembled_core.qa.qa_gates import QAResult  # noqa: F401


@dataclass
class PreTradeConfig:
    """Configuration for pre-trade checks.

    Attributes:
        max_notional_per_symbol: Maximum notional value per symbol (in currency units)
        max_weight_per_symbol: Maximum weight per symbol (as fraction, e.g., 0.1 = 10%)
            If set, orders are reduced (not blocked) when post-trade weight exceeds limit.
            SELL orders that reduce exposure are allowed even if still over limit.
        turnover_cap: Maximum turnover (as fraction of equity, e.g., 0.5 = 50%)
            If set, orders are proportionally reduced when total turnover exceeds cap.
            turnover = sum(abs(order_notional)) / equity
            order_notional = abs(qty * price)
        drawdown_threshold: Maximum drawdown threshold (as fraction, e.g., 0.2 = 20%)
            If set, orders are scaled by de_risk_scale when drawdown >= threshold.
            drawdown = 1 - current_equity / peak_equity
        de_risk_scale: Scale factor for orders when drawdown threshold exceeded (default: 0.0 = full block)
            If 0.0, all orders are blocked. If 0.25, orders are reduced to 25% of original.
        max_gross_exposure: Maximum gross exposure (sum of absolute positions)
        max_sector_exposure: Maximum sector exposure (as fraction of equity, e.g., 0.3 = 30%)
            If set, orders are reduced when sector gross_weight exceeds limit.
        max_region_exposure: Maximum region exposure (as fraction of equity, e.g., 0.5 = 50%)
            If set, orders are reduced when region gross_weight exceeds limit.
        max_fx_exposure: Maximum FX exposure (as fraction of equity, e.g., 0.2 = 20%)
            If set, orders are reduced when non-base-currency gross_weight exceeds limit.
        base_currency: Base currency for FX exposure calculation (default: "USD")
        missing_security_meta: How to handle missing security metadata (default: "raise")
            - "raise": Raise ValueError if security metadata is missing
            - "unknown": Use "UNKNOWN" as default value (not recommended)
    """

    max_notional_per_symbol: float | None = None
    max_weight_per_symbol: float | None = None
    turnover_cap: float | None = None
    drawdown_threshold: float | None = None
    de_risk_scale: float = 0.0  # Default: full block when drawdown threshold exceeded
    max_gross_exposure: float | None = None
    max_sector_exposure: float | None = None
    max_region_exposure: float | None = None
    max_fx_exposure: float | None = None
    base_currency: str = "USD"
    missing_security_meta: Literal["raise", "unknown"] = "raise"


@dataclass
class PreTradeCheckResult:
    """Result of pre-trade checks.

    Attributes:
        is_ok: True if all checks passed, False otherwise
        blocked_reasons: List of reasons why orders were blocked
        filtered_orders: Orders DataFrame after filtering (only orders that passed)
        summary: Optional dictionary with summary metrics (gross exposure, etc.)
        reduced_orders: List of dicts with reduction reasons (e.g., RISK_REDUCE_MAX_WEIGHT_PER_SYMBOL)
    """

    is_ok: bool
    blocked_reasons: list[str] = field(default_factory=list)
    filtered_orders: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    summary: dict[str, Any] = field(default_factory=dict)
    reduced_orders: list[dict[str, Any]] = field(default_factory=list)


def run_pre_trade_checks(
    orders: pd.DataFrame,
    portfolio: pd.DataFrame | None = None,
    qa_status: Any | None = None,  # QAGatesSummary (imported lazily)
    risk_summary: dict[str, Any] | None = None,
    config: PreTradeConfig | None = None,
    *,
    current_positions: pd.DataFrame | None = None,
    prices_latest: pd.DataFrame | None = None,
    equity: float | None = None,
    current_equity: float | None = None,
    peak_equity: float | None = None,
    security_meta_df: pd.DataFrame | None = None,
) -> tuple[PreTradeCheckResult, pd.DataFrame]:
    """Run pre-trade checks on orders and filter out blocked orders.

    This function performs various validation checks:
    - Position size limits (notional per symbol, weight per symbol)
    - Gross exposure limits
    - QA gate integration (if QA status blocks trading)
    - Sector/Region exposure limits (if config and portfolio data available)

    Args:
        orders: DataFrame with columns: symbol, side, qty, price (optional: timestamp)
            Must have at least symbol and qty. Price is used for notional calculation.
        portfolio: Optional DataFrame with current portfolio snapshot.
            Expected columns: symbol, qty (or weight, or value)
            Used for calculating resulting positions and weights
        qa_status: Optional QA gates summary. If overall_result is BLOCK, all orders are blocked.
        risk_summary: Optional risk summary dictionary (for future use)
        config: Optional PreTradeConfig. If None, uses default config (no limits enforced)

    Returns:
        Tuple of (PreTradeCheckResult, filtered_orders DataFrame):
        - PreTradeCheckResult with is_ok, blocked_reasons, filtered_orders, summary
        - filtered_orders: Orders that passed all checks (may be empty if all blocked)

    Example:
        >>> import pandas as pd
        >>> from src.assembled_core.execution.pre_trade_checks import (
        ...     PreTradeConfig,
        ...     run_pre_trade_checks
        ... )
        >>>
        >>> orders = pd.DataFrame({
        ...     "symbol": ["AAPL", "GOOGL"],
        ...     "side": ["BUY", "BUY"],
        ...     "qty": [100, 50],
        ...     "price": [150.0, 2500.0]
        ... })
        >>>
        >>> config = PreTradeConfig(max_notional_per_symbol=10000.0)
        >>> result, filtered = run_pre_trade_checks(orders, config=config)
        >>>
        >>> if not result.is_ok:
        ...     print(f"Orders blocked: {result.blocked_reasons}")
    """
    if config is None:
        config = PreTradeConfig()

    blocked_reasons = []
    reduced_orders = []
    filtered_orders = orders.copy()
    summary = {}

    # Handle empty orders
    if orders.empty:
        return (
            PreTradeCheckResult(
                is_ok=True,
                blocked_reasons=[],
                filtered_orders=pd.DataFrame(
                    columns=orders.columns if not orders.empty else []
                ),
                summary={"total_orders": 0, "passed_orders": 0},
            ),
            pd.DataFrame(columns=orders.columns if not orders.empty else []),
        )

    # Check required columns
    required_cols = ["symbol", "qty"]
    missing_cols = [c for c in required_cols if c not in orders.columns]
    if missing_cols:
        blocked_reasons.append(f"Orders missing required columns: {missing_cols}")
        return (
            PreTradeCheckResult(
                is_ok=False,
                blocked_reasons=blocked_reasons,
                filtered_orders=pd.DataFrame(columns=orders.columns),
                summary={"total_orders": len(orders), "passed_orders": 0},
            ),
            pd.DataFrame(columns=orders.columns),
        )

    # 1. Check QA status (if provided)
    if qa_status is not None:
        # Import lazily to avoid circular imports
        from src.assembled_core.qa.qa_gates import QAResult
        
        if qa_status.overall_result == QAResult.BLOCK:
            blocked_reasons.append("QA_BLOCK: QA gates blocked trading")
            return (
                PreTradeCheckResult(
                    is_ok=False,
                    blocked_reasons=blocked_reasons,
                    filtered_orders=pd.DataFrame(columns=orders.columns),
                    summary={
                        "total_orders": len(orders),
                        "passed_orders": 0,
                        "qa_blocked": True,
                    },
                ),
                pd.DataFrame(columns=orders.columns),
            )

    # 2. Calculate order notional values (if price available)
    orders_with_notional = filtered_orders.copy()
    if "price" in filtered_orders.columns:
        orders_with_notional["notional"] = (
            filtered_orders["qty"] * filtered_orders["price"]
        )
    else:
        # If no price, set notional to 0 (checks will be skipped)
        orders_with_notional["notional"] = 0.0

    # 3. Check max_notional_per_symbol
    if config.max_notional_per_symbol is not None:
        if "price" in filtered_orders.columns:
            # Calculate notional per symbol (sum of BUY and SELL)
            symbol_notionals = (
                orders_with_notional.groupby("symbol")["notional"].sum().abs()
            )
            exceeded_symbols = symbol_notionals[
                symbol_notionals > config.max_notional_per_symbol
            ]

            if len(exceeded_symbols) > 0:
                for symbol, notional in exceeded_symbols.items():
                    blocked_reasons.append(
                        f"max_notional_per_symbol: Symbol '{symbol}' has notional "
                        f"{notional:.2f}, exceeds limit {config.max_notional_per_symbol:.2f}"
                    )
                    # Filter out orders for this symbol
                    filtered_orders = filtered_orders[
                        filtered_orders["symbol"] != symbol
                    ]
                    orders_with_notional = orders_with_notional[
                        orders_with_notional["symbol"] != symbol
                    ]

    # 4. Check max_weight_per_symbol (requires current_positions, prices_latest, equity)
    if config.max_weight_per_symbol is not None:
        if current_positions is not None and prices_latest is not None and equity is not None:
            # Import exposure engine here to avoid circular imports
            try:
                from src.assembled_core.risk.exposure_engine import (
                    compute_exposures,
                    compute_target_positions,
                )
            except ImportError as e:
                blocked_reasons.append(
                    f"max_weight_per_symbol check failed: cannot import exposure_engine: {e}"
                )
                filtered_orders = pd.DataFrame(columns=orders.columns)
                summary["max_weight_per_symbol_check"] = f"import_error: {e}"
            else:
                # Compute post-trade exposures using exposure engine
                try:
                    # Convert current_positions to expected format (symbol, qty)
                    if current_positions.empty:
                        current_positions_df = pd.DataFrame(columns=["symbol", "qty"])
                    else:
                        # Handle different column names (qty, target_qty, etc.)
                        if "qty" in current_positions.columns:
                            current_positions_df = current_positions[["symbol", "qty"]].copy()
                        elif "target_qty" in current_positions.columns:
                            current_positions_df = current_positions[["symbol", "target_qty"]].rename(
                                columns={"target_qty": "qty"}
                            )
                        else:
                            # Skip if cannot determine current positions
                            summary["max_weight_per_symbol_check"] = "skipped_no_qty_column"
                            current_positions_df = pd.DataFrame(columns=["symbol", "qty"])

                    # Compute target positions (current + orders)
                    target_positions = compute_target_positions(
                        current_positions_df,
                        filtered_orders,
                    )

                    # Compute exposures (target + prices + equity)
                    exposures_df, _ = compute_exposures(
                        target_positions,
                        prices_latest,
                        equity,
                        missing_price_handling="raise",
                    )

                    # Check each symbol for weight limit violation
                    for _, exposure_row in exposures_df.iterrows():
                        symbol = exposure_row["symbol"]
                        weight = exposure_row["weight"]
                        abs_weight = abs(weight)

                        if abs_weight > config.max_weight_per_symbol:
                            # Find orders for this symbol
                            symbol_orders = filtered_orders[filtered_orders["symbol"] == symbol].copy()

                            if symbol_orders.empty:
                                continue

                            # Check if order reduces exposure (SELL when overweight)
                            # If SELL and current weight > limit, allow it (reduces exposure)
                            # Otherwise, reduce order qty
                            current_qty = (
                                current_positions_df[current_positions_df["symbol"] == symbol]["qty"].iloc[0]
                                if not current_positions_df.empty
                                and symbol in current_positions_df["symbol"].values
                                else 0.0
                            )
                            target_qty = exposure_row["target_qty"]
                            price = exposure_row["price"]

                            # Determine if we need to reduce
                            # If target_qty moves towards 0 (reduces exposure), allow it
                            # Otherwise, reduce to exactly hit the limit
                            if abs(current_qty) > abs(target_qty) and abs(current_qty) * price / equity > config.max_weight_per_symbol:
                                # Order reduces exposure (e.g., SELL when overweight)
                                # Allow it (don't block or reduce)
                                continue

                            # Reduce order qty so that target weight exactly equals limit
                            # max_target_notional = max_weight_per_symbol * equity
                            # max_target_qty = max_target_notional / price
                            max_target_notional = config.max_weight_per_symbol * equity
                            max_target_qty = max_target_notional / price if price > 0.0 else 0.0

                            # Calculate required reduction
                            # target_qty should be capped at max_target_qty (or -max_target_qty for shorts)
                            if target_qty > 0:
                                max_target_qty = abs(max_target_qty)
                            else:
                                max_target_qty = -abs(max_target_qty)

                            # Calculate order delta needed
                            required_target_qty = max_target_qty
                            order_delta_needed = required_target_qty - current_qty

                            # Reduce each order proportionally
                            total_order_delta = symbol_orders.apply(
                                lambda row: row["qty"] if row["side"] == "BUY" else -row["qty"],
                                axis=1,
                            ).sum()

                            if abs(total_order_delta) > 1e-10:
                                scale_factor = order_delta_needed / total_order_delta
                                scale_factor = max(0.0, min(1.0, scale_factor))  # Clamp to [0, 1]

                                # Apply reduction to orders
                                for idx in symbol_orders.index:
                                    original_qty = filtered_orders.loc[idx, "qty"]
                                    new_qty = original_qty * scale_factor

                                    if new_qty < 1e-10:
                                        # Order becomes too small, remove it
                                        filtered_orders = filtered_orders.drop(index=idx)
                                        reduced_orders.append({
                                            "reason": "RISK_REDUCE_MAX_WEIGHT_PER_SYMBOL",
                                            "symbol": symbol,
                                            "original_qty": original_qty,
                                            "new_qty": 0.0,
                                            "explain": {
                                                "current_weight": abs(current_qty * price / equity) if current_qty != 0.0 else 0.0,
                                                "target_weight": abs_weight,
                                                "limit": config.max_weight_per_symbol,
                                                "reduction_factor": scale_factor,
                                            },
                                        })
                                    else:
                                        filtered_orders.loc[idx, "qty"] = new_qty
                                        reduced_orders.append({
                                            "reason": "RISK_REDUCE_MAX_WEIGHT_PER_SYMBOL",
                                            "symbol": symbol,
                                            "original_qty": original_qty,
                                            "new_qty": new_qty,
                                            "explain": {
                                                "current_weight": abs(current_qty * price / equity) if current_qty != 0.0 else 0.0,
                                                "target_weight": abs_weight,
                                                "limit": config.max_weight_per_symbol,
                                                "reduction_factor": scale_factor,
                                            },
                                        })

                except ValueError as e:
                    # Fail-fast: equity <= 0 or missing price
                    blocked_reasons.append(
                        f"max_weight_per_symbol check failed: {e}. "
                        "Required: current_positions, prices_latest, equity > 0"
                    )
                    # Block all orders if check fails
                    filtered_orders = pd.DataFrame(columns=orders.columns)
                except Exception as e:
                    # Other errors: log and skip check
                    summary["max_weight_per_symbol_check"] = f"error: {e}"
        else:
            # Missing required inputs: skip check
            summary["max_weight_per_symbol_check"] = "skipped_missing_inputs"

    # 8. Check sector/region/FX exposure limits (requires security metadata)
    # Order: max_weight -> sector -> region -> fx -> turnover -> drawdown
    if (
        config.max_sector_exposure is not None
        or config.max_region_exposure is not None
        or config.max_fx_exposure is not None
    ):
        # Import lazily to avoid circular imports
        try:
            from src.assembled_core.risk.exposure_engine import (
                compute_exposures,
                compute_target_positions,
            )
            from src.assembled_core.risk.group_exposures import compute_group_exposures
        except ImportError as e:
            blocked_reasons.append(
                f"Group exposure checks failed: cannot import modules: {e}"
            )
            summary["group_exposure_check"] = f"import_error: {e}"
        else:
            # Check if we have required inputs
            if (
                current_positions is not None
                and prices_latest is not None
                and equity is not None
                and equity > 0.0
                and security_meta_df is not None
            ):
                try:
                    # Convert current_positions to expected format (symbol, qty)
                    if current_positions.empty:
                        current_positions_df = pd.DataFrame(columns=["symbol", "qty"])
                    else:
                        if "qty" in current_positions.columns:
                            current_positions_df = current_positions[["symbol", "qty"]].copy()
                        elif "target_qty" in current_positions.columns:
                            current_positions_df = current_positions[["symbol", "target_qty"]].rename(
                                columns={"target_qty": "qty"}
                            )
                        else:
                            current_positions_df = pd.DataFrame(columns=["symbol", "qty"])

                    # Compute target positions and exposures
                    target_positions_df = compute_target_positions(
                        current_positions_df, filtered_orders
                    )
                    exposures_df, _ = compute_exposures(
                        target_positions_df,
                        prices_latest,
                        equity,
                        missing_price_handling="raise",
                    )

                    # Check sector exposure limit
                    if config.max_sector_exposure is not None:
                        try:
                            sector_df, _ = compute_group_exposures(
                                exposures_df,
                                security_meta_df,
                                "sector",
                            )

                            # Check each sector for limit violation
                            for _, sector_row in sector_df.iterrows():
                                sector_value = sector_row["group_value"]
                                gross_weight = sector_row["gross_weight"]

                                if gross_weight > config.max_sector_exposure:
                                    # Find orders for symbols in this sector
                                    sector_symbols = security_meta_df[
                                        security_meta_df["sector"] == sector_value
                                    ]["symbol"].tolist()
                                    sector_orders = filtered_orders[
                                        filtered_orders["symbol"].isin(sector_symbols)
                                    ].copy()

                                    if sector_orders.empty:
                                        continue

                                    # Calculate scale factor to bring gross_weight to limit
                                    scale_factor = config.max_sector_exposure / gross_weight
                                    scale_factor = max(0.0, min(1.0, scale_factor))  # Clamp to [0, 1]

                                    # Apply scale factor to orders in this sector
                                    indices_to_process = list(sector_orders.index)
                                    for idx in indices_to_process:
                                        if idx not in filtered_orders.index:
                                            continue

                                        original_qty = filtered_orders.loc[idx, "qty"]
                                        symbol = filtered_orders.loc[idx, "symbol"]
                                        side = filtered_orders.loc[idx, "side"]

                                        # Apply scale factor
                                        new_qty = original_qty * scale_factor

                                        # Round deterministically: int() truncates towards zero
                                        new_qty = float(int(new_qty))

                                        # Preserve sign based on side
                                        if side == "SELL":
                                            new_qty = -abs(new_qty)
                                        else:
                                            new_qty = abs(new_qty)

                                        if abs(new_qty) < 1e-10:
                                            # Order becomes too small, remove it
                                            filtered_orders = filtered_orders.drop(index=idx)
                                            reduced_orders.append({
                                                "reason": "RISK_REDUCE_MAX_SECTOR_EXPOSURE",
                                                "symbol": symbol,
                                                "original_qty": original_qty,
                                                "new_qty": 0.0,
                                                "explain": {
                                                    "group_type": "sector",
                                                    "group_value": sector_value,
                                                    "cap": config.max_sector_exposure,
                                                    "pre_weight": gross_weight,
                                                    "post_weight": config.max_sector_exposure,
                                                    "scale_factor": scale_factor,
                                                },
                                            })
                                        else:
                                            filtered_orders.loc[idx, "qty"] = new_qty
                                            reduced_orders.append({
                                                "reason": "RISK_REDUCE_MAX_SECTOR_EXPOSURE",
                                                "symbol": symbol,
                                                "original_qty": original_qty,
                                                "new_qty": new_qty,
                                                "explain": {
                                                    "group_type": "sector",
                                                    "group_value": sector_value,
                                                    "cap": config.max_sector_exposure,
                                                    "pre_weight": gross_weight,
                                                    "post_weight": config.max_sector_exposure,
                                                    "scale_factor": scale_factor,
                                                },
                                            })
                        except ValueError as e:
                            if config.missing_security_meta == "raise":
                                # Re-raise ValueError to fail-fast
                                raise ValueError(f"Sector exposure check failed: {e}") from e
                            else:
                                summary["sector_exposure_check"] = "skipped_missing_meta"

                    # Check region exposure limit (same logic as sector)
                    if config.max_region_exposure is not None:
                        try:
                            region_df, _ = compute_group_exposures(
                                exposures_df,
                                security_meta_df,
                                "region",
                            )

                            for _, region_row in region_df.iterrows():
                                region_value = region_row["group_value"]
                                gross_weight = region_row["gross_weight"]

                                if gross_weight > config.max_region_exposure:
                                    region_symbols = security_meta_df[
                                        security_meta_df["region"] == region_value
                                    ]["symbol"].tolist()
                                    region_orders = filtered_orders[
                                        filtered_orders["symbol"].isin(region_symbols)
                                    ].copy()

                                    if region_orders.empty:
                                        continue

                                    scale_factor = config.max_region_exposure / gross_weight
                                    scale_factor = max(0.0, min(1.0, scale_factor))

                                    indices_to_process = list(region_orders.index)
                                    for idx in indices_to_process:
                                        if idx not in filtered_orders.index:
                                            continue

                                        original_qty = filtered_orders.loc[idx, "qty"]
                                        symbol = filtered_orders.loc[idx, "symbol"]
                                        side = filtered_orders.loc[idx, "side"]

                                        new_qty = original_qty * scale_factor
                                        new_qty = float(int(new_qty))

                                        if side == "SELL":
                                            new_qty = -abs(new_qty)
                                        else:
                                            new_qty = abs(new_qty)

                                        if abs(new_qty) < 1e-10:
                                            filtered_orders = filtered_orders.drop(index=idx)
                                            reduced_orders.append({
                                                "reason": "RISK_REDUCE_MAX_REGION_EXPOSURE",
                                                "symbol": symbol,
                                                "original_qty": original_qty,
                                                "new_qty": 0.0,
                                                "explain": {
                                                    "group_type": "region",
                                                    "group_value": region_value,
                                                    "cap": config.max_region_exposure,
                                                    "pre_weight": gross_weight,
                                                    "post_weight": config.max_region_exposure,
                                                    "scale_factor": scale_factor,
                                                },
                                            })
                                        else:
                                            filtered_orders.loc[idx, "qty"] = new_qty
                                            reduced_orders.append({
                                                "reason": "RISK_REDUCE_MAX_REGION_EXPOSURE",
                                                "symbol": symbol,
                                                "original_qty": original_qty,
                                                "new_qty": new_qty,
                                                "explain": {
                                                    "group_type": "region",
                                                    "group_value": region_value,
                                                    "cap": config.max_region_exposure,
                                                    "pre_weight": gross_weight,
                                                    "post_weight": config.max_region_exposure,
                                                    "scale_factor": scale_factor,
                                                },
                                            })
                        except ValueError as e:
                            if config.missing_security_meta == "raise":
                                # Re-raise ValueError to fail-fast
                                raise ValueError(f"Region exposure check failed: {e}") from e
                            else:
                                summary["region_exposure_check"] = "skipped_missing_meta"

                    # Check FX exposure limit
                    if config.max_fx_exposure is not None:
                        try:
                            # Check if all currencies are base_currency
                            if "currency" not in security_meta_df.columns:
                                if config.missing_security_meta == "raise":
                                    raise ValueError(
                                        "FX exposure check requires currency column in security_meta_df"
                                    )
                                else:
                                    summary["fx_exposure_check"] = "skipped_missing_currency"
                            else:
                                # Check for non-base currencies
                                non_base_currencies = security_meta_df[
                                    security_meta_df["currency"] != config.base_currency
                                ]["currency"].unique().tolist()

                                if non_base_currencies:
                                    # FX rates not implemented yet - fail-fast with clear message
                                    if config.missing_security_meta == "raise":
                                        raise ValueError(
                                            f"FX exposure needs currency mapping + FX rates (not implemented). "
                                            f"Non-base currencies found: {non_base_currencies}. "
                                            f"Base currency: {config.base_currency}. "
                                            f"All positions must be in base currency for now."
                                        )
                                    else:
                                        summary["fx_exposure_check"] = "skipped_non_base_currency"
                                else:
                                    # All currencies are base_currency - no FX exposure
                                    summary["fx_exposure_check"] = "all_base_currency"
                        except ValueError as e:
                            if config.missing_security_meta == "raise":
                                # Re-raise ValueError to fail-fast
                                raise ValueError(f"FX exposure check failed: {e}") from e
                            else:
                                summary["fx_exposure_check"] = "skipped_missing_meta"
                except ValueError as e:
                    # Re-raise ValueError if missing_security_meta="raise"
                    if config.missing_security_meta == "raise":
                        raise
                    else:
                        blocked_reasons.append(f"Group exposure checks failed: {e}")
                        summary["group_exposure_check"] = f"error: {e}"
                except Exception as e:
                    blocked_reasons.append(f"Group exposure checks failed: {e}")
                    summary["group_exposure_check"] = f"error: {e}"
            else:
                # Missing required inputs
                if security_meta_df is None:
                    summary["group_exposure_check"] = "skipped_no_security_meta"
                elif current_positions is None:
                    summary["group_exposure_check"] = "skipped_no_current_positions"
                elif prices_latest is None:
                    summary["group_exposure_check"] = "skipped_no_prices_latest"
                elif equity is None:
                    summary["group_exposure_check"] = "skipped_no_equity"
                elif equity <= 0.0:
                    summary["group_exposure_check"] = "skipped_equity_zero"

    # 5. Check turnover_cap (requires prices and equity)
    if config.turnover_cap is not None:
        if "price" in filtered_orders.columns and equity is not None and equity > 0.0:
            # Calculate turnover: sum(abs(order_notional)) / equity
            # order_notional = abs(qty * price)
            order_notionals = (filtered_orders["qty"].abs() * filtered_orders["price"].abs())
            total_turnover = float(order_notionals.sum() / equity)

            if total_turnover > config.turnover_cap:
                # Calculate scale factor: cap / turnover
                scale_factor = config.turnover_cap / total_turnover
                scale_factor = max(0.0, min(1.0, scale_factor))  # Clamp to [0, 1]

                # Apply reduction to all orders proportionally
                # Create a copy of indices to avoid modification during iteration
                indices_to_process = list(filtered_orders.index)
                
                for idx in indices_to_process:
                    if idx not in filtered_orders.index:
                        continue  # Already dropped
                    
                    original_qty = filtered_orders.loc[idx, "qty"]
                    symbol = filtered_orders.loc[idx, "symbol"]
                    side = filtered_orders.loc[idx, "side"]
                    
                    # Calculate new qty: scale original qty
                    new_qty = original_qty * scale_factor

                    # Round deterministically: use int() which truncates towards zero
                    # This ensures integer shares for equities
                    # For positive: 20.7 → 20, for negative: -20.7 → -20
                    new_qty = float(int(new_qty))
                    
                    # Preserve sign based on side (qty is always positive in orders, side indicates direction)
                    if side == "SELL":
                        new_qty = -abs(new_qty)
                    else:
                        new_qty = abs(new_qty)

                    if abs(new_qty) < 1e-10:
                        # Order becomes too small, remove it
                        filtered_orders = filtered_orders.drop(index=idx)
                        reduced_orders.append({
                            "reason": "RISK_REDUCE_TURNOVER_CAP",
                            "symbol": symbol,
                            "original_qty": original_qty,
                            "new_qty": 0.0,
                            "explain": {
                                "total_turnover": total_turnover,
                                "cap": config.turnover_cap,
                                "scale_factor": scale_factor,
                            },
                        })
                    else:
                        filtered_orders.loc[idx, "qty"] = new_qty
                        reduced_orders.append({
                            "reason": "RISK_REDUCE_TURNOVER_CAP",
                            "symbol": symbol,
                            "original_qty": original_qty,
                            "new_qty": new_qty,
                            "explain": {
                                "total_turnover": total_turnover,
                                "cap": config.turnover_cap,
                                "scale_factor": scale_factor,
                            },
                        })
        else:
            # Missing required inputs: skip check
            if "price" not in filtered_orders.columns:
                summary["turnover_cap_check"] = "skipped_no_price"
            elif equity is None:
                summary["turnover_cap_check"] = "skipped_no_equity"
            elif equity <= 0.0:
                summary["turnover_cap_check"] = "skipped_equity_zero"

    # 6. Check drawdown de-risking (requires current_equity and peak_equity)
    if config.drawdown_threshold is not None:
        if current_equity is not None and peak_equity is not None and peak_equity > 0.0:
            # Calculate drawdown: drawdown = 1 - current_equity / peak_equity
            drawdown = 1.0 - (current_equity / peak_equity)

            if drawdown >= config.drawdown_threshold:
                # Apply de-risk scale to all orders
                de_risk_scale = config.de_risk_scale
                de_risk_scale = max(0.0, min(1.0, de_risk_scale))  # Clamp to [0, 1]

                # Create a copy of indices to avoid modification during iteration
                indices_to_process = list(filtered_orders.index)

                for idx in indices_to_process:
                    if idx not in filtered_orders.index:
                        continue  # Already dropped

                    original_qty = filtered_orders.loc[idx, "qty"]
                    symbol = filtered_orders.loc[idx, "symbol"]
                    side = filtered_orders.loc[idx, "side"]

                    # Apply de-risk scale
                    new_qty = original_qty * de_risk_scale

                    # Round deterministically: use int() which truncates towards zero
                    new_qty = float(int(new_qty))

                    # Preserve sign based on side
                    if side == "SELL":
                        new_qty = -abs(new_qty)
                    else:
                        new_qty = abs(new_qty)

                    if abs(new_qty) < 1e-10:
                        # Order becomes too small, remove it
                        filtered_orders = filtered_orders.drop(index=idx)
                        reduced_orders.append({
                            "reason": "RISK_DERISK_DRAWDOWN",
                            "symbol": symbol,
                            "original_qty": original_qty,
                            "new_qty": 0.0,
                            "explain": {
                                "drawdown": drawdown,
                                "threshold": config.drawdown_threshold,
                                "de_risk_scale": de_risk_scale,
                                "current_equity": current_equity,
                                "peak_equity": peak_equity,
                            },
                        })
                    else:
                        filtered_orders.loc[idx, "qty"] = new_qty
                        reduced_orders.append({
                            "reason": "RISK_DERISK_DRAWDOWN",
                            "symbol": symbol,
                            "original_qty": original_qty,
                            "new_qty": new_qty,
                            "explain": {
                                "drawdown": drawdown,
                                "threshold": config.drawdown_threshold,
                                "de_risk_scale": de_risk_scale,
                                "current_equity": current_equity,
                                "peak_equity": peak_equity,
                            },
                        })
        else:
            # Missing required inputs: skip check with explicit reason
            if current_equity is None:
                summary["drawdown_derisk_check"] = "skipped_no_current_equity"
            elif peak_equity is None:
                summary["drawdown_derisk_check"] = "skipped_no_peak_equity"
            elif peak_equity <= 0.0:
                summary["drawdown_derisk_check"] = "skipped_peak_equity_zero"

    # 9. Check max_gross_exposure (check on ORIGINAL orders, before filtering)
    if config.max_gross_exposure is not None:
        # Calculate gross exposure from ORIGINAL orders (before symbol filtering)
        # Gross exposure = sum of absolute notionals for all orders
        if "price" in orders.columns:
            # Calculate on original orders
            original_notionals = orders["qty"] * orders["price"]
            gross_exposure = original_notionals.abs().sum()
            summary["gross_exposure"] = gross_exposure

            if gross_exposure > config.max_gross_exposure:
                blocked_reasons.append(
                    f"max_gross_exposure: Gross exposure {gross_exposure:.2f} "
                    f"exceeds limit {config.max_gross_exposure:.2f}"
                )
                # Block all orders if gross exposure exceeded
                filtered_orders = pd.DataFrame(columns=orders.columns)
                orders_with_notional = pd.DataFrame(
                    columns=orders_with_notional.columns
                )
        else:
            # Without prices, cannot calculate gross exposure
            # Skip this check
            summary["gross_exposure_check"] = "skipped_no_prices"

    # Note: Sector/Region/FX exposure checks are implemented in Step 8 above

    # Final summary
    summary["total_orders"] = len(orders)
    summary["passed_orders"] = len(filtered_orders)
    summary["blocked_orders"] = len(orders) - len(filtered_orders)

    # Determine is_ok: True if no reasons to block AND some orders passed
    is_ok = len(blocked_reasons) == 0 and len(filtered_orders) > 0

    # Special case: empty orders originally -> is_ok = True
    if len(orders) == 0:
        is_ok = True

    result = PreTradeCheckResult(
        is_ok=is_ok,
        blocked_reasons=blocked_reasons,
        filtered_orders=filtered_orders,
        summary=summary,
        reduced_orders=reduced_orders,
    )

    return result, filtered_orders
