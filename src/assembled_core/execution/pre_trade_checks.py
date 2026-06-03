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

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)

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
    max_cvar_95: float | None = (
        None  # Max CVaR(95%) as negative fraction (e.g. -0.05 = -5%)
    )
    base_currency: str = "USD"
    missing_security_meta: Literal["raise", "unknown"] = "raise"

    def __post_init__(self) -> None:
        # F-A3-6 R6 fix: sign-convention guard for max_cvar_95.
        # CVaR(95%) is by convention a NEGATIVE number (worst-case loss).
        # A user-supplied positive value (e.g. 0.05 instead of -0.05) would
        # silently zero all BUY orders via cvar_scale = positive/negative → clip(<0) → 0.
        # Reject at construction time so the error surfaces immediately.
        if self.max_cvar_95 is not None and self.max_cvar_95 >= 0:
            raise ValueError(
                f"max_cvar_95 must be None or negative (CVaR convention: "
                f"worst-case loss as negative fraction, e.g. -0.05 = -5%). "
                f"Got: {self.max_cvar_95}. Did you mean {-abs(self.max_cvar_95)}?"
            )


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


# ---------------------------------------------------------------------------
# run_pre_trade_checks helpers (_ptc_*)
# Convention: helpers receive mutable lists/dicts by reference and return
# updated (filtered_orders, orders_with_notional) DataFrames.
# ---------------------------------------------------------------------------


def _ptc_check_max_weight(
    filtered_orders: pd.DataFrame,
    orders_with_notional: pd.DataFrame,
    blocked_reasons: list,
    reduced_orders: list,
    summary: dict,
    config: "PreTradeConfig",
    *,
    current_positions: pd.DataFrame | None,
    prices_latest: pd.DataFrame | None,
    equity: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check max_weight_per_symbol. Modifies blocked_reasons/reduced_orders/summary in-place."""
    if config.max_weight_per_symbol is None:
        return filtered_orders, orders_with_notional
    if current_positions is None or prices_latest is None or equity is None:
        summary["max_weight_per_symbol_check"] = "skipped_missing_inputs"
        return filtered_orders, orders_with_notional

    try:
        from src.assembled_core.risk.exposure_engine import (
            compute_exposures,
            compute_target_positions,
        )
    except ImportError as e:
        logger.error(
            "PRE_TRADE: max_weight_per_symbol check skipped — cannot import exposure_engine: %s. "
            "Blocking all orders (fail-safe).",
            e,
        )
        blocked_reasons.append(
            f"max_weight_per_symbol check failed: cannot import exposure_engine: {e}"
        )
        summary["max_weight_per_symbol_check"] = f"import_error: {e}"
        return pd.DataFrame(columns=filtered_orders.columns), orders_with_notional

    try:
        if current_positions.empty:
            current_positions_df = pd.DataFrame(columns=["symbol", "qty"])
        elif "qty" in current_positions.columns:
            current_positions_df = current_positions[["symbol", "qty"]].copy()
        elif "target_qty" in current_positions.columns:
            current_positions_df = current_positions[["symbol", "target_qty"]].rename(
                columns={"target_qty": "qty"}
            )
        else:
            summary["max_weight_per_symbol_check"] = "skipped_no_qty_column"
            current_positions_df = pd.DataFrame(columns=["symbol", "qty"])

        target_positions = compute_target_positions(
            current_positions_df, filtered_orders
        )
        exposures_df, _ = compute_exposures(
            target_positions,
            prices_latest,
            equity,
            missing_price_handling="raise",
        )

        for _, exposure_row in exposures_df.iterrows():
            symbol = exposure_row["symbol"]
            weight = exposure_row["weight"]
            abs_weight = abs(weight)
            if abs_weight <= config.max_weight_per_symbol:
                continue
            symbol_orders = filtered_orders[filtered_orders["symbol"] == symbol].copy()
            if symbol_orders.empty:
                continue
            current_qty = (
                current_positions_df[current_positions_df["symbol"] == symbol][
                    "qty"
                ].iloc[0]
                if not current_positions_df.empty
                and symbol in current_positions_df["symbol"].values
                else 0.0
            )
            target_qty = exposure_row["target_qty"]
            price = exposure_row["price"]
            if (
                abs(current_qty) > abs(target_qty)
                and abs(current_qty) * price / equity > config.max_weight_per_symbol
            ):
                continue
            max_target_notional = config.max_weight_per_symbol * equity
            max_target_qty = max_target_notional / price if price > 0.0 else 0.0
            max_target_qty = (
                abs(max_target_qty) if target_qty > 0 else -abs(max_target_qty)
            )
            order_delta_needed = max_target_qty - current_qty
            total_order_delta = symbol_orders.apply(
                lambda row: row["qty"] if row["side"] == "BUY" else -row["qty"], axis=1
            ).sum()
            if abs(total_order_delta) <= 1e-10:
                continue
            scale_factor = max(0.0, min(1.0, order_delta_needed / total_order_delta))
            for idx in symbol_orders.index:
                original_qty = filtered_orders.loc[idx, "qty"]
                new_qty = original_qty * scale_factor
                explain = {
                    "current_weight": (
                        abs(current_qty * price / equity) if current_qty != 0.0 else 0.0
                    ),
                    "target_weight": abs_weight,
                    "limit": config.max_weight_per_symbol,
                    "reduction_factor": scale_factor,
                }
                if new_qty < 1e-10:
                    filtered_orders = filtered_orders.drop(index=idx)
                    reduced_orders.append(
                        {
                            "reason": "RISK_REDUCE_MAX_WEIGHT_PER_SYMBOL",
                            "symbol": symbol,
                            "original_qty": original_qty,
                            "new_qty": 0.0,
                            "explain": explain,
                        }
                    )
                else:
                    filtered_orders.loc[idx, "qty"] = new_qty
                    reduced_orders.append(
                        {
                            "reason": "RISK_REDUCE_MAX_WEIGHT_PER_SYMBOL",
                            "symbol": symbol,
                            "original_qty": original_qty,
                            "new_qty": new_qty,
                            "explain": explain,
                        }
                    )
    except ValueError as e:
        blocked_reasons.append(
            f"max_weight_per_symbol check failed: {e}. "
            "Required: current_positions, prices_latest, equity > 0"
        )
        filtered_orders = pd.DataFrame(columns=filtered_orders.columns)
    except Exception as e:
        logger.error(
            "[PreTrade] max_weight_per_symbol check crashed: %s — blocking all orders (fail-safe).",
            e,
        )
        blocked_reasons.append(
            f"max_weight_per_symbol check errored (fail-closed): {e}"
        )
        filtered_orders = pd.DataFrame(columns=filtered_orders.columns)
        summary["max_weight_per_symbol_check"] = f"error_fail_closed: {e}"
    return filtered_orders, orders_with_notional


def _ptc_check_group_exposures(
    filtered_orders: pd.DataFrame,
    orders_with_notional: pd.DataFrame,
    blocked_reasons: list,
    reduced_orders: list,
    summary: dict,
    config: "PreTradeConfig",
    *,
    current_positions: pd.DataFrame | None,
    prices_latest: pd.DataFrame | None,
    equity: float | None,
    security_meta_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check sector/region/FX group exposure limits. Modifies state in-place."""
    if (
        config.max_sector_exposure is None
        and config.max_region_exposure is None
        and config.max_fx_exposure is None
    ):
        return filtered_orders, orders_with_notional

    try:
        from src.assembled_core.risk.exposure_engine import (
            compute_exposures,
            compute_target_positions,
        )
        from src.assembled_core.risk.group_exposures import compute_group_exposures
    except ImportError as e:
        logger.error(
            "PRE_TRADE: group exposure check skipped — cannot import modules: %s. Blocking all orders.",
            e,
        )
        blocked_reasons.append(
            f"Group exposure checks failed: cannot import modules: {e}"
        )
        summary["group_exposure_check"] = f"import_error: {e}"
        return pd.DataFrame(columns=filtered_orders.columns), orders_with_notional

    if not (
        current_positions is not None
        and prices_latest is not None
        and equity is not None
        and equity > 0.0
        and security_meta_df is not None
    ):
        if security_meta_df is None:
            summary["group_exposure_check"] = "skipped_no_security_meta"
        elif current_positions is None:
            summary["group_exposure_check"] = "skipped_no_current_positions"
        elif prices_latest is None:
            summary["group_exposure_check"] = "skipped_no_prices_latest"
        elif equity is None:
            summary["group_exposure_check"] = "skipped_no_equity"
        else:
            summary["group_exposure_check"] = "skipped_equity_zero"
        return filtered_orders, orders_with_notional

    try:
        if current_positions.empty:
            current_positions_df = pd.DataFrame(columns=["symbol", "qty"])
        elif "qty" in current_positions.columns:
            current_positions_df = current_positions[["symbol", "qty"]].copy()
        elif "target_qty" in current_positions.columns:
            current_positions_df = current_positions[["symbol", "target_qty"]].rename(
                columns={"target_qty": "qty"}
            )
        else:
            current_positions_df = pd.DataFrame(columns=["symbol", "qty"])

        # F-A3-2 MAJOR fix (R4): factor out recomputation so we can re-derive
        # exposures_df after each group-check mutates filtered_orders.
        # Previously: exposures_df computed ONCE here, then sector→region→fx
        # all consumed the stale snapshot, even after _apply_group_scale
        # reduced qty in filtered_orders.
        def _recompute_exposures() -> pd.DataFrame:
            tgt = compute_target_positions(current_positions_df, filtered_orders)
            exp, _ = compute_exposures(
                tgt, prices_latest, equity, missing_price_handling="raise"
            )
            return exp

        exposures_df = _recompute_exposures()

        def _apply_group_scale(
            group_type: str,
            group_value: Any,
            gross_weight: float,
            cap: float,
            reason_tag: str,
        ) -> None:
            nonlocal filtered_orders
            group_symbols = security_meta_df[
                security_meta_df[group_type] == group_value
            ]["symbol"].tolist()
            group_orders = filtered_orders[
                filtered_orders["symbol"].isin(group_symbols)
            ].copy()
            if group_orders.empty:
                return
            scale_factor = max(0.0, min(1.0, cap / gross_weight))
            for idx in list(group_orders.index):
                if idx not in filtered_orders.index:
                    continue
                original_qty = filtered_orders.loc[idx, "qty"]
                symbol = filtered_orders.loc[idx, "symbol"]
                side = filtered_orders.loc[idx, "side"]
                new_qty = float(round(original_qty * scale_factor))
                new_qty = -abs(new_qty) if side == "SELL" else abs(new_qty)
                explain = {
                    "group_type": group_type,
                    "group_value": group_value,
                    "cap": cap,
                    "pre_weight": gross_weight,
                    "post_weight": cap,
                    "scale_factor": scale_factor,
                }
                if abs(new_qty) < 1e-10:
                    filtered_orders = filtered_orders.drop(index=idx)
                    reduced_orders.append(
                        {
                            "reason": reason_tag,
                            "symbol": symbol,
                            "original_qty": original_qty,
                            "new_qty": 0.0,
                            "explain": explain,
                        }
                    )
                else:
                    filtered_orders.loc[idx, "qty"] = new_qty
                    reduced_orders.append(
                        {
                            "reason": reason_tag,
                            "symbol": symbol,
                            "original_qty": original_qty,
                            "new_qty": new_qty,
                            "explain": explain,
                        }
                    )

        if config.max_sector_exposure is not None:
            try:
                sector_df, _ = compute_group_exposures(
                    exposures_df, security_meta_df, "sector"
                )
                for _, row in sector_df.iterrows():
                    if row["gross_weight"] > config.max_sector_exposure:
                        _apply_group_scale(
                            "sector",
                            row["group_value"],
                            row["gross_weight"],
                            config.max_sector_exposure,
                            "RISK_REDUCE_MAX_SECTOR_EXPOSURE",
                        )
                # F-A3-2: refresh exposures after sector mutations
                exposures_df = _recompute_exposures()
            except ValueError as e:
                if config.missing_security_meta == "raise":
                    raise ValueError(f"Sector exposure check failed: {e}") from e
                else:
                    summary["sector_exposure_check"] = "skipped_missing_meta"

        if config.max_region_exposure is not None:
            try:
                region_df, _ = compute_group_exposures(
                    exposures_df, security_meta_df, "region"
                )
                for _, row in region_df.iterrows():
                    if row["gross_weight"] > config.max_region_exposure:
                        _apply_group_scale(
                            "region",
                            row["group_value"],
                            row["gross_weight"],
                            config.max_region_exposure,
                            "RISK_REDUCE_MAX_REGION_EXPOSURE",
                        )
                # F-A3-2: refresh exposures after region mutations
                exposures_df = _recompute_exposures()
            except ValueError as e:
                if config.missing_security_meta == "raise":
                    raise ValueError(f"Region exposure check failed: {e}") from e
                else:
                    summary["region_exposure_check"] = "skipped_missing_meta"

        if config.max_fx_exposure is not None:
            try:
                if "currency" not in security_meta_df.columns:
                    if config.missing_security_meta == "raise":
                        raise ValueError(
                            "FX exposure check requires currency column in security_meta_df"
                        )
                    else:
                        summary["fx_exposure_check"] = "skipped_missing_currency"
                else:
                    non_base = (
                        security_meta_df[
                            security_meta_df["currency"] != config.base_currency
                        ]["currency"]
                        .unique()
                        .tolist()
                    )
                    if non_base:
                        if config.missing_security_meta == "raise":
                            raise ValueError(
                                f"FX exposure needs currency mapping + FX rates (not implemented). "
                                f"Non-base currencies found: {non_base}. Base currency: {config.base_currency}."
                            )
                        else:
                            summary["fx_exposure_check"] = "skipped_non_base_currency"
                    else:
                        summary["fx_exposure_check"] = "all_base_currency"
            except ValueError as e:
                if config.missing_security_meta == "raise":
                    raise ValueError(f"FX exposure check failed: {e}") from e
                else:
                    summary["fx_exposure_check"] = "skipped_missing_meta"

    except ValueError as e:
        if config.missing_security_meta == "raise":
            raise
        else:
            blocked_reasons.append(f"Group exposure checks failed: {e}")
            summary["group_exposure_check"] = f"error: {e}"
    except Exception as e:
        blocked_reasons.append(f"Group exposure checks failed: {e}")
        summary["group_exposure_check"] = f"error: {e}"
    return filtered_orders, orders_with_notional


def _ptc_check_turnover(
    filtered_orders: pd.DataFrame,
    orders_with_notional: pd.DataFrame,
    blocked_reasons: list,
    reduced_orders: list,
    summary: dict,
    config: "PreTradeConfig",
    *,
    equity: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check turnover_cap. Modifies reduced_orders/summary in-place."""
    if config.turnover_cap is None:
        return filtered_orders, orders_with_notional
    if "price" not in filtered_orders.columns or equity is None or equity <= 0.0:
        if "price" not in filtered_orders.columns:
            summary["turnover_cap_check"] = "skipped_no_price"
        elif equity is None:
            summary["turnover_cap_check"] = "skipped_no_equity"
        else:
            summary["turnover_cap_check"] = "skipped_equity_zero"
        return filtered_orders, orders_with_notional

    order_notionals = filtered_orders["qty"].abs() * filtered_orders["price"].abs()
    total_turnover = float(order_notionals.sum() / equity)
    if total_turnover <= config.turnover_cap:
        return filtered_orders, orders_with_notional

    scale_factor = max(0.0, min(1.0, config.turnover_cap / total_turnover))
    for idx in list(filtered_orders.index):
        if idx not in filtered_orders.index:
            continue
        original_qty = filtered_orders.loc[idx, "qty"]
        symbol = filtered_orders.loc[idx, "symbol"]
        side = filtered_orders.loc[idx, "side"]
        new_qty = float(round(original_qty * scale_factor))
        new_qty = -abs(new_qty) if side == "SELL" else abs(new_qty)
        explain = {
            "total_turnover": total_turnover,
            "cap": config.turnover_cap,
            "scale_factor": scale_factor,
        }
        if abs(new_qty) < 1e-10:
            filtered_orders = filtered_orders.drop(index=idx)
            reduced_orders.append(
                {
                    "reason": "RISK_REDUCE_TURNOVER_CAP",
                    "symbol": symbol,
                    "original_qty": original_qty,
                    "new_qty": 0.0,
                    "explain": explain,
                }
            )
        else:
            filtered_orders.loc[idx, "qty"] = new_qty
            reduced_orders.append(
                {
                    "reason": "RISK_REDUCE_TURNOVER_CAP",
                    "symbol": symbol,
                    "original_qty": original_qty,
                    "new_qty": new_qty,
                    "explain": explain,
                }
            )
    return filtered_orders, orders_with_notional


def _ptc_check_drawdown(
    filtered_orders: pd.DataFrame,
    orders_with_notional: pd.DataFrame,
    blocked_reasons: list,
    reduced_orders: list,
    summary: dict,
    config: "PreTradeConfig",
    *,
    current_equity: float | None,
    peak_equity: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check drawdown de-risking. Modifies reduced_orders/summary in-place."""
    if config.drawdown_threshold is None:
        return filtered_orders, orders_with_notional
    if current_equity is None or peak_equity is None or peak_equity <= 0.0:
        if current_equity is None:
            summary["drawdown_derisk_check"] = "skipped_no_current_equity"
        elif peak_equity is None:
            summary["drawdown_derisk_check"] = "skipped_no_peak_equity"
        else:
            summary["drawdown_derisk_check"] = "skipped_peak_equity_zero"
        return filtered_orders, orders_with_notional

    drawdown = 1.0 - (current_equity / peak_equity)
    if drawdown < config.drawdown_threshold:
        return filtered_orders, orders_with_notional

    de_risk_scale = max(0.0, min(1.0, config.de_risk_scale))
    for idx in list(filtered_orders.index):
        if idx not in filtered_orders.index:
            continue
        original_qty = filtered_orders.loc[idx, "qty"]
        symbol = filtered_orders.loc[idx, "symbol"]
        side = filtered_orders.loc[idx, "side"]
        new_qty = float(round(original_qty * de_risk_scale))
        new_qty = -abs(new_qty) if side == "SELL" else abs(new_qty)
        explain = {
            "drawdown": drawdown,
            "threshold": config.drawdown_threshold,
            "de_risk_scale": de_risk_scale,
            "current_equity": current_equity,
            "peak_equity": peak_equity,
        }
        if abs(new_qty) < 1e-10:
            filtered_orders = filtered_orders.drop(index=idx)
            reduced_orders.append(
                {
                    "reason": "RISK_DERISK_DRAWDOWN",
                    "symbol": symbol,
                    "original_qty": original_qty,
                    "new_qty": 0.0,
                    "explain": explain,
                }
            )
        else:
            filtered_orders.loc[idx, "qty"] = new_qty
            reduced_orders.append(
                {
                    "reason": "RISK_DERISK_DRAWDOWN",
                    "symbol": symbol,
                    "original_qty": original_qty,
                    "new_qty": new_qty,
                    "explain": explain,
                }
            )
    return filtered_orders, orders_with_notional


def _ptc_check_gross_exposure(
    filtered_orders: pd.DataFrame,
    orders_with_notional: pd.DataFrame,
    blocked_reasons: list,
    summary: dict,
    config: "PreTradeConfig",
    *,
    orders_original: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check max_gross_exposure against original orders. Modifies blocked_reasons/summary in-place."""
    if config.max_gross_exposure is None:
        return filtered_orders, orders_with_notional
    if "price" not in orders_original.columns:
        summary["gross_exposure_check"] = "skipped_no_prices"
        return filtered_orders, orders_with_notional

    gross_exposure = (orders_original["qty"] * orders_original["price"]).abs().sum()
    summary["gross_exposure"] = gross_exposure
    if gross_exposure > config.max_gross_exposure:
        blocked_reasons.append(
            f"max_gross_exposure: Gross exposure {gross_exposure:.2f} "
            f"exceeds limit {config.max_gross_exposure:.2f}"
        )
        empty_df = pd.DataFrame(columns=orders_original.columns)
        return empty_df, pd.DataFrame(columns=orders_with_notional.columns)
    return filtered_orders, orders_with_notional


def _ptc_check_cvar(
    filtered_orders: pd.DataFrame,
    orders_with_notional: pd.DataFrame,
    blocked_reasons: list,
    reduced_orders: list,
    summary: dict,
    config: "PreTradeConfig",
    *,
    risk_summary: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check CVaR limit. Scales down BUY orders if portfolio CVaR exceeds limit."""
    if (
        config.max_cvar_95 is None
        or risk_summary is None
        or risk_summary.get("cvar_95") is None
    ):
        return filtered_orders, orders_with_notional

    portfolio_cvar = float(risk_summary["cvar_95"])
    # CVaR values are negative: -0.07 means 7% tail loss (worse than -0.05 limit).
    # Trigger scaling when portfolio_cvar < config.max_cvar_95 (i.e. worse than limit).
    if portfolio_cvar >= config.max_cvar_95:
        # Within limit — no action required.
        return filtered_orders, orders_with_notional

    # portfolio_cvar == 0.0 means no positions (cash-only) → no tail risk → pass through.
    # Also guards against division by zero.
    if portfolio_cvar == 0.0:
        return filtered_orders, orders_with_notional

    # Scale = limit / actual (both negative, so ratio is positive and < 1 when breached).
    # E.g. -0.05 / -0.07 ≈ 0.714 → reduce BUY orders to 71.4% of requested size.
    cvar_scale = config.max_cvar_95 / portfolio_cvar
    cvar_scale = max(0.0, min(cvar_scale, 1.0))
    if (
        cvar_scale < 1.0
        and not filtered_orders.empty
        and "qty" in filtered_orders.columns
    ):
        buy_mask = filtered_orders["side"] == "BUY"
        filtered_orders.loc[buy_mask, "qty"] = (
            filtered_orders.loc[buy_mask, "qty"] * cvar_scale
        )
        for _, row in filtered_orders[buy_mask].iterrows():
            reduced_orders.append(
                {
                    "symbol": row.get("symbol", "?"),
                    "reason": "RISK_REDUCE_CVAR_95",
                    "details": {
                        "portfolio_cvar_95": round(portfolio_cvar, 4),
                        "limit": config.max_cvar_95,
                        "scale": round(cvar_scale, 4),
                    },
                }
            )
    summary["cvar_95"] = portfolio_cvar
    summary["cvar_95_limit"] = config.max_cvar_95
    return filtered_orders, orders_with_notional


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

    blocked_reasons: list = []
    reduced_orders: list = []
    filtered_orders = orders.copy()
    summary: dict = {}

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

    if qa_status is not None:
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

    orders_with_notional = filtered_orders.copy()
    if "price" in filtered_orders.columns:
        orders_with_notional["notional"] = (
            filtered_orders["qty"] * filtered_orders["price"]
        ).abs()
    else:
        if (
            config.max_notional_per_symbol is not None
            or config.max_gross_exposure is not None
        ):
            blocked_reasons.append(
                "pre_trade_missing_price: notional/gross-exposure caps configured "
                "but 'price' column missing from orders — blocking all orders (fail-safe)."
            )
            return (
                PreTradeCheckResult(
                    is_ok=False,
                    blocked_reasons=blocked_reasons,
                    filtered_orders=pd.DataFrame(columns=orders.columns),
                    summary={
                        "total_orders": len(orders),
                        "passed_orders": 0,
                        "pre_trade_missing_price": True,
                    },
                ),
                pd.DataFrame(columns=orders.columns),
            )
        orders_with_notional["notional"] = 0.0

    # max_notional_per_symbol
    if (
        config.max_notional_per_symbol is not None
        and "price" in filtered_orders.columns
    ):
        symbol_notionals = (
            orders_with_notional.groupby("symbol")["notional"].sum().abs()
        )
        exceeded = symbol_notionals[symbol_notionals > config.max_notional_per_symbol]
        for symbol, notional in exceeded.items():
            blocked_reasons.append(
                f"max_notional_per_symbol: Symbol '{symbol}' has notional "
                f"{notional:.2f}, exceeds limit {config.max_notional_per_symbol:.2f}"
            )
            filtered_orders = filtered_orders[filtered_orders["symbol"] != symbol]
            orders_with_notional = orders_with_notional[
                orders_with_notional["symbol"] != symbol
            ]

    filtered_orders, orders_with_notional = _ptc_check_max_weight(
        filtered_orders,
        orders_with_notional,
        blocked_reasons,
        reduced_orders,
        summary,
        config,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
    )
    filtered_orders, orders_with_notional = _ptc_check_group_exposures(
        filtered_orders,
        orders_with_notional,
        blocked_reasons,
        reduced_orders,
        summary,
        config,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        security_meta_df=security_meta_df,
    )
    filtered_orders, orders_with_notional = _ptc_check_turnover(
        filtered_orders,
        orders_with_notional,
        blocked_reasons,
        reduced_orders,
        summary,
        config,
        equity=equity,
    )
    filtered_orders, orders_with_notional = _ptc_check_drawdown(
        filtered_orders,
        orders_with_notional,
        blocked_reasons,
        reduced_orders,
        summary,
        config,
        current_equity=current_equity,
        peak_equity=peak_equity,
    )
    filtered_orders, orders_with_notional = _ptc_check_gross_exposure(
        filtered_orders,
        orders_with_notional,
        blocked_reasons,
        summary,
        config,
        orders_original=orders,
    )
    filtered_orders, orders_with_notional = _ptc_check_cvar(
        filtered_orders,
        orders_with_notional,
        blocked_reasons,
        reduced_orders,
        summary,
        config,
        risk_summary=risk_summary,
    )

    summary["total_orders"] = len(orders)
    summary["passed_orders"] = len(filtered_orders)
    summary["blocked_orders"] = len(orders) - len(filtered_orders)

    is_ok = len(blocked_reasons) == 0 and len(filtered_orders) > 0
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


# ---------------------------------------------------------------------------
# ADV Cap Utility (market microstructure — standalone, does not modify above)
# ---------------------------------------------------------------------------


def apply_adv_cap(
    orders: pd.DataFrame,
    prices: pd.DataFrame,
    adv_fraction: float = 0.10,
    adv_window: int = 20,
    amihud_threshold: float | None = None,
    symbol_col: str = "symbol",
    qty_col: str = "quantity",
    side_col: str = "side",
    timestamp_col: str = "timestamp",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Cap order quantities to a fraction of Average Daily Volume (ADV).

    Optionally applies the cap only to symbols above an Amihud illiquidity
    threshold (when ``amihud_threshold`` is set and prices contains
    ``amihud_illiq_{adv_window}d`` from ``add_amihud_illiquidity``).

    Args:
        orders: Orders DataFrame with at least symbol, quantity, side columns.
        prices: Daily OHLCV panel used to compute ADV (rolling mean of volume).
        adv_fraction: Maximum order size as fraction of ADV (default: 0.10 = 10%).
        adv_window: Look-back window in days for ADV calculation (default: 20).
        amihud_threshold: If set, cap is only applied to symbols where the most
            recent Amihud illiquidity exceeds this value (default: None = cap all).
        symbol_col: Symbol column name in orders and prices (default: "symbol").
        qty_col: Quantity column name in orders (default: "quantity").
        side_col: Side column name (default: "side"). BUY/SELL both capped.
        timestamp_col: Timestamp column name in prices (default: "timestamp").
        volume_col: Volume column name in prices (default: "volume").

    Returns:
        Orders DataFrame with quantities capped. A new column ``adv_cap_applied``
        (bool) is added to indicate which orders were actually reduced.
    """
    if orders.empty or prices.empty:
        if not orders.empty:
            orders = orders.copy()
            orders["adv_cap_applied"] = False
        return orders

    orders = orders.copy()
    orders["adv_cap_applied"] = False

    prices_ts = prices.copy()
    prices_ts[timestamp_col] = pd.to_datetime(prices_ts[timestamp_col])

    # Compute ADV per symbol: rolling mean of volume over adv_window days
    adv_map: dict = {}
    amihud_map: dict = {}
    amihud_col = f"amihud_illiq_{adv_window}d"

    for sym, grp in prices_ts.groupby(symbol_col):
        grp = grp.sort_values(timestamp_col)
        if volume_col in grp.columns and len(grp) >= 1:
            adv_val = grp[volume_col].tail(adv_window).mean()
            adv_map[sym] = float(adv_val) if pd.notna(adv_val) else None
        if amihud_threshold is not None and amihud_col in grp.columns:
            last_val = grp[amihud_col].dropna()
            amihud_map[sym] = float(last_val.iloc[-1]) if len(last_val) > 0 else None

    for idx, row in orders.iterrows():
        sym = row.get(symbol_col)
        adv = adv_map.get(sym)
        if adv is None or adv <= 0:
            continue

        # Skip if amihud_threshold set and symbol is liquid enough
        if amihud_threshold is not None:
            illiq = amihud_map.get(sym)
            if illiq is not None and illiq < amihud_threshold:
                continue

        max_qty = adv * adv_fraction
        current_qty = abs(float(row[qty_col]))
        if current_qty > max_qty:
            sign = 1 if float(row[qty_col]) >= 0 else -1
            orders.at[idx, qty_col] = sign * max_qty
            orders.at[idx, "adv_cap_applied"] = True

    logger.debug(
        "[ADV Cap] adv_fraction=%.2f window=%dd — %d/%d orders capped",
        adv_fraction,
        adv_window,
        int(orders["adv_cap_applied"].sum()),
        len(orders),
    )
    return orders


# ---------------------------------------------------------------------------
# Gawande-style explicit-raise gate (audit C2-070)
# ---------------------------------------------------------------------------


class PreTradeGateBlocked(RuntimeError):
    """Raised by ``pre_trade_gate`` when any must-pass check refuses orders.

    Holds the list of blocking reasons in ``reasons`` for the caller and an
    optional ``check`` field naming which guard tripped first.
    """

    def __init__(
        self,
        message: str,
        *,
        reasons: list[str] | None = None,
        check: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reasons = reasons or []
        self.check = check


def pre_trade_gate(
    orders: "pd.DataFrame",
    *,
    config: PreTradeConfig | None = None,
    portfolio: "pd.DataFrame | None" = None,
    qa_status: Any | None = None,
    enforce_kill_switch: bool = True,
    **run_kwargs: Any,
) -> "pd.DataFrame":
    """Single-call must-pass pre-trade gate (Gawande-checklist pattern).

    Wraps :func:`run_pre_trade_checks` plus an explicit kill-switch read and
    converts ``is_ok=False`` into a named exception (``PreTradeGateBlocked``)
    instead of returning a result struct the caller might forget to inspect.

    This is the *opinionated* path for production submission pipelines —
    research/notebooks should keep using ``run_pre_trade_checks`` directly.

    Args:
        orders: orders DataFrame as accepted by ``run_pre_trade_checks``.
        config: PreTradeConfig — same semantics as the underlying function.
        portfolio, qa_status, **run_kwargs: forwarded.
        enforce_kill_switch: if True (default), the gate also rejects when
            the kill-switch is engaged with ``throttle_pct == 0`` — caller
            does not need a second read.

    Returns:
        Filtered/reduced orders DataFrame (only orders that passed).

    Raises:
        PreTradeGateBlocked: any blocking reason → caller MUST handle.
    """
    if enforce_kill_switch:
        try:
            from src.assembled_core.execution.kill_switch import (
                _persistent_state_corrupt,
                get_kill_switch_state,
            )

            # Fail CLOSED on a corrupt/unreadable-but-PRESENT persistent state
            # file, BEFORE the healthy throttle read below. ``get_kill_switch_state``
            # flattens a corrupt file to ``engaged=False, throttle_pct=1.0`` (via
            # ``_read_state() -> {}``), so reading ``state["engaged"]`` here would
            # see False and let corrupt-present state fail OPEN. A present-yet-broken
            # file means the switch state is UNKNOWN — block conservatively, matching
            # ``is_kill_switch_engaged()``'s fail-closed contract. A legitimately
            # MISSING file is NOT corrupt and stays disengaged (handled below).
            if _persistent_state_corrupt():
                logger.error(
                    "[PreTradeGate] Persistent kill-switch state file is PRESENT "
                    "but unreadable/corrupt — switch state is UNKNOWN, blocking "
                    "orders conservatively (fail-closed / treated as ENGAGED)."
                )
                raise PreTradeGateBlocked(
                    "kill-switch state corrupt (present-but-unreadable)",
                    reasons=["kill_switch_engaged"],
                    check="kill_switch",
                )

            state = get_kill_switch_state()
            if state["engaged"] and state["throttle_pct"] <= 0.0:
                raise PreTradeGateBlocked(
                    "kill-switch fully engaged (throttle_pct=0)",
                    reasons=["kill_switch_engaged"],
                    check="kill_switch",
                )
        except PreTradeGateBlocked:
            raise
        except Exception as exc:  # noqa: BLE001
            # kill-switch read MUST not silently no-op — surface as gate block.
            raise PreTradeGateBlocked(
                f"kill-switch state read failed: {exc}",
                reasons=[f"kill_switch_unreadable: {exc}"],
                check="kill_switch",
            ) from exc

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=portfolio,
        qa_status=qa_status,
        config=config,
        **run_kwargs,
    )
    if not result.is_ok:
        first = result.blocked_reasons[0] if result.blocked_reasons else "blocked"
        raise PreTradeGateBlocked(
            f"pre-trade checks blocked orders: {first}",
            reasons=list(result.blocked_reasons),
            check="run_pre_trade_checks",
        )
    return filtered


# ---------------------------------------------------------------------------
# Pre-Trade Stress Test (Plan 7.2)
# ---------------------------------------------------------------------------


@dataclass
class StressScenario:
    """A pre-trade stress scenario."""

    name: str
    equity_shock: float = 0.0  # e.g. -0.05 = -5%
    vix_shock_pct: float = 0.0  # e.g. 0.50 = +50%
    credit_spread_bps: float = 0.0  # e.g. 200 = +200bps
    rate_shock_bps: float = 0.0  # e.g. 50 = +50bps


DEFAULT_STRESS_SCENARIOS = [
    StressScenario("mild_selloff", equity_shock=-0.05, vix_shock_pct=0.50),
    StressScenario(
        "moderate_crash", equity_shock=-0.10, vix_shock_pct=1.00, credit_spread_bps=200
    ),
    StressScenario(
        "severe_crisis", equity_shock=-0.20, vix_shock_pct=2.00, credit_spread_bps=500
    ),
    StressScenario("rate_shock", equity_shock=-0.03, rate_shock_bps=100),
]


def run_pre_trade_stress_test(
    portfolio_weights: dict[str, float],
    portfolio_value: float,
    betas: dict[str, float] | None = None,
    scenarios: list[StressScenario] | None = None,
    max_stress_loss_pct: float = 0.15,
) -> dict:
    """Run stress tests on proposed portfolio before trade execution.

    For each scenario, estimates portfolio loss using:
    ``loss_i = weight_i × beta_i × equity_shock``

    If any scenario's loss > max_stress_loss_pct → block BUY orders.

    Args:
        portfolio_weights: Symbol → weight.
        portfolio_value: Total portfolio value in dollars.
        betas: Symbol → market beta. Defaults to 1.0 for all.
        scenarios: List of stress scenarios. Uses defaults if None.
        max_stress_loss_pct: Maximum acceptable loss (default 15%).

    Returns:
        Dict with ``passed``, ``worst_scenario``, ``worst_loss_pct``,
        ``scenario_results`` list.
    """
    scenarios = scenarios or DEFAULT_STRESS_SCENARIOS
    betas = betas or {}

    results = []
    worst_loss = 0.0
    worst_name = ""

    for sc in scenarios:
        portfolio_loss = 0.0
        for sym, w in portfolio_weights.items():
            beta = betas.get(sym, 1.0)
            # Equity component
            sym_loss = w * beta * sc.equity_shock
            portfolio_loss += sym_loss

        loss_pct = abs(portfolio_loss)
        results.append(
            {
                "scenario": sc.name,
                "loss_pct": round(loss_pct, 6),
                "loss_dollars": round(loss_pct * portfolio_value, 2),
                "breached": loss_pct > max_stress_loss_pct,
            }
        )

        if loss_pct > worst_loss:
            worst_loss = loss_pct
            worst_name = sc.name

    passed = all(not r["breached"] for r in results)

    if not passed:
        logger.warning(
            "[StressTest] FAILED — worst scenario '%s' loss %.1f%% > limit %.1f%%",
            worst_name,
            worst_loss * 100,
            max_stress_loss_pct * 100,
        )

    return {
        "passed": passed,
        "worst_scenario": worst_name,
        "worst_loss_pct": round(worst_loss, 6),
        "scenario_results": results,
    }
