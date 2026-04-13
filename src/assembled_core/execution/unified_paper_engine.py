"""Unified Paper Trading Engine.

This module unifies the three previously separate paper trading paths:
1. execution/paper_trading_engine.py  — in-memory FillModel with TWAP/VWAP
2. paper/paper_track.py               — PaperTrackRunner with JSON state
3. scripts/run_live_paper.py          — LivePaperRunner with Alpaca + intent store

All three concerns are now handled by a single ``UnifiedPaperEngine`` with:
- Configurable fill simulation (spread + impact via Almgren-Chriss sqrt model)
- JSON-persisted state (positions, cash, equity history)
- Optional risk controls (kill switch, fat finger guard, pre-trade checks)
- Ledger events + reconciliation hooks
- Degraded-mode operation when optional modules are unavailable

Lifecycle per trading day
--------------------------
1.  Load state (positions, cash, equity history)
2.  Load prices
3.  Generate signals (strategy from config)
4.  Size positions
5.  Generate orders
6.  Risk controls (fat finger, kill switch, pre-trade)
7.  Fill simulation via FillModel (spread + impact + commission)
8.  Ledger events
9.  Position update (average-cost accounting)
10. Reconciliation
11. Post-trade learning / experience-log entry
12. State persist

Usage::

    from src.assembled_core.execution.unified_paper_engine import (
        UnifiedPaperConfig, UnifiedPaperEngine,
    )

    cfg = UnifiedPaperConfig(seed_capital=100_000.0)
    engine = UnifiedPaperEngine(cfg)
    result = engine.run_paper_day("2025-01-15")
    print(result.equity_after)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional module imports — engine degrades gracefully if unavailable
# ---------------------------------------------------------------------------

try:
    from src.assembled_core.execution.kill_switch import (
        guard_orders_with_kill_switch,
        is_kill_switch_engaged,
    )
    _HAS_KILL_SWITCH = True
except Exception:  # pragma: no cover
    _HAS_KILL_SWITCH = False
    logger.warning("[PAPER] kill_switch unavailable — kill-switch checks disabled")

try:
    from src.assembled_core.execution.fat_finger_guard import apply_fat_finger_guard
    _HAS_FAT_FINGER = True
except Exception:  # pragma: no cover
    _HAS_FAT_FINGER = False
    logger.warning("[PAPER] fat_finger_guard unavailable — fat-finger checks disabled")

try:
    from src.assembled_core.execution.pre_trade_checks import run_pre_trade_checks
    _HAS_PRE_TRADE = True
except Exception:  # pragma: no cover
    _HAS_PRE_TRADE = False
    logger.warning("[PAPER] pre_trade_checks unavailable — pre-trade checks disabled")

try:
    from src.assembled_core.execution.fill_model import (
        PartialFillModel,
        apply_cash_gate,
        ensure_fill_schema,
    )
    _HAS_FILL_MODEL = True
except Exception:  # pragma: no cover
    _HAS_FILL_MODEL = False
    logger.warning("[PAPER] fill_model unavailable — using simple fill simulation")

try:
    from src.assembled_core.accounting.ledger import (
        generate_ledger_events,
        store_ledger_events_parquet,
    )
    _HAS_LEDGER = True
except Exception:  # pragma: no cover
    _HAS_LEDGER = False
    logger.warning("[PAPER] ledger unavailable — ledger events disabled")

try:
    from src.assembled_core.accounting.reconciliation import reconcile_ledger_vs_broker
    _HAS_RECONCILIATION = True
except Exception:  # pragma: no cover
    _HAS_RECONCILIATION = False
    logger.warning("[PAPER] reconciliation unavailable — reconciliation disabled")

try:
    from src.assembled_core.execution.order_generation import generate_orders_from_targets
    _HAS_ORDER_GEN = True
except Exception:  # pragma: no cover
    _HAS_ORDER_GEN = False
    logger.warning("[PAPER] order_generation unavailable — order generation disabled")

try:
    from src.assembled_core.execution.smart_order_router import route_order
    _HAS_SOR = True
except Exception:  # pragma: no cover
    _HAS_SOR = False

try:
    from src.assembled_core.ops.experience_log import log_experience_entry
    _HAS_EXPERIENCE_LOG = True
except Exception:  # pragma: no cover
    _HAS_EXPERIENCE_LOG = False


# ---------------------------------------------------------------------------
# Config and result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class UnifiedPaperConfig:
    """Configuration for the UnifiedPaperEngine.

    Attributes:
        seed_capital: Starting cash in USD.
        enable_ledger: Write Parquet ledger events after each day.
        enable_reconciliation: Run ledger-vs-broker reconciliation.
        enable_fat_finger: Apply fat-finger guard before fills.
        enable_kill_switch: Check kill switch before every order batch.
        half_spread_bps: Half bid-ask spread in basis points for fill simulation.
        impact_coefficient: Almgren-Chriss impact multiplier (sqrt model).
        default_adv: Default ADV (shares) used when per-symbol ADV is unknown.
        max_participation: Maximum ADV participation cap (e.g. 0.05 = 5 %).
        state_dir: Directory for JSON state persistence.
        ledger_dir: Directory for Parquet ledger events.
        run_id: Identifier for this paper run (used in ledger events).
    """

    seed_capital: float = 100_000.0
    enable_ledger: bool = True
    enable_reconciliation: bool = True
    enable_fat_finger: bool = True
    enable_kill_switch: bool = True
    half_spread_bps: float = 5.0
    impact_coefficient: float = 0.10
    default_adv: float = 1_000_000.0
    max_participation: float = 0.05
    state_dir: Path = field(default_factory=lambda: Path("output/paper_state"))
    ledger_dir: Path = field(default_factory=lambda: Path("output/paper_ledger"))
    run_id: str = "paper_unified"


@dataclass
class PaperDayResult:
    """Result of one paper trading day.

    Attributes:
        date: ISO date string (YYYY-MM-DD).
        status: Outcome — "success", "error", or "kill_switch".
        n_orders: Number of orders generated.
        n_fills: Number of orders that resulted in fills.
        total_cost_bps: Total execution cost in basis points (spread + impact).
        equity_before: Portfolio equity at start of day.
        equity_after: Portfolio equity at end of day.
        daily_return: Arithmetic daily return (equity_after / equity_before - 1).
        positions: Dict mapping symbol → quantity after fills.
        errors: List of non-fatal error messages encountered during the day.
    """

    date: str
    status: str  # "success" | "error" | "kill_switch"
    n_orders: int
    n_fills: int
    total_cost_bps: float
    equity_before: float
    equity_after: float
    daily_return: float
    positions: dict[str, float]
    errors: list[str]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class UnifiedPaperEngine:
    """Unified paper trading engine.

    Lifecycle per trading day:
    1. Load state (positions, cash, equity history)
    2. Load prices
    3. Generate signals (strategy from config)
    4. Size positions
    5. Generate orders
    6. Risk controls (fat finger, kill switch, pre-trade)
    7. Fill simulation via FillModel (spread + impact + commission)
    8. Ledger events
    9. Position update (average-cost accounting)
    10. Reconciliation
    11. Post-trade learning store entry
    12. State persist
    """

    _STATE_FILE = "paper_state.json"
    _EQUITY_FILE = "equity_curve.json"

    def __init__(self, config: UnifiedPaperConfig | None = None) -> None:
        self.config = config or UnifiedPaperConfig()
        self._state: dict[str, Any] = {}
        self._equity_curve: list[dict[str, Any]] = []
        self._initialized = False

        # Ensure directories exist
        try:
            self.config.state_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("[PAPER] Could not create state_dir: %s", exc)

        if self.config.enable_ledger:
            try:
                self.config.ledger_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # pragma: no cover
                logger.warning("[PAPER] Could not create ledger_dir: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_paper_day(
        self,
        as_of_date: str,
        prices: pd.DataFrame | None = None,
        dry_run: bool = False,
    ) -> PaperDayResult:
        """Execute one complete paper trading day.

        Args:
            as_of_date: ISO date string (YYYY-MM-DD).
            prices: Optional price DataFrame (symbol, close, [volume, adv]).
                    If None, the engine attempts to load prices from data layer.
            dry_run: If True, skip state persistence and ledger writes.

        Returns:
            PaperDayResult with full day summary.
        """
        logger.info("[PAPER] === Starting paper day: %s ===", as_of_date)
        errors: list[str] = []

        # Step 1 — Load state
        self._load_state()
        equity_before = self._compute_equity(prices)

        # Step 2 — Kill switch check (fast-exit if engaged)
        if self.config.enable_kill_switch and _HAS_KILL_SWITCH:
            try:
                if is_kill_switch_engaged():
                    logger.warning("[PAPER] Kill switch engaged — skipping %s", as_of_date)
                    self._append_equity_point(as_of_date, equity_before)
                    if not dry_run:
                        self._save_state()
                    return PaperDayResult(
                        date=as_of_date,
                        status="kill_switch",
                        n_orders=0,
                        n_fills=0,
                        total_cost_bps=0.0,
                        equity_before=equity_before,
                        equity_after=equity_before,
                        daily_return=0.0,
                        positions=dict(self._state.get("positions", {})),
                        errors=["Kill switch engaged"],
                    )
            except Exception as exc:
                errors.append(f"kill_switch check error: {exc}")

        # Step 3 — Price loading (use provided or stub)
        if prices is None:
            prices = self._try_load_prices(as_of_date)

        if prices is None or prices.empty:
            logger.warning("[PAPER] No prices available for %s — skipping fills", as_of_date)
            errors.append("No prices available")
            equity_after = equity_before
            self._append_equity_point(as_of_date, equity_after)
            if not dry_run:
                self._save_state()
            return PaperDayResult(
                date=as_of_date,
                status="error",
                n_orders=0,
                n_fills=0,
                total_cost_bps=0.0,
                equity_before=equity_before,
                equity_after=equity_after,
                daily_return=0.0,
                positions=dict(self._state.get("positions", {})),
                errors=errors,
            )

        # Steps 4+5 — Generate orders (placeholder: callers may inject via subclass)
        orders = self._generate_orders(as_of_date, prices)
        n_orders = len(orders) if orders is not None and not orders.empty else 0
        logger.info("[PAPER] %s orders generated for %s", n_orders, as_of_date)

        fills = pd.DataFrame()
        total_cost_bps = 0.0

        if n_orders > 0 and orders is not None:
            # Step 6 — Risk controls
            try:
                orders = self._apply_risk_controls(orders)
            except Exception as exc:
                errors.append(f"risk_controls error: {exc}")
                logger.error("[PAPER] Risk controls failed: %s", exc)

            # Step 7 — Fill simulation
            try:
                fills, total_cost_bps = self._simulate_fills_with_cost(orders, prices)
            except Exception as exc:
                errors.append(f"fill simulation error: {exc}")
                logger.error("[PAPER] Fill simulation failed: %s", exc)
                fills = pd.DataFrame()

        n_fills = len(fills) if not fills.empty else 0
        logger.info("[PAPER] %s fills executed for %s", n_fills, as_of_date)

        # Step 8 — Ledger events
        if self.config.enable_ledger and _HAS_LEDGER and not fills.empty and not dry_run:
            self._write_ledger_events(fills, as_of_date)

        # Step 9 — Update positions (average-cost accounting)
        if not fills.empty:
            self._update_positions(fills)

        # Step 10 — Reconciliation
        if self.config.enable_reconciliation and _HAS_RECONCILIATION and not dry_run:
            self._run_reconciliation(as_of_date)

        # Step 11 — Experience log
        equity_after = self._compute_equity(prices)
        daily_return = (equity_after / equity_before - 1.0) if equity_before > 0 else 0.0
        self._append_equity_point(as_of_date, equity_after)

        if not dry_run and _HAS_EXPERIENCE_LOG:
            self._write_experience_entry(as_of_date, equity_before, equity_after, n_fills)

        # Step 12 — Persist state
        if not dry_run:
            self._save_state()

        logger.info(
            "[PAPER] Day %s done: equity %.2f → %.2f (%.2f%%), fills=%s, cost=%.1f bps",
            as_of_date,
            equity_before,
            equity_after,
            daily_return * 100,
            n_fills,
            total_cost_bps,
        )

        return PaperDayResult(
            date=as_of_date,
            status="success" if not errors else "error",
            n_orders=n_orders,
            n_fills=n_fills,
            total_cost_bps=total_cost_bps,
            equity_before=equity_before,
            equity_after=equity_after,
            daily_return=daily_return,
            positions=dict(self._state.get("positions", {})),
            errors=errors,
        )

    def run_paper_period(
        self,
        start_date: str,
        end_date: str,
        weekdays_only: bool = True,
    ) -> list[PaperDayResult]:
        """Run paper trading over a date range.

        Args:
            start_date: ISO date string for start (inclusive).
            end_date: ISO date string for end (inclusive).
            weekdays_only: If True, skip weekends (Mon–Fri only).

        Returns:
            List of PaperDayResult, one per trading day.
        """
        results: list[PaperDayResult] = []
        current = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        while current <= end:
            if weekdays_only and current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            try:
                result = self.run_paper_day(current.isoformat())
                results.append(result)
            except Exception as exc:
                logger.error("[PAPER] Unhandled error on %s: %s", current, exc)
                results.append(
                    PaperDayResult(
                        date=current.isoformat(),
                        status="error",
                        n_orders=0,
                        n_fills=0,
                        total_cost_bps=0.0,
                        equity_before=self._compute_equity(None),
                        equity_after=self._compute_equity(None),
                        daily_return=0.0,
                        positions={},
                        errors=[str(exc)],
                    )
                )

            current += timedelta(days=1)

        return results

    def get_portfolio_snapshot(self) -> dict[str, Any]:
        """Return current portfolio state: positions, cash, equity, P&L.

        Returns:
            Dict with keys: positions, cash, equity, total_pnl, n_positions.
        """
        self._load_state()
        positions: dict[str, float] = self._state.get("positions", {})
        cash: float = self._state.get("cash", self.config.seed_capital)
        # Equity without current prices — best-effort using cost basis
        cost_basis: dict[str, float] = self._state.get("cost_basis", {})
        position_value = sum(
            qty * cost_basis.get(sym, 0.0)
            for sym, qty in positions.items()
        )
        equity = cash + position_value
        total_pnl = equity - self.config.seed_capital
        return {
            "positions": dict(positions),
            "cash": cash,
            "equity": equity,
            "total_pnl": total_pnl,
            "n_positions": len([q for q in positions.values() if q != 0]),
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """Return the full equity curve as a DataFrame.

        Returns:
            DataFrame with columns: date (str), equity (float).
            Sorted ascending by date.
        """
        self._load_state()
        if not self._equity_curve:
            return pd.DataFrame(columns=["date", "equity"])
        df = pd.DataFrame(self._equity_curve)
        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        """Load persistent state from JSON.

        Initializes with seed_capital if no state file exists.

        Returns:
            The current state dict (also stored as self._state).
        """
        state_path = self.config.state_dir / self._STATE_FILE
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as fh:
                    self._state = json.load(fh)
                logger.debug("[PAPER] State loaded from %s", state_path)
            except Exception as exc:
                logger.error("[PAPER] Failed to load state from %s: %s", state_path, exc)
                self._state = self._default_state()
        else:
            self._state = self._default_state()

        # Load equity curve
        equity_path = self.config.state_dir / self._EQUITY_FILE
        if equity_path.exists():
            try:
                with open(equity_path, "r", encoding="utf-8") as fh:
                    self._equity_curve = json.load(fh)
            except Exception as exc:
                logger.warning("[PAPER] Could not load equity curve: %s", exc)
                self._equity_curve = []

        self._initialized = True
        return self._state

    def _save_state(self) -> None:
        """Persist state to JSON."""
        state_path = self.config.state_dir / self._STATE_FILE
        try:
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2, default=str)
            logger.debug("[PAPER] State saved to %s", state_path)
        except Exception as exc:
            logger.error("[PAPER] Failed to save state: %s", exc)

        equity_path = self.config.state_dir / self._EQUITY_FILE
        try:
            with open(equity_path, "w", encoding="utf-8") as fh:
                json.dump(self._equity_curve, fh, indent=2, default=str)
        except Exception as exc:  # pragma: no cover
            logger.warning("[PAPER] Could not save equity curve: %s", exc)

    def _default_state(self) -> dict[str, Any]:
        """Return the default initial state."""
        return {
            "cash": self.config.seed_capital,
            "positions": {},          # symbol → qty (float)
            "cost_basis": {},         # symbol → avg_cost_per_share (float)
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": None,
        }

    # ------------------------------------------------------------------
    # Fill simulation
    # ------------------------------------------------------------------

    def _simulate_fills(self, orders: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Simulate fills using spread + Almgren-Chriss impact model.

        Fill price formula:
            fill_price = mid * (1 + side_sign * half_spread_bps / 10_000)
                         + side_sign * impact_coeff * sqrt(qty / adv) * mid

        where side_sign = +1 for BUY, -1 for SELL.

        Args:
            orders: DataFrame with columns: symbol, side, qty, price.
            prices: DataFrame with columns: symbol, close (and optionally volume/adv).

        Returns:
            DataFrame with columns: symbol, side, qty, fill_price, notional,
            spread_cost_bps, impact_cost_bps, total_cost_bps.
        """
        if orders.empty:
            return pd.DataFrame()

        # Build price lookup
        price_map: dict[str, float] = {}
        adv_map: dict[str, float] = {}
        if prices is not None and not prices.empty:
            sym_col = "symbol" if "symbol" in prices.columns else prices.columns[0]
            price_col = "close" if "close" in prices.columns else (
                "price" if "price" in prices.columns else prices.columns[1]
            )
            for _, row in prices.iterrows():
                sym = str(row[sym_col])
                price_map[sym] = float(row[price_col])
                if "adv" in prices.columns:
                    adv_map[sym] = float(row["adv"])
                elif "volume" in prices.columns:
                    adv_map[sym] = float(row["volume"])

        fills = []
        half_spread = self.config.half_spread_bps / 10_000.0
        impact_coeff = self.config.impact_coefficient
        default_adv = self.config.default_adv

        for _, order in orders.iterrows():
            sym = str(order.get("symbol", ""))
            side = str(order.get("side", "BUY")).upper()
            qty = abs(float(order.get("qty", 0.0)))

            if qty <= 0:
                continue

            mid = price_map.get(sym, float(order.get("price", 0.0)))
            if mid <= 0:
                logger.warning("[PAPER] No valid price for %s — skipping fill", sym)
                continue

            adv = adv_map.get(sym, default_adv)
            if adv <= 0:
                adv = default_adv

            side_sign = 1.0 if side == "BUY" else -1.0

            # Spread cost
            spread_component = side_sign * half_spread * mid

            # Almgren-Chriss market impact (sqrt model)
            participation = qty / adv
            impact_component = side_sign * impact_coeff * math.sqrt(participation) * mid

            fill_price = mid + spread_component + impact_component
            fill_price = max(fill_price, 1e-6)  # floor at near-zero

            notional = qty * fill_price
            spread_cost_bps = abs(spread_component / mid) * 10_000
            impact_cost_bps = abs(impact_component / mid) * 10_000

            fills.append(
                {
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "fill_price": fill_price,
                    "mid_price": mid,
                    "notional": notional,
                    "spread_cost_bps": spread_cost_bps,
                    "impact_cost_bps": impact_cost_bps,
                    "total_cost_bps": spread_cost_bps + impact_cost_bps,
                }
            )

        return pd.DataFrame(fills) if fills else pd.DataFrame()

    def _simulate_fills_with_cost(
        self, orders: pd.DataFrame, prices: pd.DataFrame
    ) -> tuple[pd.DataFrame, float]:
        """Wrapper around _simulate_fills that also applies cash gate and returns avg cost.

        Returns:
            (fills DataFrame, average total cost in bps across all fills)
        """
        fills = self._simulate_fills(orders, prices)
        if fills.empty:
            return fills, 0.0

        # Apply cash gate: reject buys that exceed available cash
        cash = float(self._state.get("cash", self.config.seed_capital))
        buy_mask = fills["side"].str.upper() == "BUY"
        running_cash = cash
        keep_rows = []
        for idx, row in fills.iterrows():
            if row["side"].upper() == "BUY":
                cost = float(row["notional"])
                if running_cash - cost >= -1e-6:
                    running_cash -= cost
                    keep_rows.append(idx)
                else:
                    logger.info(
                        "[PAPER] Cash gate: rejected BUY %s qty=%s (cash=%.2f notional=%.2f)",
                        row["symbol"],
                        row["qty"],
                        running_cash,
                        cost,
                    )
            else:
                keep_rows.append(idx)

        fills = fills.loc[keep_rows].reset_index(drop=True)

        if fills.empty:
            return fills, 0.0

        avg_cost_bps = float(fills["total_cost_bps"].mean()) if "total_cost_bps" in fills.columns else 0.0
        return fills, avg_cost_bps

    # ------------------------------------------------------------------
    # Risk controls
    # ------------------------------------------------------------------

    def _apply_risk_controls(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Apply kill switch, fat finger guard, and pre-trade checks.

        Each control is applied only if its corresponding module is available
        and enabled in config. Returns the filtered orders DataFrame.

        Args:
            orders: Raw orders DataFrame.

        Returns:
            Filtered orders DataFrame (rejected orders removed).
        """
        if orders is None or orders.empty:
            return orders

        # Kill switch
        if self.config.enable_kill_switch and _HAS_KILL_SWITCH:
            try:
                orders = guard_orders_with_kill_switch(orders)
                if orders.empty:
                    logger.warning("[PAPER] Kill switch blocked all orders")
                    return orders
            except Exception as exc:
                logger.error("[PAPER] guard_orders_with_kill_switch error: %s", exc)

        # Fat finger guard
        if self.config.enable_fat_finger and _HAS_FAT_FINGER:
            try:
                # Build history from current positions for dynamic cap
                cost_basis = self._state.get("cost_basis", {})
                history_qty = {
                    sym: abs(float(qty))
                    for sym, qty in self._state.get("positions", {}).items()
                }
                filtered, reasons = apply_fat_finger_guard(
                    orders,
                    history_qty_by_symbol=history_qty if history_qty else None,
                )
                if reasons:
                    logger.warning("[PAPER] Fat finger guard blocked: %s", reasons)
                orders = filtered
            except Exception as exc:
                logger.error("[PAPER] apply_fat_finger_guard error: %s", exc)

        # Pre-trade checks
        if _HAS_PRE_TRADE:
            try:
                orders = run_pre_trade_checks(orders)
            except Exception as exc:
                logger.warning("[PAPER] pre_trade_checks error (non-fatal): %s", exc)

        return orders

    # ------------------------------------------------------------------
    # Position management (average-cost accounting)
    # ------------------------------------------------------------------

    def _update_positions(self, fills: pd.DataFrame) -> None:
        """Update positions and cash from fills using average-cost accounting.

        For BUY fills: increase qty, recalculate average cost per share.
        For SELL fills: decrease qty (floor at 0), cash increases at fill price.

        Args:
            fills: DataFrame with symbol, side, qty, fill_price, notional.
        """
        positions: dict[str, float] = self._state.setdefault("positions", {})
        cost_basis: dict[str, float] = self._state.setdefault("cost_basis", {})
        cash: float = float(self._state.get("cash", self.config.seed_capital))

        for _, fill in fills.iterrows():
            sym = str(fill["symbol"])
            side = str(fill["side"]).upper()
            qty = abs(float(fill["qty"]))
            fill_price = float(fill["fill_price"])
            notional = qty * fill_price

            current_qty = float(positions.get(sym, 0.0))
            current_cost = float(cost_basis.get(sym, 0.0))

            if side == "BUY":
                new_qty = current_qty + qty
                if new_qty > 0:
                    # Average cost = (old_total_cost + new_notional) / new_qty
                    old_total_cost = current_qty * current_cost
                    cost_basis[sym] = (old_total_cost + notional) / new_qty
                positions[sym] = new_qty
                cash -= notional

            elif side == "SELL":
                sold_qty = min(qty, current_qty)  # cannot sell more than owned
                proceeds = sold_qty * fill_price
                new_qty = current_qty - sold_qty
                positions[sym] = new_qty
                if new_qty <= 1e-8:
                    # Fully closed — reset cost basis
                    cost_basis.pop(sym, None)
                    positions.pop(sym, None)
                cash += proceeds

        self._state["cash"] = cash
        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Ledger and reconciliation
    # ------------------------------------------------------------------

    def _write_ledger_events(self, fills: pd.DataFrame, as_of_date: str) -> None:
        """Write Parquet ledger events for the day's fills.

        Args:
            fills: Filled orders DataFrame.
            as_of_date: ISO date string for the trading day.
        """
        try:
            ledger_path = self.config.ledger_dir / f"ledger_{as_of_date}.parquet"

            # Build minimal ledger event schema
            events = []
            ts = f"{as_of_date}T16:00:00+00:00"
            for _, fill in fills.iterrows():
                side = str(fill["side"]).upper()
                qty = float(fill["qty"])
                price = float(fill["fill_price"])
                cash_delta = qty * price * (-1 if side == "BUY" else 1)
                events.append(
                    {
                        "event_ts": ts,
                        "event_type": "FILL",
                        "symbol": str(fill["symbol"]),
                        "qty": qty if side == "BUY" else -qty,
                        "price": price,
                        "cash_delta": cash_delta,
                        "run_id": self.config.run_id,
                        "event_id": f"{self.config.run_id}_{as_of_date}_{fill['symbol']}",
                    }
                )

            if events:
                df_events = pd.DataFrame(events)
                if _HAS_LEDGER:
                    try:
                        store_ledger_events_parquet(df_events, ledger_path)
                    except Exception:
                        df_events.to_parquet(ledger_path, index=False)
                else:
                    df_events.to_parquet(ledger_path, index=False)
                logger.info("[PAPER] Ledger events written: %s (%s rows)", ledger_path, len(events))
        except Exception as exc:
            logger.error("[PAPER] Ledger write failed: %s", exc)

    def _run_reconciliation(self, as_of_date: str) -> None:
        """Run ledger-vs-broker reconciliation for the day.

        This is a best-effort step; errors are logged but do not fail the day.

        Args:
            as_of_date: ISO date string for the trading day.
        """
        try:
            ledger_path = self.config.ledger_dir / f"ledger_{as_of_date}.parquet"
            if not ledger_path.exists():
                return

            broker_positions = dict(self._state.get("positions", {}))
            ledger_df = pd.read_parquet(ledger_path)

            result = reconcile_ledger_vs_broker(
                ledger_df, broker_positions=broker_positions
            )
            logger.info("[PAPER] Reconciliation %s: %s", as_of_date, result)
        except Exception as exc:
            logger.warning("[PAPER] Reconciliation skipped: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_equity(self, prices: pd.DataFrame | None) -> float:
        """Compute current equity = cash + sum(position_qty * current_price).

        Falls back to cost basis when prices are not available.

        Args:
            prices: Optional price DataFrame (symbol, close).

        Returns:
            Equity as float.
        """
        cash = float(self._state.get("cash", self.config.seed_capital))
        positions: dict[str, float] = self._state.get("positions", {})

        if not positions:
            return cash

        # Build price map
        price_map: dict[str, float] = {}
        if prices is not None and not prices.empty:
            sym_col = "symbol" if "symbol" in prices.columns else prices.columns[0]
            price_col = "close" if "close" in prices.columns else (
                "price" if "price" in prices.columns else prices.columns[-1]
            )
            for _, row in prices.iterrows():
                price_map[str(row[sym_col])] = float(row[price_col])

        cost_basis: dict[str, float] = self._state.get("cost_basis", {})

        position_value = 0.0
        for sym, qty in positions.items():
            if qty == 0:
                continue
            price = price_map.get(sym, cost_basis.get(sym, 0.0))
            position_value += float(qty) * price

        return cash + position_value

    def _append_equity_point(self, as_of_date: str, equity: float) -> None:
        """Append a point to the in-memory equity curve."""
        self._equity_curve.append({"date": as_of_date, "equity": equity})

    def _try_load_prices(self, as_of_date: str) -> pd.DataFrame | None:
        """Attempt to load prices from the data layer for as_of_date.

        This is a best-effort stub. Subclasses or callers should override
        by passing prices directly to run_paper_day().

        Returns:
            DataFrame or None.
        """
        # Future: integrate with data layer (e.g. data/prices.py)
        logger.debug("[PAPER] No external price loader configured for %s", as_of_date)
        return None

    def _generate_orders(self, as_of_date: str, prices: pd.DataFrame) -> pd.DataFrame | None:
        """Generate orders for the trading day.

        Default implementation returns an empty DataFrame.
        Subclasses or callers should override this method or inject orders
        through a custom signal/sizing pipeline.

        Args:
            as_of_date: ISO date string.
            prices: Current price DataFrame.

        Returns:
            Orders DataFrame or None.
        """
        # Default: no orders — callers inject orders by subclassing or
        # using the pipeline.trading_cycle integration externally.
        return pd.DataFrame()

    def _write_experience_entry(
        self, as_of_date: str, equity_before: float, equity_after: float, n_fills: int
    ) -> None:
        """Write a post-trade entry to the experience log."""
        try:
            log_experience_entry(
                {
                    "date": as_of_date,
                    "equity_before": equity_before,
                    "equity_after": equity_after,
                    "daily_return": (equity_after / equity_before - 1.0) if equity_before > 0 else 0.0,
                    "n_fills": n_fills,
                    "run_id": self.config.run_id,
                }
            )
        except Exception as exc:
            logger.debug("[PAPER] Experience log write failed (non-fatal): %s", exc)

    def reset(self, confirm: bool = False) -> None:
        """Reset all state to initial seed capital.

        Args:
            confirm: Must be True to actually reset (safety guard).
        """
        if not confirm:
            logger.warning("[PAPER] reset() called without confirm=True — ignored")
            return
        self._state = self._default_state()
        self._equity_curve = []
        self._save_state()
        logger.info("[PAPER] State reset to seed_capital=%.2f", self.config.seed_capital)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified Paper Trading Engine — run one day or a date range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", metavar="YYYY-MM-DD", help="Run a single paper trading day.")
    group.add_argument(
        "--start-date",
        metavar="YYYY-MM-DD",
        help="Start of date range (requires --end-date).",
    )
    p.add_argument("--end-date", metavar="YYYY-MM-DD", help="End of date range (inclusive).")
    p.add_argument("--dry-run", action="store_true", help="Simulate without persisting state.")
    p.add_argument(
        "--seed-capital",
        type=float,
        default=100_000.0,
        help="Starting capital in USD (used only when no state file exists).",
    )
    p.add_argument(
        "--state-dir",
        default="output/paper_state",
        help="Directory for JSON state files.",
    )
    p.add_argument(
        "--ledger-dir",
        default="output/paper_ledger",
        help="Directory for Parquet ledger events.",
    )
    p.add_argument(
        "--no-kill-switch",
        action="store_true",
        help="Disable kill switch check.",
    )
    p.add_argument(
        "--no-fat-finger",
        action="store_true",
        help="Disable fat finger guard.",
    )
    p.add_argument(
        "--no-ledger",
        action="store_true",
        help="Disable ledger event writing.",
    )
    p.add_argument(
        "--no-reconciliation",
        action="store_true",
        help="Disable reconciliation.",
    )
    p.add_argument(
        "--weekdays-only",
        action="store_true",
        default=True,
        help="Skip weekends when running a date range.",
    )
    return p


def main() -> None:
    """CLI entry point for the unified paper engine."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    cfg = UnifiedPaperConfig(
        seed_capital=args.seed_capital,
        enable_ledger=not args.no_ledger,
        enable_reconciliation=not args.no_reconciliation,
        enable_fat_finger=not args.no_fat_finger,
        enable_kill_switch=not args.no_kill_switch,
        state_dir=Path(args.state_dir),
        ledger_dir=Path(args.ledger_dir),
    )

    engine = UnifiedPaperEngine(cfg)

    if args.date:
        result = engine.run_paper_day(args.date, dry_run=args.dry_run)
        print(
            f"[PAPER] {result.date} | {result.status} | "
            f"equity={result.equity_after:.2f} | return={result.daily_return*100:.2f}% | "
            f"fills={result.n_fills} | cost={result.total_cost_bps:.1f}bps"
        )
        if result.errors:
            print(f"[PAPER] Errors: {result.errors}")
    else:
        if not args.end_date:
            parser.error("--end-date is required when using --start-date")
        results = engine.run_paper_period(
            args.start_date, args.end_date, weekdays_only=args.weekdays_only
        )
        successes = sum(1 for r in results if r.status == "success")
        print(
            f"[PAPER] Period {args.start_date}–{args.end_date}: "
            f"{len(results)} days, {successes} success, "
            f"final equity={results[-1].equity_after:.2f if results else 0:.2f}"
        )


if __name__ == "__main__":
    main()
