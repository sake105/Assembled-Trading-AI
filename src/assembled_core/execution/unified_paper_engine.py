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
    from src.assembled_core.execution.symbol_kill_switch import (
        filter_orders_from_policy as _symbol_kill_filter,
    )
    _HAS_SYMBOL_KILL = True
except Exception:  # pragma: no cover
    _HAS_SYMBOL_KILL = False
    logger.warning("[PAPER] symbol_kill_switch unavailable — per-symbol halt disabled")

try:
    _HAS_FILL_MODEL = True
except Exception:  # pragma: no cover
    _HAS_FILL_MODEL = False
    logger.warning("[PAPER] fill_model unavailable — using simple fill simulation")

try:
    from src.assembled_core.accounting.ledger import (
        store_ledger_events_parquet,
    )
    _HAS_LEDGER = True
except Exception:  # pragma: no cover
    _HAS_LEDGER = False
    logger.warning("[PAPER] ledger unavailable — ledger events disabled")

try:
    from src.assembled_core.accounting.reconciliation import (
        ReconcileSLO,
        evaluate_reconcile_slo,
        reconcile_ledger_vs_broker,
    )
    _HAS_RECONCILIATION = True
except Exception:  # pragma: no cover
    _HAS_RECONCILIATION = False
    ReconcileSLO = None  # type: ignore[assignment,misc]
    logger.warning("[PAPER] reconciliation unavailable — reconciliation disabled")

try:
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

try:
    from src.assembled_core.execution.order_lifecycle import (
        OrderLifecycleTracker,
        OrderState,
    )
    _HAS_LIFECYCLE = True
except Exception:  # pragma: no cover
    _HAS_LIFECYCLE = False
    logger.warning("[PAPER] order_lifecycle unavailable — lifecycle tracking disabled")

try:
    from src.assembled_core.ops.replay_snapshot import RunSnapshot, make_rng
    _HAS_REPLAY = True
except Exception:  # pragma: no cover
    _HAS_REPLAY = False
    logger.warning("[PAPER] replay_snapshot unavailable — determinism helpers disabled")

try:
    from src.assembled_core.ops.run_index import append_run_index
    from src.assembled_core.ops.run_manifest import (
        compute_config_hash,
        write_run_manifest,
    )
    _HAS_MANIFEST = True
except Exception:  # pragma: no cover
    _HAS_MANIFEST = False

try:
    from src.assembled_core.accounting.attribution import (
        compute_cost_attribution,
        compute_factor_attribution,
        compute_regime_attribution,
    )
    _HAS_ATTRIBUTION = True
except Exception:  # pragma: no cover
    _HAS_ATTRIBUTION = False
    logger.warning("[PAPER] run_manifest unavailable — manifest writing disabled")

try:
    from src.assembled_core.execution.fill_model import (
        apply_adversarial_fill_adjustment,
        check_circuit_breaker,
        compute_adversarial_fill_cost,
    )
    _HAS_CIRCUIT_BREAKER = True
except Exception:  # pragma: no cover
    _HAS_CIRCUIT_BREAKER = False
    logger.warning("[PAPER] circuit breaker / adversarial fill unavailable")

try:
    from src.assembled_core.execution.borrow_costs import (
        BorrowRateTable,
        compute_borrow_cost_for_positions,
    )
    _HAS_BORROW = True
except Exception:  # pragma: no cover
    _HAS_BORROW = False
    logger.warning("[PAPER] borrow_costs unavailable — short borrow disabled")

try:
    from src.assembled_core.data.corporate_actions import (
        adjust_prices_for_splits,
    )
    _HAS_CORP_ACTIONS = True
except Exception:  # pragma: no cover
    _HAS_CORP_ACTIONS = False
    logger.warning("[PAPER] corporate_actions unavailable — CA adjustments disabled")

try:
    from src.assembled_core.costs import get_tier_costs_for_symbol
    _HAS_COST_TIERS = True
except Exception:  # pragma: no cover
    _HAS_COST_TIERS = False
    logger.warning("[PAPER] cost_tiers unavailable — flat cost fallback")


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
    # Per-symbol halt list (sidecar state at ``symbol_kill_state_path``).
    # Default OFF — opt-in via policy or test harness (Tier-1 activation plan).
    enable_symbol_kill_switch: bool = False
    symbol_kill_state_path: Path | None = None
    enable_lifecycle_tracking: bool = True
    enable_partial_fills: bool = False
    enable_circuit_breaker: bool = False
    enable_adversarial_fill: bool = False
    kyle_lambda: float = 0.10
    max_adversarial_bps: float = 50.0
    market_benchmark: str = "SPY"
    enable_sor: bool = False
    sor_regime: str = "bull"
    sor_urgency: float = 0.5
    sor_max_venues: int = 3
    sor_allow_dark_pools: bool = True
    enable_borrow_costs: bool = True
    borrow_rate_default_bps: float = 50.0
    borrow_rate_htb_bps: float = 500.0
    borrow_rate_overrides: dict[str, float] = field(default_factory=dict)
    htb_symbols: tuple[str, ...] = ()
    enable_corporate_actions: bool = True
    corporate_actions_path: Path | None = None
    # Phase 6 — Alpaca parity: intent store + hardened reconciliation
    enable_intent_store: bool = False
    intent_store_path: Path | None = None
    reconcile_alerts_dir: Path = field(
        default_factory=lambda: Path("output/reconciliation_alerts")
    )
    reconcile_slo: Any = None  # ReconcileSLO instance; None → defaults
    shadow_mode: bool = False
    shadow_compare_dir: Path = field(
        default_factory=lambda: Path("output/shadow_compare")
    )
    shadow_broker: Any = None  # optional broker adapter with .submit(...)→dict
    # Phase 7 — TCA output
    enable_tca: bool = True
    tca_dir: Path = field(default_factory=lambda: Path("output/paper_tca"))
    # Phase 9 — Attribution drilldown
    enable_attribution: bool = True
    attribution_dir: Path = field(
        default_factory=lambda: Path("output/paper_attribution")
    )
    # Phase 8 — Run manifest + cross-run index
    enable_manifest: bool = True
    manifests_dir: Path = field(default_factory=lambda: Path("output/manifests"))
    run_index_path: Path = field(
        default_factory=lambda: Path("output/manifests/index.csv")
    )
    half_spread_bps: float = 5.0
    impact_coefficient: float = 0.10
    default_adv: float = 100_000.0
    enable_cost_tiers: bool = False
    reject_unknown_adv: bool = False
    max_participation: float = 0.05
    min_fill_qty: float = 0.0
    # B2 — state-save batching. 1 = save every day (paper/live safe default).
    # Backtest callers can pass e.g. 5 to amortise JSON I/O over a week.
    # ``run_paper_period`` always forces a final save so the on-disk state at
    # end-of-run is identical regardless of the batching interval.
    state_save_every_n_days: int = 1
    manifest_every_n_days: int = 1
    state_dir: Path = field(default_factory=lambda: Path("output/paper_state"))
    ledger_dir: Path = field(default_factory=lambda: Path("output/paper_ledger"))
    lifecycle_dir: Path = field(default_factory=lambda: Path("output/paper_lifecycle"))
    replay_snapshot_dir: Path | None = None
    random_seed: int | None = None
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

        # B3: last-run order/fill buffers live as plain attributes (NOT inside
        # ``self._state``) because ``_save_state`` serialises the state dict
        # via ``json.dump(default=str)``. A DataFrame written there becomes a
        # repr string on reload, silently breaking fill-rate and slippage SLO
        # evaluation in ``_run_reconciliation``.
        self._last_fills: pd.DataFrame = pd.DataFrame()
        self._last_orders_n: int = 0
        self._last_slippage_obs: list[float] = []
        self._last_rejection_counts: dict[str, int] = {}

        # B2 — batched state-save bookkeeping. Counters are days *elapsed since
        # last flush*, not absolute day indices, so they stay correct even if
        # callers mix ``run_paper_day`` and ``run_paper_period``.
        self._days_since_state_save: int = 0
        self._days_since_manifest: int = 0

        # Optional lifecycle tracker — one per engine instance; the tracker
        # accumulates orders across days so the on-disk dump is per-day but
        # the in-memory view is run-scoped.
        self._lifecycle: OrderLifecycleTracker | None = None
        self._lifecycle_dumped_ids: set[str] = set()
        if self.config.enable_lifecycle_tracking and _HAS_LIFECYCLE:
            self._lifecycle = OrderLifecycleTracker()

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

        if self._lifecycle is not None:
            try:
                self.config.lifecycle_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # pragma: no cover
                logger.warning("[PAPER] Could not create lifecycle_dir: %s", exc)

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
        run_started_utc = datetime.now(timezone.utc).isoformat()

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

        # Phase 5: apply corporate actions (splits adjust qty/cost-basis and
        # prices; dividends credit cash). Runs before snapshot so the frozen
        # input already reflects any split-adjusted level.
        prices = self._apply_corporate_actions(as_of_date, prices)

        # Deterministic replay snapshot of the day's inputs.
        self._maybe_save_replay_snapshot(as_of_date, prices)

        # Steps 4+5 — Generate orders (placeholder: callers may inject via subclass)
        orders = self._generate_orders(as_of_date, prices)
        n_orders = len(orders) if orders is not None and not orders.empty else 0
        logger.info("[PAPER] %s orders generated for %s", n_orders, as_of_date)

        # Lifecycle: CREATED for every order generated today.
        orders = self._lifecycle_attach(orders, as_of_date)
        pre_risk_ids = (
            list(orders["order_id"]) if orders is not None and not orders.empty
            and "order_id" in orders.columns else []
        )

        fills = pd.DataFrame()
        total_cost_bps = 0.0

        if n_orders > 0 and orders is not None:
            # Step 6 — Risk controls (incl. market-wide circuit breaker)
            try:
                market_return = self._extract_benchmark_return(prices)
                orders = self._apply_risk_controls(
                    orders, market_return_today=market_return
                )
            except Exception as exc:
                errors.append(f"risk_controls error: {exc}")
                logger.error("[PAPER] Risk controls failed: %s", exc)
                # Fail-closed: an exception in risk controls must not let the
                # unfiltered pre-risk order set flow into fill simulation.
                orders = pd.DataFrame(columns=orders.columns)

            # Lifecycle: VALIDATED for survivors, REJECTED for those dropped.
            post_risk_ids = (
                list(orders["order_id"]) if orders is not None and not orders.empty
                and "order_id" in orders.columns else []
            )
            self._lifecycle_mark_validation(pre_risk_ids, post_risk_ids)

            # Step 7 — Fill simulation
            submit_keys: list[tuple[str, str]] = []
            try:
                self._lifecycle_mark_submitted(post_risk_ids)
                # Phase 6: idempotent ORDER_SUBMIT intents before simulation.
                submit_keys = self._record_submit_intents(orders)
                fills, total_cost_bps = self._simulate_fills_with_cost(orders, prices)
            except Exception as exc:
                errors.append(f"fill simulation error: {exc}")
                logger.error("[PAPER] Fill simulation failed: %s", exc)
                fills = pd.DataFrame()

            # Lifecycle: terminal state per order (FILLED / PARTIAL_FILL / REJECTED).
            self._lifecycle_mark_fills(orders, fills, post_risk_ids)

            # Phase 6: pair ORDER_COMPLETE intents with their submits.
            self._record_complete_intents(orders, fills, submit_keys)

            # Phase 6: optional shadow-broker compare (observability only).
            if self.config.shadow_mode and not dry_run:
                try:
                    self._run_shadow_compare(as_of_date, orders, fills)
                except Exception as exc:  # pragma: no cover
                    logger.warning("[PAPER] shadow compare skipped: %s", exc)

            # Phase 6: expose latest orders/fills to reconciliation SLO step.
            # B3: stored as plain attrs — not persisted via _save_state.
            self._last_orders_n = int(len(orders)) if orders is not None else 0
            self._last_fills = fills if isinstance(fills, pd.DataFrame) else pd.DataFrame()
            # Slippage observations: (fill_price - mid_price) / mid_price * 10_000
            if isinstance(fills, pd.DataFrame) and not fills.empty and {"fill_price", "mid_price", "status"}.issubset(fills.columns):
                _f = fills[fills["status"].isin({"filled", "partial"})]
                if not _f.empty:
                    _mid = _f["mid_price"]
                    self._last_slippage_obs = ((_f["fill_price"] - _mid) / _mid.where(_mid != 0, other=float("nan")) * 10_000).dropna().tolist()
                else:
                    self._last_slippage_obs = []
                # Rejection counts keyed by reject_reason
                if "reject_reason" in fills.columns:
                    _rej = fills[fills["status"] == "rejected"]
                    if not _rej.empty:
                        self._last_rejection_counts = _rej["reject_reason"].value_counts().to_dict()
                    else:
                        self._last_rejection_counts = {}
                else:
                    self._last_rejection_counts = {}
            else:
                self._last_slippage_obs = []
                self._last_rejection_counts = {}

        n_fills = len(fills) if not fills.empty else 0
        logger.info("[PAPER] %s fills executed for %s", n_fills, as_of_date)

        # Step 8 — Ledger events
        if self.config.enable_ledger and _HAS_LEDGER and not fills.empty and not dry_run:
            self._write_ledger_events(fills, as_of_date)

        # Step 9 — Update positions (average-cost accounting)
        if not fills.empty:
            self._update_positions(fills)

        # Step 9b — Phase 5: Accrue one day of borrow cost for short positions.
        try:
            self._apply_borrow_costs(as_of_date, prices)
        except Exception as exc:  # pragma: no cover
            logger.warning("[PAPER] borrow cost accrual error: %s", exc)

        # Step 10 — Reconciliation
        if self.config.enable_reconciliation and _HAS_RECONCILIATION and not dry_run:
            self._run_reconciliation(as_of_date)

        # Step 10b — Phase 7: TCA + fill-quality artifacts.
        if not dry_run:
            try:
                self._write_tca_artifacts(as_of_date, orders, fills)
            except Exception as exc:  # pragma: no cover
                logger.warning("[PAPER] TCA artifact write skipped: %s", exc)

        # Step 10c — Phase 9: attribution drilldown (cost / regime / factor).
        if not dry_run:
            try:
                self._write_attribution_artifacts(as_of_date, fills)
            except Exception as exc:  # pragma: no cover
                logger.warning("[PAPER] attribution write skipped: %s", exc)

        # Step 11 — Experience log
        equity_after = self._compute_equity(prices)
        daily_return = (equity_after / equity_before - 1.0) if equity_before > 0 else 0.0
        self._append_equity_point(as_of_date, equity_after)

        if not dry_run and _HAS_EXPERIENCE_LOG:
            self._write_experience_entry(as_of_date, equity_before, equity_after, n_fills)

        # Step 11b — Lifecycle dump (per-day JSONL snapshot)
        if not dry_run:
            self._lifecycle_dump(as_of_date)

        # Step 12 — Persist state (B2: batched by ``state_save_every_n_days``)
        if not dry_run:
            self._maybe_save_state()

        # Step 12b — Phase 8: Run manifest + cross-run index
        # (B2: batched by ``manifest_every_n_days``).
        if not dry_run:
            try:
                self._maybe_write_manifest(
                    as_of_date=as_of_date,
                    run_started_utc=run_started_utc,
                    status="success" if not errors else "error",
                    equity_after=equity_after,
                    total_return=daily_return,
                    n_fills=n_fills,
                    total_cost_bps=total_cost_bps,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("[PAPER] manifest/index write skipped: %s", exc)

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

        # B2 — force a final flush so the on-disk state at end-of-run is
        # identical regardless of ``state_save_every_n_days``. Without this,
        # a run that ends on a non-flush boundary would leave the last few
        # days only in memory.
        if self._days_since_state_save > 0:
            self._save_state()
            self._days_since_state_save = 0

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
                # A corrupted state.json (power loss before fsync, partial
                # write, tampered file) must NEVER silently revert to a
                # fresh $10k seed — the very next _save_state would then
                # overwrite the recoverable corrupt file with the default,
                # permanently destroying the prior position book. Rename
                # the bad file for forensic recovery and refuse to boot.
                corrupt_suffix = datetime.now(timezone.utc).strftime(
                    ".corrupt.%Y%m%dT%H%M%S"
                )
                corrupt_path = state_path.with_name(state_path.name + corrupt_suffix)
                try:
                    state_path.rename(corrupt_path)
                except Exception:
                    corrupt_path = state_path
                logger.critical(
                    "[PAPER] state load failed: %s — preserved at %s; refusing to boot",
                    exc,
                    corrupt_path,
                )
                raise RuntimeError(
                    f"paper state unreadable: preserved at {corrupt_path}; "
                    f"investigate before restart"
                ) from exc
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

    def _maybe_save_state(self) -> None:
        """B2 — batched wrapper around :meth:`_save_state`.

        Increments the day counter and only flushes to disk once
        ``state_save_every_n_days`` days have elapsed since the last flush.
        Callers that need a guaranteed flush (``run_paper_period`` end-of-run,
        kill-switch path, crash recovery) must call :meth:`_save_state`
        directly.
        """
        self._days_since_state_save += 1
        every = max(1, int(getattr(self.config, "state_save_every_n_days", 1)))
        if self._days_since_state_save >= every:
            self._save_state()
            self._days_since_state_save = 0

    def _maybe_write_manifest(
        self,
        as_of_date: str,
        run_started_utc: str,
        status: str,
        equity_after: float,
        total_return: float,
        n_fills: int,
        total_cost_bps: float,
    ) -> None:
        """B2 — batched wrapper around :meth:`_write_manifest_and_index`."""
        self._days_since_manifest += 1
        every = max(1, int(getattr(self.config, "manifest_every_n_days", 1)))
        if self._days_since_manifest >= every:
            self._write_manifest_and_index(
                as_of_date=as_of_date,
                run_started_utc=run_started_utc,
                status=status,
                equity_after=equity_after,
                total_return=total_return,
                n_fills=n_fills,
                total_cost_bps=total_cost_bps,
            )
            self._days_since_manifest = 0

    def _save_state(self) -> None:
        """Persist state to JSON atomically.

        Writes to a ``.tmp`` sibling first, fsyncs, then ``os.replace`` onto
        the target. On POSIX and Windows (Py >=3.3) ``os.replace`` is atomic
        for paths on the same filesystem, so a crash mid-save cannot leave
        a corrupt primary file. If the temp write fails it is cleaned up.
        """
        self._atomic_write_json(
            self.config.state_dir / self._STATE_FILE,
            self._state,
            log_failure_level=logging.ERROR,
            log_label="state",
        )
        self._atomic_write_json(
            self.config.state_dir / self._EQUITY_FILE,
            self._equity_curve,
            log_failure_level=logging.WARNING,
            log_label="equity curve",
        )

    @staticmethod
    def _atomic_write_json(
        target_path: Path,
        payload: Any,
        *,
        log_failure_level: int = logging.ERROR,
        log_label: str = "file",
    ) -> None:
        """Write ``payload`` as JSON to ``target_path`` atomically."""
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, default=str)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # Some filesystems (network, tmpfs) reject fsync — accept
                    # the weaker durability guarantee rather than losing the
                    # whole write.
                    pass
            os.replace(tmp_path, target_path)
            logger.debug("[PAPER] %s saved atomically to %s", log_label, target_path)
        except Exception as exc:
            logger.log(
                log_failure_level,
                "[PAPER] atomic save of %s to %s failed: %s",
                log_label,
                target_path,
                exc,
            )
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

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

        # B4 — vectorised price_map build. ``iterrows`` was ~3× slower per
        # symbol than a zipped Series scan and is hit once per ``run_paper_day``
        # plus once per rebalance bar in backtests.
        price_map: dict[str, float] = {}
        adv_map: dict[str, float] = {}
        if prices is not None and not prices.empty:
            sym_col = "symbol" if "symbol" in prices.columns else prices.columns[0]
            price_col = "close" if "close" in prices.columns else (
                "price" if "price" in prices.columns else prices.columns[1]
            )
            syms = prices[sym_col].astype(str).tolist()
            closes = prices[price_col].astype(float).tolist()
            price_map = dict(zip(syms, closes, strict=False))
            if "adv" in prices.columns:
                advs = prices["adv"].astype(float).tolist()
                adv_map = dict(zip(syms, advs, strict=False))
            elif "volume" in prices.columns:
                vols = prices["volume"].astype(float).tolist()
                adv_map = dict(zip(syms, vols, strict=False))

        fills = []
        default_half_spread = self.config.half_spread_bps / 10_000.0
        impact_coeff = self.config.impact_coefficient
        default_adv = self.config.default_adv
        enable_partial = bool(self.config.enable_partial_fills)
        enable_tiers = bool(self.config.enable_cost_tiers) and _HAS_COST_TIERS
        reject_unknown_adv = bool(self.config.reject_unknown_adv)
        max_participation = float(self.config.max_participation)
        min_fill_qty = float(self.config.min_fill_qty)
        enable_adversarial = (
            bool(self.config.enable_adversarial_fill) and _HAS_CIRCUIT_BREAKER
        )
        kyle_lambda = float(self.config.kyle_lambda)
        max_adv_bps = float(self.config.max_adversarial_bps)
        enable_sor = bool(self.config.enable_sor) and _HAS_SOR
        sor_regime = str(self.config.sor_regime)
        sor_urgency = float(self.config.sor_urgency)
        sor_max_venues = int(self.config.sor_max_venues)
        sor_allow_dark = bool(self.config.sor_allow_dark_pools)

        # B4 — pre-extract hot columns as plain Python lists. Row-by-row
        # ``iterrows()`` creates a fresh pandas Series per iteration and is the
        # dominant cost in the fill-sim profile. Pulling to lists once keeps
        # the per-iteration work to arithmetic + dict lookups, which is
        # 2-4× faster and still deterministic.
        _n_orders = len(orders)
        _sym_arr = orders["symbol"].astype(str).tolist() if "symbol" in orders.columns else [""] * _n_orders
        _side_arr = orders["side"].astype(str).str.upper().tolist() if "side" in orders.columns else ["BUY"] * _n_orders
        _qty_arr = orders["qty"].astype(float).abs().tolist() if "qty" in orders.columns else [0.0] * _n_orders
        _order_id_arr = orders["order_id"].tolist() if "order_id" in orders.columns else [None] * _n_orders
        _price_arr = orders["price"].astype(float).tolist() if "price" in orders.columns else [0.0] * _n_orders
        if "signal_strength" in orders.columns:
            _sig_arr = orders["signal_strength"].fillna(0.0).astype(float).tolist()
        else:
            _sig_arr = [0.0] * _n_orders

        for _i in range(_n_orders):
            sym = _sym_arr[_i]
            side = _side_arr[_i]
            qty = _qty_arr[_i]
            order_id = _order_id_arr[_i]
            signal_strength = _sig_arr[_i]

            if qty <= 0:
                continue

            mid = price_map.get(sym, _price_arr[_i])
            if mid <= 0:
                logger.warning("[PAPER] No valid price for %s — skipping fill", sym)
                continue

            adv_known = sym in adv_map and adv_map[sym] > 0
            adv = adv_map.get(sym, default_adv)
            if adv <= 0:
                adv = default_adv
            if reject_unknown_adv and not adv_known:
                fills.append(
                    {
                        "symbol": sym,
                        "side": side,
                        "qty": qty,
                        "fill_qty": 0.0,
                        "remaining_qty": qty,
                        "fill_price": mid,
                        "mid_price": mid,
                        "notional": 0.0,
                        "spread_cost_bps": 0.0,
                        "impact_cost_bps": 0.0,
                        "total_cost_bps": 0.0,
                        "status": "rejected",
                        "reject_reason": "UNKNOWN_ADV",
                        "order_id": order_id,
                    }
                )
                continue

            # E0.2 — per-symbol cost tiers. When enabled, derive half_spread
            # and commission from the ADV-in-USD bucket. Legacy path
            # (enable_cost_tiers=False) keeps the flat config value so
            # existing equity curves stay bit-identical.
            tier_commission_bps = 0.0
            tier_name: str | None = None
            if enable_tiers:
                adv_usd = adv * mid
                tier_name, tier_costs = get_tier_costs_for_symbol(sym, adv_usd)
                half_spread = tier_costs["half_spread_bps"] / 10_000.0
                tier_commission_bps = tier_costs["commission_bps"]
            else:
                half_spread = default_half_spread

            # Phase 2: participation cap → partial fills + reject-on-min-qty.
            # When disabled, fill_qty == qty and status == "filled", preserving
            # the legacy full-fill behaviour bit-for-bit.
            if enable_partial and adv > 0 and max_participation > 0:
                max_fill = max_participation * adv
                fill_qty = min(qty, max_fill)
                if fill_qty < min_fill_qty:
                    # Reject the whole order. Emit a REJECT row so the lifecycle
                    # tracker and ledger can observe the decision.
                    fills.append(
                        {
                            "symbol": sym,
                            "side": side,
                            "qty": qty,
                            "fill_qty": 0.0,
                            "remaining_qty": qty,
                            "fill_price": mid,
                            "mid_price": mid,
                            "notional": 0.0,
                            "spread_cost_bps": 0.0,
                            "impact_cost_bps": 0.0,
                            "total_cost_bps": 0.0,
                            "status": "rejected",
                            "reject_reason": "MIN_FILL_QTY",
                            "order_id": order_id,
                        }
                    )
                    continue
                status = "filled" if fill_qty >= qty - 1e-9 else "partial"
                remaining_qty = max(qty - fill_qty, 0.0)
            else:
                fill_qty = qty
                status = "filled"
                remaining_qty = 0.0

            side_sign = 1.0 if side == "BUY" else -1.0

            # Spread cost (independent of fill_qty; expressed as price offset).
            # When tiers are enabled, a tier-driven commission term is added
            # as an additive bps cost below — this is what the fill_price
            # and ledger observe.
            spread_component = side_sign * half_spread * mid

            # Almgren-Chriss market impact (sqrt model). Impact is computed on
            # the *executed* quantity, not the intended quantity — that is
            # what actually crosses the book.
            participation = fill_qty / adv
            impact_component = side_sign * impact_coeff * math.sqrt(participation) * mid

            fill_price = mid + spread_component + impact_component
            fill_price = max(fill_price, 1e-6)  # floor at near-zero

            spread_cost_bps = abs(spread_component / mid) * 10_000
            impact_cost_bps = abs(impact_component / mid) * 10_000

            # Tier commission: applied when enable_cost_tiers=True.
            # Translated from bps to a side-signed price adjustment so the
            # ledger's cash delta reflects the full cost.
            if enable_tiers and tier_commission_bps > 0:
                commission_adjustment = side_sign * mid * tier_commission_bps / 10_000.0
                fill_price = fill_price + commission_adjustment
                fill_price = max(fill_price, 1e-6)

            # Phase 3: Kyle-lambda adversarial fill cost. Informed orders
            # (high abs(signal_strength)) get worse fills on top of spread and
            # impact. Skipped silently when disabled or signal_strength is 0.
            adversarial_cost_bps = 0.0
            if enable_adversarial and abs(signal_strength) > 0 and fill_qty > 0:
                try:
                    adversarial_cost_bps = float(
                        compute_adversarial_fill_cost(
                            order_size=fill_qty * mid,
                            signal_strength=abs(signal_strength),
                            adv=adv * mid,
                            kyle_lambda=kyle_lambda,
                            max_cost_bps=max_adv_bps,
                        )
                    )
                    fill_price = apply_adversarial_fill_adjustment(
                        fill_price, side, adversarial_cost_bps
                    )
                except Exception as exc:  # pragma: no cover
                    logger.error("[PAPER] adversarial fill cost error: %s", exc)
                    adversarial_cost_bps = 0.0

            # Phase 4: Smart-Order-Router cost adjustment. Opt-in via
            # ``enable_sor``; defaults preserve legacy behaviour bit-for-bit.
            # The SOR decides venue mix and returns an expected cost in bps
            # (spread + fees - rebates under the current regime). We treat
            # it as an additive cost term on top of the baseline model and
            # adjust fill_price accordingly, so the cash gate and ledger
            # observe the routing cost.
            sor_cost_bps = 0.0
            sor_venues: str | None = None
            if enable_sor and fill_qty > 0:
                try:
                    routing = route_order(
                        order_size=fill_qty,
                        signal_urgency=sor_urgency,
                        adv=adv,
                        regime=sor_regime,
                        price=mid,
                        allow_dark_pools=sor_allow_dark,
                        max_venues=sor_max_venues,
                    )
                    sor_cost_bps = float(routing.total_expected_cost_bps)
                    # Apply cost as a side-adjusted price delta.
                    sor_price_adjustment = side_sign * mid * sor_cost_bps / 10_000.0
                    fill_price = max(fill_price + sor_price_adjustment, 1e-6)
                    sor_venues = "|".join(a.venue for a in routing.allocations)
                except Exception as exc:  # pragma: no cover
                    logger.error("[PAPER] SOR routing error: %s", exc)
                    sor_cost_bps = 0.0

            notional = fill_qty * fill_price

            fills.append(
                {
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "fill_qty": fill_qty,
                    "remaining_qty": remaining_qty,
                    "fill_price": fill_price,
                    "mid_price": mid,
                    "notional": notional,
                    "spread_cost_bps": spread_cost_bps,
                    "impact_cost_bps": impact_cost_bps,
                    "adversarial_cost_bps": adversarial_cost_bps,
                    "sor_cost_bps": sor_cost_bps,
                    "sor_venues": sor_venues,
                    "commission_bps": tier_commission_bps,
                    "tier": tier_name,
                    "total_cost_bps": spread_cost_bps
                    + impact_cost_bps
                    + adversarial_cost_bps
                    + sor_cost_bps
                    + tier_commission_bps,
                    "status": status,
                    "order_id": order_id,
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

        # Apply cash gate: reject buys that exceed available cash.
        # B4 — iterate via pre-extracted lists. ``iterrows`` was redundant here
        # because the loop only reads five columns; the vectorised form also
        # keeps ``running_cash`` accumulation strictly sequential so the cash
        # gate decision stays deterministic.
        cash = float(self._state.get("cash", self.config.seed_capital))
        buy_mask = fills["side"].str.upper() == "BUY"  # noqa: F841
        running_cash = cash
        _side_list = fills["side"].astype(str).str.upper().tolist()
        _notional_list = fills["notional"].astype(float).tolist()
        _symbol_list = fills["symbol"].astype(str).tolist() if "symbol" in fills.columns else [""] * len(fills)
        _qty_list = fills["qty"].astype(float).tolist() if "qty" in fills.columns else [0.0] * len(fills)
        _index_list = list(fills.index)
        keep_rows: list = []
        for _i in range(len(fills)):
            _s = _side_list[_i]
            _idx = _index_list[_i]
            if _s == "BUY":
                cost = _notional_list[_i]
                if running_cash - cost >= -1e-6:
                    running_cash -= cost
                    keep_rows.append(_idx)
                else:
                    logger.info(
                        "[PAPER] Cash gate: rejected BUY %s qty=%s (cash=%.2f notional=%.2f)",
                        _symbol_list[_i],
                        _qty_list[_i],
                        running_cash,
                        cost,
                    )
            else:
                keep_rows.append(_idx)

        fills = fills.loc[keep_rows].reset_index(drop=True)

        if fills.empty:
            return fills, 0.0

        avg_cost_bps = float(fills["total_cost_bps"].mean()) if "total_cost_bps" in fills.columns else 0.0
        return fills, avg_cost_bps

    # ------------------------------------------------------------------
    # Risk controls
    # ------------------------------------------------------------------

    def _apply_risk_controls(
        self,
        orders: pd.DataFrame,
        market_return_today: float | None = None,
    ) -> pd.DataFrame:
        """Apply kill switch, fat finger guard, pre-trade checks, and (optional)
        market-wide circuit breaker.

        Each control is applied only if its corresponding module is available
        and enabled in config. Returns the filtered orders DataFrame.

        Args:
            orders: Raw orders DataFrame.
            market_return_today: Intraday return of the market benchmark,
                used by the NYSE Rule 80B circuit breaker when enabled.
                ``None`` means "unknown" and skips the check.

        Returns:
            Filtered orders DataFrame (rejected orders removed).
        """
        if orders is None or orders.empty:
            return orders

        # Phase 3: market-wide circuit breaker (before kill switch so that a
        # halted market is always visible, even if kill switch is disabled).
        if (
            self.config.enable_circuit_breaker
            and _HAS_CIRCUIT_BREAKER
            and market_return_today is not None
        ):
            try:
                halted, reason = check_circuit_breaker(
                    market_return_today=market_return_today
                )
                if halted:
                    logger.warning(
                        "[PAPER] Circuit breaker halt — blocking all orders: %s",
                        reason,
                    )
                    self._last_circuit_breaker_reason = reason
                    return orders.iloc[0:0].copy()
            except Exception as exc:  # pragma: no cover
                logger.error("[PAPER] circuit breaker check error: %s", exc)

        # Kill switch
        if self.config.enable_kill_switch and _HAS_KILL_SWITCH:
            try:
                orders = guard_orders_with_kill_switch(orders)
                if orders.empty:
                    logger.warning("[PAPER] Kill switch blocked all orders")
                    return orders
            except Exception as exc:
                logger.error("[PAPER] guard_orders_with_kill_switch error: %s", exc)

        # Per-symbol kill switch (Tier-1 R5 wiring).
        # Policy-gated; default OFF. Rejected rows never reach fills.
        if (
            self.config.enable_symbol_kill_switch
            and _HAS_SYMBOL_KILL
            and not orders.empty
        ):
            try:
                filtered, reasons = _symbol_kill_filter(
                    orders,
                    policy={"symbol_kill_switch": {"enabled": True}},
                    state_path=self.config.symbol_kill_state_path,
                )
                if reasons:
                    logger.warning("[PAPER] symbol_kill_switch blocked: %s", reasons)
                orders = filtered
            except Exception as exc:
                logger.error("[PAPER] symbol_kill_switch error: %s", exc)

        # Fat finger guard
        if self.config.enable_fat_finger and _HAS_FAT_FINGER:
            try:
                # Build history from current positions for dynamic cap
                cost_basis = self._state.get("cost_basis", {})  # noqa: F841
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

        # Pre-trade checks. The upstream helper returns a
        # ``(PreTradeCheckResult, filtered_orders)`` tuple; we only need the
        # filtered orders here.
        if _HAS_PRE_TRADE:
            try:
                result = run_pre_trade_checks(orders)
                if isinstance(result, tuple) and len(result) == 2:
                    _, orders = result
                else:
                    orders = result  # defensive: older/alt signatures
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

        Uses ``fill_qty`` (partial-fill-aware) when present, falling back to ``qty``
        for legacy full-fill paths. Rows with ``status == "rejected"`` or
        ``fill_qty <= 0`` are skipped — they moved no shares and no cash.

        Args:
            fills: DataFrame with symbol, side, qty, fill_price, notional, and
                optionally fill_qty and status.
        """
        positions: dict[str, float] = self._state.setdefault("positions", {})
        cost_basis: dict[str, float] = self._state.setdefault("cost_basis", {})
        cash: float = float(self._state.get("cash", self.config.seed_capital))

        for _, fill in fills.iterrows():
            status = str(fill.get("status", "filled")).lower()
            if status == "rejected":
                continue

            sym = str(fill["symbol"])
            side = str(fill["side"]).upper()
            # Partial-fill-aware: use fill_qty when available, else fall back to qty
            raw_fill_qty = fill.get("fill_qty", fill["qty"])
            qty = abs(float(raw_fill_qty)) if pd.notna(raw_fill_qty) else abs(float(fill["qty"]))
            if qty <= 0:
                continue

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

        Event-type mapping:
          - status == "filled" or "partial" (or absent) → FILL event, qty is the
            actually filled share count (fill_qty), with a deterministic event_id
            that includes side and symbol to avoid collisions on multi-leg days.
          - status == "rejected" → REJECT event with qty=0, cash_delta=0, and
            reject_reason preserved when available.

        Args:
            fills: Filled orders DataFrame. May contain fill_qty, status,
                reject_reason columns; legacy paths without these default to
                full fill.
            as_of_date: ISO date string for the trading day.
        """
        try:
            ledger_path = self.config.ledger_dir / f"ledger_{as_of_date}.parquet"

            events = []
            ts = f"{as_of_date}T16:00:00+00:00"
            for _, fill in fills.iterrows():
                sym = str(fill["symbol"])
                side = str(fill["side"]).upper()
                status = str(fill.get("status", "filled")).lower()
                reject_reason = str(fill.get("reject_reason", "") or "")

                # Partial-fill-aware: use fill_qty when present
                raw_fill_qty = fill.get("fill_qty", fill["qty"])
                fill_qty = (
                    float(raw_fill_qty)
                    if pd.notna(raw_fill_qty)
                    else float(fill["qty"])
                )
                fill_qty = abs(fill_qty)
                price = float(fill["fill_price"])

                if status == "rejected":
                    event_type = "REJECT"
                    qty_signed = 0.0
                    cash_delta = 0.0
                else:
                    event_type = "FILL"
                    qty_signed = fill_qty if side == "BUY" else -fill_qty
                    cash_delta = fill_qty * price * (-1 if side == "BUY" else 1)

                # event_id must be unique per run/day/symbol/side to avoid
                # collisions when the same symbol has both BUY and SELL on the
                # same day (rare in paper, but real).
                event_id = (
                    f"{self.config.run_id}_{as_of_date}_{sym}_{side}_{event_type}"
                )

                event = {
                    "event_ts": ts,
                    "event_type": event_type,
                    "symbol": sym,
                    "qty": qty_signed,
                    "price": price,
                    "cash_delta": cash_delta,
                    "run_id": self.config.run_id,
                    "event_id": event_id,
                }
                if reject_reason:
                    event["reject_reason"] = reject_reason
                events.append(event)

            if events:
                df_events = pd.DataFrame(events)
                if _HAS_LEDGER:
                    try:
                        store_ledger_events_parquet(df_events, ledger_path)
                    except Exception:
                        df_events.to_parquet(ledger_path, index=False)
                else:
                    df_events.to_parquet(ledger_path, index=False)
                logger.info(
                    "[PAPER] Ledger events written: %s (%s rows)",
                    ledger_path,
                    len(events),
                )
        except Exception as exc:
            logger.error("[PAPER] Ledger write failed: %s", exc)

    # ------------------------------------------------------------------
    # Phase 5 — Borrow costs + corporate actions
    # ------------------------------------------------------------------

    def _apply_borrow_costs(
        self, as_of_date: str, prices: pd.DataFrame | None
    ) -> float:
        """Accrue one day of borrow cost for every short position and
        decrement cash. Returns the total cost in USD (always ≥ 0).

        Opt-in via ``enable_borrow_costs``. When disabled or when no short
        positions exist the call is a silent noop.
        """
        if not (self.config.enable_borrow_costs and _HAS_BORROW):
            return 0.0
        positions = self._state.get("positions", {}) or {}
        if not any(qty < 0 for qty in positions.values()):
            return 0.0

        price_map: dict[str, float] = {}
        if prices is not None and not prices.empty:
            sym_col = "symbol" if "symbol" in prices.columns else prices.columns[0]
            price_col = (
                "close" if "close" in prices.columns
                else ("price" if "price" in prices.columns else prices.columns[-1])
            )
            for _, row in prices.iterrows():
                price_map[str(row[sym_col])] = float(row[price_col])

        table = BorrowRateTable(
            default_rate_bps=float(self.config.borrow_rate_default_bps),
            htb_rate_bps=float(self.config.borrow_rate_htb_bps),
            overrides=dict(self.config.borrow_rate_overrides or {}),
            htb_symbols=set(self.config.htb_symbols or ()),
        )
        costs = compute_borrow_cost_for_positions(
            positions, price_map, rate_table=table, days_held=1
        )
        total = float(sum(costs.values()))
        if total > 0:
            self._state["cash"] = (
                float(self._state.get("cash", self.config.seed_capital)) - total
            )
            logger.info(
                "[PAPER] Borrow cost %s: %.2f USD across %d shorts",
                as_of_date, total, len(costs),
            )
            self._state.setdefault("borrow_cost_history", []).append(
                {"date": as_of_date, "cost_usd": total, "per_symbol": costs}
            )
        return total

    def _apply_corporate_actions(
        self, as_of_date: str, prices: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        """Apply splits to positions and credit dividends to cash.

        Opt-in via ``enable_corporate_actions``. Returns (possibly adjusted)
        prices. When disabled, passes through unchanged.

        - Splits: position qty *= split_ratio, cost_basis /= split_ratio.
          Prices are split-adjusted in-place via ``adjust_prices_for_splits``.
        - Dividends: cash += qty * dividend_cash for each long position.
        """
        if not self.config.enable_corporate_actions:
            logger.warning(
                "[PAPER] Corporate actions DISABLED — splits/dividends ignored. "
                "Results may be misleading."
            )
            return prices
        if not _HAS_CORP_ACTIONS:
            return prices
        ca_path = self.config.corporate_actions_path
        if ca_path is None or not Path(ca_path).exists():
            logger.warning(
                "[PAPER] corporate_actions_path not set or missing — "
                "CA adjustments disabled despite enable_corporate_actions=True. "
                "Provide a valid path to apply splits/dividends."
            )
            return prices
        try:
            actions = pd.read_csv(ca_path, dtype={"symbol": "string", "action_type": "string"})
        except Exception as exc:
            logger.warning("[PAPER] Could not read CA file: %s", exc)
            return prices
        if actions.empty:
            return prices

        actions = actions.copy()
        if "effective_date" in actions.columns:
            actions["effective_date"] = pd.to_datetime(
                actions["effective_date"], utc=True, errors="coerce"
            )
        as_of_ts = pd.Timestamp(as_of_date, tz="UTC")
        today_actions = actions[actions.get("effective_date") == as_of_ts]

        positions: dict[str, float] = self._state.setdefault("positions", {})
        cost_basis: dict[str, float] = self._state.setdefault("cost_basis", {})
        cash = float(self._state.get("cash", self.config.seed_capital))

        # Splits first (adjust qty and cost basis)
        splits = today_actions[today_actions.get("action_type") == "SPLIT"]
        for _, row in splits.iterrows():
            sym = str(row["symbol"])
            if sym not in positions:
                continue
            ratio = float(row.get("split_ratio", 1.0))
            if ratio <= 0:
                continue
            positions[sym] = positions[sym] * ratio
            if sym in cost_basis and cost_basis[sym] != 0:
                cost_basis[sym] = cost_basis[sym] / ratio

        # Dividends (long holders earn cash; shorts pay out)
        divs = today_actions[today_actions.get("action_type") == "DIVIDEND"]
        for _, row in divs.iterrows():
            sym = str(row["symbol"])
            if sym not in positions:
                continue
            per_share = float(row.get("dividend_cash", 0.0))
            cash += positions[sym] * per_share

        self._state["cash"] = cash

        # Split-adjust prices so downstream math sees the new price level.
        try:
            if not splits.empty and prices is not None and not prices.empty:
                prices = adjust_prices_for_splits(prices, splits)
        except Exception as exc:  # pragma: no cover
            logger.warning("[PAPER] split adjust error: %s", exc)

        return prices

    def _run_reconciliation(self, as_of_date: str) -> dict | None:
        """Run ledger-vs-broker reconciliation for the day and evaluate SLO.

        Compares the engine-internal ledger/state (our "source of truth") against
        a broker snapshot. When no external broker is configured the snapshot is
        the engine state itself — so a clean internal run produces a noop pass.

        On SLO violation, a JSON alert is written to
        ``config.reconcile_alerts_dir / reconcile_alert_{run_id}_{date}.json``.

        Args:
            as_of_date: ISO date string for the trading day.

        Returns:
            Dict with keys ``severity``, ``violations``, ``cash_diff_bps``,
            ``max_qty_diff``, ``reconcile``. ``None`` if reconciliation was
            skipped (e.g. missing ledger or hard error — errors are logged).
        """
        try:
            ledger_path = self.config.ledger_dir / f"ledger_{as_of_date}.parquet"

            positions = dict(self._state.get("positions", {}))
            cash = float(self._state.get("cash", self.config.seed_capital))

            ledger_positions_df = pd.DataFrame(
                [
                    {"symbol": sym, "qty": float(qty)}
                    for sym, qty in positions.items()
                ]
            )
            if ledger_positions_df.empty:
                ledger_positions_df = pd.DataFrame(columns=["symbol", "qty"])

            # Prefer an external broker snapshot if a shadow broker exposes one;
            # otherwise the engine state is its own snapshot (self-consistent noop).
            broker_positions_df = ledger_positions_df.copy()
            broker_cash = cash
            if self.config.shadow_broker is not None:
                try:
                    snap = self.config.shadow_broker.get_snapshot()  # type: ignore[attr-defined]
                    broker_positions_df = pd.DataFrame(
                        snap.get("positions", []),
                        columns=["symbol", "qty"],
                    )
                    broker_cash = float(snap.get("cash", cash))
                except Exception as exc:  # pragma: no cover
                    logger.warning("[PAPER] shadow broker snapshot failed: %s", exc)

            recon = reconcile_ledger_vs_broker(
                ledger_positions_df,
                cash,
                broker_positions_df,
                broker_cash,
                fail_fast=False,
            )

            pos_diffs = recon.get("position_diffs_df")
            max_qty_diff = 0.0
            if pos_diffs is not None and not pos_diffs.empty:
                max_qty_diff = float(pos_diffs["diff_qty"].abs().max())

            # Fill-rate + slippage: pulled from this run's fills if available.
            fill_rate: float | None = None
            slippage_p99: float | None = None
            last_fills = self._last_fills
            last_orders_n = int(self._last_orders_n)
            if isinstance(last_fills, pd.DataFrame) and not last_fills.empty:
                filled = last_fills[
                    last_fills.get("status", pd.Series(["filled"] * len(last_fills)))
                    != "rejected"
                ]
                if last_orders_n > 0:
                    fill_rate = float(len(filled)) / float(last_orders_n)
                if "arrival_price" in last_fills.columns and "fill_price" in last_fills.columns:
                    ap = last_fills["arrival_price"].astype(float)
                    fp = last_fills["fill_price"].astype(float)
                    non_zero = ap.abs() > 0
                    if non_zero.any():
                        slip = ((fp - ap).abs() / ap.abs() * 10_000.0)[non_zero]
                        if len(slip) > 0:
                            slippage_p99 = float(slip.quantile(0.99))

            slo = self.config.reconcile_slo or ReconcileSLO()
            verdict = evaluate_reconcile_slo(
                cash_diff=float(recon.get("cash_diff", 0.0)),
                broker_cash=broker_cash,
                max_qty_diff=max_qty_diff,
                fill_rate=fill_rate,
                slippage_p99_bps=slippage_p99,
                slo=slo,
            )
            verdict["reconcile"] = {
                "cash_match": recon.get("cash_match"),
                "ok": recon.get("ok"),
                "ledger_exists": ledger_path.exists(),
                "n_positions": int(len(ledger_positions_df)),
            }

            if verdict["severity"] != "ok":
                self._write_reconcile_alert(as_of_date, verdict)

            logger.info(
                "[PAPER] Reconciliation %s severity=%s cash_diff_bps=%.2f max_qty_diff=%.4f",
                as_of_date,
                verdict["severity"],
                verdict["cash_diff_bps"],
                verdict["max_qty_diff"],
            )
            return verdict
        except Exception as exc:
            logger.warning("[PAPER] Reconciliation skipped: %s", exc)
            return None

    def _write_reconcile_alert(self, as_of_date: str, verdict: dict) -> Path:
        out_dir = Path(self.config.reconcile_alerts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"reconcile_alert_{self.config.run_id}_{as_of_date}.json"
        payload = {
            "run_id": self.config.run_id,
            "date": as_of_date,
            "severity": verdict["severity"],
            "cash_diff_bps": verdict["cash_diff_bps"],
            "max_qty_diff": verdict["max_qty_diff"],
            "violations": verdict["violations"],
            "reconcile": verdict.get("reconcile", {}),
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        from src.assembled_core.utils.atomic_io import atomic_write_json
        atomic_write_json(out_path, payload)
        logger.warning(
            "[PAPER] Reconcile alert written: %s (severity=%s)",
            out_path, verdict["severity"],
        )
        return out_path

    # ------------------------------------------------------------------
    # Phase 7 — TCA artifacts
    # ------------------------------------------------------------------

    def _write_tca_artifacts(
        self,
        as_of_date: str,
        orders: pd.DataFrame,
        fills: pd.DataFrame,
    ) -> tuple[Path, Path] | None:
        """Write per-order TCA CSV and aggregate JSON for the day.

        Per-order columns: symbol, side, qty, fill_qty, arrival_price, fill_price,
        arrival_slippage_bps, spread_cost_bps, impact_cost_bps,
        adversarial_cost_bps, sor_cost_bps, total_cost_bps, status.

        Aggregate JSON:
          * n_orders, n_fills, fill_rate
          * p50/p90/p99 arrival_slippage_bps
          * avg/total per-category cost bps
        """
        if not self.config.enable_tca:
            return None
        if (fills is None or fills.empty) and (orders is None or orders.empty):
            return None

        # Per-order frame (empty frame = still write an empty CSV so downstream
        # tooling can safely assume the artifact exists after a run).
        cols = [
            "date", "symbol", "side", "qty", "fill_qty",
            "arrival_price", "fill_price",
            "arrival_slippage_bps",
            "spread_cost_bps", "impact_cost_bps",
            "adversarial_cost_bps", "sor_cost_bps", "total_cost_bps",
            "status",
        ]
        rows: list[dict] = []
        if fills is not None and not fills.empty:
            for _, f in fills.iterrows():
                ap = float(f.get("arrival_price", float("nan")))
                fp = float(f.get("fill_price", float("nan")))
                slip_bps = float("nan")
                side_sign = 1.0 if str(f.get("side", "BUY")).upper() == "BUY" else -1.0
                if ap == ap and fp == fp and ap > 0:
                    slip_bps = side_sign * (fp - ap) / ap * 10_000.0
                rows.append({
                    "date": as_of_date,
                    "symbol": str(f.get("symbol", "")),
                    "side": str(f.get("side", "")),
                    "qty": float(f.get("qty", 0.0)),
                    "fill_qty": float(f.get("fill_qty", f.get("qty", 0.0))),
                    "arrival_price": ap,
                    "fill_price": fp,
                    "arrival_slippage_bps": slip_bps,
                    "spread_cost_bps": float(f.get("spread_cost_bps", 0.0)),
                    "impact_cost_bps": float(f.get("impact_cost_bps", 0.0)),
                    "adversarial_cost_bps": float(f.get("adversarial_cost_bps", 0.0)),
                    "sor_cost_bps": float(f.get("sor_cost_bps", 0.0)),
                    "total_cost_bps": float(f.get("total_cost_bps", 0.0)),
                    "status": str(f.get("status", "filled")),
                })
        per_order = pd.DataFrame(rows, columns=cols)

        out_dir = Path(self.config.tca_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"tca_{self.config.run_id}_{as_of_date}.csv"
        json_path = out_dir / f"tca_{self.config.run_id}_{as_of_date}.json"
        per_order.to_csv(csv_path, index=False)

        # Aggregate metrics
        n_orders = int(len(orders)) if orders is not None else 0
        n_fills = int((per_order["status"] != "rejected").sum()) if not per_order.empty else 0
        fill_rate = (n_fills / n_orders) if n_orders > 0 else 0.0

        def _pct(series: pd.Series, q: float) -> float:
            s = series.dropna()
            if s.empty:
                return 0.0
            return float(s.abs().quantile(q))

        def _avg(series: pd.Series) -> float:
            s = series.dropna()
            return float(s.mean()) if not s.empty else 0.0

        def _total(series: pd.Series) -> float:
            s = series.dropna()
            return float(s.sum()) if not s.empty else 0.0

        aggregate = {
            "run_id": self.config.run_id,
            "date": as_of_date,
            "n_orders": n_orders,
            "n_fills": n_fills,
            "fill_rate": fill_rate,
            "slippage_bps": {
                "p50": _pct(per_order.get("arrival_slippage_bps", pd.Series(dtype=float)), 0.50),
                "p90": _pct(per_order.get("arrival_slippage_bps", pd.Series(dtype=float)), 0.90),
                "p99": _pct(per_order.get("arrival_slippage_bps", pd.Series(dtype=float)), 0.99),
            },
            "cost_bps_avg": {
                "spread": _avg(per_order.get("spread_cost_bps", pd.Series(dtype=float))),
                "impact": _avg(per_order.get("impact_cost_bps", pd.Series(dtype=float))),
                "adversarial": _avg(per_order.get("adversarial_cost_bps", pd.Series(dtype=float))),
                "sor": _avg(per_order.get("sor_cost_bps", pd.Series(dtype=float))),
                "total": _avg(per_order.get("total_cost_bps", pd.Series(dtype=float))),
            },
            "cost_bps_sum": {
                "spread": _total(per_order.get("spread_cost_bps", pd.Series(dtype=float))),
                "impact": _total(per_order.get("impact_cost_bps", pd.Series(dtype=float))),
                "adversarial": _total(per_order.get("adversarial_cost_bps", pd.Series(dtype=float))),
                "sor": _total(per_order.get("sor_cost_bps", pd.Series(dtype=float))),
                "total": _total(per_order.get("total_cost_bps", pd.Series(dtype=float))),
            },
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        from src.assembled_core.utils.atomic_io import atomic_write_json
        atomic_write_json(json_path, aggregate)
        logger.info("[PAPER] TCA artifacts written: %s", csv_path)
        return csv_path, json_path

    # ------------------------------------------------------------------
    # Phase 9 — Attribution drilldown (cost / regime / factor)
    # ------------------------------------------------------------------

    def _write_attribution_artifacts(
        self,
        as_of_date: str,
        fills: pd.DataFrame,
        regime_history: list[dict] | None = None,
    ) -> tuple[Path, Path] | None:
        """Write per-symbol attribution CSV + aggregate JSON for the day.

        CSV: cost attribution per symbol (notional-weighted bps + cash).
        JSON: ``{total, regime, factor}`` where ``regime``/``factor`` are lists
        of rows produced by the attribution helpers.
        """
        if not getattr(self.config, "enable_attribution", False):
            return None
        if not _HAS_ATTRIBUTION:
            return None
        if fills is None or fills.empty:
            return None

        cost = compute_cost_attribution(fills)
        regime = compute_regime_attribution(fills, regime_history or [])
        factor = compute_factor_attribution(fills)

        out_dir = Path(self.config.attribution_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = self.config.run_id
        csv_path = out_dir / f"attribution_{run_id}_{as_of_date}.csv"
        json_path = out_dir / f"attribution_{run_id}_{as_of_date}.json"

        per_symbol: pd.DataFrame = cost["per_symbol"]
        per_symbol.to_csv(csv_path, index=False)

        payload = {
            "run_id": run_id,
            "date": as_of_date,
            "total": cost["total"],
            "regime": regime.to_dict(orient="records"),
            "factor": factor.to_dict(orient="records"),
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        from src.assembled_core.utils.atomic_io import atomic_write_json
        atomic_write_json(json_path, payload)
        logger.info("[PAPER] Attribution artifacts written: %s", csv_path)
        return csv_path, json_path

    # ------------------------------------------------------------------
    # Phase 8 — Run manifest + cross-run index
    # ------------------------------------------------------------------

    def _write_manifest_and_index(
        self,
        *,
        as_of_date: str,
        run_started_utc: str,
        status: str,
        equity_after: float,
        total_return: float,
        n_fills: int,
        total_cost_bps: float,
    ) -> Path | None:
        if not self.config.enable_manifest or not _HAS_MANIFEST:
            return None
        run_id = self.config.run_id
        candidates: dict[str, Path] = {
            "ledger": self.config.ledger_dir / f"ledger_{as_of_date}.parquet",
            "lifecycle": (
                self.config.lifecycle_dir / f"lifecycle_{run_id}_{as_of_date}.jsonl"
            ),
            "tca_csv": self.config.tca_dir / f"tca_{run_id}_{as_of_date}.csv",
            "tca_json": self.config.tca_dir / f"tca_{run_id}_{as_of_date}.json",
            "attribution_csv": (
                self.config.attribution_dir
                / f"attribution_{run_id}_{as_of_date}.csv"
            ),
            "attribution_json": (
                self.config.attribution_dir
                / f"attribution_{run_id}_{as_of_date}.json"
            ),
            "reconcile_alert": (
                self.config.reconcile_alerts_dir
                / f"reconcile_alert_{run_id}_{as_of_date}.json"
            ),
            "shadow_compare": (
                self.config.shadow_compare_dir
                / f"shadow_compare_{run_id}_{as_of_date}.csv"
            ),
        }
        if self.config.replay_snapshot_dir is not None:
            candidates["replay_snapshot"] = (
                Path(self.config.replay_snapshot_dir)
                / f"replay_{run_id}_{as_of_date}.json"
            )

        metrics = {
            "final_equity": float(equity_after),
            "total_return": float(total_return),
            "n_fills": int(n_fills),
            "avg_cost_bps": float(total_cost_bps),
        }

        manifest_path = write_run_manifest(
            run_id=run_id,
            date=as_of_date,
            started_at_utc=run_started_utc,
            status=status,
            config=self.config,
            artifacts=candidates,
            metrics=metrics,
            phase_versions={"paper_engine": "phase8"},
            manifests_dir=self.config.manifests_dir,
        )

        # Re-derive sha/hash the same way the manifest did, for the index row.
        from src.assembled_core.ops.run_manifest import (
            _compute_git_sha,
        )
        append_run_index(
            run_id=run_id,
            date=as_of_date,
            status=status,
            metrics=metrics,
            git_sha=_compute_git_sha(),
            config_hash=compute_config_hash(self.config),
            manifest_path=manifest_path,
            index_path=self.config.run_index_path,
        )
        return manifest_path

    @staticmethod
    def _order_pair_key(order_id: Any, pos: int) -> str:
        """Stable per-order pairing key.

        H1: ORDER_SUBMIT / ORDER_COMPLETE pairing must be 1:1 per order. Keying
        by ``(symbol, side)`` collides whenever SOR child splits or a strategy
        emits duplicate (symbol, side) rows in the same day. We key by
        ``order_id`` when present (lifecycle already attaches these) and fall
        back to a stable positional tag otherwise.
        """
        if order_id is not None and pd.notna(order_id):
            return f"oid::{str(order_id)}"
        return f"pos::{int(pos)}"

    def _record_submit_intents(self, orders: pd.DataFrame) -> list[tuple[str, str]]:
        """Write ORDER_SUBMIT intents — returns ``[(pair_key, intent_key), ...]``.

        The outer list is positionally aligned with ``orders.iterrows()`` so
        the complete step can walk both in lock-step. The ``pair_key`` lets
        the complete step resolve the matching fill even when multiple orders
        share ``(symbol, side)``.
        """
        if not self.config.enable_intent_store:
            return []
        try:
            from src.assembled_core.execution.intent_store import record_order_submit
        except Exception:
            return []
        keys: list[tuple[str, str]] = []
        for pos, (_, row) in enumerate(orders.iterrows()):
            pair_key = self._order_pair_key(row.get("order_id"), pos)
            try:
                rec = record_order_submit(
                    str(row.get("symbol", "")),
                    str(row.get("side", "")),
                    float(row.get("qty", 0.0)),
                    store_path=self.config.intent_store_path,
                )
                keys.append((pair_key, rec["idempotency_key"]))
            except Exception as exc:  # pragma: no cover
                logger.warning("[PAPER] intent submit failed: %s", exc)
                keys.append((pair_key, ""))
        return keys

    def _record_complete_intents(
        self,
        orders: pd.DataFrame,
        fills: pd.DataFrame,
        submit_keys: list[tuple[str, str]],
    ) -> None:
        """Pair ORDER_COMPLETE intents with their submits 1:1 per order."""
        if not self.config.enable_intent_store or not submit_keys:
            return
        try:
            from src.assembled_core.execution.intent_store import (
                record_order_complete,
            )
        except Exception:
            return

        # H1: index fills by the same pair_key used on submit so SOR splits or
        # duplicate (symbol, side) orders cannot share a fill.
        fill_by_key: dict[str, pd.Series] = {}
        if fills is not None and not fills.empty:
            has_fill_oid = "order_id" in fills.columns
            for pos, (_, f) in enumerate(fills.iterrows()):
                fill_oid = f.get("order_id") if has_fill_oid else None
                fill_by_key[self._order_pair_key(fill_oid, pos)] = f

        for (_, order), (pair_key, key) in zip(
            orders.iterrows(), submit_keys, strict=False
        ):
            if not key:
                continue
            sym = str(order.get("symbol", ""))
            side = str(order.get("side", ""))
            fill = fill_by_key.get(pair_key)
            filled_qty = (
                float(fill.get("fill_qty", fill.get("qty", 0.0))) if fill is not None else 0.0
            )
            filled_price = (
                float(fill.get("fill_price", 0.0)) if fill is not None else None
            )
            status = str(fill.get("status", "filled")) if fill is not None else "rejected"
            try:
                record_order_complete(
                    sym,
                    side,
                    float(order.get("qty", 0.0)),
                    filled_qty=filled_qty,
                    filled_price=filled_price,
                    status=status,
                    intent_key=key,
                    store_path=self.config.intent_store_path,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("[PAPER] intent complete failed: %s", exc)

    def _run_shadow_compare(
        self,
        as_of_date: str,
        orders: pd.DataFrame,
        fills: pd.DataFrame,
    ) -> Path | None:
        """Optional: compare engine fills against a live shadow broker's fills.

        Writes ``shadow_compare_{run_id}_{date}.csv``. No decision influence —
        purely observability. If no shadow_broker is configured the call is a
        noop.
        """
        if not self.config.shadow_mode or self.config.shadow_broker is None:
            return None
        if orders is None or orders.empty:
            return None
        rows: list[dict] = []
        for _, order in orders.iterrows():
            sym = str(order.get("symbol", ""))
            side = str(order.get("side", ""))
            qty = float(order.get("qty", 0.0))
            sim_fill = fills[
                (fills.get("symbol") == sym) & (fills.get("side") == side)
            ]
            sim_price = (
                float(sim_fill.iloc[0]["fill_price"])
                if not sim_fill.empty and "fill_price" in sim_fill.columns
                else float("nan")
            )
            sim_status = (
                str(sim_fill.iloc[0].get("status", "filled"))
                if not sim_fill.empty else "rejected"
            )
            try:
                live = self.config.shadow_broker.submit(  # type: ignore[attr-defined]
                    symbol=sym, side=side, qty=qty
                )
                live_price = float(live.get("fill_price", float("nan")))
                live_status = str(live.get("status", "unknown"))
            except Exception as exc:  # pragma: no cover
                logger.warning("[PAPER] shadow submit failed: %s", exc)
                live_price = float("nan")
                live_status = "error"
            diff_bps = float("nan")
            if sim_price and live_price and sim_price == sim_price and live_price == live_price:
                if sim_price != 0:
                    diff_bps = (live_price - sim_price) / sim_price * 10_000.0
            rows.append({
                "date": as_of_date,
                "symbol": sym,
                "side": side,
                "qty": qty,
                "sim_fill_price": sim_price,
                "live_fill_price": live_price,
                "diff_bps": diff_bps,
                "sim_status": sim_status,
                "live_status": live_status,
            })
        out_dir = Path(self.config.shadow_compare_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"shadow_compare_{self.config.run_id}_{as_of_date}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        logger.info("[PAPER] Shadow compare written: %s", out_path)
        return out_path

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

    # ------------------------------------------------------------------
    # Phase 3 helpers — circuit breaker benchmark lookup
    # ------------------------------------------------------------------

    def _extract_benchmark_return(self, prices: pd.DataFrame | None) -> float | None:
        """Return today's benchmark intraday return or ``None`` if unavailable.

        Looks for the configured ``market_benchmark`` symbol in ``prices`` and
        uses either an explicit ``return`` column, or falls back to
        ``close/open - 1`` if open/high/low are present. Returns ``None`` when
        the benchmark is missing or data is insufficient (the circuit breaker
        then skips the check rather than emitting a false positive).
        """
        if prices is None or prices.empty:
            return None
        sym_col = "symbol" if "symbol" in prices.columns else prices.columns[0]
        row = prices[prices[sym_col] == self.config.market_benchmark]
        if row.empty:
            return None
        row = row.iloc[0]
        if "return" in prices.columns and pd.notna(row["return"]):
            try:
                return float(row["return"])
            except (TypeError, ValueError):
                return None
        if "open" in prices.columns and "close" in prices.columns:
            try:
                open_px = float(row["open"])
                close_px = float(row["close"])
                if open_px > 0:
                    return close_px / open_px - 1.0
            except (TypeError, ValueError):
                return None
        return None

    # ------------------------------------------------------------------
    # Determinism helpers (Phase 1.5)
    # ------------------------------------------------------------------

    def _rng(self, as_of_date: str):
        """Return a numpy Generator seeded deterministically for this day.

        The seed is derived from ``run_id + as_of_date + config.random_seed``.
        When ``random_seed`` is None, determinism still holds across runs
        with the same ``run_id``/date — that is the "deterministic-by-default"
        mode for paper trading.

        Returns ``None`` when numpy / replay_snapshot is not importable, in
        which case callers must fall back to non-stochastic paths.
        """
        if not _HAS_REPLAY:
            return None
        return make_rng(self.config.run_id, as_of_date, self.config.random_seed)

    def _maybe_save_replay_snapshot(
        self,
        as_of_date: str,
        prices: pd.DataFrame | None,
        signals: pd.DataFrame | None = None,
        context: dict | None = None,
    ) -> None:
        """Persist a replay snapshot when ``replay_snapshot_dir`` is configured.

        Best-effort; failures are logged and do not fail the day.
        """
        if not _HAS_REPLAY or self.config.replay_snapshot_dir is None:
            return
        if prices is None or prices.empty:
            return
        try:
            snap = RunSnapshot(
                run_id=self.config.run_id,
                as_of_date=as_of_date,
                seed=self.config.random_seed,
                prices=prices,
                signals=signals,
                context=context or {},
            )
            snap.save(self.config.replay_snapshot_dir)
        except Exception as exc:  # pragma: no cover
            logger.warning("[PAPER] replay snapshot save failed: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle tracking helpers
    # ------------------------------------------------------------------

    def _lifecycle_attach(
        self, orders: pd.DataFrame | None, as_of_date: str
    ) -> pd.DataFrame | None:
        """Attach order_id column and create CREATED events in the tracker.

        If orders already carry an order_id column (e.g. injected by caller),
        that id is reused. Otherwise a stable id of the form
        ``{run_id}-{as_of_date}-{i}`` is assigned.

        Returns orders unchanged when lifecycle tracking is disabled or no
        tracker is configured.
        """
        if self._lifecycle is None or orders is None or orders.empty:
            return orders

        if "order_id" not in orders.columns:
            orders = orders.copy()
            orders["order_id"] = [
                f"{self.config.run_id}-{as_of_date}-{i}" for i in range(len(orders))
            ]

        for _, row in orders.iterrows():
            oid = str(row["order_id"])
            if self._lifecycle.get_order(oid) is not None:
                continue
            self._lifecycle.create(
                symbol=str(row.get("symbol", "")),
                side=str(row.get("side", "")),
                quantity=float(row.get("qty", 0.0)),
                price=(float(row["price"]) if pd.notna(row.get("price")) else None),
                source=str(row.get("source", "PAPER")),
                order_id=oid,
            )
        return orders

    def _lifecycle_mark_validation(
        self, pre_ids: list[str], post_ids: list[str]
    ) -> None:
        """Mark survivors VALIDATED and dropped orders REJECTED.

        Uses set semantics: any id in pre_ids but not in post_ids is treated
        as blocked by risk controls.
        """
        if self._lifecycle is None:
            return
        post_set = set(post_ids)
        for oid in pre_ids:
            order = self._lifecycle.get_order(oid)
            if order is None:
                continue
            if order.current_state != OrderState.CREATED:
                continue
            if oid in post_set:
                try:
                    self._lifecycle.transition(oid, OrderState.VALIDATED)
                except ValueError:
                    pass
            else:
                try:
                    self._lifecycle.transition(
                        oid, OrderState.REJECTED, reason="risk_control_block"
                    )
                except ValueError:
                    pass

    def _lifecycle_mark_submitted(self, order_ids: list[str]) -> None:
        """Transition VALIDATED → SUBMITTED for each id just before fill sim."""
        if self._lifecycle is None:
            return
        for oid in order_ids:
            order = self._lifecycle.get_order(oid)
            if order is None or order.current_state != OrderState.VALIDATED:
                continue
            try:
                self._lifecycle.transition(oid, OrderState.SUBMITTED)
            except ValueError:
                pass

    def _lifecycle_mark_fills(
        self,
        orders: pd.DataFrame,
        fills: pd.DataFrame,
        submitted_ids: list[str],
    ) -> None:
        """Mark terminal states based on fill outcome.

        Matching strategy: if fills carries an order_id column, match by id;
        otherwise fall back to first-match on symbol+side. Submitted orders
        with no fill row are marked CANCELLED (cash gate / missing price).
        """
        if self._lifecycle is None or not submitted_ids:
            return

        matched: set[str] = set()

        if fills is not None and not fills.empty:
            has_oid_col = "order_id" in fills.columns
            for _, fill in fills.iterrows():
                status = str(fill.get("status", "filled")).lower()
                fill_qty = fill.get("fill_qty", fill.get("qty", 0.0))
                try:
                    fill_qty_f = float(fill_qty) if pd.notna(fill_qty) else 0.0
                except (TypeError, ValueError):
                    fill_qty_f = 0.0
                fill_price = fill.get("fill_price")
                try:
                    fill_price_f = float(fill_price) if pd.notna(fill_price) else None
                except (TypeError, ValueError):
                    fill_price_f = None

                # Resolve target order_id
                oid: str | None = None
                if has_oid_col and pd.notna(fill.get("order_id")):
                    oid = str(fill["order_id"])
                else:
                    sym = str(fill.get("symbol", ""))
                    side = str(fill.get("side", "")).upper()
                    for cand in submitted_ids:
                        if cand in matched:
                            continue
                        order = self._lifecycle.get_order(cand)
                        if order is None:
                            continue
                        if order.symbol == sym and order.side.upper() == side:
                            oid = cand
                            break
                if oid is None:
                    continue
                matched.add(oid)

                order = self._lifecycle.get_order(oid)
                if order is None or order.current_state != OrderState.SUBMITTED:
                    continue

                if status == "rejected":
                    target = OrderState.REJECTED
                    reason = str(fill.get("reject_reason", "") or "fill_rejected")
                    try:
                        self._lifecycle.transition(oid, target, reason=reason)
                    except ValueError:
                        pass
                elif status == "partial":
                    try:
                        self._lifecycle.transition(
                            oid,
                            OrderState.PARTIAL_FILL,
                            fill_price=fill_price_f,
                            fill_qty=fill_qty_f,
                        )
                    except ValueError:
                        pass
                else:
                    try:
                        self._lifecycle.transition(
                            oid,
                            OrderState.FILLED,
                            fill_price=fill_price_f,
                            fill_qty=fill_qty_f,
                        )
                    except ValueError:
                        pass

        # Any submitted-but-not-matched order → CANCELLED at EOD.
        for oid in submitted_ids:
            if oid in matched:
                continue
            order = self._lifecycle.get_order(oid)
            if order is None or order.current_state != OrderState.SUBMITTED:
                continue
            try:
                self._lifecycle.transition(
                    oid, OrderState.CANCELLED, reason="eod_no_fill"
                )
            except ValueError:
                pass

    def _lifecycle_dump(self, as_of_date: str) -> None:
        """Write lifecycle snapshots for orders that went terminal this run.

        Output: ``lifecycle_dir / lifecycle_{run_id}_{date}.jsonl`` — one line
        per tracked order that reached a terminal state and has not yet been
        dumped. The in-memory ``_lifecycle_dumped_ids`` set ensures that
        multi-day runs do not re-emit the same order across daily files.
        Best-effort; errors are logged but do not fail the day.
        """
        if self._lifecycle is None:
            return
        try:
            path = self.config.lifecycle_dir / (
                f"lifecycle_{self.config.run_id}_{as_of_date}.jsonl"
            )
            lines = []
            fresh_ids: list[str] = []
            for order in self._lifecycle.get_all_orders():
                if not order.is_terminal:
                    continue
                if order.order_id in self._lifecycle_dumped_ids:
                    continue
                lines.append(json.dumps(order.to_dict()))
                fresh_ids.append(order.order_id)
            if lines:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                self._lifecycle_dumped_ids.update(fresh_ids)
                logger.debug(
                    "[PAPER] Lifecycle dump %s (%d rows)", path, len(lines)
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("[PAPER] Lifecycle dump failed: %s", exc)

    # ------------------------------------------------------------------

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
