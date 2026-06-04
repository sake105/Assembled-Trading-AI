"""Reconciliation engine: Compare ledger state vs broker snapshots (Sprint 13 L3).

This module provides functions to reconcile ledger positions/cash against
broker snapshots (paper or live), detecting mismatches and missing positions.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


_DEFAULT_RECON_AUDIT = Path("output/ops/reconciliation_audit.jsonl")


def _recon_audit_path() -> Path:
    override = os.environ.get("ASSEMBLED_RECONCILE_AUDIT", "")
    return Path(override) if override else _DEFAULT_RECON_AUDIT


def _append_recon_audit(event: dict) -> None:
    """Append a JSONL entry with fsync (audit C3-072).

    Used by evaluate_reconcile_slo to leave a record of every reconciliation
    outcome (ok / warn / fail). Best-effort — write failures are logged but
    must not crash the reconciliation itself.
    """
    p = _recon_audit_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        event["ts"] = datetime.now(timezone.utc).isoformat()
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError as exc:
                logger.debug("[recon-audit] fsync failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("[recon-audit] write failed for %s: %s", p, exc)


@dataclass
class ReconcileSLO:
    """Service-level objectives for ledger-vs-broker reconciliation.

    Warn thresholds are informational; fail thresholds should block production
    and surface as alerts. Units:

    - ``cash_diff_bps_*``: absolute cash diff in basis points of broker_cash.
    - ``position_qty_diff_*``: absolute share-count diff per symbol.
    - ``fill_rate_min_*``: minimum fraction of submitted orders that must fill.
    - ``slippage_p99_bps_*``: 99th-percentile per-fill slippage against arrival.

    The defaults follow the Phase 6 plan: warn-at-lenient, fail-at-strict.
    """

    cash_diff_bps_warn: float = 5.0
    cash_diff_bps_fail: float = 25.0
    position_qty_diff_warn: float = 1.0
    position_qty_diff_fail: float = 10.0
    fill_rate_min_warn: float = 0.80
    fill_rate_min_fail: float = 0.50
    slippage_p99_bps_warn: float = 30.0
    slippage_p99_bps_fail: float = 100.0


def evaluate_reconcile_slo(
    *,
    cash_diff: float,
    broker_cash: float,
    max_qty_diff: float,
    fill_rate: float | None,
    slippage_p99_bps: float | None,
    slo: ReconcileSLO,
    suppress_side_effects: bool = False,
) -> dict:
    """Classify a reconciliation result against SLO thresholds.

    Returns a dict with ``severity`` in {"ok", "warn", "fail"} and a list of
    ``violations`` explaining which SLOs were breached.

    Args:
        suppress_side_effects: When True, skip BOTH the AlertManager().fire(...)
            dispatch and the reconciliation-audit JSONL append. The pure
            classification (severity / violations / cash_diff_bps / max_qty_diff)
            is computed and returned identically. Defaults to False so every
            existing live/paper caller keeps firing alerts and writing the
            ops-audit trail. Backtest callers (E-035) pass True so replaying
            historical drift does NOT emit real production alerts nor pollute
            the live ops audit log.
    """
    violations: list[dict] = []

    # Cash diff in bps of broker_cash (guard divide-by-zero)
    denom = max(abs(broker_cash), 1.0)
    cash_bps = abs(cash_diff) / denom * 10_000.0

    if cash_bps >= slo.cash_diff_bps_fail:
        violations.append(
            {
                "metric": "cash_diff_bps",
                "value": cash_bps,
                "threshold": slo.cash_diff_bps_fail,
                "severity": "fail",
            }
        )
    elif cash_bps >= slo.cash_diff_bps_warn:
        violations.append(
            {
                "metric": "cash_diff_bps",
                "value": cash_bps,
                "threshold": slo.cash_diff_bps_warn,
                "severity": "warn",
            }
        )

    if max_qty_diff >= slo.position_qty_diff_fail:
        violations.append(
            {
                "metric": "position_qty_diff",
                "value": max_qty_diff,
                "threshold": slo.position_qty_diff_fail,
                "severity": "fail",
            }
        )
    elif max_qty_diff >= slo.position_qty_diff_warn:
        violations.append(
            {
                "metric": "position_qty_diff",
                "value": max_qty_diff,
                "threshold": slo.position_qty_diff_warn,
                "severity": "warn",
            }
        )

    if fill_rate is not None:
        if fill_rate < slo.fill_rate_min_fail:
            violations.append(
                {
                    "metric": "fill_rate",
                    "value": fill_rate,
                    "threshold": slo.fill_rate_min_fail,
                    "severity": "fail",
                }
            )
        elif fill_rate < slo.fill_rate_min_warn:
            violations.append(
                {
                    "metric": "fill_rate",
                    "value": fill_rate,
                    "threshold": slo.fill_rate_min_warn,
                    "severity": "warn",
                }
            )

    if slippage_p99_bps is not None:
        if slippage_p99_bps >= slo.slippage_p99_bps_fail:
            violations.append(
                {
                    "metric": "slippage_p99_bps",
                    "value": slippage_p99_bps,
                    "threshold": slo.slippage_p99_bps_fail,
                    "severity": "fail",
                }
            )
        elif slippage_p99_bps >= slo.slippage_p99_bps_warn:
            violations.append(
                {
                    "metric": "slippage_p99_bps",
                    "value": slippage_p99_bps,
                    "threshold": slo.slippage_p99_bps_warn,
                    "severity": "warn",
                }
            )

    if any(v["severity"] == "fail" for v in violations):
        severity = "fail"
    elif violations:
        severity = "warn"
    else:
        severity = "ok"

    # E-035 backtest-safety: a backtest replaying historical drift must classify
    # SILENTLY. When suppress_side_effects is True we skip BOTH the alert dispatch
    # and the live ops-audit append — the verdict returned below is identical
    # either way. Live/paper callers leave it False and keep both side effects.
    if not suppress_side_effects:
        # Dispatch alert when SLO is breached. AlertManager is cooldown-aware
        # (configs/alerting.yaml), so a repeat fail does not spam channels —
        # the manager handles strike-counting / suppression itself.
        if severity in ("warn", "fail"):
            try:
                from src.assembled_core.ops.alerting import AlertManager

                rule_name = (
                    "reconciliation_fail"
                    if severity == "fail"
                    else "reconciliation_warn"
                )
                AlertManager().fire(
                    rule_name,
                    {
                        "cash_diff_bps": cash_bps,
                        "max_qty_diff": max_qty_diff,
                        "violation_count": len(violations),
                        "first_violation": (
                            violations[0]["metric"] if violations else ""
                        ),
                    },
                )
            except Exception as _ae:  # alerting must never block reconciliation
                logger.debug("[Reconciliation] alert dispatch failed: %s", _ae)

        # Append every SLO evaluation to the audit log so a regulator (or a
        # future post-mortem) can reconstruct what the reconciler saw at each
        # tick — not just the breaches.
        _append_recon_audit(
            {
                "kind": "slo_eval",
                "severity": severity,
                "cash_diff_bps": cash_bps,
                "max_qty_diff": max_qty_diff,
                "violation_count": len(violations),
                "first_violation": violations[0]["metric"] if violations else "",
            }
        )

    return {
        "severity": severity,
        "violations": violations,
        "cash_diff_bps": cash_bps,
        "max_qty_diff": max_qty_diff,
    }


def reconcile_ledger_vs_broker(
    ledger_positions_df: pd.DataFrame,
    ledger_cash: float,
    broker_positions_df: pd.DataFrame,
    broker_cash: float,
    *,
    cash_tol: float = 1e-8,
    qty_tol: float = 1e-6,
    fail_fast: bool = True,
) -> dict:
    """Reconcile ledger state vs broker snapshot.

    This function compares:
    1. Cash balances (within cash_tol tolerance)
    2. Positions (symbol-by-symbol, within qty_tol tolerance)
    3. Missing positions (in ledger but not broker, or vice versa)

    Args:
        ledger_positions_df: Ledger positions DataFrame with columns: symbol, qty
            (may have additional columns like avg_price, realized_pnl, etc.)
        ledger_cash: Ledger cash balance (float)
        broker_positions_df: Broker positions DataFrame with columns: symbol, qty
            (may have additional columns)
        broker_cash: Broker cash balance (float)
        cash_tol: Cash tolerance (default: 1e-8)
            Differences <= cash_tol are treated as zero
        qty_tol: Quantity tolerance (default: 1e-6)
            Differences <= qty_tol are treated as zero
        fail_fast: If True, raise ValueError on mismatch (default: True)
            If False, return reconciliation report with ok=False

    Returns:
        Dictionary with keys:
        - ok: bool (True if all checks pass)
        - cash_match: bool (True if cash within tolerance)
        - cash_diff: float (ledger_cash - broker_cash)
        - position_diffs_df: DataFrame with columns:
            - symbol: str
            - ledger_qty: float
            - broker_qty: float
            - diff_qty: float (ledger_qty - broker_qty)
            Only includes symbols with differences > qty_tol
        - missing_in_ledger: list[str] (symbols in broker but not ledger)
        - missing_in_broker: list[str] (symbols in ledger but not broker)
        - message: str (human-readable summary)

    Raises:
        ValueError: If fail_fast=True and mismatch detected
        ValueError: If required columns (symbol, qty) are missing
    """
    # Validate inputs
    if ledger_positions_df.empty:
        ledger_positions_df = pd.DataFrame(columns=["symbol", "qty"])
    if broker_positions_df.empty:
        broker_positions_df = pd.DataFrame(columns=["symbol", "qty"])

    # Validate required columns
    required_cols = ["symbol", "qty"]
    missing_ledger = [
        col for col in required_cols if col not in ledger_positions_df.columns
    ]
    missing_broker = [
        col for col in required_cols if col not in broker_positions_df.columns
    ]
    if missing_ledger:
        raise ValueError(
            f"Missing required columns in ledger_positions_df: {missing_ledger}"
        )
    if missing_broker:
        raise ValueError(
            f"Missing required columns in broker_positions_df: {missing_broker}"
        )

    # Normalize positions: trim symbols, deterministic sort
    ledger_normalized = ledger_positions_df.copy()
    broker_normalized = broker_positions_df.copy()

    # Trim symbol strings
    ledger_normalized["symbol"] = ledger_normalized["symbol"].astype(str).str.strip()
    broker_normalized["symbol"] = broker_normalized["symbol"].astype(str).str.strip()

    # Ensure qty is float
    ledger_normalized["qty"] = ledger_normalized["qty"].astype(float)
    broker_normalized["qty"] = broker_normalized["qty"].astype(float)

    # Remove zero positions (threshold: qty_tol) - filter both sides consistently
    ledger_normalized = ledger_normalized[
        ledger_normalized["qty"].abs() > qty_tol
    ].copy()
    broker_normalized = broker_normalized[
        broker_normalized["qty"].abs() > qty_tol
    ].copy()

    # Deterministic sort by symbol
    ledger_normalized = ledger_normalized.sort_values(
        "symbol", kind="mergesort"
    ).reset_index(drop=True)
    broker_normalized = broker_normalized.sort_values(
        "symbol", kind="mergesort"
    ).reset_index(drop=True)

    # Get symbol sets
    ledger_symbols = set(ledger_normalized["symbol"].unique())
    broker_symbols = set(broker_normalized["symbol"].unique())

    # Find missing symbols
    missing_in_ledger = sorted(list(broker_symbols - ledger_symbols))
    missing_in_broker = sorted(list(ledger_symbols - broker_symbols))

    # Check cash match
    cash_diff = ledger_cash - broker_cash
    cash_match = abs(cash_diff) <= cash_tol

    # Build position differences
    position_diffs_list = []

    # Compare common symbols
    common_symbols = sorted(list(ledger_symbols & broker_symbols))
    for symbol in common_symbols:
        ledger_filtered = ledger_normalized[ledger_normalized["symbol"] == symbol]
        broker_filtered = broker_normalized[broker_normalized["symbol"] == symbol]
        if ledger_filtered.empty or broker_filtered.empty:
            continue
        ledger_row = ledger_filtered.iloc[0]
        broker_row = broker_filtered.iloc[0]

        ledger_qty = float(ledger_row["qty"])
        broker_qty = float(broker_row["qty"])
        diff_qty = ledger_qty - broker_qty

        # Only include if difference exceeds tolerance
        if abs(diff_qty) > qty_tol:
            position_diffs_list.append(
                {
                    "symbol": symbol,
                    "ledger_qty": ledger_qty,
                    "broker_qty": broker_qty,
                    "diff_qty": diff_qty,
                }
            )

    # Build position_diffs_df
    if position_diffs_list:
        position_diffs_df = pd.DataFrame(position_diffs_list)
        position_diffs_df = position_diffs_df.sort_values(
            "symbol", kind="mergesort"
        ).reset_index(drop=True)
    else:
        position_diffs_df = pd.DataFrame(
            columns=["symbol", "ledger_qty", "broker_qty", "diff_qty"]
        )

    # Determine overall match
    positions_match = (
        len(position_diffs_list) == 0
        and len(missing_in_ledger) == 0
        and len(missing_in_broker) == 0
    )
    ok = cash_match and positions_match

    # Build message (include key differences for fail_fast)
    message_parts = []
    if not cash_match:
        message_parts.append(
            f"Cash mismatch: diff={cash_diff:.6f} (ledger={ledger_cash:.6f}, broker={broker_cash:.6f})"
        )
    if len(position_diffs_list) > 0:
        # Include all symbols with mismatches (not just first 5)
        mismatch_symbols = sorted([d["symbol"] for d in position_diffs_list])
        message_parts.append(
            f"Position qty mismatches: {len(position_diffs_list)} symbol(s): {mismatch_symbols}"
        )
    if len(missing_in_ledger) > 0:
        # missing_in_ledger is already sorted
        message_parts.append(
            f"Missing in ledger: {len(missing_in_ledger)} symbol(s): {missing_in_ledger}"
        )
    if len(missing_in_broker) > 0:
        # missing_in_broker is already sorted
        message_parts.append(
            f"Missing in broker: {len(missing_in_broker)} symbol(s): {missing_in_broker}"
        )

    if ok:
        message = "Reconciliation OK: cash and positions match"
    else:
        message = "Reconciliation FAILED: " + "; ".join(message_parts)

    # Fail-fast if requested
    if fail_fast and not ok:
        raise ValueError(message)

    return {
        "ok": ok,
        "cash_match": cash_match,
        "cash_diff": cash_diff,
        "position_diffs_df": position_diffs_df,
        "missing_in_ledger": missing_in_ledger,
        "missing_in_broker": missing_in_broker,
        "message": message,
    }


# ── Daily P&L Reconciliation (Plan 8.1) ──────────────────────────────


def reconcile_daily_pnl(
    positions: dict[str, float],
    position_prices_start: dict[str, float],
    position_prices_end: dict[str, float],
    portfolio_return: float,
    *,
    tolerance_pct: float = 0.001,
) -> dict:
    """Reconcile daily P&L: sum of position-level returns vs portfolio return.

    Checks that:
        ``sum(position_i × return_i) ≈ portfolio_return ± tolerance``

    If the two diverge, the break is flagged with diagnostic info.

    Args:
        positions: Dict mapping symbol → weight (or dollar exposure).
        position_prices_start: Symbol → start-of-day price.
        position_prices_end: Symbol → end-of-day price.
        portfolio_return: Reported portfolio return for the day.
        tolerance_pct: Maximum acceptable deviation (0.001 = 0.1%).

    Returns:
        Dict with ``ok``, ``explained_return``, ``unexplained_return``,
        ``break_pct``, ``position_contributions``.
    """
    contributions: dict[str, float] = {}
    # A symbol with missing start/end price or p_start==0 silently booked
    # contribution=0 before. For a held position that looks indistinguishable
    # from a flat-price day, so a price-feed gap on a moving symbol becomes
    # invisible "unexplained_return" that downstream attribution cannot
    # separate from fees / timing. Track which symbols were skipped so the
    # reason is surfaced and the break-analysis can be trusted.
    skipped_symbols: list[str] = []
    explained = 0.0

    # B-acct-4: a literal ``== 0`` on a float start price is fragile (a 1e-15
    # residual price would slip through and divide). Guard with an explicit
    # tolerance and treat a (near-)zero start price as a skip, NOT a divide.
    price_eps = 1e-12

    for sym, weight in positions.items():
        p_start = position_prices_start.get(sym)
        p_end = position_prices_end.get(sym)

        if p_start is None or p_end is None or abs(float(p_start)) <= price_eps:
            contributions[sym] = 0.0
            if abs(float(weight)) > 0:
                skipped_symbols.append(sym)
            continue

        sym_return = (p_end - p_start) / p_start
        contrib = weight * sym_return
        contributions[sym] = round(contrib, 8)
        explained += contrib

    unexplained = portfolio_return - explained
    break_pct = abs(unexplained)
    # B-acct-4: a price-feed gap on a HELD symbol (skipped above) is a degraded
    # reconciliation — its return is silently absent from ``explained``, so a
    # numerically-small break is not trustworthy. Surface a distinct ``degraded``
    # flag AND downgrade ``ok`` so a feed gap can never be read as a clean pass.
    degraded = len(skipped_symbols) > 0
    ok = (break_pct <= tolerance_pct) and not degraded

    # Break reason analysis
    break_reason = ""
    if abs(unexplained) > 0.01:
        break_reason = (
            "LARGE_BREAK: possible missing position, corporate action, or fee"
        )
    elif abs(unexplained) > tolerance_pct:
        break_reason = "MINOR_BREAK: rounding, timing, or cash drag"
    elif degraded:
        # ok was forced False purely by a price-feed gap, not a numeric break.
        break_reason = (
            "DEGRADED: price-feed gap on held symbol(s); attribution incomplete"
        )

    result = {
        "ok": ok,
        "explained_return": round(explained, 8),
        "unexplained_return": round(unexplained, 8),
        "break_pct": round(break_pct, 8),
        "portfolio_return": round(portfolio_return, 8),
        "tolerance_pct": tolerance_pct,
        "break_reason": break_reason,
        "position_contributions": contributions,
        "skipped_symbols": skipped_symbols,
        "degraded": degraded,
    }

    if not ok:
        logger.warning(
            "[Reconcile] P&L break: explained=%.6f, portfolio=%.6f, break=%.6f%% (%s)",
            explained,
            portfolio_return,
            break_pct * 100,
            break_reason,
        )
    if skipped_symbols:
        logger.warning(
            "[Reconcile] %d held-position symbol(s) skipped due to missing "
            "start/end price — attribution for these positions is missing "
            "from explained_return: %s",
            len(skipped_symbols),
            skipped_symbols,
        )

    return result
